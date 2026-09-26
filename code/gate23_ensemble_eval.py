"""Gate 23: does the generative model capture conformational heterogeneity? (EigenFold / SimpleFold protocol)

For each target of an apo/holo or fold-switch set (gate22 files): sample K structures from the sequence,
then report
  TM_A, TM_B     mean over targets of the best-of-K TM to state A / state B (Foldseek TM-align, exhaustive)
  TM-ens         mean over targets of the mean over the two states of the best-of-K TM  (SimpleFold Tab. 3)
  TM-single      TM of the first sample to its nearest state (accuracy of one draw)
  flex global    Pearson r between per-residue sample RMSF (Kabsch-aligned to the sample mean) and the
                 per-residue |A - B| deviation (states aligned on shared residues), pooled over targets
  flex per-tgt   mean over targets of the per-target Pearson r
  diversity      mean pairwise Kabsch TM-like score between samples (lower = more diverse)
  coverage       fraction of (target, state) pairs with a Foldseek hit
Run at several CFG weights: lower w = more diverse samples.
  ESM_PROAE_MAX_LEN=512 python gate23_ensemble_eval.py --ckpt last_pf_459M_p128x8_long512_scratch.ckpt --set apo --k 5 --cfg-w 1,2
"""
import os, sys, json, time, argparse, tempfile, shutil, numpy as np, torch, h5py
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/code"); os.chdir(ROOT + "/ProteinAE_v1"); sys.path.insert(0, ".")
import gate7_latent_flow as G
from gate6_fape_train import PROJECT, _write_pseudo_backbone_pdb, _foldseek_tm
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--set", default="apo"); ap.add_argument("--k", type=int, default=5)
ap.add_argument("--cfg-w", default="1,2"); ap.add_argument("--steps", type=int, default=50); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--n", type=int, default=0)
a = ap.parse_args(); dev = torch.device("cuda"); D = PROJECT / "data/phase1_dataset"

def kabsch_tm(P, Q, mask):
    m = mask.unsqueeze(-1).double(); L = m.sum(1); p = P.double() * m; q = Q.double() * m
    p = p - p.sum(1, keepdim=True) / L.unsqueeze(-1); q = q - q.sum(1, keepdim=True) / L.unsqueeze(-1); p, q = p * m, q * m
    U, S, Vt = torch.linalg.svd(p.transpose(1, 2) @ q); d = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    Dm = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1)); R = Vt.transpose(1, 2) @ Dm @ U.transpose(1, 2)
    pr = (R @ p.transpose(1, 2)).transpose(1, 2); dist2 = ((pr - q) ** 2).sum(-1)
    d0 = (1.24 * (L.squeeze(-1) - 15).clamp(min=1) ** (1 / 3) - 1.8).clamp(min=0.5)
    return ((1 / (1 + dist2 / d0.unsqueeze(-1) ** 2) * mask.double()).sum(1) / L.squeeze(-1)).float()

def kabsch_align(P, Q):
    """Rotate/translate P (n,3) onto Q (n,3); returns aligned P."""
    pc, qc = P.mean(0), Q.mean(0); p, q = P - pc, Q - qc
    U, S, Vt = np.linalg.svd(p.T @ q); d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return (R @ p.T).T + qc

# ---- model ----
ck = D / a.ckpt
if a.ckpt.endswith(".ckpt"):
    st = torch.load(str(ck), weights_only=False, map_location="cpu"); arch, weights, ex = st["arch"], st["ema"], dict(st.get("extra_arch") or {}); model = st.get("model")
else:
    meta = json.load(open(str(ck) + ".meta.json")); arch = {k: meta[k] for k in ("d_model", "n_layers", "n_heads", "dropout", "self_cond", "d_cond")}
    weights = torch.load(str(ck), weights_only=True); ex = {k: meta[k] for k in ("d_pair", "n_pair_blocks", "pair_contact", "pair_fused") if k in meta}; model = meta.get("model")
for k in ("recycle", "p_rec", "rec_every"): ex.pop(k, None)
if model in ("PairFlowNet", "_Net") or ex:
    import gate10_pair_flow as G10; G10.install(); net = G10.PairFlowNet(**arch, **ex).to(dev)
else:
    net = G.LatentFlowNet(**arch).to(dev)
net.load_state_dict(G.extend_pos_table(G.adapt_state_dict(weights, net.state_dict().keys()), net.state_dict())); net.eval()
dec = G.load_decoder(dev)

# ---- data ----
base = h5py.File(str(D / f"dataset_{a.set}.h5"), "r")["val"]; emb = h5py.File(str(D / f"dataset_{a.set}_esmc.h5"), "r")["val"]
names = sorted(base.keys())[: a.n or None]; print(f"{a.set}: {len(names)} targets, K={a.k}, {a.steps} steps", flush=True)
results = {}
for w in [float(x) for x in a.cfg_w.split(",")]:
    work = tempfile.mkdtemp(prefix=f"ens_{a.set}_"); t0 = time.perf_counter()
    for st_ in ("A", "B"): os.makedirs(f"{work}/gt{st_}")
    for k in range(a.k): os.makedirs(f"{work}/pr{k}")
    gen = torch.Generator(device="cuda").manual_seed(0); per = {}
    with torch.no_grad():
        for s in range(0, len(names), a.bs):
            nb = names[s:s + a.bs]; Lmax = max(len(base[n].attrs["sequence"]) for n in nb); Lmax = (Lmax + 7) // 8 * 8
            esm = torch.zeros(len(nb), Lmax, 2560, dtype=torch.float16); mask = torch.zeros(len(nb), Lmax, dtype=torch.bool)
            for i, n in enumerate(nb):
                e = emb[n]["esm2_emb"][:]; esm[i, :len(e)] = torch.from_numpy(e.astype(np.float16)); mask[i, :len(e)] = True
            esm, mask = esm.to(dev), mask.to(dev); samples = []
            for k in range(a.k):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    z = G.sample(net, esm, mask, a.steps, w, gen); ca = dec(z.float(), mask).float()
                samples.append(ca.cpu())
                for i, n in enumerate(nb):
                    L = int(mask[i].sum()); _write_pseudo_backbone_pdb(ca[i, :L].cpu().numpy(), f"{work}/pr{k}/{n}.pdb")
            S = torch.stack(samples, 1)  # (B, K, L, 3)
            for i, n in enumerate(nb):
                g = base[n]; L = int(mask[i].sum()); idxA = g["seqidx_A"][:]; caA = g["ca_A"][:]; okA = idxA >= 0
                _write_pseudo_backbone_pdb(caA[okA], f"{work}/gtA/{n}.pdb")
                rec = {"L": L}
                if "seqidx_B" in g:
                    idxB = g["seqidx_B"][:]; caB = g["ca_B"][:]; okB = idxB >= 0; _write_pseudo_backbone_pdb(caB[okB], f"{work}/gtB/{n}.pdb")
                    # state deviation on residues observed in both, after superposition
                    posA = {int(p): j for j, p in enumerate(idxA) if p >= 0}; posB = {int(p): j for j, p in enumerate(idxB) if p >= 0}
                    common = sorted(set(posA) & set(posB))
                    if len(common) >= 20:
                        A_ = caA[[posA[p] for p in common]]; B_ = caB[[posB[p] for p in common]]; B_al = kabsch_align(B_, A_)
                        dev_ab = np.linalg.norm(A_ - B_al, axis=1)
                        # sample RMSF on the same residues: align each sample to the mean iteratively (2 passes)
                        X = S[i, :, :L].numpy()[:, common]; ref = X[0]
                        for _ in range(2):
                            X = np.stack([kabsch_align(x, ref) for x in X]); ref = X.mean(0)
                        rmsf = np.sqrt(((X - ref) ** 2).sum(-1).mean(0))
                        rec["dev_ab"] = dev_ab.tolist(); rec["rmsf"] = rmsf.tolist()
                        rec["flex_r"] = float(np.corrcoef(rmsf, dev_ab)[0, 1]) if np.std(rmsf) > 1e-6 and np.std(dev_ab) > 1e-6 else float("nan")
                mm = mask[i:i+1, :L].cpu().repeat(a.k * (a.k - 1) // 2, 1); P, Q = [], []
                for x in range(a.k):
                    for y in range(x + 1, a.k): P.append(S[i, x, :L]); Q.append(S[i, y, :L])
                rec["diversity"] = float(kabsch_tm(torch.stack(P), torch.stack(Q), mm).mean()) if P else float("nan")
                per[n] = rec
    tmA = [_foldseek_tm(f"{work}/pr{k}", f"{work}/gtA", f"{work}/tmA{k}") for k in range(a.k)]
    tmB = [_foldseek_tm(f"{work}/pr{k}", f"{work}/gtB", f"{work}/tmB{k}") for k in range(a.k)] if os.listdir(f"{work}/gtB") else None
    shutil.rmtree(work, ignore_errors=True)
    bestA = np.array([max(t.get(n, 0.0) for t in tmA) for n in names]); firstA = np.array([tmA[0].get(n, 0.0) for n in names])
    cov = np.mean([n in tmA[0] for n in names]); out = {"n": len(names), "k": a.k, "cfg_w": w, "TM_A_best": float(bestA.mean()), "TM_A_first": float(firstA.mean()), "coverage": float(cov),
                                                          "diversity_pairwise_tm": float(np.nanmean([per[n]["diversity"] for n in names]))}
    if tmB:
        bestB = np.array([max(t.get(n, 0.0) for t in tmB) for n in names]); firstB = np.array([tmB[0].get(n, 0.0) for n in names])
        out.update({"TM_B_best": float(bestB.mean()), "TM_ens": float(((bestA + bestB) / 2).mean()), "TM_single_nearest": float(np.maximum(firstA, firstB).mean()),
                    "flex_r_per_target": float(np.nanmean([per[n].get("flex_r", np.nan) for n in names])),
                    "flex_r_global": float(np.corrcoef(np.concatenate([per[n]["rmsf"] for n in names if "rmsf" in per[n]]), np.concatenate([per[n]["dev_ab"] for n in names if "rmsf" in per[n]]))[0, 1]),
                    "coverage_B": float(np.mean([n in tmB[0] for n in names]))})
    print(f"w={w}: " + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in out.items()) + f"  [{time.perf_counter()-t0:.0f}s]", flush=True)
    results[str(w)] = out
json.dump({"ckpt": a.ckpt, "set": a.set, "results": results}, open(PROJECT / "notes" / f"gate23_{a.set}_{a.ckpt.replace('.ckpt','').replace('.pt','')}.json", "w"), indent=1)
