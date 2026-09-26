"""Gate 24: evaluate the co-design model (gate20) in its three modes on validation proteins.

  fold          t_seq = 1 (full sequence): standard latent sampling -> TM to truth (sanity: the joint model still folds)
  inverse fold  clean latent (t_str = 1) + fully masked sequence: iterative unmasking (--unmask-steps) ->
                native sequence recovery; then the designed sequence is folded with the SAME model and
                compared with the truth's decoded structure (scTM, self-consistency).
  co-design     all-masked sequence + noise latent; 25 Euler structure steps with the sequence unmasked
                linearly along the way (ESMC re-embeds after each unmasking); the final sequence is folded
                from scratch with the model and compared with the co-designed structure (scTM), plus
                pairwise-TM diversity among designs and native-sequence identity (should be low).
Designed sequences are written to notes/<label>_designs.fasta for an external folder (ESMFold2) check.
  python gate24_codesign_eval.py --ckpt best_cd_174M_p64x6_esmc.pt --n 100
"""
import os, sys, json, time, argparse, tempfile, shutil, numpy as np, torch, torch.nn.functional as F, h5py
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/code"); os.chdir(ROOT + "/ProteinAE_v1"); sys.path.insert(0, ".")
import gate7_latent_flow as G, gate10_pair_flow as G10, gate20_codesign as G20
from gate6_fape_train import PROJECT, _write_pseudo_backbone_pdb, _foldseek_tm
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=100); ap.add_argument("--offset", type=int, default=1000)
ap.add_argument("--steps", type=int, default=25); ap.add_argument("--cfg-w", type=float, default=2.0); ap.add_argument("--unmask-steps", type=int, default=10); ap.add_argument("--bs", type=int, default=10)
a = ap.parse_args(); dev = torch.device("cuda"); D = PROJECT / "data/phase1_dataset"
meta = json.load(open(str(D / a.ckpt) + ".meta.json"))
arch = {k: meta[k] for k in ("d_model", "n_layers", "n_heads", "dropout", "self_cond", "d_cond")}
ex = {k: meta[k] for k in ("d_pair", "n_pair_blocks", "pair_contact", "pair_fused") if k in meta}
G20.install(); net = G20.CodesignNet(**arch, **ex).to(dev)
net.load_state_dict(G.adapt_state_dict(torch.load(str(D / a.ckpt), weights_only=True), net.state_dict().keys())); net.eval()
dec = G.load_decoder(dev); emb = G.OnlineESM(meta.get("esm_path", str(PROJECT / "data/esmc6b")), layer_mix=False, device=dev, kind="esmc")
AA = G20.AA
h = h5py.File(str(D / "dataset_100k_esmc.h5"), "r")["val"]; names = list(h.keys()); torch.manual_seed(42); idx = torch.randperm(len(names))[a.offset:a.offset + a.n].tolist()
targets = [names[i] for i in idx]; print(f"{len(targets)} validation proteins, {a.steps} structure steps, {a.unmask_steps} unmasking steps", flush=True)

@torch.no_grad()
def embed_masked(seqs, masks, L):
    """masks: list of bool arrays (True = masked)."""
    strs = ["".join(emb.tok.mask_token if m else c for c, m in zip(s, mk)) for s, mk in zip(seqs, masks)]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16): return emb(strs, L=L)

@torch.no_grad()
def fold(seqs, L):
    mask = torch.zeros(len(seqs), L, dtype=torch.bool, device=dev)
    for i, s in enumerate(seqs): mask[i, :len(s)] = True
    e = embed_masked(seqs, [np.zeros(len(s), bool) for s in seqs], L)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        z = G.sample(net, e, mask, a.steps, a.cfg_w); ca = dec(z.float(), mask).float()
    return ca.cpu(), mask.cpu()

@torch.no_grad()
def unmask_round(x_t, t, e, mask, cur, msk, frac):
    """One unmasking step: predict logits, unmask the `frac` most confident masked positions per protein."""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _, logits = net(x_t, t, e, mask, None, x_t if net.self_cond else None, return_logits=True)
    prob = torch.softmax(logits.float(), -1); conf, pred = prob.max(-1)
    for i in range(len(cur)):
        pos = np.where(msk[i])[0]
        if len(pos) == 0: continue
        n_un = max(1, int(round(frac * msk[i].sum())))
        order = pos[np.argsort(-conf[i, pos].cpu().numpy())][:n_un]
        s = list(cur[i])
        for p in order: s[p] = AA[int(pred[i, p])]; msk[i][p] = False
        cur[i] = "".join(s)
    return cur, msk

work = tempfile.mkdtemp(prefix="cd_"); dirs = {k: os.path.join(work, k) for k in ("truth", "fold", "inv_fold", "cd_struct", "cd_fold")}
for d in dirs.values(): os.makedirs(d)
rec = {}
with torch.no_grad():
    for s in range(0, len(targets), a.bs):
        nb = targets[s:s + a.bs]; seqs = [str(h[n].attrs["sequence"]) for n in nb]; L = ((max(len(x) for x in seqs) + 7) // 8) * 8
        mask = torch.zeros(len(nb), L, dtype=torch.bool, device=dev); z_true = torch.zeros(len(nb), L, 8, device=dev)
        for i, n in enumerate(nb): mask[i, :len(seqs[i])] = True; z_true[i, :len(seqs[i])] = torch.from_numpy(h[n]["z"][:]).to(dev)
        ca_true = dec(z_true, mask).float().cpu()                                   # decoded truth (0.999 to the real structure)
        # 1. fold
        ca_f, _ = fold(seqs, L)
        # 2. inverse fold from the clean latent
        cur = ["".join("A" for _ in x) for x in seqs]; msk = [np.ones(len(x), bool) for x in seqs]; t1 = torch.ones(len(nb), device=dev)
        for r in range(a.unmask_steps):
            e = embed_masked(cur, msk, L); cur, msk = unmask_round(z_true, t1, e, mask, cur, msk, 1.0 / (a.unmask_steps - r))
        inv = cur; rec_id = [np.mean([x == y for x, y in zip(d_, s_)]) for d_, s_ in zip(inv, seqs)]
        ca_inv, _ = fold(inv, L)                                                     # fold the designed sequence back
        # 3. co-design: noise latent + all masked, unmask linearly during the structure ODE
        cur = ["".join("A" for _ in x) for x in seqs]; msk = [np.ones(len(x), bool) for x in seqs]
        x = torch.randn(len(nb), L, 8, device=dev); x_sc = None; ts = torch.linspace(0, 1, a.steps + 1, device=dev); e = embed_masked(cur, msk, L)
        every = max(1, a.steps // a.unmask_steps); ones = torch.ones(len(nb), dtype=torch.bool, device=dev)
        for i in range(a.steps):
            t = ts[i].expand(len(nb)); dt = ts[i + 1] - ts[i]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                v = net(x, t, e, mask, None, x_sc); v_u = net(x, t, e, mask, ones, x_sc); v = v_u + a.cfg_w * (v - v_u)
            if net.self_cond: x_sc = x + (1 - ts[i]) * v
            x = x + v * dt
            if (i + 1) % every == 0 and any(m.any() for m in msk):
                remaining = max(1, (a.steps - i - 1) // every + 1)
                cur, msk = unmask_round(x_sc if x_sc is not None else x, ts[i + 1].expand(len(nb)), e, mask, cur, msk, 1.0 / remaining); e = embed_masked(cur, msk, L)
        z_cd = F.layer_norm(x, (8,)) * mask.unsqueeze(-1); ca_cd = dec(z_cd, mask).float().cpu(); cd_seq = cur
        ca_cdf, _ = fold(cd_seq, L)                                                  # refold the co-designed sequence
        for i, n in enumerate(nb):
            Ln = len(seqs[i])
            _write_pseudo_backbone_pdb(ca_true[i, :Ln].numpy(), f"{dirs['truth']}/{n}.pdb"); _write_pseudo_backbone_pdb(ca_f[i, :Ln].numpy(), f"{dirs['fold']}/{n}.pdb")
            _write_pseudo_backbone_pdb(ca_inv[i, :Ln].numpy(), f"{dirs['inv_fold']}/{n}.pdb"); _write_pseudo_backbone_pdb(ca_cd[i, :Ln].numpy(), f"{dirs['cd_struct']}/{n}.pdb"); _write_pseudo_backbone_pdb(ca_cdf[i, :Ln].numpy(), f"{dirs['cd_fold']}/{n}.pdb")
            rec[n] = {"len": Ln, "inv_recovery": float(rec_id[i]), "inv_seq": inv[i], "cd_seq": cd_seq[i], "cd_native_identity": float(np.mean([x == y for x, y in zip(cd_seq[i], seqs[i])]))}
        print(f"  {s+len(nb)}/{len(targets)}", flush=True)
tm_fold = _foldseek_tm(dirs["fold"], dirs["truth"], f"{work}/t1"); tm_inv = _foldseek_tm(dirs["inv_fold"], dirs["truth"], f"{work}/t2"); tm_cd = _foldseek_tm(dirs["cd_fold"], dirs["cd_struct"], f"{work}/t3")
tm_cd_truth = _foldseek_tm(dirs["cd_struct"], dirs["truth"], f"{work}/t4")
shutil.rmtree(work, ignore_errors=True)
out = {"n": len(targets), "fold_TM": float(np.mean([tm_fold.get(n, 0) for n in targets])), "fold_cov": float(np.mean([n in tm_fold for n in targets])),
       "inverse_fold_recovery": float(np.mean([rec[n]["inv_recovery"] for n in targets])), "inverse_fold_scTM": float(np.mean([tm_inv.get(n, 0) for n in targets])),
       "codesign_scTM": float(np.mean([tm_cd.get(n, 0) for n in targets])), "codesign_scTM_gt0.5": float(np.mean([tm_cd.get(n, 0) > 0.5 for n in targets])),
       "codesign_native_identity": float(np.mean([rec[n]["cd_native_identity"] for n in targets])), "codesign_TM_to_native_structure": float(np.mean([tm_cd_truth.get(n, 0) for n in targets]))}
print(json.dumps(out, indent=1))
label = a.ckpt.replace("best_", "").replace(".pt", "")
with open(PROJECT / "notes" / f"{label}_designs.fasta", "w") as f:
    for n in targets: f.write(f">{n}|inverse_fold\n{rec[n]['inv_seq']}\n>{n}|codesign\n{rec[n]['cd_seq']}\n")
json.dump({"ckpt": a.ckpt, "summary": out, "per_protein": rec}, open(PROJECT / "notes" / f"gate24_{label}.json", "w"), indent=1)
