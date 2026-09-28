"""Gate 29: the designability loop. Does a structure we generated in latent space correspond
to a real, foldable protein?

  latent  ->  Ca structure (frozen ProteinAE decoder, from gate28)
          ->  sequence     (our co-design head, inverse folding at temperature 1)
          ->  structure    (ESMFold2-Fast, an INDEPENDENT model that never saw our latent)
          ->  TM back to the structure we started from   = scTM

scTM is only interpretable against a control, because our inverse-folding head is weak (0.18
native recovery). So the same loop runs on the REAL experimental structures of the same
proteins: whatever a real protein scores through this pipeline is the ceiling, and a generated
structure that matches it is as designable as a real one.

  python gate29_designability.py --modes real,uncond --max-len 256
"""
import os, sys, json, argparse, tempfile, shutil, time
import numpy as np, torch, torch.nn.functional as F
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/code")
import gate7_latent_flow as G
from gate6_fape_train import PROJECT, _write_pseudo_backbone_pdb, _foldseek_tm

ap = argparse.ArgumentParser()
ap.add_argument("--cd-ckpt", default="best_cd_174M_p64x6_esmc.pt")
ap.add_argument("--in-dir", default=str(PROJECT / "notes" / "gate28"))
ap.add_argument("--modes", default="real,uncond")
ap.add_argument("--max-len", type=int, default=256, help="the co-design head was trained at a 256 window")
ap.add_argument("--unmask-steps", type=int, default=10); ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--bs", type=int, default=8); ap.add_argument("--esmfold", default=str(PROJECT / "data/esmfold2_fast"))
ap.add_argument("--loops", type=int, default=3); ap.add_argument("--fold-steps", type=int, default=50)
a = ap.parse_args(); dev = torch.device("cuda")
D = PROJECT / "data/phase1_dataset"

# ---- co-design model (inverse folding) ----
import gate20_codesign as G20
meta = json.loads(open(str(D / a.cd_ckpt) + ".meta.json").read())
arch = {k: meta[k] for k in ("d_model", "n_layers", "n_heads", "dropout", "self_cond", "d_cond")}
ex = {k: meta[k] for k in ("d_pair", "n_pair_blocks", "pair_contact", "pair_fused") if k in meta}
G20.install(); net = G20.CodesignNet(**arch, **ex).to(dev)
net.load_state_dict(G.adapt_state_dict(torch.load(str(D / a.cd_ckpt), weights_only=True), net.state_dict().keys())); net.eval()
emb = G.OnlineESM(meta.get("esm_path", str(PROJECT / "data/esmc6b")), layer_mix=False, device=dev, kind="esmc")
AA = G20.AA
print(f"inverse-folding head {a.cd_ckpt} (epoch {meta['epoch']}), temperature {a.temp}", flush=True)

@torch.no_grad()
def embed_masked(seqs, masks, L):
    strs = ["".join(emb.tok.mask_token if m else c for c, m in zip(s, mk)) for s, mk in zip(seqs, masks)]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16): return emb(strs, L=L)

@torch.no_grad()
def inverse_fold(z, mask):
    """Clean latent (t = 1) + fully masked sequence -> iterative confidence-ordered unmasking."""
    B, L = mask.shape
    cur = ["A" * int(mask[i].sum()) for i in range(B)]
    msk = [np.ones(int(mask[i].sum()), bool) for i in range(B)]
    t1 = torch.ones(B, device=dev)
    for r in range(a.unmask_steps):
        e = embed_masked(cur, msk, L)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, logits = net(z, t1, e, mask, None, z if net.self_cond else None, return_logits=True)
        prob = torch.softmax(logits.float(), -1); conf, pred = prob.max(-1)
        if a.temp > 0:
            ps = torch.softmax(logits.float() / a.temp, -1)
            pred = torch.multinomial(ps.reshape(-1, ps.shape[-1]), 1).reshape(ps.shape[:-1])
        remaining = a.unmask_steps - r
        for i in range(B):
            pos = np.where(msk[i])[0]
            if len(pos) == 0: continue
            k = max(1, int(round(msk[i].sum() / remaining)))
            order = pos[np.argsort(-conf[i, pos].cpu().numpy())][:k]
            s = list(cur[i])
            for p in order: s[p] = AA[int(pred[i, p])]; msk[i][p] = False
            cur[i] = "".join(s)
    return cur

# ---- collect the structures gate28 produced ----
modes = [m.strip() for m in a.modes.split(",")]
work = tempfile.mkdtemp(prefix="g29_")
results = {}
for mode in modes:
    dirp = f"{a.in_dir}/{mode}"
    npz = f"{dirp}/latents.npz"
    if not os.path.exists(npz): print(f"  {mode}: no latents.npz, skipped", flush=True); continue
    Z = np.load(npz)
    names = [n for n in Z.files if Z[n].shape[0] <= a.max_len]
    print(f"\n== {mode}: {len(names)} of {len(Z.files)} structures at <= {a.max_len} residues", flush=True)
    if not names: continue
    seqs = {}
    for s0 in range(0, len(names), a.bs):
        nb = names[s0:s0 + a.bs]
        Ls = [Z[n].shape[0] for n in nb]; L = max(Ls)
        z = torch.zeros(len(nb), L, G.D_LAT, device=dev); mask = torch.zeros(len(nb), L, dtype=torch.bool, device=dev)
        for i, n in enumerate(nb):
            z[i, :Ls[i]] = torch.from_numpy(Z[n]).to(dev); mask[i, :Ls[i]] = True
        out = inverse_fold(z, mask)
        for i, n in enumerate(nb): seqs[n] = out[i][:Ls[i]]
        print(f"  designed {min(s0+a.bs, len(names))}/{len(names)}", flush=True)
    results[mode] = {"seqs": seqs, "dir": dirp}
    with open(PROJECT / "notes" / f"gate29_{mode}_designs.fasta", "w") as f:
        for n, s in seqs.items(): f.write(f">{n}\n{s}\n")

del net, emb; torch.cuda.empty_cache()

# ---- fold the designed sequences with ESMFold2-Fast ----
from transformers import EsmFold2Model
from transformers.models.esmfold2.protein_utils import prepare_protein_features, output_to_pdb
t0 = time.perf_counter()
fold = EsmFold2Model.from_pretrained(a.esmfold, dtype=torch.float32).eval().to(dev)
print(f"\nloaded ESMFold2-Fast in {time.perf_counter()-t0:.0f}s", flush=True)

summary = {}
for mode, r in results.items():
    od = f"{work}/{mode}_refold"; os.makedirs(od, exist_ok=True)
    comp = {}
    for k, (n, s) in enumerate(r["seqs"].items()):
        with torch.no_grad():
            feats = prepare_protein_features(s, device=dev)
            out = fold.fold(**feats, num_loops=a.loops, num_sampling_steps=a.fold_steps, num_diffusion_samples=1)
        pdb = output_to_pdb(out, feats)
        ca = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in pdb.splitlines()
                       if l.startswith("ATOM") and l[12:16].strip() == "CA"], dtype=np.float32)
        if ca.shape[0] != len(s): continue
        _write_pseudo_backbone_pdb(ca, f"{od}/{n}.pdb"); comp[n] = True
        if (k + 1) % 20 == 0: print(f"  {mode}: folded {k+1}/{len(r['seqs'])}", flush=True)
    tm = _foldseek_tm(od, r["dir"], f"{work}/w_{mode}")
    v = [tm.get(n, 0.0) for n in comp]
    summary[mode] = {"n": len(comp), "scTM": float(np.mean(v)), "scTM_gt0.5": float(np.mean([x > 0.5 for x in v])),
                     "coverage": float(np.mean([n in tm for n in comp]))}
    print(f"  {mode}: scTM {summary[mode]['scTM']:.3f}, {summary[mode]['scTM_gt0.5']:.0%} above 0.5, coverage {summary[mode]['coverage']:.2f}", flush=True)

print(f"\n{'mode':>10s} {'n':>4s} {'scTM':>7s} {'>0.5':>6s}")
for m, s in summary.items(): print(f"{m:>10s} {s['n']:4d} {s['scTM']:7.3f} {s['scTM_gt0.5']:6.2f}")
print("\nread scTM against the 'real' row: that is what this pipeline scores on genuine structures")
json.dump(summary, open(PROJECT / "notes" / "gate29_designability.json", "w"), indent=1)
shutil.rmtree(work, ignore_errors=True)
print("ALLDONE")
