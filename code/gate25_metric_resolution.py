"""Gate 25: can single-sample TM against ONE reference resolve a 0.02 difference between methods?

Three measurements, no GPU:
  1. paired comparison of our per-protein TM against ESMFold2-Fast's on the same CASP domains:
     mean difference, bootstrap 95 % CI, sign test, and per-target spread.
  2. reference-vs-reference TM: two experimental structures of the SAME protein (apo/holo and
     fold-switch pairs), superposed on their corresponding residues. This is the intrinsic width
     of "the correct answer" in the same units the benchmark reports.
  3. our own sample-to-sample spread on the same targets (already stored as best-of-K minus mean).
"""
import os, sys, json, argparse, numpy as np, h5py
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/code")
N_ = ROOT + "/notes"; D = ROOT + "/data/phase1_dataset"
ap = argparse.ArgumentParser()
ap.add_argument("--ours", default="tm_last_pf_840M_p128x8_long512_1p3M_sweep_named_casp_domains_le256.json")
ap.add_argument("--theirs", default="gate12_esmfold2_fast_casp.json")
ap.add_argument("--set-key", default="casp_domains_le256")
ap.add_argument("--names", default="casp_domains_le256.txt")
ap.add_argument("--boot", type=int, default=20000)
ap.add_argument("--ours-h5", default=None, help="derive the ours-ordering as the seed-42 permutation of this HDF5's val split (for --offset scores with no stored names)")
ap.add_argument("--ours-offset", type=int, default=0)
ap.add_argument("--skip-refs", action="store_true")
a = ap.parse_args()
rng = np.random.default_rng(0)

def tm_score(P, Q):
    """TM-score with a FIXED residue correspondence, optimal rigid superposition (Kabsch).
    Normalised by the number of corresponding residues, d0 as in TM-align."""
    P = P - P.mean(0); Q = Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(U @ Vt))
    Rm = U @ np.diag([1, 1, d]) @ Vt
    Pa = P @ Rm
    L = len(P)
    d0 = 1.24 * (max(L, 19) - 15) ** (1 / 3) - 1.8
    di = np.linalg.norm(Pa - Q, axis=1)
    return float(np.mean(1.0 / (1.0 + (di / d0) ** 2)))

# ---------- 1. paired model comparison ----------
if a.ours_h5:
    import torch, h5py
    hh = h5py.File(a.ours_h5 if os.path.isabs(a.ours_h5) else f"{D}/{a.ours_h5}", "r")["val"]
    allnames = list(hh.keys())
    torch.manual_seed(42); perm = torch.randperm(len(allnames)).tolist()
    names = [allnames[i] for i in perm][a.ours_offset:]
else:
    names = [l.strip() for l in open(f"{N_}/{a.names}") if l.strip()]
ours = json.load(open(f"{N_}/{a.ours}"))
row = ours["rows"][sorted(ours["rows"], key=lambda w: abs(float(w) - 2.0))[0]]
mine = np.array(row["tm_per_protein"], dtype=float)
names = names[:len(mine)]
assert len(mine) == len(names), f"{len(mine)} scores vs {len(names)} names"
theirs_j = json.load(open(f"{N_}/{a.theirs}"))["per_protein"]
keep = [i for i, n in enumerate(names) if n in theirs_j]
mine = mine[keep]; nm = [names[i] for i in keep]
theirs = np.array([theirs_j[n]["tm"] for n in nm], dtype=float)
diff = theirs - mine
bs = np.array([np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(a.boot)])
lo, hi = np.percentile(bs, [2.5, 97.5])
wins = int((diff < 0).sum()); losses = int((diff > 0).sum())
print(f"== paired comparison on {len(nm)} targets ({a.set_key})")
print(f"  ours {mine.mean():.4f}   ESMFold2-Fast {theirs.mean():.4f}   mean difference {diff.mean():+.4f}")
print(f"  bootstrap 95 % CI of the difference: [{lo:+.4f}, {hi:+.4f}]  (excludes 0: {not (lo < 0 < hi)})")
print(f"  per-target difference sd {diff.std(ddof=1):.3f}, sem {diff.std(ddof=1)/np.sqrt(len(diff)):.4f}")
from math import comb
n_eff = wins + losses
p_sign = min(1.0, 2 * sum(comb(n_eff, i) for i in range(0, min(wins, losses) + 1)) / 2 ** n_eff)
print(f"  targets we win {wins}, lose {losses}  (exact sign test p = {p_sign:.4f})")
print(f"  targets where |difference| > 0.10: {int((np.abs(diff)>0.10).sum())} of {len(diff)}")
print(f"  our best-of-{row['k']} {row['tm_best_of_k']:.4f} (oracle headroom {row['tm_best_of_k']-mine.mean():+.4f})")

# ---------- 2. reference-vs-reference ----------
print("\n== intrinsic width of 'the correct answer': two experimental structures, same protein")
for tag in ([] if a.skip_refs else ("apo", "codnas")):
    p = f"{D}/dataset_{tag}.h5"
    if not os.path.exists(p): print(f"  {tag}: missing"); continue
    h = h5py.File(p, "r")["val"]; vals = []; lens = []
    for k in h:
        g = h[k]
        ia, ib = g["seqidx_A"][:], g["seqidx_B"][:]
        ca, cb = g["ca_A"][:], g["ca_B"][:]
        common = np.intersect1d(ia, ib)
        if len(common) < 20: continue
        ma = {int(v): i for i, v in enumerate(ia)}; mb = {int(v): i for i, v in enumerate(ib)}
        A = ca[[ma[int(c)] for c in common]]; B = cb[[mb[int(c)] for c in common]]
        vals.append(tm_score(A, B)); lens.append(len(common))
    v = np.array(vals)
    print(f"  {tag}: n={len(v)}  A-vs-B TM mean {v.mean():.3f}  median {np.median(v):.3f}  "
          f"10th pct {np.percentile(v,10):.3f}  fraction < 0.9 {np.mean(v<0.9):.2f}  mean residues {np.mean(lens):.0f}")
    print(f"        1 - mean = {1-v.mean():.3f}: the reference itself moves this much between experiments")
