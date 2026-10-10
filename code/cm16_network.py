"""cm16: does the complex-level effect of cm12 extend to the wider physical interaction network?

cm12's weakness is coverage: only 2,858 of 11,570 proteins are in a Complex Portal complex with a measured
partner, so the result is measured on a quarter of the data and says nothing about the rest. This repeats the
test on STRING v12 *physical* links -- the subnetwork of proteins reported to be in the same physical complex,
rather than merely associated -- at several confidence cutoffs, which trades specificity for coverage.

Caveats carried through to the report: STRING's physical scores include evidence transferred from other organisms
and from curated databases, so the graph is not independent of Complex Portal, and a high-degree hub's partner
mean is an average over a large, heterogeneous set. The point of the comparison is coverage, not a cleaner graph.

Design is identical to cm12 and deliberately the conservative variant throughout: a protein's partner mean is
taken only from TRAINING-fold proteins, so no held-out measurement is used and no partner can be in the test
protein's own 30 % identity cluster. Null by shuffling residuals within fold.

Writes data/campaign/cm16_network.json and reports/figures/campaign/network_residual.png.
"""
import os, re, csv, gzip, json
import numpy as np, pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
from pdpipe import plots as pl

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae"); C = f"{ROOT}/data/campaign"
FIG = f"{ROOT}/reports/figures/campaign"; SD = f"{ROOT}/data/external/string"
meta = json.load(open(f"{C}/targets_meta.json")); Tz = np.load(f"{C}/targets.npz", allow_pickle=True)
names = list(Tz["names"].astype(str)); Y = Tz["Y"].astype(np.float64); n = len(Y)
F = np.load(f"{C}/folds.npz"); outer, grp = F["outer"], F["group"]
U = pd.read_csv(f"{C}/union.tsv", sep="\t")["accession"].astype(str).tolist(); pos = {a: i for i, a in enumerate(U)}
FC = ["GA20_salt75", "GA20_salt150", "GA22_ip_salt60", "GA22_ip_salt120", "GA24_ip_salt60", "GA24_ip_salt120",
      "GA33_A1_heat37", "GA33_A1_heat43", "GA33_B11_heat37", "GA33_B11_heat43", "FS76_mg"]

P = np.full((n, len(names)), np.nan)
oof = np.load(f"{C}/runs/s2/full_oof.npy")
rows = (np.load(f"{C}/runs/s2/full_rows.npy") if os.path.exists(f"{C}/runs/s2/full_rows.npy")
        else np.nonzero(F["has_features"] & np.isfinite(Y).any(1))[0])
for j, t in enumerate(names): P[rows, names.index(t)] = oof[:, j]
o5 = np.load(f"{C}/runs/s5/ip_fold_changes_oof.npy"); r5 = np.load(f"{C}/runs/s5/ip_fold_changes_rows.npy")
for j, t in enumerate(FC): P[r5, names.index(t)] = o5[:, j]

# ---- STRING ENSP -> UniProt, then the physical edge list restricted to measured proteins
ens2u = {}
with gzip.open(f"{SD}/9606.protein.aliases.v12.0.txt.gz", "rt") as fh:
    for l in fh:
        if l.startswith("#"): continue
        p_, a_, src = l.rstrip("\n").split("\t")[:3]
        if src in ("UniProt_AC", "Ensembl_UniProt") and a_ in pos: ens2u.setdefault(p_, a_)
print(f"{len(ens2u)} STRING proteins map to a measured accession", flush=True)
E = []
with gzip.open(f"{SD}/9606.protein.physical.links.v12.0.txt.gz", "rt") as fh:
    next(fh)
    for l in fh:
        a_, b_, s_ = l.split()
        ia, ib = ens2u.get(a_), ens2u.get(b_)
        if ia is not None and ib is not None and ia != ib: E.append((pos[ia], pos[ib], int(s_)))
E = np.array(E, dtype=np.int32); print(f"{len(E)} physical edges between measured proteins", flush=True)

# ---- Complex Portal, for the side-by-side
cp = np.zeros((n, n), dtype=bool)
for r in csv.reader(open(f"{ROOT}/data/external/complexes/complexportal_9606.tsv"), delimiter="\t"):
    if r[0].startswith("#"): continue
    mem = set(re.findall(r"([OPQ]\d[A-Z0-9]{3}\d|[A-NR-Z]\d(?:[A-Z][A-Z0-9]{2}\d){1,2})(?:-\d+)?\(", r[18]))
    idx = sorted({pos[m] for m in mem if m in pos})
    if len(idx) >= 2: cp[np.ix_(idx, idx)] = True
np.fill_diagonal(cp, False)

GRAPHS = {"Complex Portal": sparse.csr_matrix(cp.astype(np.float32))}
for cut in (900, 700, 400):
    e = E[E[:, 2] >= cut]
    r_ = np.concatenate([e[:, 0], e[:, 1]]); c_ = np.concatenate([e[:, 1], e[:, 0]])
    A = sparse.csr_matrix((np.ones(len(r_), np.float32), (r_, c_)), shape=(n, n))
    A.data[:] = 1.0; A.setdiag(0); A.eliminate_zeros()
    GRAPHS[f"STRING physical >= {cut}"] = A


def blocks(k, A):
    """per outer fold: (test rows, training-row weight vector, neighbour count). The adjacency stays sparse and
    shared; slicing a dense test-by-train block per fold, per target, per permutation is what made the first two
    versions of this script unusable -- 50 s per target on the SMALLEST graph."""
    B = []
    for f in np.unique(outer):
        i = np.nonzero(k & (outer == f))[0]; w = (k & (outer != f)).astype(np.float32)
        if len(i) < 3 or w.sum() < 10: continue
        c = A @ w                                              # neighbours in the training folds, per protein
        if (c[i] > 0).sum() >= 10: B.append((i, w, c[i]))
    return B


def partner_mean(v, B, A):
    """training-fold partners only -- never in the test protein's own identity cluster"""
    pm = np.full(n, np.nan)
    vf = np.nan_to_num(v).astype(np.float32)
    for i, w, c in B:
        s_ = (A @ (vf * w))[i]; ok = c > 0
        m = np.full(len(i), np.nan); m[ok] = s_[ok] / c[ok]; pm[i] = m
    return pm


J = {}
for gname, A in GRAPHS.items():
    deg = np.asarray(A.sum(1)).ravel(); cover = deg > 0
    g = dict(n_covered=int(cover.sum()), median_degree=float(np.median(deg[cover])) if cover.any() else 0.0, targets={})
    for t in names:
        j = names.index(t); y, p = Y[:, j], P[:, j]
        k = np.isfinite(y) & np.isfinite(p) & cover
        if k.sum() < 300: continue
        res = y - p; B = blocks(k, A)
        if not B: continue
        pm = partner_mean(res, B, A); q = k & np.isfinite(pm)
        if q.sum() < 200: continue
        r = float(np.corrcoef(res[q], pm[q])[0, 1])
        # 25 draws, not cm12's 500: cm12 already pinned this null at ~0.00 with 33 targets x 500 draws, and the
        # STRING blocks are 4-8x larger, so the permutation is the whole cost here. Fold index lists are hoisted.
        fidx = [np.nonzero(k & (outer == f))[0] for f in np.unique(outer)]
        rng = np.random.default_rng(0); null = []
        for _ in range(25):
            vp = res.copy()
            for ii in fidx:                     # one shuffle per fold covers every row exactly once, test and
                if len(ii): vp[ii] = res[rng.permutation(ii)]        # training side alike; folds are preserved
            pp = partner_mean(vp, B, A); qq = k & np.isfinite(pp)
            null.append(np.corrcoef(vp[qq], pp[qq])[0, 1])
        p2 = p.copy()
        for f in np.unique(outer):
            te, tr = q & (outer == f), q & (outer != f)
            if te.sum() < 5 or tr.sum() < 20: continue
            b = np.linalg.lstsq(np.column_stack([pm[tr], np.ones(int(tr.sum()))]), res[tr], rcond=None)[0]
            p2[te] = p[te] + pm[te] * b[0] + b[1]
        ss = ((y[q] - y[q].mean()) ** 2).sum(); cl = meta[t]["ceiling"]
        print(f"    {gname[:22]:22} {t:22} r={r:6.3f} null={np.mean(null):+.3f} n={int(q.sum())}", flush=True)
        g["targets"][t] = dict(r=r, null=float(np.mean(null)), n=int(q.sum()),
                               base=float(1 - ((y - p) ** 2)[q].sum() / ss) / cl,
                               plus=float(1 - ((y - p2) ** 2)[q].sum() / ss) / cl)
    tt = g["targets"]
    g["mean_r"] = float(np.mean([d["r"] for d in tt.values()])); g["mean_base"] = float(np.mean([d["base"] for d in tt.values()]))
    g["mean_plus"] = float(np.mean([d["plus"] for d in tt.values()]))
    g["mean_n"] = float(np.mean([d["n"] for d in tt.values()]))
    J[gname] = g
    print(f"{gname:24} covers {g['n_covered']:5d} proteins (median degree {g['median_degree']:.0f})  "
          f"mean residual r {g['mean_r']:.3f}  R2/ceil {g['mean_base']:.3f} -> {g['mean_plus']:.3f} "
          f"(+{g['mean_plus'] - g['mean_base']:.3f}) over {len(tt)} targets, mean n {g['mean_n']:.0f}", flush=True)

json.dump(J, open(f"{C}/cm16_network.json", "w"), indent=1)

ts = [t for t in names if all(t in J[g]["targets"] for g in J)]
f, ax = plt.subplots(1, 2, figsize=(17, 5.4))
a = ax[0]; x = np.arange(len(ts)); w = .8 / len(J)
for i, (gname, g) in enumerate(J.items()):
    a.bar(x + (i - (len(J) - 1) / 2) * w, [g["targets"][t]["r"] for t in ts], w,
          label=f"{gname} ({g['n_covered']} proteins)")
a.set_xticks(x); a.set_xticklabels(ts, rotation=70, ha="right", fontsize=6.5); a.axhline(0, color="k", lw=.6)
a.set_ylabel("r(residual, partner mean residual)"); a.legend(fontsize=7)
a.set_title("Residual sharing across interaction graphs, by coverage", fontsize=10)
a = ax[1]
for gname, g in J.items():
    a.scatter([g["targets"][t]["base"] for t in ts], [g["targets"][t]["plus"] for t in ts], s=22, label=gname)
lim = [0, max(max(g["targets"][t]["plus"] for t in ts) for g in J.values()) * 1.05]
a.plot(lim, lim, "k--", lw=.8); a.set_xlim(lim); a.set_ylim(lim)
a.set_xlabel("R2 / ceiling, sequence alone"); a.set_ylabel("with the partner term"); a.legend(fontsize=7)
a.set_title("What the partner term is worth", fontsize=10)
f.tight_layout(); f.savefig(f"{FIG}/network_residual.png", dpi=140); plt.close(f)
print("CM16_DONE")
