"""ga01: parse GA_33 (co-chaperone pull-downs across temperature) and build fold-change targets.

pd_data/GA_33_report_out.tsv, one row per protein group, raw log2 intensities, NaN for missing.
Column names encode the design:

    X20260805_GA_33_<chap>_STAU_<s>_T_<temp>_BR<n>_TR1_<well>_1_<injection>

    chap   21A = DNAJA1, 24 = DNAJB11 (the two co-chaperone baits)
    s      0 = control, 10 = treated
    temp   35, 37, 43 C
    BR     1-6 biological replicates

2 x 2 x 3 x 6 = 72 samples. One sample, 24_STAU_10_T_43_BR2, was injected twice (2872, 2933);
the two injections are averaged in log2 space into that one biological replicate, since they
are technical repeats of the same sample, not an extra replicate.

Normalisation. Run medians rise ~1 log2 unit from 35 to 43 C. That is either loading or a real
global increase in binding, and the two cannot be told apart from these data, so each run is
median-normalised (median of its log-ratios to each protein's across-run mean, over proteins
detected in >= 90 % of runs). A global shift then appears only as the intercept of a fold
change, which a model with an intercept absorbs anyway; what is removed is run-to-run loading
noise. The per-run offsets are saved so the raw shift is not lost.

Targets, log2 fold changes, each a difference of biological-replicate means:
    stau_<chap>_<T>       STAU_10 vs STAU_0 (control) at one temperature   6 targets
    temp_<chap>_<T>v35    T vs 35 C within the controls                    4 targets
A protein is kept for a target when both groups have >= 4 of 6 real values.

Ceiling per target from split-half agreement by biological replicate (all 10 ways of splitting
six replicates into two disjoint triples), Spearman-Brown corrected to all six (k = 2).
"""
import os, re, csv, json, itertools
import numpy as np, pandas as pd

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
SRC = f"{ROOT}/pd_data/GA_33_report_out.tsv"
OUT = f"{ROOT}/data/ga_data"
MIN_REAL = 4

df = pd.read_csv(SRC, sep="\t", low_memory=False)
runs = list(df.columns[8:])
pat = re.compile(r"GA_33_(21A|24)_STAU_(\d+)_T_(\d+)_BR(\d+)_TR\d+_.*_(\d+)$")
lab = [pat.search(c).groups() for c in runs]
V = df[runs].apply(pd.to_numeric, errors="coerce").values.astype(float)
print(f"{len(df)} protein groups x {len(runs)} runs; missing {np.isnan(V).mean():.3f}", flush=True)

# --- median normalisation over well-detected proteins
det = np.isfinite(V).mean(1) >= 0.9
with np.errstate(invalid="ignore"):
    pm = np.nanmean(V[det], 1, keepdims=True)
off = np.nanmedian(V[det] - pm, 0)
Vn = V - off
print(f"normalisation over {det.sum()} proteins detected in >=90% of runs; "
      f"offsets range {off.min():+.2f} .. {off.max():+.2f}", flush=True)
offs = pd.DataFrame(dict(run=runs, chap=[l[0] for l in lab], stau=[l[1] for l in lab],
                         temp=[l[2] for l in lab], br=[l[3] for l in lab], offset=off))
print("  mean offset by temperature:", offs.groupby("temp")["offset"].mean().round(3).to_dict(), flush=True)

# --- assemble (chap, stau, temp) -> (n, 6) matrices, averaging the duplicate injection
CH, ST, TE = ("21A", "24"), ("0", "10"), ("35", "37", "43")
M = {}
for c, s, t in itertools.product(CH, ST, TE):
    A = np.full((len(df), 6), np.nan)
    for b in range(1, 7):
        cols = [j for j, l in enumerate(lab) if l[:4] == (c, s, t, str(b))]
        assert cols, (c, s, t, b)
        with np.errstate(invalid="ignore"):
            A[:, b - 1] = np.nanmean(Vn[:, cols], 1) if len(cols) > 1 else Vn[:, cols[0]]
        if len(cols) > 1:
            print(f"  {c} STAU_{s} {t}C BR{b}: {len(cols)} injections averaged", flush=True)
    M[(c, s, t)] = A


def contrast(B, A):
    nb, na = np.isfinite(B).sum(1), np.isfinite(A).sum(1)
    with np.errstate(invalid="ignore"):
        y = np.nanmean(B, 1) - np.nanmean(A, 1)
        se = np.sqrt(np.nanvar(B, 1, ddof=1) / nb + np.nanvar(A, 1, ddof=1) / na)
    keep = (nb >= MIN_REAL) & (na >= MIN_REAL)
    return y, se, keep


def ceiling(B, A, keep):
    rs = []
    for tri in itertools.combinations(range(6), 3):
        if 0 not in tri:
            continue                                   # each split once
        o = [i for i in range(6) if i not in tri]
        with np.errstate(invalid="ignore"):
            y1 = np.nanmean(B[:, list(tri)], 1) - np.nanmean(A[:, list(tri)], 1)
            y2 = np.nanmean(B[:, o], 1) - np.nanmean(A[:, o], 1)
        k = keep & np.isfinite(y1) & np.isfinite(y2)
        rs.append(np.corrcoef(y1[k], y2[k])[0, 1])
    rh = float(np.mean(rs))
    rel = 2 * rh / (1 + rh)
    return rh, rel, float(np.sqrt(max(rel, 0))), len(rs)


T = {}
for c, t in itertools.product(CH, TE):
    T[f"stau_{c}_{t}"] = (M[(c, "10", t)], M[(c, "0", t)])
for c, t in itertools.product(CH, ("37", "43")):
    T[f"temp_{c}_{t}v35"] = (M[(c, "0", t)], M[(c, "0", "35")])

acc_raw = df["Protein.Group"].astype(str).values
acc = np.array([a.split(";")[0] for a in acc_raw])
gene = df["Genes"].astype(str).values
out = dict(accession=acc, accession_raw=acc_raw, gene=gene,
           npeptides=df["n_peptides"].values, proteotypic=df["Proteotypic"].values,
           run_offset=off, run_names=np.array(runs))
for (c, s, t), A in M.items():
    out[f"br_{c}_{s}_{t}"] = A
summary = {}
print(f"\n{'target':<18}{'n kept':>8}{'sd(y)':>8}{'median SE':>11}{'noise share':>13}"
      f"{'half r':>8}{'reliab.':>9}{'max r':>7}", flush=True)
for nm, (B, A) in T.items():
    y, se, keep = contrast(B, A)
    rh, rel, mx, ns = ceiling(B, A, keep)
    yk = y[keep]
    share = float(np.nanmean(se[keep] ** 2) / np.nanvar(yk))
    out[f"y_{nm}"], out[f"se_{nm}"], out[f"keep_{nm}"] = y, se, keep
    summary[nm] = dict(n=int(keep.sum()), sd=float(np.nanstd(yk)), mean=float(np.nanmean(yk)),
                       median_se=float(np.nanmedian(se[keep])), noise_share=share,
                       half_r=rh, reliability=rel, max_r=mx, n_splits=ns)
    print(f"{nm:<18}{keep.sum():>8}{np.nanstd(yk):>8.3f}{np.nanmedian(se[keep]):>11.3f}"
          f"{share:>13.3f}{rh:>8.3f}{rel:>9.3f}{mx:>7.3f}", flush=True)

# --- are biological replicates paired across conditions (shared prep)?
print("\nBR pairing: sd of per-BR differences, matched index vs rolled", flush=True)
for nm in ("stau_21A_37", "stau_24_37", "temp_21A_43v35"):
    B, A = T[nm]
    d = lambda X, Y: np.nanmean([np.nanstd(X[:, i] - Y[:, i] - np.nanmean(X - Y, 1)) for i in range(6)])
    mt = d(B, A)
    mis = np.mean([d(B, np.roll(A, s, 1)) for s in range(1, 6)])
    print(f"  {nm:<16} matched {mt:.3f}  rolled {mis:.3f}  -> {'PAIRED' if mt < 0.9 * mis else 'not paired'}", flush=True)
    summary[nm]["br_paired_ratio"] = float(mt / mis)

# --- how the targets relate
names = list(T)
Y = np.column_stack([out[f"y_{n}"] for n in names])
kk = np.all([out[f"keep_{n}"] for n in names], 0)
C = np.corrcoef(Y[kk].T)
print(f"\ncorrelation between targets ({kk.sum()} proteins kept for all ten)", flush=True)
print(" " * 18 + "".join(f"{n[-9:]:>10}" for n in names), flush=True)
for i, n in enumerate(names):
    print(f"{n:<18}" + "".join(f"{C[i, j]:>+10.2f}" for j in range(len(names))), flush=True)
summary["_target_corr"] = dict(names=names, corr=C.tolist(), n=int(kk.sum()))

np.savez_compressed(f"{OUT}/blocks.npz", **out)
with open(f"{OUT}/targets.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["accession", "accession_raw", "gene"] + [f"y_{n}" for n in names])
    for i in range(len(df)):
        w.writerow([acc[i], acc_raw[i], gene[i]] + [f"{out[f'y_{n}'][i]:.5f}" for n in names])
offs.to_csv(f"{OUT}/run_offsets.tsv", sep="\t", index=False)
json.dump(summary, open(f"{OUT}/ga01_summary.json", "w"), indent=1)
print(f"\nwrote {OUT}/blocks.npz, targets.tsv, run_offsets.tsv, ga01_summary.json\nGA01_DONE", flush=True)
