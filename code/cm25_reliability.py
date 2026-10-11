"""cm25: how reliable is the co-fold's own interface score?

The pilot (cm23) found no relation between a predicted HSPB1-ACD / client-window interface and
measured HSPB1 binding. Before that is called a result about biology, it has to be separated from a
result about the measuring instrument. Two very different conclusions produce the same null:

  (a) the score is a stable property of the window, and it simply does not predict binding;
  (b) the score is mostly sampling noise, in which case the pilot never had the resolution to
      detect anything and the null is uninformative.

Boltz-2 defaults to no seeding, so the pilot run was one unseeded draw. Re-running 600 of the same
windows gives a second independent draw from the same model on the same input. The correlation
between the two draws is the test-retest reliability, and its square root bounds the correlation the
score could have shown with anything -- the same "report against the ceiling" discipline used for
the pull-down targets themselves.
"""
import os, json, glob
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import spearmanr, pearsonr

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
C = f"{ROOT}/data/campaign"
import importlib.util
spec = importlib.util.spec_from_file_location("cm23", f"{ROOT}/code/cm23_pilot_test.py")

# reuse the parser without running cm23's analysis: import the two functions by exec of the file head
src = open(f"{ROOT}/code/cm23_pilot_test.py").read().split("# ------------")[0]
ns = {}
exec(compile(src, "cm23_head", "exec"), ns)
parse_one, GROOVE = ns["parse_one"], ns["GROOVE"]

dirs = sorted(glob.glob(f"{C}/boltz/rep/out/shard_*/boltz_results_shard_*/predictions/*"))
print(f"parsing {len(dirs)} replicate predictions ...", flush=True)
with ProcessPoolExecutor(max_workers=12) as ex:
    recs = [r for r in ex.map(parse_one, dirs, chunksize=16) if r]
B = pd.DataFrame(recs).set_index("name")
A = pd.read_csv(f"{C}/rep_ref.tsv", sep="\t").set_index("name")
M = A.join(B, rsuffix="_2", how="inner").dropna(subset=["iptm_pep", "iptm_pep_2"])
print(f"{len(M)} windows with both draws")

print("\n---- test-retest reliability of the window score (two unseeded draws) ----")
out = {}
for c in ("iptm_pep", "groove_recall", "n_contact"):
    x, y = M[c].values, M[f"{c}_2"].values
    pr = pearsonr(x, y).statistic
    sr = spearmanr(x, y).statistic
    out[c] = dict(pearson=float(pr), spearman=float(sr), ceiling=float(np.sqrt(max(pr, 0))))
    print(f"  {c:16s} r = {pr:+.3f}   rho = {sr:+.3f}   "
          f"=> ceiling on any correlation: {np.sqrt(max(pr,0)):.3f}")
    print(f"  {'':16s} draw 1 mean {x.mean():.3f} sd {x.std():.3f} | "
          f"draw 2 mean {y.mean():.3f} sd {y.std():.3f} | "
          f"mean |difference| {np.abs(x-y).mean():.3f}")

d = M.iptm_pep.values - M.iptm_pep_2.values
print(f"\n  a window's ipTM moves by {np.abs(d).mean():.3f} on average between two runs of the SAME input,")
print(f"  against a spread across different windows of {M.iptm_pep.std():.3f}.")
frac = 1 - out["iptm_pep"]["pearson"]
print(f"  so {100*frac:.0f}% of the variance in the pilot's window scores is run-to-run noise.")

json.dump(out, open(f"{C}/cm25_reliability.json", "w"), indent=1)
M.to_csv(f"{C}/rep_pairs.tsv", sep="\t")
print(f"\nwrote {C}/cm25_reliability.json")
print("CM25_DONE")

# ---------------------------------------------------------------- aggregate reliability
# The pilot scores a PROTEIN by aggregating ~22 windows, so the quantity that needs a ceiling is the
# aggregate, not the window. Measured by split-half over each protein's own windows: aggregate the
# odd windows and the even windows separately, correlate across proteins, Spearman-Brown correct for
# the halving. This asks the honest question -- if a different half of the protein had been tiled,
# would the protein have got the same score? -- and it carries the run-to-run noise along with it.
#
# (An earlier attempt simulated windows as i.i.d. draws per protein. That builds in the absence of a
# protein-level component, i.e. it assumes the null, so it measures reliability under the null
# rather than a ceiling. It is not used.)
print("\n---- split-half reliability of the PER-PROTEIN aggregations ----")
W = pd.read_csv(f"{C}/pilot_windows.tsv", sep="\t")
w = W[(W.kind == "window") & W.iptm_pep.notna()].copy()
rng = np.random.default_rng(0)
w["half"] = rng.integers(0, 2, len(w))


def aggregate(v, how):
    v = np.asarray(v)
    if how == "max":
        return v.max()
    if how == "mean":
        return v.mean()
    if how == "top3":
        return np.sort(v)[-3:].mean()
    return float((v >= HI).mean())


res = {}
for col in ("iptm_pep", "groove_recall"):
    HI = w[col].quantile(0.90)
    for how in ("max", "mean", "top3", "fhi"):
        a, b = [], []
        for _, g in w.groupby("accession"):
            g0, g1 = g[g.half == 0][col].values, g[g.half == 1][col].values
            if len(g0) >= 3 and len(g1) >= 3:
                a.append(aggregate(g0, how)); b.append(aggregate(g1, how))
        a, b = np.array(a), np.array(b)
        rh = pearsonr(a, b).statistic
        sb = 2 * rh / (1 + rh)                     # Spearman-Brown: halves -> full set
        res[f"{col}_{how}"] = dict(half=float(rh), full=float(sb), n=len(a))
        print(f"  {col}_{how:5s} split-half r {rh:+.3f} -> full-set reliability {sb:+.3f} "
              f"=> ceiling on |rho| {np.sqrt(max(sb,0)):.3f}   (n={len(a)})")
out["aggregate_reliability"] = res
json.dump(out, open(f"{C}/cm25_reliability.json", "w"), indent=1)

print("\nA null only means something where the ceiling is high. cm23's observed |rho| against it:")
J = json.load(open(f"{C}/cm23_pilot.json"))
for k, v in res.items():
    o = J["p1"].get(k, {}).get("rho")
    if o is None:
        continue
    c = np.sqrt(max(v["full"], 0))
    verdict = "informative null" if c > 0.45 else "TOO NOISY to interpret"
    print(f"  {k:22s} observed {o:+.3f}   ceiling {c:.3f}   {verdict}")
