"""ga12: how do the two co-chaperones differ? Targets and their measurement ceilings.

Two kinds of difference between the DNAJB11 ("24") and DNAJA1 ("21A") pull-downs:

  spec_<T>       bait specificity at temperature T without staurosporine:
                 mean(DNAJB11 replicates) - mean(DNAJA1 replicates); > 0 = prefers DNAJB11
  spec_avg       the same averaged over 35, 37 and 43 C (18 replicates per bait)
  dheat_<T>      difference of heat responses, DNAJB11 minus DNAJA1, (T vs 35 C)
  dstau_<T>      difference of staurosporine responses, DNAJB11 minus DNAJA1, at T

Runs are already median-normalised (ga01), so specificity is relative to each pull-down as a whole.
A protein is kept when every group entering a target has >= 4 of 6 values. The two baits themselves
are excluded. Ceiling: split-half agreement over the 10 ways of dividing six replicates into two
triples (the same replicate split applied to every group; replicates are not paired across groups,
so the assignment is arbitrary), Spearman-Brown corrected to six replicates.

Folds: scikit-learn GroupKFold over the 30 %-identity clusters, written alongside, so the GPU fits
in ga13 use exactly the folds every number here refers to.
"""
import os, csv, itertools, json
import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
D = f"{ROOT}/data/ga_data"
blk = np.load(f"{D}/blocks.npz", allow_pickle=True)
st = {r["sheet_accession"]: r for r in csv.DictReader(open(f"{D}/sequences_status.tsv"), delimiter="\t")}
acc = np.array([st[str(a)]["resolved_accession"] if str(a) in st else str(a) for a in blk["accession"]])
has = np.array([st.get(str(a), {}).get("status", "notfound") != "notfound" for a in blk["accession"]])
gene = np.array([str(g) for g in blk["gene"]])
bait = np.isin(gene, ["DNAJA1", "DNAJB11"])
cl = {r["accession"]: r["cluster"] for r in csv.DictReader(open(f"{D}/clusters.tsv"), delimiter="\t")}
groups = np.array([cl.get(a, a) for a in acc])
M = lambda c, s, t: blk[f"br_{c}_{s}_{t}"]

# each target: list of (sign, matrix) whose replicate means are summed
TG = {}
for t in ("35", "37", "43"):
    TG[f"spec_{t}"] = [(+1, M("24", "0", t)), (-1, M("21A", "0", t))]
TG["spec_avg"] = [(sg / 3, m) for t in ("35", "37", "43") for sg, m in TG[f"spec_{t}"]]
for t in ("37", "43"):
    TG[f"dheat_{t}"] = [(+1, M("24", "0", t)), (-1, M("24", "0", "35")), (-1, M("21A", "0", t)), (+1, M("21A", "0", "35"))]
for t in ("35", "37", "43"):
    TG[f"dstau_{t}"] = [(+1, M("24", "10", t)), (-1, M("24", "0", t)), (-1, M("21A", "10", t)), (+1, M("21A", "0", t))]


def value(terms, cols=None):
    with np.errstate(invalid="ignore"):
        return sum(sg * np.nanmean(m if cols is None else m[:, cols], 1) for sg, m in terms)


out, summ = {"groups": groups, "accession": acc, "gene": gene}, {}
print(f"{'target':<11}{'n':>6}{'sd':>7}{'mean':>8}{'half r':>8}{'reliability':>13}{'noise share':>13}")
for nm, terms in TG.items():
    keep = has & ~bait & np.all([np.isfinite(m).sum(1) >= 4 for _, m in terms], 0)
    y = value(terms)
    keep &= np.isfinite(y)
    rs = []
    for tri in itertools.combinations(range(6), 3):
        if 0 not in tri:
            continue
        o = [i for i in range(6) if i not in tri]
        a_, b_ = value(terms, list(tri)), value(terms, o)
        k = keep & np.isfinite(a_) & np.isfinite(b_)
        rs.append(np.corrcoef(a_[k], b_[k])[0, 1])
    rh = float(np.mean(rs)); rel = 2 * rh / (1 + rh)
    rows = np.nonzero(keep)[0]
    fid = np.full(len(rows), -1)
    for f, (_, te) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(rows)), None, groups[rows])):
        fid[te] = f
    out[f"rows_{nm}"], out[f"fold_{nm}"], out[f"y_{nm}"] = rows, fid, y[rows]
    summ[nm] = dict(n=int(len(rows)), sd=float(y[rows].std()), mean=float(y[rows].mean()), half_r=rh, reliability=rel)
    print(f"{nm:<11}{len(rows):>6}{y[rows].std():>7.3f}{y[rows].mean():>+8.3f}{rh:>8.3f}{rel:>13.3f}{1-rel:>13.3f}")
# how much do the specificities at the three temperatures agree?
common = np.intersect1d(np.intersect1d(out["rows_spec_35"], out["rows_spec_37"]), out["rows_spec_43"])
Ys = np.column_stack([value(TG[f"spec_{t}"])[common] for t in ("35", "37", "43")])
print(f"\nspecificity at 35/37/43 C, correlations on {len(common)} proteins:\n{np.round(np.corrcoef(Ys.T), 3)}")
summ["_spec_temp_corr"] = np.corrcoef(Ys.T).tolist()
np.savez(f"{D}/spec.npz", **out)
json.dump(summ, open(f"{D}/ga12_summary.json", "w"), indent=1)
print("GA12_DONE")
