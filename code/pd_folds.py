"""Write the outer cross-validation folds to disk so that any environment can use them.

The fitting scripts select folds with scikit-learn's GroupKFold over MMseqs2 clusters. The GPU
readouts run in the `proteinae` env, which has the Blackwell-capable torch build but no
scikit-learn, while `dendritic_env` has scikit-learn but a torch build without sm_120 kernels.
Re-implementing GroupKFold would risk silently different folds, and with them comparisons to the
ridge numbers that are not like-for-like. So the folds are computed once, here, with
scikit-learn itself, for every `keep_<target>` mask in a blocks file, and loaded by index.

  python pd_folds.py --data data/pd_data            # salt titration
  python pd_folds.py --data data/ga_data            # GA_33 temperature series
"""
import os, csv, argparse
import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=f"{ROOT}/data/pd_data")
ap.add_argument("--n", type=int, default=5)
a = ap.parse_args()

blk = np.load(f"{a.data}/blocks.npz", allow_pickle=True)
acc = np.array([str(x) for x in blk["accession"]])
resolved = {r["sheet_accession"]: r["resolved_accession"]
            for r in csv.DictReader(open(f"{a.data}/sequences_status.tsv"), delimiter="\t")}
cl = {r["accession"]: r["cluster"] for r in csv.DictReader(open(f"{a.data}/clusters.tsv"), delimiter="\t")}
groups = np.array([cl.get(resolved.get(s, s), s) for s in acc])
out = {}
for key in [k for k in blk.files if k.startswith("keep_")]:
    t = key[5:]
    y = blk[f"y_{t}"].astype(float)
    k = blk[key].astype(bool) & np.isfinite(y)
    fid = np.full(k.sum(), -1)
    for f, (_, te) in enumerate(GroupKFold(n_splits=a.n).split(np.zeros(k.sum()), None, groups[k])):
        fid[te] = f
    assert (fid >= 0).all()
    G = groups[k]
    for f in range(a.n):
        assert not set(G[fid == f]) & set(G[fid != f])
    out[f"fold_{t}"] = fid
    out[f"rows_{t}"] = np.nonzero(k)[0]
    print(f"  {t:<18} {k.sum():>6} rows  fold sizes {np.bincount(fid).tolist()}", flush=True)
out["groups"] = groups
np.savez(f"{a.data}/folds.npz", **out)
print(f"wrote {a.data}/folds.npz")
