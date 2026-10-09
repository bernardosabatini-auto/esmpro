"""cm_summary: rank the groups of one or more campaign sweeps against the ridge baselines.

  python cm_summary.py data/campaign/runs/s1 [more run dirs] [--top 15]

For every group: mean R2 and mean R2 / ceiling over all targets, and per target kind (baseline, abundance,
enrichment, salt, temperature, Mg); the same for ridge on layer 80 (best single layer overall) and for the
per-target best ridge input (optimistic: chosen on out-of-fold R2). Writes <dir>/summary_table.tsv.
"""
import os, sys, json, glob
import numpy as np, pandas as pd

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae"); C = f"{ROOT}/data/campaign"
dirs = [d for d in sys.argv[1:] if not d.startswith("--")]; top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 15
meta = json.load(open(f"{C}/targets_meta.json")); names = list(meta)
kind = {t: meta[t]["kind"] for t in names}; KINDS = ["baseline", "abundance", "enrichment", "salt", "temperature", "Mg"]
ridge = json.load(open(f"{C}/ridge.json"))["per_target"]
rows = []


def summarise(label, r2):
    out = dict(group=label, mean_r2=np.nanmean([r2[t] for t in names]), mean_r2_over_ceiling=np.nanmean([r2[t] / meta[t]["ceiling"] for t in names]))
    for k in KINDS:
        ts = [t for t in names if kind[t] == k]; out[k] = np.nanmean([r2[t] / meta[t]["ceiling"] for t in ts])
    return out


rows.append(summarise("RIDGE layer80", {t: ridge[t]["layer80"] for t in names}))
rows.append(summarise("RIDGE best input per target (optimistic)", {t: max(v for k, v in ridge[t].items() if k.startswith("layer") or k == "sae") for t in names}))
for d in dirs:
    for f in glob.glob(f"{d}/*.json"):
        if f.endswith("summary.json") or "/cm04" in f: continue
        r = json.load(open(f))
        if "r2" not in r or set(names) - set(r["r2"]): continue          # only groups trained on every target
        rows.append({**summarise(r["group"], r["r2"]), "minutes": r["seconds"] / 60, "peak_gb": r["peak_gb"], "dir": os.path.basename(d)})
T = pd.DataFrame(rows).sort_values("mean_r2_over_ceiling", ascending=False)
pd.set_option("display.width", 220)
print(T.head(top).round(3).to_string(index=False))
print(f"\n{len(T) - 2} groups; kinds are mean R2/ceiling within the kind")
T.to_csv(f"{dirs[0]}/summary_table.tsv", sep="\t", index=False)
