"""Figure for the main report: heat response against Meltome melting temperature (deciles), with
standard errors. A: 43 vs 35 C as measured, its lysate-like part and its selective part (ad11), and the
selective 37 vs 35 C response. B: the selective 43 vs 35 C response for complex subunits and other proteins."""
import os, csv, json, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
GA = f"{ROOT}/data/ga_data"
HUM = ["HepG2", "Jurkat", "K562", "U937", "HAOEC", "HL60", "HEK293T", "colon_cancer_spheroids", "HaCaT", "pTcells"]
dm = json.load(open(f"{ROOT}/data/external/meltome/full_dataset.json"))
per = collections.defaultdict(lambda: collections.defaultdict(list))
for x in dm:
    if x["runName"] in HUM and x["uniprotAccession"] and x["meltingPoint"] is not None:
        per[x["runName"]][x["uniprotAccession"].split("-")[0]].append(x["meltingPoint"])
rm = {r: {a: float(np.median(v)) for a, v in per[r].items()} for r in HUM}
cen = {r: float(np.median(list(rm[r].values()))) for r in HUM}; grand = float(np.median(list(cen.values())))
LL = collections.defaultdict(list)
for r in HUM:
    for a, t in rm[r].items(): LL[a].append(t - cen[r])
TM = {a: float(np.median(v)) + grand for a, v in LL.items()}
blk = np.load(f"{GA}/blocks.npz", allow_pickle=True)
st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{GA}/sequences_status.tsv"), delimiter="\t")}
acc = np.array([st.get(str(a), str(a)) for a in blk["accession"]])
FG = np.load(f"{GA}/folds_ga05.npz", allow_pickle=True)
import re
member = set()
for r in csv.reader(open(f"{ROOT}/data/external/complexes/complexportal_9606.tsv"), delimiter="\t"):
    if r[0].startswith("#"): continue
    mem = set(re.findall(r"([OPQ]\d[A-Z0-9]{3}\d|[A-NR-Z]\d(?:[A-Z][A-Z0-9]{2}\d){1,2})(?:-\d+)?\(", r[18]))
    if len(mem) >= 2: member |= mem


SH = np.load(f"{GA}/selective_heat.npz")
rows = FG["rows_temp_avg_43v35"]
inrows = np.zeros(len(acc), bool); inrows[rows] = True
tm = np.array([TM.get(a, np.nan) for a in acc])
base = inrows & np.isfinite(tm) & np.isfinite(SH["sel43"])
edges = np.percentile(tm[base], np.linspace(0, 100, 11))
bins = np.clip(np.digitize(tm, edges[1:-1]), 0, 9)


def deciles(y, mask=None):
    ok = base & np.isfinite(y) & (mask if mask is not None else True)
    xs, ms, se = [], [], []
    for i in range(10):
        s = ok & (bins == i)
        xs.append(np.median(tm[s])); ms.append(y[s].mean()); se.append(y[s].std() / np.sqrt(s.sum()))
    return np.array(xs), np.array(ms), np.array(se)


fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
for y, lab, col, fmt, alpha in ((SH["raw43"], "43 vs 35 °C, as measured", "#999999", "o--", 1),
                                (SH["fit43"], "43 vs 35 °C, lysate-like part", "#c9a66b", "^:", 1),
                                (SH["sel43"], "43 vs 35 °C, selective", "#b5442b", "o-", 1),
                                (SH["sel37"], "37 vs 35 °C, selective", "#3b6ea5", "s-", 1)):
    x, m, s = deciles(y)
    ax[0].errorbar(x, m, yerr=s, fmt=fmt, color=col, label=lab, capsize=3, alpha=alpha)
    print(lab, np.round(m, 2), flush=True)
ax[0].axhline(0, color="grey", lw=0.6)
ax[0].set_xlabel("melting temperature (Meltome, °C), decile median"); ax[0].set_ylabel("mean change in co-chaperone binding (log2)")
ax[0].set_title("A. Heat response by thermal stability", fontsize=10, loc="left"); ax[0].legend(frameon=False, fontsize=7.5, loc="lower left")
mem = np.isin(acc, list(member))
xc, mc, sc = deciles(SH["sel43"], mem); xo, mo, so = deciles(SH["sel43"], ~mem)
ax[1].errorbar(xc, mc, yerr=sc, fmt="o-", color="#6a3d9a", label="complex subunits", capsize=3)
ax[1].errorbar(xo, mo, yerr=so, fmt="o-", color="#999999", label="other proteins", capsize=3)
ax[1].axhline(0, color="grey", lw=0.6)
ax[1].set_xlabel("melting temperature (Meltome, °C), decile median"); ax[1].set_ylabel("selective change at 43 vs 35 °C (log2)")
ax[1].set_title("B. Complex subunits gain more", fontsize=10, loc="left"); ax[1].legend(frameon=False, fontsize=8, loc="lower left")
for a_ in ax:
    a_.spines["top"].set_visible(False); a_.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{ROOT}/reports/figures/heat_tm_selective.png", dpi=200)
print("subunits:", np.round(mc, 2), "\nothers:", np.round(mo, 2), "\nedges:", np.round(edges, 1))
