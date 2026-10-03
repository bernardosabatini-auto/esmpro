"""Figure for the main report: heat response against Meltome melting temperature (deciles), 43 and
37 C, with standard errors; and the same split into complex subunits and other proteins at 43 C."""
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


def deciles(t, mask=None, nb=10, edges=None):
    rows = FG[f"rows_{t}"]; y = FG[f"y_{t}"].astype(float); tm = np.array([TM.get(a, np.nan) for a in acc[rows]])
    ok = np.isfinite(tm) if mask is None else np.isfinite(tm) & mask(acc[rows])
    if edges is None:
        edges = np.percentile(tm[np.isfinite(tm)], np.linspace(0, 100, nb + 1))
    b = np.clip(np.digitize(tm, edges[1:-1]), 0, len(edges) - 2)
    xs, ms, se = [], [], []
    for i in range(len(edges) - 1):
        s = ok & (b == i)
        xs.append(np.median(tm[s])); ms.append(y[s].mean()); se.append(y[s].std() / np.sqrt(s.sum()))
    return np.array(xs), np.array(ms), np.array(se), edges


fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
x, m, s, edges = deciles("temp_avg_43v35")
x7, m7, s7, _ = deciles("temp_avg_37v35", edges=edges)
ax[0].axvspan(47.9, 52.2, color="#f2e2c4", zorder=0)
ax[0].errorbar(x, m, yerr=s, fmt="o-", color="#b5442b", label="43 vs 35 °C", capsize=3)
ax[0].errorbar(x7, m7, yerr=s7, fmt="s-", color="#3b6ea5", label="37 vs 35 °C", capsize=3)
ax[0].axhline(0, color="grey", lw=0.6)
ax[0].text(50.05, -0.045, "binding window", ha="center", fontsize=9, color="#8a6a2f")
ax[0].set_xlabel("melting temperature (Meltome, °C), decile median"); ax[0].set_ylabel("mean change in co-chaperone binding (log2)")
ax[0].set_title("A. Heat response by thermal stability", fontsize=10, loc="left"); ax[0].legend(frameon=False, fontsize=8, loc="upper right")
q = np.percentile(np.array([v for v in TM.values()]), [0, 33.3, 66.7, 100])
xc, mc, sc, _ = deciles("temp_avg_43v35", mask=lambda a: np.isin(a, list(member)), edges=edges)
xo, mo, so, _ = deciles("temp_avg_43v35", mask=lambda a: ~np.isin(a, list(member)), edges=edges)
ax[1].axvspan(47.9, 52.2, color="#f2e2c4", zorder=0)
ax[1].errorbar(xc, mc, yerr=sc, fmt="o-", color="#6a3d9a", label="complex subunits", capsize=3)
ax[1].errorbar(xo, mo, yerr=so, fmt="o-", color="#999999", label="other proteins", capsize=3)
ax[1].axhline(0, color="grey", lw=0.6)
ax[1].set_xlabel("melting temperature (Meltome, °C), decile median"); ax[1].set_ylabel("mean change at 43 vs 35 °C (log2)")
ax[1].set_title("B. Complex subunits show the window more sharply", fontsize=10, loc="left"); ax[1].legend(frameon=False, fontsize=8, loc="upper right")
for a_ in ax:
    a_.spines["top"].set_visible(False); a_.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{ROOT}/reports/figures/heat_tm_window.png", dpi=200)
print("43C:", np.round(m, 2), "\n37C:", np.round(m7, 2), "\nsubunits:", np.round(mc, 2), "\nothers:", np.round(mo, 2), "\nedges:", np.round(edges, 1))
