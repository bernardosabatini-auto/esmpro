"""pd01: parse the HSPB1 salt-titration workbook into fold-change targets.

The sheet is an HSPB1 pull-down across a salt titration. Layout (1-based columns, header on
row 2, data from row 3):

    A UniProt ID   B Genes   C entry name   D npeptides   E XIC
    F-O   0 mM salt   (baseline)   P  = block mean, CONTAMINATED, ignored
    Q-Z   150 mM salt              AA = block mean, CONTAMINATED, ignored
    AB-AK 75 mM salt               AL = block mean, CONTAMINATED, ignored
    AM-AS limma stats for 75 vs 0  (logFC CI.L CI.R t B P.Value adj.P.Val)
    AT-AZ limma stats for 150 vs 0

Each block is 2 technical groups x 5 biological replicates, in the order
BR1..BR5 (group _01_) then BR1..BR5 (group _02_), so the technical pair for BR*i* is
(col0 + i, col0 + 5 + i). Values are ALREADY log2 intensities.

Two properties of this file drive the code:

  * A global find/replace of "NA" -> "0" was applied, so **missing values are literal 0.0**.
    A log2 intensity of exactly zero is not physically possible. Every 0 becomes NaN here.
    The block means in P/AA/AL were computed including those zeros and are therefore biased
    downward for ~1,900 rows; they are read only to confirm that, never used.
  * The same replace damaged 8 UniProt accessions, 426 gene symbols and 367 entry names.
    Only column A matters downstream, and pd02 repairs it.

openpyxl is not installed in this environment, so the sheet is streamed with stdlib
zipfile + iterparse, which never holds more than one row in memory.

  python pd01_parse.py                      # writes data/pd_data/{targets.tsv,blocks.npz}
  python pd01_parse.py --min-real 5         # sensitivity of the keep rule
"""
import os, sys, csv, json, zipfile, argparse
import xml.etree.ElementTree as ET
import numpy as np

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

ap = argparse.ArgumentParser()
ap.add_argument("--xlsx", default=f"{ROOT}/pd_data/GA_20_results_bs.xlsx")
ap.add_argument("--out-dir", default=f"{ROOT}/data/pd_data")
ap.add_argument("--min-real", type=int, default=7,
                help="minimum non-missing replicates required in BOTH blocks of a comparison")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)


def col_to_idx(ref):
    """'A' -> 0, 'AK' -> 36. Accepts a full cell ref like 'AB17'."""
    n = 0
    for ch in ref:
        if not ch.isalpha():
            break
        n = n * 26 + (ord(ch.upper()) - 64)
    return n - 1


# column blocks, 0-based, end-exclusive
C_ACC, C_GENE, C_ENTRY, C_NPEP, C_XIC = 0, 1, 2, 3, 4
BLOCKS = {                      # name -> (first col, salt in mM)
    "base": (col_to_idx("F"), 0),
    "s150": (col_to_idx("Q"), 150),
    "s75":  (col_to_idx("AB"), 75),
}
MEAN_COL = {"base": col_to_idx("P"), "s150": col_to_idx("AA"), "s75": col_to_idx("AL")}
LIMMA = {                       # comparison -> (logFC col, P.Value col)
    "s75":  (col_to_idx("AM"), col_to_idx("AR")),
    "s150": (col_to_idx("AT"), col_to_idx("AY")),
}
N_W = 63                        # populated width (A..BK)


def shared_strings(z):
    out = []
    with z.open("xl/sharedStrings.xml") as fh:
        for ev, el in ET.iterparse(fh, events=("end",)):
            if el.tag == NS + "si":
                out.append("".join(t.text or "" for t in el.iter(NS + "t")))
                el.clear()
    return out


def rows(z, sst):
    """Yield (row_number, [cell values as str|None]) one row at a time."""
    with z.open("xl/worksheets/sheet1.xml") as fh:
        for ev, el in ET.iterparse(fh, events=("end",)):
            if el.tag != NS + "row":
                continue
            vals = [None] * N_W
            for c in el.findall(NS + "c"):
                j = col_to_idx(c.get("r", ""))
                if not (0 <= j < N_W):
                    continue
                v = c.find(NS + "v")
                if v is None or v.text is None:
                    continue
                vals[j] = sst[int(v.text)] if c.get("t") == "s" else v.text
            yield int(el.get("r")), vals
            el.clear()


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


print(f"reading {os.path.basename(a.xlsx)}", flush=True)
z = zipfile.ZipFile(a.xlsx)
sst = shared_strings(z)

ids, genes, entries, npep, xic = [], [], [], [], []
raw = {k: [] for k in BLOCKS}          # 10 raw replicate values per block
stored_mean = {k: [] for k in BLOCKS}  # the contaminated means, for the audit
limma = {k: [] for k in LIMMA}
header = None

for rnum, v in rows(z, sst):
    if rnum == 2:
        header = v
        continue
    if rnum < 3:
        continue
    acc = (v[C_ACC] or "").strip()
    if not acc:
        continue
    ids.append(acc)
    genes.append((v[C_GENE] or "").strip())
    entries.append((v[C_ENTRY] or "").strip())
    npep.append(f(v[C_NPEP]))
    xic.append(f(v[C_XIC]))
    for k, (c0, _salt) in BLOCKS.items():
        raw[k].append([f(v[c0 + i]) for i in range(10)])
        stored_mean[k].append(f(v[MEAN_COL[k]]))
    for k, (cfc, cp) in LIMMA.items():
        limma[k].append((f(v[cfc]), f(v[cp])))

n = len(ids)
print(f"  {n} proteins, header row has {sum(x is not None for x in header)} populated cells", flush=True)

X = {k: np.asarray(raw[k], dtype=np.float64) for k in BLOCKS}       # (n,10)
stored = {k: np.asarray(stored_mean[k], dtype=np.float64) for k in BLOCKS}
lim = {k: np.asarray(limma[k], dtype=np.float64) for k in LIMMA}    # (n,2)

# ---- audit: confirm the zeros are what spoiled the stored means --------------
print("\n-- audit: do the stored block means equal the NAIVE 10-value mean (zeros included)?", flush=True)
for k in BLOCKS:
    naive = X[k].mean(1)
    ok = np.isfinite(stored[k]) & np.isfinite(naive)
    d = np.abs(naive[ok] - stored[k][ok])
    print(f"   {k:5s}: max |naive - stored| = {d.max():.2e} over {ok.sum()} rows "
          f"-> {'MATCH, stored means include the zeros' if d.max() < 1e-6 else 'MISMATCH'}", flush=True)

# ---- zeros become missing ----------------------------------------------------
zero_counts = {}
for k in BLOCKS:
    zmask = X[k] == 0.0
    zero_counts[k] = int(zmask.sum())
    X[k][zmask] = np.nan
print("\n-- missingness (zeros reinterpreted as NaN)", flush=True)
for k, (_c0, salt) in BLOCKS.items():
    nz = np.isfinite(X[k]).sum(1)
    print(f"   {k:5s} ({salt:3d} mM): {zero_counts[k]:5d} zero cells ({100*zero_counts[k]/X[k].size:.2f}%), "
          f"{int((nz < 10).sum()):4d} rows with >=1 missing, {int((nz == 0).sum()):3d} rows all missing", flush=True)


def block_stats(M):
    """Average technical pairs within each biological replicate, then summarise over the 5 BRs.

    Columns are BR1..BR5 of technical group 01 followed by BR1..BR5 of group 02, so the pair
    for BR i is (i, i+5). Averaging the pair first makes the biological replicate the unit of
    replication; treating all ten columns as independent would understate the standard error.
    """
    pair = np.nanmean(np.stack([M[:, 0:5], M[:, 5:10]]), axis=0)     # (n,5), NaN where both absent
    with np.errstate(invalid="ignore"):
        nbr = np.isfinite(pair).sum(1)
        mean = np.nanmean(pair, axis=1)
        sd = np.nanstd(pair, axis=1, ddof=1)
    se = np.where(nbr > 1, sd / np.sqrt(np.maximum(nbr, 1)), np.nan)
    return pair, mean, se, nbr


pair, mean, se, nbr = {}, {}, {}, {}
for k in BLOCKS:
    pair[k], mean[k], se[k], nbr[k] = block_stats(X[k])

n_real = {k: np.isfinite(X[k]).sum(1) for k in BLOCKS}

targets = {}
for cond in ("s75", "s150"):
    y = mean[cond] - mean["base"]
    sey = np.sqrt(np.nan_to_num(se[cond]) ** 2 + np.nan_to_num(se["base"]) ** 2)
    sey[~np.isfinite(se[cond]) | ~np.isfinite(se["base"])] = np.nan
    targets[cond] = (y, sey)

print("\n-- how many proteins each keep rule retains", flush=True)
print(f"   {'rule':>12s} {'75 mM':>8s} {'150 mM':>8s} {'both':>8s}", flush=True)
keep_at = {}
for thr in (5, a.min_real, 10) if a.min_real not in (5, 10) else (5, 10):
    k75 = (n_real["base"] >= thr) & (n_real["s75"] >= thr) & np.isfinite(targets["s75"][0])
    k150 = (n_real["base"] >= thr) & (n_real["s150"] >= thr) & np.isfinite(targets["s150"][0])
    keep_at[thr] = (k75, k150)
    print(f"   {'>= ' + str(thr) + ' of 10':>12s} {int(k75.sum()):8d} {int(k150.sum()):8d} {int((k75 & k150).sum()):8d}", flush=True)
keep75, keep150 = keep_at[a.min_real]
print(f"   using --min-real {a.min_real}", flush=True)

# ---- cross-check against limma ----------------------------------------------
print("\n-- cross-check: fresh log2 fold change vs the limma logFC already in the sheet", flush=True)
for cond, lk in (("s75", "s75"), ("s150", "s150")):
    y = targets[cond][0]
    lfc = lim[lk][:, 0]
    complete = (n_real["base"] == 10) & (n_real[cond] == 10) & np.isfinite(lfc) & np.isfinite(y)
    d = y[complete] - lfc[complete]
    r = np.corrcoef(y[complete], lfc[complete])[0, 1]
    print(f"   {cond:5s}: n={int(complete.sum()):5d} complete rows  r={r:.5f}  "
          f"mean diff {d.mean():+.4f}  max |diff| {np.abs(d).max():.4f}", flush=True)
    part = np.isfinite(lfc) & np.isfinite(y) & ~complete
    if part.sum():
        dp = y[part] - lfc[part]
        print(f"          rows WITH missing values: n={int(part.sum()):5d}  "
              f"r={np.corrcoef(y[part], lfc[part])[0,1]:.4f}  mean diff {dp.mean():+.4f} "
              f"(limma used the zeros, we dropped them, so these should differ)", flush=True)

print("\n-- target structure", flush=True)
both = keep75 & keep150
r_doses = np.corrcoef(targets["s75"][0][both], targets["s150"][0][both])[0, 1]
print(f"   y_75 vs y_150 over {int(both.sum())} shared proteins: r={r_doses:.4f}", flush=True)
for cond in ("s75", "s150"):
    y, sey = targets[cond]
    kp = keep_at[a.min_real][0 if cond == "s75" else 1]
    print(f"   {cond:5s}: mean {y[kp].mean():+.3f}  sd {y[kp].std():.3f}  "
          f"median SE {np.nanmedian(sey[kp]):.3f}  range [{y[kp].min():+.2f}, {y[kp].max():+.2f}]", flush=True)

# ---- write -------------------------------------------------------------------
tsv = f"{a.out_dir}/targets.tsv"
with open(tsv, "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["accession", "accession_raw", "gene", "entry", "npeptides", "xic",
                "n_real_base", "n_real_s75", "n_real_s150",
                "mean_base", "mean_s75", "mean_s150",
                "y_s75", "se_s75", "y_s150", "se_s150",
                "limma_logfc_s75", "limma_p_s75", "limma_logfc_s150", "limma_p_s150",
                "keep_s75", "keep_s150"])
    for i, acc in enumerate(ids):
        first = acc.split(";")[0]          # protein groups: first accession is the representative
        w.writerow([first, acc, genes[i], entries[i], npep[i], xic[i],
                    int(n_real["base"][i]), int(n_real["s75"][i]), int(n_real["s150"][i]),
                    f"{mean['base'][i]:.6f}", f"{mean['s75'][i]:.6f}", f"{mean['s150'][i]:.6f}",
                    f"{targets['s75'][0][i]:.6f}", f"{targets['s75'][1][i]:.6f}",
                    f"{targets['s150'][0][i]:.6f}", f"{targets['s150'][1][i]:.6f}",
                    f"{lim['s75'][i,0]:.6f}", f"{lim['s75'][i,1]:.6g}",
                    f"{lim['s150'][i,0]:.6f}", f"{lim['s150'][i,1]:.6g}",
                    int(keep75[i]), int(keep150[i])])

npz = f"{a.out_dir}/blocks.npz"
np.savez_compressed(
    npz,
    accession=np.array([s.split(";")[0] for s in ids]),
    accession_raw=np.array(ids), gene=np.array(genes), entry=np.array(entries),
    npeptides=np.asarray(npep), xic=np.asarray(xic),
    raw_base=X["base"], raw_s75=X["s75"], raw_s150=X["s150"],          # (n,10) NaN = missing
    br_base=pair["base"], br_s75=pair["s75"], br_s150=pair["s150"],    # (n,5) technical pairs averaged
    y_s75=targets["s75"][0], se_s75=targets["s75"][1],
    y_s150=targets["s150"][0], se_s150=targets["s150"][1],
    keep_s75=keep75, keep_s150=keep150,
    limma_s75=lim["s75"], limma_s150=lim["s150"],
    min_real=a.min_real,
)
print(f"\nwrote {tsv}\nwrote {npz}", flush=True)

groups = [s for s in ids if ";" in s]
bad_acc = [s for s in ids if not (6 <= len(s.split(";")[0]) <= 10) or not s[0].isalpha()]
print(f"\n-- for pd02: {len(groups)} protein groups {groups}", flush=True)
print(f"   {len(bad_acc)} accessions with a suspicious length/shape: {bad_acc}", flush=True)
print("PD01_DONE")
