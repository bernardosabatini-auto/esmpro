"""fs00: convert the FS screen workbooks (pd_data/FS*.xlsx) to TSV, one file per sheet, in data/fs_data/.
Read-only streaming (openpyxl read_only) so the 50 MB sheets load without building the full object model."""
import os, sys, csv, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".pylib"))
import openpyxl
ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
OUT = f"{ROOT}/data/fs_data"; os.makedirs(OUT, exist_ok=True)
for f in sorted(glob.glob(f"{ROOT}/pd_data/FS*.xlsx")):
    wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
    stem = os.path.basename(f)[:-5]
    for ws in wb.worksheets:
        ws.reset_dimensions()   # some writers store a wrong <dimension>, which truncates read_only iteration to one row
        p = f"{OUT}/{stem}__{ws.title}.tsv"; n = 0
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            for row in ws.iter_rows(values_only=True):
                w.writerow(["" if v is None else v for v in row]); n += 1
        print(f"{stem} / {ws.title}: {n} rows -> {p}", flush=True)
