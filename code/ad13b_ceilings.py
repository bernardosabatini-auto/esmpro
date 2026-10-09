"""ad13b: measurement ceilings for the ad13 targets.

Enrichment (per co-chaperone, average, and with abundance removed): split-half reliability of the
pull-down term (BR1-3 vs BR4-6), Spearman-Brown to six; the input term is shared by both halves but its
noise is negligible (reliability 0.997). Heat responses: raw split-half reliability; for the selective part,
1 - (noise variance of the raw response) / (variance of the selective response), the noise variance taken
from the two halves (var(half A - half B) / 4) -- approximate, since removing the lysate-like part also
moves some 35 C noise.
"""
import os, json
import numpy as np
ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
GA = f"{ROOT}/data/ga_data"
blk = np.load(f"{GA}/blocks.npz", allow_pickle=True); EN = np.load(f"{GA}/enrichment.npz", allow_pickle=True)
SH = np.load(f"{GA}/selective_heat.npz"); FG = np.load(f"{GA}/folds_ga05.npz", allow_pickle=True)
inp, con = EN["input"], EN["contaminant"].astype(bool)
A, B = [0, 1, 2], [3, 4, 5]
def bm(c, s, T, cols):
    with np.errstate(invalid="ignore"):
        return np.nanmean(blk[f"br_{c}_{s}_{T}"][:, cols], 1)
def sb(x, y):
    m = np.isfinite(x) & np.isfinite(y) & ~con; r = np.corrcoef(x[m], y[m])[0, 1]; return 2 * r / (1 + r)
def detr(y):
    m = np.isfinite(y) & np.isfinite(inp) & ~con; b = np.polyfit(inp[m], y[m], 1); return y - np.polyval(b, inp)
out = {}
h = {c: (bm(c, "0", "35", A) - inp, bm(c, "0", "35", B) - inp) for c in ("21A", "24")}
out["enrichment 35 DNAJA1"] = sb(*h["21A"]); out["enrichment 35 DNAJB11"] = sb(*h["24"])
av = ((h["21A"][0] + h["24"][0]) / 2, (h["21A"][1] + h["24"][1]) / 2); out["enrichment 35 avg"] = sb(*av)
out["enrichment 35 DNAJA1, abundance removed"] = sb(detr(h["21A"][0]), detr(h["21A"][1]))
out["enrichment 35 DNAJB11, abundance removed"] = sb(detr(h["24"][0]), detr(h["24"][1]))
out["enrichment 35 avg, abundance removed"] = sb(detr(av[0]), detr(av[1]))
for T, sel in (("43", SH["sel43"]), ("37", SH["sel37"])):
    ya = np.mean([bm(c, "0", T, A) - bm(c, "0", "35", A) for c in ("21A", "24")], 0)
    yb = np.mean([bm(c, "0", T, B) - bm(c, "0", "35", B) for c in ("21A", "24")], 0)
    m = np.isfinite(ya) & np.isfinite(yb) & np.isfinite(sel) & ~con
    out[f"heat {T} raw"] = sb(ya, yb)
    out[f"heat {T} selective (approx.)"] = float(1 - np.var(ya[m] - yb[m]) / 4 / np.var(sel[m]))
for k, v in out.items(): print(f"  {k:<44} ceiling {100*v:.1f}%")
json.dump(out, open(f"{GA}/ad13b_ceilings.json", "w"), indent=1)
