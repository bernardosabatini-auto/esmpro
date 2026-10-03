"""ad05: proteins that appear or vanish between conditions.

The fold-change targets keep a protein only when both groups have >= 4 of 6 values (GA_33) or >= 7 of
10 (salt). Proteins detected in one condition and essentially absent in the other are excluded, though
they may be the strongest responses of all. Here, per contrast:
    appears  : detected in >= 4 of 6 replicates of the condition, <= 1 of 6 of the reference
    vanishes : the reverse
(salt: >= 7 of 10 vs <= 2 of 10.) For GA_33 the two co-chaperone pull-downs are independent samples,
so an event is called reproducible when both show it; agreement beyond chance is the test of whether
on/off calls are biology or detection-floor flicker. Intensity of the detected side shows whether the
events sit at the detection floor.
"""
import os, csv, json, itertools
import numpy as np
from scipy import stats

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
GA, PD = f"{ROOT}/data/ga_data", f"{ROOT}/data/pd_data"
g = np.load(f"{GA}/blocks.npz", allow_pickle=True)
B = lambda c, s, t: g[f"br_{c}_{s}_{t}"]
CON = {"heat 43": (("0", "43"), ("0", "35")), "heat 37": (("0", "37"), ("0", "35")),
       "stau 35": (("10", "35"), ("0", "35")), "stau 37": (("10", "37"), ("0", "37")), "stau 43": (("10", "43"), ("0", "43"))}
out = {}
print("GA_33: on/off events per contrast, per co-chaperone and agreement between the two independent pull-downs", flush=True)
print(f"  {'contrast':<9}{'event':<9}{'DNAJA1':>8}{'DNAJB11':>9}{'both':>6}{'expected':>10}{'x chance':>10}{'median log2 int. (detected side), events / kept':>50}", flush=True)
for nm, ((s1, t1), (s0, t0)) in CON.items():
    rec = {}
    calls = {}
    for c in ("21A", "24"):
        n1 = np.isfinite(B(c, s1, t1)).sum(1); n0 = np.isfinite(B(c, s0, t0)).sum(1)
        calls[c] = dict(appears=(n1 >= 4) & (n0 <= 1), vanishes=(n0 >= 4) & (n1 <= 1), kept=(n1 >= 4) & (n0 >= 4),
                        seen=(n1 >= 1) | (n0 >= 1))
    for ev in ("appears", "vanishes"):
        a, b = calls["21A"][ev], calls["24"][ev]
        seen = calls["21A"]["seen"] & calls["24"]["seen"]
        exp = a[seen].mean() * b[seen].mean() * seen.sum()
        both = (a & b).sum()
        side = (s1, t1) if ev == "appears" else (s0, t0)
        I = np.nanmean(np.hstack([B(c, *side) for c in ("21A", "24")]), 1)
        Ik = np.nanmean(np.hstack([B(c, *side) for c in ("21A", "24")]), 1)
        kept = calls["21A"]["kept"] & calls["24"]["kept"]
        rec[ev] = dict(dnaja1=int(a.sum()), dnajb11=int(b.sum()), both=int(both), expected=float(exp),
                       enrichment=float(both / exp) if exp > 0 else None,
                       median_intensity_events=float(np.nanmedian(I[a & b])) if both else None,
                       median_intensity_kept=float(np.nanmedian(Ik[kept])))
        print(f"  {nm:<9}{ev:<9}{a.sum():>8}{b.sum():>9}{both:>6}{exp:>10.1f}{(both/exp if exp else float('nan')):>10.1f}"
              f"{rec[ev]['median_intensity_events'] if both else float('nan'):>34.2f} / {rec[ev]['median_intensity_kept']:.2f}", flush=True)
    out[nm] = rec
# salt
p = np.load(f"{PD}/blocks.npz", allow_pickle=True)
print("\nSalt titration (HSPB1): >= 7 of 10 in one condition, <= 2 of 10 in the other", flush=True)
for c, nm in (("s150", "salt 150"), ("s75", "salt 75")):
    n1 = np.isfinite(p[f"raw_{c}"]).sum(1); n0 = np.isfinite(p["raw_base"]).sum(1)
    ap, va, kp = (n1 >= 7) & (n0 <= 2), (n0 >= 7) & (n1 <= 2), (n1 >= 7) & (n0 >= 7)
    I1 = np.nanmean(p[f"raw_{c}"], 1); I0 = np.nanmean(p["raw_base"], 1)
    out[nm] = dict(appears=int(ap.sum()), vanishes=int(va.sum()), kept=int(kp.sum()),
                   med_int_appears=float(np.nanmedian(I1[ap])) if ap.any() else None, med_int_vanishes=float(np.nanmedian(I0[va])) if va.any() else None,
                   med_int_kept=float(np.nanmedian(I0[kp])))
    print(f"  {nm}: appear {ap.sum()}, vanish {va.sum()}, kept {kp.sum()}; median log2 intensity of detected side: appearing "
          f"{out[nm]['med_int_appears'] if ap.any() else float('nan'):.2f}, vanishing {out[nm]['med_int_vanishes'] if va.any() else float('nan'):.2f}, kept {out[nm]['med_int_kept']:.2f}", flush=True)
json.dump(out, open(f"{GA}/ad05_counts.json", "w"), indent=1)
print("\nAD05_DONE")
