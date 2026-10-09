"""ad14: do the heat class effects survive removal of the lysate-like component (ad11)?

Earlier (pd15-style SAE reading, main report section 4): folded cytoplasmic enzymes gain at 43 C,
disordered and membrane proteins lose. Classes here, from UniProt and the ESMC SAE (layer 60, as ad01b):
transmembrane, nucleus only, cytoplasmic soluble, enzymes (EC-class keywords), disorder feature 654 and membrane-helix
feature 10715 (top quartile of activation). Mean response of each class minus the rest, raw and selective,
with 95 % intervals resampling sequence clusters.
"""
import os, csv
import numpy as np
ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
GA = f"{ROOT}/data/ga_data"
blk = np.load(f"{GA}/blocks.npz", allow_pickle=True); FG = np.load(f"{GA}/folds_ga05.npz", allow_pickle=True)
SH = np.load(f"{GA}/selective_heat.npz"); EN = np.load(f"{GA}/enrichment.npz", allow_pickle=True)
st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{GA}/sequences_status.tsv"), delimiter="\t")}
acc = np.array([st.get(str(a), str(a)) for a in blk["accession"]])
ann = {r["Entry"]: r for r in csv.DictReader(open(f"{ROOT}/data/annot/uniprot_annot.tsv"), delimiter="\t")}
S = np.load(f"{ROOT}/data/sae/sae_l60.npz", allow_pickle=True); SMAX = S["max"]; spos = {str(a): i for i, a in enumerate(S["accession"])}
rng = np.random.default_rng(0)
rows = FG["rows_temp_avg_43v35"]; G = FG["groups"]
m0 = np.zeros(len(acc), bool); m0[rows] = True; m0 &= np.isfinite(SH["sel43"])
loc = [ann.get(a, {}).get("Subcellular location [CC]", "") for a in acc]; kw = [ann.get(a, {}).get("Keywords", "") for a in acc]
def sae(f):
    v = np.array([np.log1p(float(SMAX[spos[a], f])) if a in spos else np.nan for a in acc])
    return v > np.nanpercentile(v[m0], 75)                    # top quartile of activation among the response proteins
CL = {"transmembrane": np.array(["Transmembrane" in k for k in kw]),
      "nucleus only": np.array([("Nucleus" in l) and ("Cytoplasm" not in l) for l in loc]),
      "cytoplasmic, soluble": np.array([("Cytoplasm" in l) and ("Nucleus" not in l) and ("Transmembrane" not in k) for l, k in zip(loc, kw)]),
      "enzymes (EC keywords)": np.array([any(x in k for x in ("Oxidoreductase", "Transferase", "Hydrolase", "Lyase", "Isomerase", "Ligase")) for k in kw]),
      "SAE disorder feature, top quartile": sae(654), "SAE membrane-helix feature 10715, top quartile": sae(10715)}
print(f"{m0.sum()} proteins; class minus rest, 43 vs 35 C [95 % cluster bootstrap]", flush=True)
ug, inv = np.unique(G[m0], return_inverse=True); mem = [np.nonzero(inv == k)[0] for k in range(len(ug))]
idx = np.nonzero(m0)[0]
for nm, c in CL.items():
    line = []
    for lab, y in (("raw", SH["raw43"]), ("lysate-like", SH["fit43"]), ("selective", SH["sel43"])):
        yy, cc = y[idx], c[idx]
        d = yy[cc].mean() - yy[~cc].mean()
        bs = []
        for _ in range(500):
            s = np.concatenate([mem[k] for k in rng.integers(0, len(ug), len(ug))]); bs.append(yy[s][cc[s]].mean() - yy[s][~cc[s]].mean())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        line.append(f"{lab} {d:+.2f} [{lo:+.2f},{hi:+.2f}]")
    print(f"  {nm:<28} n {c[idx].sum():>5}: " + "   ".join(line), flush=True)

# ---- 2. the DNAJB11 - DNAJA1 preference by quartile of average 35 C enrichment (abundance removed).
# Both pull-downs share one input, so the input cancels from the preference; the question is whether the
# preference is carried by proteins bound beyond their lysate level, or spread across the background too.
print("\n2. DNAJB11 - DNAJA1 preference (35 C control) by quartile of 35 C enrichment at fixed abundance", flush=True)
inp, con = EN["input"], EN["contaminant"].astype(bool)
def bm(c, cols):
    with np.errstate(invalid="ignore"):
        return np.nanmean(blk[f"br_{c}_0_35"][:, cols], 1)
ok4 = lambda c: np.isfinite(blk[f"br_{c}_0_35"]).sum(1) >= 4
ea = (bm("21A", list(range(6))) + bm("24", list(range(6)))) / 2 - inp
m = np.isfinite(ea) & ok4("21A") & ok4("24") & ~con
b_ = np.polyfit(inp[m], ea[m], 1); ed = ea - np.polyval(b_, inp)
p1 = bm("24", [0, 1, 2]) - bm("21A", [0, 1, 2]); p2 = bm("24", [3, 4, 5]) - bm("21A", [3, 4, 5]); pref = (p1 + p2) / 2
q = np.percentile(ed[m], [25, 50, 75]); qb = np.digitize(ed, q)
for k in range(4):
    s = m & (qb == k) & np.isfinite(p1) & np.isfinite(p2)
    r = np.corrcoef(p1[s], p2[s])[0, 1]
    print(f"  quartile {k+1} (enrichment {np.median(ed[s]):+.2f}): n {s.sum()}, sd of preference {pref[s].std():.3f}, "
          f"reliability {2*r/(1+r):.3f}, true-signal sd {pref[s].std()*np.sqrt(max(2*r/(1+r),0)):.3f}", flush=True)
print("AD14_DONE")
