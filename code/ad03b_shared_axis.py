"""ad03b: the shared-axis analysis redone without the two flaws found in ad03.

ad03 (a) measured "shared with the others" against the first component of all other responses,
which the three staurosporine targets dominate, and (b) corrected correlations for noise as if each
response's noise were independent. Within GA_33 it is not: heat, staurosporine and preference reuse
the same control samples (heat 43 = control 43 - control 35; staurosporine at T = treated T - control T),
so shared noise enters their correlations with a sign set by the arithmetic (heat 37 vs stau 35
+0.14, vs stau 37 -0.13 in ad03 is that signature).

Here:
  * GA_33 pairs: each response computed from replicates 1-3 and from 4-6 (the same split for every
    sample group); the correlation of A from one half with B from the other, averaged over both
    directions, is free of shared noise; corrected for attenuation by the half-replicate reliability
    of each response, r_half = R / (2 - R).
  * salt vs GA_33: independent experiments, so the full-data correlation corrected by full
    reliabilities is valid.
  * "shared": how much of each response the OTHER experiment predicts (cross-validated multiple
    regression on all of its responses), and how much ESMC predicts of what is left.
"""
import os, csv, json
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
PD, GA = f"{ROOT}/data/pd_data", f"{ROOT}/data/ga_data"


def resolved(d):
    blk = np.load(f"{d}/blocks.npz", allow_pickle=True)
    st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{d}/sequences_status.tsv"), delimiter="\t")}
    return blk, np.array([st.get(str(a), str(a)) for a in blk["accession"]])


pblk, pacc = resolved(PD); gblk, gacc = resolved(GA)
FG = np.load(f"{GA}/folds_ga05.npz", allow_pickle=True); SP = np.load(f"{GA}/spec.npz", allow_pickle=True)
ga5 = json.load(open(f"{GA}/ga05_results.json")); g12 = json.load(open(f"{GA}/ga12_summary.json"))
M = lambda c, s, t, cols: gblk[f"br_{c}_{s}_{t}"][:, cols]
CH = ("21A", "24")


def ga_resp(name, cols):
    """GA_33 response from a subset of replicate columns (both co-chaperones averaged)."""
    with np.errstate(invalid="ignore"):
        if name.startswith("heat"):
            T = name.split()[1]
            return np.mean([np.nanmean(M(c, "0", T, cols), 1) - np.nanmean(M(c, "0", "35", cols), 1) for c in CH], 0)
        if name.startswith("stau"):
            T = name.split()[1]
            return np.mean([np.nanmean(M(c, "10", T, cols), 1) - np.nanmean(M(c, "0", T, cols), 1) for c in CH], 0)
        return np.mean([np.nanmean(M("24", "0", T, cols), 1) - np.nanmean(M("21A", "0", T, cols), 1) for T in ("35", "37", "43")], 0)


GAN = ["heat 37", "heat 43", "stau 35", "stau 37", "stau 43", "DNAJB11 pref"]
REL = {"heat 37": ga5["temp_avg_37v35"]["ceiling"], "heat 43": ga5["temp_avg_43v35"]["ceiling"], "stau 35": ga5["stau_avg_35"]["ceiling"],
       "stau 37": ga5["stau_avg_37"]["ceiling"], "stau 43": ga5["stau_avg_43"]["ceiling"], "DNAJB11 pref": g12["spec_avg"]["reliability"],
       "salt 75": 0.986, "salt 150": 0.993}
full = {}
for t, nm in (("temp_avg_37v35", "heat 37"), ("temp_avg_43v35", "heat 43"), ("stau_avg_35", "stau 35"), ("stau_avg_37", "stau 37"), ("stau_avg_43", "stau 43")):
    full[nm] = dict(zip(gacc[FG[f"rows_{t}"]], FG[f"y_{t}"].astype(float)))
full["DNAJB11 pref"] = dict(zip(gacc[SP["rows_spec_avg"]], SP["y_spec_avg"].astype(float)))
for c, nm in (("s75", "salt 75"), ("s150", "salt 150")):
    k = pblk[f"keep_{c}"].astype(bool) & np.isfinite(pblk[f"y_{c}"])
    full[nm] = dict(zip(pacc[k], pblk[f"y_{c}"][k].astype(float)))
names = ["salt 75", "salt 150"] + GAN
common = sorted(set.intersection(*[set(v) for v in full.values()]))
gpos = {a: i for i, a in enumerate(gacc)}; gi = np.array([gpos[a] for a in common])
print(f"{len(common)} proteins in all eight responses", flush=True)

# GA_33 halves
H1 = {n: ga_resp(n, [0, 1, 2])[gi] for n in GAN}
H2 = {n: ga_resp(n, [3, 4, 5])[gi] for n in GAN}
Y = {n: np.array([full[n][a] for a in common]) for n in names}


def rr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[ok], b[ok])[0, 1]


nN = len(names)
Craw = np.eye(nN); Ccor = np.eye(nN)
for i in range(nN):
    for j in range(i + 1, nN):
        a, b = names[i], names[j]
        Craw[i, j] = Craw[j, i] = rr(Y[a], Y[b])
        if a in GAN and b in GAN:
            d = 0.5 * (rr(H1[a], H2[b]) + rr(H2[a], H1[b]))
            ha, hb = REL[a] / (2 - REL[a]), REL[b] / (2 - REL[b])
            Ccor[i, j] = Ccor[j, i] = d / np.sqrt(ha * hb)
        else:
            Ccor[i, j] = Ccor[j, i] = Craw[i, j] / np.sqrt(REL[a] * REL[b])
print("\n1. CORRELATIONS (upper: raw, same samples; lower: free of shared noise and corrected for measurement noise)", flush=True)
print(" " * 13 + "".join(f"{n:>13}" for n in names), flush=True)
for i, n in enumerate(names):
    print(f"{n:<13}" + "".join(f"{(Craw[i,j] if j > i else (Ccor[i,j] if j < i else 1)):>+13.2f}" for j in range(nN)), flush=True)
w, V = np.linalg.eigh(Ccor); o = np.argsort(-w); w, V = w[o], V[:, o]
print(f"\n2. COMPONENTS of the corrected correlation matrix (eigenvalues {', '.join(f'{x:.2f}' for x in w)})", flush=True)
for k in range(3):
    s = np.sign(V[0, k]) or 1
    print(f"  C{k+1}: {100*w[k]/nN:4.1f}%; " + ", ".join(f"{n} {s*V[i,k]:+.2f}" for i, n in enumerate(names)), flush=True)

# 3. what the other experiment predicts, and what ESMC predicts of the rest
cl = {r["accession"]: r["cluster"] for r in csv.DictReader(open(f"{GA}/clusters.tsv"), delimiter="\t")}
G = np.array([cl.get(a, a) for a in common]); folds = list(GroupKFold(n_splits=5).split(np.zeros(len(common)), None, G))
E = np.load(f"{GA}/emb_L50.npy")[gi]


def cvr2(X, y, model):
    p = np.zeros(len(y))
    for tr, te in folds:
        p[te] = model().fit(X[tr], y[tr]).predict(X[te])
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2), p


def esmc(y):
    return max(cvr2(E, y, lambda: make_pipeline(StandardScaler(), Ridge(alpha=al)))[0] for al in (1e3, 1e4, 1e5))


print(f"\n3. WHAT THE OTHER EXPERIMENT EXPLAINS, AND WHAT SEQUENCE EXPLAINS OF THE REST (cross-validated, cluster folds)", flush=True)
print(f"  {'response':<13}{'other experiment explains':>27}{'ESMC on response':>18}{'ESMC on the rest':>18}", flush=True)
res = {}
for n in names:
    others = GAN if n.startswith("salt") else ["salt 75", "salt 150"]
    Xo = np.column_stack([Y[o] for o in others]); y = Y[n]
    r_o, p_o = cvr2(Xo, y, LinearRegression)
    rest = y - p_o
    res[n] = dict(other_experiment=float(r_o), esmc=float(esmc(y)), esmc_rest=float(esmc(rest)))
    print(f"  {n:<13}{100*r_o:>26.1f}%{100*res[n]['esmc']:>17.1f}%{100*res[n]['esmc_rest']:>17.1f}%", flush=True)
json.dump(dict(n=len(common), names=names, corr_raw=Craw.tolist(), corr_corrected=Ccor.tolist(), eig=w.tolist(),
               loadings=V[:, :3].tolist(), cross_experiment=res), open(f"{GA}/ad03b_results.json", "w"), indent=1)
print("\nAD03B_DONE", flush=True)

# ---- what the salt-heat axis is, with intervals (appended)
rng = np.random.default_rng(0)
ug, inv = np.unique(G, return_inverse=True); mem = [np.nonzero(inv == k)[0] for k in range(len(ug))]
boot = [np.concatenate([mem[k] for k in rng.integers(0, len(ug), len(ug))]) for _ in range(1000)]


def ci_r(a, b):
    v = [np.corrcoef(a[s], b[s])[0, 1] for s in boot]
    return float(np.corrcoef(a, b)[0, 1]), [float(x) for x in np.percentile(v, [2.5, 97.5])]


print("\n4. KEY CORRELATIONS, 95 % intervals over sequence clusters", flush=True)
key = {}
for a, b in (("salt 150", "heat 43"), ("salt 75", "heat 43"), ("salt 150", "heat 37"), ("salt 150", "DNAJB11 pref"), ("salt 150", "stau 37")):
    r, c = ci_r(Y[a], Y[b]); key[f"{a} | {b}"] = [r, c]
    print(f"  {a} vs {b}: {r:+.3f} [{c[0]:+.3f},{c[1]:+.3f}]", flush=True)
for a, b in (("heat 43", "DNAJB11 pref"), ("heat 37", "DNAJB11 pref"), ("stau 35", "DNAJB11 pref")):
    ok = np.isfinite(H1[a]) & np.isfinite(H2[b]) & np.isfinite(H2[a]) & np.isfinite(H1[b])
    v = [0.5 * (np.corrcoef(H1[a][s][ok[s]], H2[b][s][ok[s]])[0, 1] + np.corrcoef(H2[a][s][ok[s]], H1[b][s][ok[s]])[0, 1]) for s in boot]
    d = 0.5 * (rr(H1[a], H2[b]) + rr(H2[a], H1[b]))
    key[f"{a} | {b} (disjoint halves, uncorrected)"] = [float(d), [float(x) for x in np.percentile(v, [2.5, 97.5])]]
    print(f"  {a} vs {b} (disjoint halves, before attenuation correction): {d:+.3f} [{np.percentile(v,2.5):+.3f},{np.percentile(v,97.5):+.3f}]", flush=True)

ax = (Y["salt 150"] - Y["salt 150"].mean()) / Y["salt 150"].std() - (Y["heat 43"] - Y["heat 43"].mean()) / Y["heat 43"].std()
ann = {r["Entry"]: r for r in csv.DictReader(open(f"{ROOT}/data/annot/uniprot_annot.tsv"), delimiter="\t")}
kw = [set(k.strip() for k in ann.get(a, {}).get("Keywords", "").split(";")) for a in common]
loc = [ann.get(a, {}).get("Subcellular location [CC]", "") for a in common]
print("\n5. WHAT THE SALT-HEAT AXIS IS: mean of (salt 150 - heat 43, standardised) by class; > 0 = holds HSPB1 with salt, falls with heat", flush=True)
cls = {"Transmembrane": np.array(["Transmembrane" in k for k in kw]), "Signal peptide": np.array(["Signal" in k for k in kw]),
       "Mitochondrion": np.array(["Mitochondri" in l for l in loc]), "Nucleus only": np.array([("Nucleus" in l) and ("Cytoplasm" not in l) for l in loc]),
       "Cytoplasm only": np.array([("Cytoplasm" in l) and ("Nucleus" not in l) for l in loc]), "Ribosomal / RNA-binding": np.array([("Ribonucleoprotein" in k) or ("RNA-binding" in k) for k in kw]),
       "Kinase": np.array(["Kinase" in k for k in kw])}
from scipy import stats as st_
cres = {}
for nm, m in cls.items():
    cres[nm] = dict(n=int(m.sum()), mean=float(ax[m].mean()), salt=float(Y["salt 150"][m].mean()), heat=float(Y["heat 43"][m].mean()),
                    p=float(st_.mannwhitneyu(ax[m], ax[~m]).pvalue))
    print(f"  {nm:<26} n {m.sum():>4}  axis {ax[m].mean():+.2f}  (salt 150 {Y['salt 150'][m].mean():+.2f}, heat 43 {Y['heat 43'][m].mean():+.2f})  p {cres[nm]['p']:.0e}", flush=True)
tm = cls["Transmembrane"]
rest = ~tm
r_all, c_all = ci_r(Y["salt 150"], Y["heat 43"])
r_nt = float(np.corrcoef(Y["salt 150"][rest], Y["heat 43"][rest])[0, 1])
print(f"\n  salt-heat correlation: all {r_all:+.3f}; without transmembrane proteins {r_nt:+.3f} (n {rest.sum()})", flush=True)
o = json.load(open(f"{GA}/ad03b_results.json")); o.update(key_correlations=key, axis_classes=cres, salt_heat_without_tm=r_nt)
json.dump(o, open(f"{GA}/ad03b_results.json", "w"), indent=1)
