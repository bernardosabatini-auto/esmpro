"""ga11: which proteins are well predicted, and does that change with temperature or family?

Reads out-of-fold predictions (ga10 for GA_33, pd09 for the salt titration). "The model" is the
50/50 average of the neural-network ensemble and ridge, the best general form; ridge alone is
reported alongside where it differs.

Per protein, the explained part of its squared deviation:
    c_i = (y_i - mean y)^2 - (y_i - p_i)^2
positive when the model accounts for that protein's deviation from the mean. Summed over proteins
and divided by the total sum of squares this is exactly R2, so c_i apportions the explained variance
among proteins, and therefore among temperatures and families.

1. Across temperatures. For each pair of targets: correlation of the true responses, of the
   predictions, and of c_i; how much the 5 % best-explained proteins overlap beyond chance; and how
   well one temperature's predictions predict another temperature's response.
2. Across families (UniProt "protein families", top level, >= 15 members) and compartments
   (UniProt subcellular location). How much of the response variance lies between families (eta^2),
   how much the model captures by getting family averages right (R2 of the per-family mean
   prediction) versus within families, and which families carry the most explained variance per
   protein.
"""
import os, csv, json, re
import numpy as np
from scipy import stats

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
GA, PD = f"{ROOT}/data/ga_data", f"{ROOT}/data/pd_data"
rng = np.random.default_rng(0)
MINF = 15

ann = {r["Entry"]: r for r in csv.DictReader(open(f"{ROOT}/data/annot/uniprot_annot.tsv"), delimiter="\t")}


def family(acc):
    f = ann.get(acc, {}).get("Protein families", "").strip()
    return f.split(",")[0].split(";")[0].strip() if f else ""


COMP = [("Nucleus", r"\bNucleus\b|nucleol|Nucleus speckle"), ("Cytoplasm", r"\bCytoplasm\b|cytosol"),
        ("Mitochondrion", r"Mitochondri"), ("Endoplasmic reticulum", r"Endoplasmic reticulum"),
        ("Golgi", r"Golgi"), ("Membrane", r"[Mm]embrane"), ("Secreted", r"\bSecreted\b"),
        ("Cytoskeleton", r"[Cc]ytoskeleton"), ("Ribosome / translation", r"[Rr]ibosom")]


def comps(acc):
    loc = ann.get(acc, {}).get("Subcellular location [CC]", "")
    return {c for c, pat in COMP if re.search(pat, loc)}


def resolve(data_dir):
    blk = np.load(f"{data_dir}/blocks.npz", allow_pickle=True)
    st = {r["sheet_accession"]: r["resolved_accession"]
          for r in csv.DictReader(open(f"{data_dir}/sequences_status.tsv"), delimiter="\t")}
    acc = np.array([st.get(str(a), str(a)) for a in blk["accession"]])
    gene = np.array([str(g) for g in blk["gene"]])
    return acc, gene


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


sets = {}
# --- GA_33 from ga10
acc_ga, gene_ga = resolve(GA)
for famname in ("heat", "stau"):
    fn = f"{GA}/ga10_{famname}.npz"
    if not os.path.exists(fn):
        print(f"(missing {fn})"); continue
    Z = np.load(fn, allow_pickle=True)
    rows, Y, TG = Z["rows"], Z["Y"], [str(t) for t in Z["targets"]]
    blend = 0.5 * Z["separate"] + 0.5 * Z["ridge"]
    for k, t in enumerate(TG):
        sets[t] = dict(acc=acc_ga[rows], gene=gene_ga[rows], y=Y[:, k], p=blend[:, k], pr=Z["ridge"][:, k], group=famname)
# --- salt from pd09
acc_pd, gene_pd = resolve(PD)
FP = np.load(f"{PD}/folds.npz", allow_pickle=True)
for c, lab in (("s75", "salt_75"), ("s150", "salt_150")):
    Yv, rid, mlp = np.load(f"{PD}/pd09_oof_{c}.npy")
    rows = FP[f"rows_{c}"]
    sets[lab] = dict(acc=acc_pd[rows], gene=gene_pd[rows], y=Yv, p=0.5 * rid + 0.5 * mlp, pr=rid, group="salt")
print("targets:", {k: (len(v["y"]), round(100 * r2(v["y"], v["p"]), 2)) for k, v in sets.items()}, flush=True)
out = {"pairs": {}, "families": {}, "compartments": {}}


def contrib(y, p):
    return (y - y.mean()) ** 2 - (y - p) ** 2


# ======================================================================= 1. across temperatures
print("\n1. ACROSS TEMPERATURES (same proteins, model = MLP+ridge average)", flush=True)
print(f"{'pair':<34}{'r(y)':>7}{'r(pred)':>9}{'r(c_i)':>8}{'top-5% overlap':>17}"
      f"{'own R2':>9}{'R2 from other':>15}", flush=True)
pairs = [("stau_avg_35", "stau_avg_37"), ("stau_avg_37", "stau_avg_43"), ("stau_avg_35", "stau_avg_43"),
         ("temp_avg_37v35", "temp_avg_43v35"), ("salt_75", "salt_150")]
for A, B in pairs:
    if A not in sets or B not in sets:
        continue
    a_, b_ = sets[A], sets[B]
    common = np.intersect1d(a_["acc"], b_["acc"])
    ia = {x: i for i, x in enumerate(a_["acc"])}; ib = {x: i for i, x in enumerate(b_["acc"])}
    ia_, ib_ = np.array([ia[x] for x in common]), np.array([ib[x] for x in common])
    yA, yB, pA, pB = a_["y"][ia_], b_["y"][ib_], a_["p"][ia_], b_["p"][ib_]
    cA, cB = contrib(yA, pA), contrib(yB, pB)
    k5 = int(0.05 * len(common))
    tA, tB = set(np.argsort(-cA)[:k5]), set(np.argsort(-cB)[:k5])
    ov = len(tA & tB) / (k5 * k5 / len(common))
    # how well do A's predictions, linearly rescaled, predict B? (rescaling is the 'new readout')
    sl, ic = np.polyfit(pA, yB, 1)
    r_cross = r2(yB, sl * pA + ic)
    rec = dict(n=int(len(common)), r_y=float(stats.pearsonr(yA, yB)[0]), r_pred=float(stats.pearsonr(pA, pB)[0]),
               r_contrib=float(stats.spearmanr(cA, cB)[0]), top5_overlap_x=float(ov),
               own_r2_B=float(r2(yB, pB)), r2_B_from_A=float(r_cross))
    out["pairs"][f"{A}|{B}"] = rec
    print(f"{A+' vs '+B:<34}{rec['r_y']:>7.2f}{rec['r_pred']:>9.2f}{rec['r_contrib']:>8.2f}"
          f"{ov:>13.1f}x   {100*rec['own_r2_B']:>7.2f}%{100*r_cross:>13.2f}%", flush=True)

# ======================================================================= 2. across families
print(f"\n2. ACROSS PROTEIN FAMILIES (UniProt top-level family, >= {MINF} members)", flush=True)
for t in ("salt_150", "temp_avg_43v35", "stau_avg_37"):
    if t not in sets:
        continue
    s = sets[t]
    y, p = s["y"], s["p"]
    fam = np.array([family(x) for x in s["acc"]])
    c = contrib(y, p)
    sst = np.sum((y - y.mean()) ** 2)
    names, cnt = np.unique(fam[fam != ""], return_counts=True)
    big = set(names[cnt >= MINF])
    inb = np.array([f in big for f in fam])
    # between-family share of the TRUE response, and of what the model captures
    ym_f = {f: y[fam == f].mean() for f in big}
    pm_f = {f: p[fam == f].mean() for f in big}
    yf = np.array([ym_f.get(f, np.nan) for f in fam]); pf = np.array([pm_f.get(f, np.nan) for f in fam])
    eta2 = np.sum((yf[inb] - y[inb].mean()) ** 2) / np.sum((y[inb] - y[inb].mean()) ** 2)
    r2_all = r2(y[inb], p[inb])
    r2_fammean = r2(y[inb], pf[inb])             # the model, with every protein given its family's mean prediction
    r2_within = stats.pearsonr(y[inb] - yf[inb], p[inb] - pf[inb])[0]
    print(f"\n  {t}: {inb.sum()} proteins in {len(big)} families of >= {MINF}  (model R2 on them {100*r2_all:.1f}%)", flush=True)
    print(f"    share of the true response that is between families (eta2):   {100*eta2:.1f}%", flush=True)
    print(f"    R2 from family-average predictions alone:                      {100*r2_fammean:.1f}%  "
          f"(= {100*r2_fammean/max(r2_all,1e-9):.0f}% of the model's)", flush=True)
    print(f"    within-family correlation of prediction with truth:            {r2_within:+.3f}", flush=True)
    rows_ = []
    for f in big:
        m = fam == f
        rows_.append(dict(family=f, n=int(m.sum()), mean_y=float(y[m].mean()), mean_p=float(p[m].mean()),
                          share_expl=float(c[m].sum() / max(c.sum(), 1e-12)), share_prot=float(m.mean()),
                          r2_fam=float(1 - np.sum((y[m] - p[m]) ** 2) / np.sum((y[m] - y.mean()) ** 2)),
                          r_within=float(stats.pearsonr(y[m], p[m])[0]) if m.sum() > 3 and p[m].std() > 0 else float("nan")))
    rows_.sort(key=lambda d: -d["r2_fam"])
    hdr = f"    {'family':<52}{'n':>5}{'mean y':>8}{'mean pred':>10}{'R2 here':>9}{'r within':>9}"
    print("    best predicted families:\n" + hdr, flush=True)
    for d in rows_[:8]:
        print(f"    {d['family'][:51]:<52}{d['n']:>5}{d['mean_y']:>+8.2f}{d['mean_p']:>+10.2f}{100*d['r2_fam']:>8.1f}%{d['r_within']:>+9.2f}", flush=True)
    print("    worst predicted families:\n" + hdr, flush=True)
    for d in rows_[-6:]:
        print(f"    {d['family'][:51]:<52}{d['n']:>5}{d['mean_y']:>+8.2f}{d['mean_p']:>+10.2f}{100*d['r2_fam']:>8.1f}%{d['r_within']:>+9.2f}", flush=True)
    # compartments
    print(f"    by compartment:{'n':>42}{'mean y':>8}{'mean pred':>10}{'R2 here':>9}{'r within':>9}", flush=True)
    cm = [comps(x) for x in s["acc"]]
    crow = []
    for cname, _ in COMP:
        m = np.array([cname in z for z in cm])
        if m.sum() < 50:
            continue
        d = dict(compartment=cname, n=int(m.sum()), mean_y=float(y[m].mean()), mean_p=float(p[m].mean()),
                 r2_here=float(1 - np.sum((y[m] - p[m]) ** 2) / np.sum((y[m] - y.mean()) ** 2)),
                 r_within=float(stats.pearsonr(y[m], p[m])[0]))
        crow.append(d)
        print(f"      {cname:<46}{d['n']:>5}{d['mean_y']:>+8.2f}{d['mean_p']:>+10.2f}{100*d['r2_here']:>8.1f}%{d['r_within']:>+9.2f}", flush=True)
    out["families"][t] = dict(eta2=float(eta2), r2=float(r2_all), r2_family_mean=float(r2_fammean),
                              r_within=float(r2_within), n_families=len(big), table=rows_)
    out["compartments"][t] = crow

json.dump(out, open(f"{GA}/ga11_results.json", "w"), indent=1)
print("\nGA11_DONE", flush=True)
