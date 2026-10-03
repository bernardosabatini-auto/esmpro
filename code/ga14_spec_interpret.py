"""ga14: what distinguishes the proteins DNAJB11 pulls down from those DNAJA1 pulls down, and which
proteins shift between them with heat?

DNAJB11 (ERdj3) is an ER-lumenal J-protein that binds unfolded secretory proteins; DNAJA1 is
cytosolic. For each specificity target (ga12; > 0 = prefers DNAJB11) and for the change with heat:
  1. UniProt keywords: mean target for proteins carrying each keyword vs the rest (Mann-Whitney),
     keywords carried by >= 40 proteins, ranked by effect.
  2. Compartments (UniProt subcellular location) and families (>= 15 members): means, and how much of
     the best model's success is family averages.
  3. SAE features: distinct strongest features (near-duplicates collapsed at |r| > 0.7), labels
     checked against UniProt keywords of the proteins they fire on here, and category enrichment of
     the top 100 each way against 500 random features.
"""
import os, csv, json, re, collections
import numpy as np
from scipy import stats

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
D, SD = f"{ROOT}/data/ga_data", f"{ROOT}/data/sae"
SP = np.load(f"{D}/spec.npz", allow_pickle=True)
R = json.load(open(f"{D}/ga13_results.json"))
acc = np.array([str(a) for a in SP["accession"]])
ann = {r["Entry"]: r for r in csv.DictReader(open(f"{ROOT}/data/annot/uniprot_annot.tsv"), delimiter="\t")}
kw = {a: set(k.strip() for k in r.get("Keywords", "").split(";") if k.strip()) for a, r in ann.items()}
gene_of = {a: r.get("Gene Names (primary)", "") for a, r in ann.items()}
A = {}
for l in open(f"{SD}/feature_annot.jsonl"):
    d = json.loads(l); A[d["feature_index"]] = d
base = [int(x) for x in open(f"{SD}/features_baseline_sample.txt").read().split()]
S = np.load(f"{SD}/sae_l60.npz", allow_pickle=True); spos = {str(a): i for i, a in enumerate(S["accession"])}; MAX = S["max"]
COMP = [("Endoplasmic reticulum", r"Endoplasmic reticulum"), ("Golgi", r"Golgi"), ("Secreted", r"\bSecreted\b"),
        ("Lysosome / endosome", r"Lysosom|Endosom"), ("Mitochondrion", r"Mitochondri"), ("Nucleus", r"\bNucleus\b"),
        ("Cytoplasm", r"\bCytoplasm\b|cytosol"), ("Membrane", r"[Mm]embrane"), ("Cytoskeleton", r"[Cc]ytoskeleton")]
fam = lambda a: (ann.get(a, {}).get("Protein families", "").split(",")[0].split(";")[0].strip())
out = {}
for t in ("spec_avg", "dheat_43", "spec_35", "spec_43"):
    rows = SP[f"rows_{t}"]; y = SP[f"y_{t}"].astype(float); a_ = acc[rows]
    print(f"\n{'='*112}\n{t}  (n {len(y)}, sd {y.std():.3f}; > 0 = {'prefers DNAJB11' if t.startswith('spec') else 'shifts toward DNAJB11 at 43 C'})")
    # 1. keywords
    cnt = collections.Counter(k for x in a_ for k in kw.get(x, ()))
    kres = []
    for k, c in cnt.items():
        if c < 40 or c > 0.6 * len(y):
            continue
        m = np.array([k in kw.get(x, ()) for x in a_])
        d = y[m].mean() - y[~m].mean()
        kres.append((d, k, int(m.sum()), float(stats.mannwhitneyu(y[m], y[~m]).pvalue)))
    kres.sort()
    print("  keywords, toward DNAJB11: " + "; ".join(f"{k} {d:+.2f} (n {n}, p {p:.0e})" for d, k, n, p in kres[::-1][:9]))
    print("  keywords, toward DNAJA1:  " + "; ".join(f"{k} {d:+.2f} (n {n}, p {p:.0e})" for d, k, n, p in kres[:9]))
    # 2. compartments
    locs = [ann.get(x, {}).get("Subcellular location [CC]", "") for x in a_]
    crow = []
    for c, pat in COMP:
        m = np.array([bool(re.search(pat, l)) for l in locs])
        if m.sum() >= 50:
            crow.append((c, int(m.sum()), float(y[m].mean()), float(stats.mannwhitneyu(y[m], y[~m]).pvalue)))
    print("  compartments: " + "; ".join(f"{c} {mu:+.2f} (n {n}, p {p:.0e})" for c, n, mu, p in sorted(crow, key=lambda z: -z[2])))
    # families
    F = np.array([fam(x) for x in a_]); nm, ct = np.unique(F[F != ""], return_counts=True); big = nm[ct >= 15]
    fr = sorted(((y[F == f].mean(), f, int((F == f).sum())) for f in big))
    print("  families toward DNAJB11: " + "; ".join(f"{f[:38]} {m:+.2f} ({n})" for m, f, n in fr[::-1][:6]))
    print("  families toward DNAJA1:  " + "; ".join(f"{f[:38]} {m:+.2f} ({n})" for m, f, n in fr[:6]))
    oof = np.load(f"{D}/ga13_oof_{t}.npy"); best = int(np.argmax([1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2) for p in oof[1:]])) + 1
    p = oof[best]; inb = np.isin(F, big)
    pf = np.array([p[F == f].mean() if f in set(big) else np.nan for f in F]); yf = np.array([y[F == f].mean() if f in set(big) else np.nan for f in F])
    r2 = lambda yy, pp: 1 - np.sum((yy - pp) ** 2) / np.sum((yy - yy.mean()) ** 2)
    fam_dec = dict(r2_model=float(r2(y[inb], p[inb])), r2_family_means=float(r2(y[inb], pf[inb])),
                   eta2=float(np.sum((yf[inb] - y[inb].mean()) ** 2) / np.sum((y[inb] - y[inb].mean()) ** 2)),
                   r_within=float(stats.pearsonr(y[inb] - yf[inb], p[inb] - pf[inb])[0]))
    print(f"  {inb.sum()} proteins in {len(big)} families: between-family share {100*fam_dec['eta2']:.0f}%; model R2 {100*fam_dec['r2_model']:.1f}%, "
          f"from family averages {100*fam_dec['r2_family_means']:.1f}%; within-family r {fam_dec['r_within']:+.2f}")
    # 3. SAE features
    ix = np.array([spos[x] for x in a_])
    top = R[t]["top_features"]
    Xf = {d["feature"]: np.log1p(MAX[ix, d["feature"]].astype(np.float64)) for d in top}
    picked, absorbed = [], collections.Counter()
    for d in top:
        f = d["feature"]; v = Xf[f]
        dup = next((q["feature"] for q in picked if np.std(v) > 0 and abs(np.corrcoef(v, Xf[q["feature"]])[0, 1]) > 0.7), None)
        if dup is not None:
            absorbed[dup] += 1; continue
        if len(picked) < 12:
            picked.append(d)
    allk = collections.Counter(k for x in a_ for k in kw.get(x, ()))
    srows = []
    print(f"  SAE: {R[t]['n_fdr05']} features at FDR < 0.05; distinct strongest:")
    for d in picked:
        f = d["feature"]; a = A.get(f, {}); thr = a.get("threshold") or 0.0
        act = MAX[ix, f].astype(np.float32) > thr
        kc = collections.Counter(k for x in a_[act] for k in kw.get(x, ()))
        enr = sorted(((np.log((c + .5) / (act.sum() - c + .5)) - np.log((allk[k] + .5) / (len(a_) - allk[k] + .5)), k)
                      for k, c in kc.items() if c >= 5), reverse=True)[:3]
        topp = [gene_of.get(a_[i], a_[i]) for i in np.argsort(-MAX[ix, f].astype(np.float32))[:4]]
        srows.append(dict(feature=f, r=d["r"], stands_for=1 + absorbed[f], n_active=int(act.sum()), label=a.get("label"),
                          category=a.get("category"), keywords=[k for _, k in enr], top=topp))
        print(f"   {f:>5} r {d['r']:+.3f} (x{1+absorbed[f]:<2}) n {act.sum():>4} | {str(a.get('category'))[:20]:<20} | {str(a.get('label'))[:44]:<44} | {', '.join(k for _, k in enr)} | {', '.join(topp)}")
    cb = collections.Counter(A[f]["category"] for f in base if f in A and A[f].get("category")); nb = sum(cb.values())
    cats = {}
    for sign, sel in (("toward DNAJB11", [d["feature"] for d in top if d["r"] > 0][:100]), ("toward DNAJA1", [d["feature"] for d in top if d["r"] < 0][:100])):
        ctg = collections.Counter(A[f]["category"] for f in sel if f in A and A[f].get("category")); ns = sum(ctg.values())
        res = sorted((stats.fisher_exact([[ctg[c], ns - ctg[c]], [cb[c], nb - cb[c]]])[1], c, ctg[c], ns, cb[c]) for c in set(ctg) | set(cb))
        cats[sign] = [dict(category=c, n=k, of=n_, base=b, base_of=nb, p=float(pp)) for pp, c, k, n_, b in res]
        print(f"   categories {sign} ({ns}): " + ("; ".join(f"{c} {k}/{n_} vs {b}/{nb} (p {pp:.0e})" for pp, c, k, n_, b in res if pp < 0.05 and k / max(n_, 1) > b / nb) or "none enriched"))
    out[t] = dict(keywords=[dict(keyword=k, diff=d, n=n, p=p) for d, k, n, p in kres], compartments=crow,
                  families=[dict(family=f, mean=m, n=n) for m, f, n in fr], family_decomposition=fam_dec, sae=srows, categories=cats)
json.dump(out, open(f"{D}/ga14_results.json", "w"), indent=1)
print("\nGA14_DONE")
