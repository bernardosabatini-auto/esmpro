"""fs05: are the drug-sensitive HSPB1 pull-down proteins schizophrenia or bipolar risk genes?

Risk genes come from exome sequencing (rare coding variants, gene-level burden tests), downloaded from
the Broad browsers' public files:
  SCHEMA  schizophrenia, 24,248 cases (Singh et al. 2022): case-control + de novo meta-analysis, P meta and Q meta.
  BipEx   bipolar disorder, 13,933 cases (Palmer et al. 2022), group "Bipolar Disorder": Fisher tests for
          protein-truncating (PTV) and damaging-missense variants against gnomAD non-psychiatric controls.
          Gene p = min(PTV p, damaging-missense p) x number of tests available.
Ensembl gene -> UniProt via HGNC (data/annot/hgnc_complete_set.txt.gz).

Drug sensitivity (one value per protein and drug, -log10 p):
  FS73  carbamazepine / lamotrigine / topiramate: 2-df F test, both doses against control (same fit as fs01).
  FS76  lithium: dose trend at 0.25 mM Mg (the reproducible Li test); Mg2+ effect as a comparison.
Drug hits: FS73 q <= 0.05 at either dose; FS76 the 30 Li hits of fs02; Li-specific = fs04 calls.

Questions
  1. Do drug hits carry more schizophrenia / bipolar association than other pull-down proteins? (QQ plots,
     mean -log10 p against length-matched random sets: long genes collect more variants, so get smaller p)
  2. Across all detected proteins, does drug sensitivity rise with genetic association? (scatter + OLS,
     partial Spearman controlling for length and pull-down level)
  3. Which of the established risk genes (SCHEMA FDR 5 %, top BipEx) are in the pull-down, and do any respond?
"""
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, fisher_exact, rankdata
from pdpipe import stats as st, plots as pl, io

warnings.filterwarnings("ignore")
ROOT = io.ROOT; A = f"{ROOT}/data/annot"
FIG = f"{ROOT}/reports/figures/psych"; OUT = f"{ROOT}/data/fs_data/psych"
os.makedirs(FIG, exist_ok=True); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0); J = {}

# ---------------------------------------------------------------- 1. exome results keyed by UniProt
h = pd.read_csv(f"{A}/hgnc_complete_set.txt.gz", sep="\t", usecols=["symbol", "ensembl_gene_id", "uniprot_ids"], dtype=str)
h = h.dropna(subset=["ensembl_gene_id", "uniprot_ids"])
e2a = [(e, a) for e, u in zip(h.ensembl_gene_id, h.uniprot_ids) for a in u.split("|")]
e2a = pd.DataFrame(e2a, columns=["ens", "acc"])

sc = pd.read_csv(f"{A}/exome/SCHEMA_gene_results.tsv.bgz", sep="\t", compression="gzip")
sc = sc.rename(columns={"gene_id": "ens", "P meta": "scz_p", "Q meta": "scz_q", "OR (PTV)": "scz_or_ptv"})[["ens", "scz_p", "scz_q", "scz_or_ptv"]]
bp = pd.read_csv(f"{A}/exome/BipEx_gene_results.tsv.bgz", sep="\t", compression="gzip")
bp = bp[bp["group"] == "Bipolar Disorder"].rename(columns={"gene_id": "ens", "ptv_fisher_gnom_non_psych_pval": "bd_p_ptv",
                                                           "damaging_missense_fisher_gnom_non_psych_pval": "bd_p_dmis",
                                                           "ptv_fisher_gnom_non_psych_OR": "bd_or_ptv"})
pp = bp[["bd_p_ptv", "bd_p_dmis"]]
bp["bd_p"] = np.minimum(pp.min(1) * pp.notna().sum(1), 1).where(pp.notna().any(axis=1))
bp = bp[["ens", "bd_p", "bd_p_ptv", "bd_p_dmis", "bd_or_ptv"]]
G = e2a.merge(sc, on="ens", how="left").merge(bp, on="ens", how="left")
G = G.sort_values("scz_p").groupby("acc").first()           # an accession with several Ensembl genes keeps its best-tested one
J["exome"] = dict(schema_genes=int(sc["scz_p"].notna().sum()), schema_fdr5=int((sc["scz_q"] <= .05).sum()),
                  bipex_genes=int(bp["bd_p"].notna().sum()), bipex_p_1e4=int((bp["bd_p"] <= 1e-4).sum()))

# ---------------------------------------------------------------- 2. drug sensitivity per protein
D73 = io.load_proteoda(io.load_config("fs73"))
Dm, names = st.design(D73.samples, block=True); fit = st.ebayes(st.lmfit(D73.X, Dm))


def cvec(w):
    c = np.zeros(len(names))
    for k, v in w.items(): c[names.index(k)] += v
    return c


DRUG = {"car": "carbamazepine", "lam": "lamotrigene", "top": "topiramate"}
NICE = {"car": "carbamazepine", "lam": "lamotrigine", "top": "topiramate", "li": "lithium", "mg": "Mg2+ (comparison)"}
c73 = pd.read_csv(f"{ROOT}/data/fs_data/FS73/contrasts.tsv", sep="\t", index_col=0)
c73 = c73.loc[:, ~c73.columns.duplicated()]
sens, hits, tval = {}, {}, {}
for d, nm in DRUG.items():
    F = st.ftest(fit, np.stack([cvec({f"{nm}|{x}": 1, "ctl|0": -1}) for x in (10, 100)], 1))
    sens[d] = -np.log10(F["p"])
    hits[d] = (c73[f"{d}_10:q"] <= .05) | (c73[f"{d}_100:q"] <= .05)
    for x in (10, 100): tval[f"{d} {x}"] = c73[f"{d}_{x}:t"]
t76 = pd.read_csv(f"{ROOT}/data/fs_data/FS76/fs02_tests.tsv", sep="\t", index_col=0)
sens["li"] = -np.log10(t76["trend Mg0.25:p"]); sens["mg"] = -np.log10(t76["Mg effect:p"])
hits["li"] = (t76[["trend Mg0.25:q", "trend Mg2.5:q", "any dose Mg0.25:q", "any dose Mg2.5:q"]] <= .05).any(axis=1)
hits["mg"] = pd.Series(False, index=t76.index)
tval["Li, Mg 0.25"] = t76["trend Mg0.25:t"]; tval["Li, Mg 2.5"] = t76["trend Mg2.5:t"]; tval["Mg2+"] = t76["Mg effect:t"]
f04 = pd.read_csv(f"{ROOT}/data/fs_data/FS76/fs04_li_specific.tsv", sep="\t")
lispec = set(f04.loc[f04["call"].fillna("").str.startswith("Li-specific"), "accession"]) - {"Q9ULD6"}   # INTU: one sample
assert sum(hits["li"]) == 30
D76 = io.load_proteoda(io.load_config("fs76"))
meta = pd.concat([D73.proteins[["gene", "length"]], D76.proteins[["gene", "length"]]]).groupby(level=0).first()
level = {**{d: D73.X[D73.samples.loc[D73.samples["cond"] == "ctl|0", "column"]].mean(1) for d in DRUG},
         **{d: D76.X[D76.samples.loc[D76.samples["cond"] == "0.25|0", "column"]].mean(1) for d in ("li", "mg")}}

# ---------------------------------------------------------------- 3. tests
DIS = {"scz": "schizophrenia (SCHEMA)", "bd": "bipolar (BipEx)"}


def frame(d):
    f = pd.DataFrame({"sens": sens[d], "hit": hits[d].reindex(sens[d].index).fillna(False)}).join(G[["scz_p", "bd_p"]]).join(meta)
    f["level"] = level[d].reindex(f.index); f["loglen"] = np.log10(f["length"])
    return f.dropna(subset=["sens", "length"])


def matched_null(f, sel, col, n=20000):
    """mean -log10 p of `col` over random sets the size of `sel`, drawn within protein-length deciles."""
    ok = f[col].notna(); z = -np.log10(f.loc[ok, col]).values; dec = pd.qcut(f.loc[ok, "loglen"], 10, labels=False).values
    s = sel[ok.values]; obs = z[s].mean(); counts = np.bincount(dec[s], minlength=10)
    pools = [np.nonzero(dec == k)[0] for k in range(10)]
    null = np.zeros(n)
    for k, c in enumerate(counts):
        if c: null += z[pools[k][rng.random((n, len(pools[k]))).argsort(1)[:, :c]]].sum(1)   # c without replacement
    null /= s.sum()
    return dict(n=int(s.sum()), obs=float(obs), null_mean=float(null.mean()), null95=float(np.percentile(null, 95)),
                p=float((1 + (null >= obs).sum()) / (n + 1)), n_p05=int((f.loc[ok, col].values[s] <= .05).sum()),
                expected_p05=float(s.sum() * (f.loc[ok, col] <= .05).mean()))


def partial_spearman(f, col):
    ok = f[col].notna() & f["level"].notna()
    y = rankdata(f.loc[ok, "sens"]); x = rankdata(-np.log10(f.loc[ok, col]))
    Z = np.column_stack([np.ones(ok.sum()), rankdata(f.loc[ok, "loglen"]), rankdata(f.loc[ok, "level"])])
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]; rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    r = np.corrcoef(rx, ry)[0, 1]; n = int(ok.sum())
    from scipy.stats import t as tdist
    tt = r * np.sqrt((n - 5) / (1 - r * r)); return float(r), float(2 * tdist.sf(abs(tt), n - 5)), n


J["tests"] = {}
for d in sens:
    f = frame(d); J["tests"][d] = {}
    for dc in ("scz", "bd"):
        col = f"{dc}_p"; ok = f[col].notna()
        rs = spearmanr(f.loc[ok, "sens"], -np.log10(f.loc[ok, col]))
        pr, pp_, n = partial_spearman(f, col)
        R = dict(n_universe=int(ok.sum()), spearman=float(rs.correlation), p_spearman=float(rs.pvalue), partial_spearman=pr, p_partial=pp_)
        # genetic association by drug sensitivity: top 1 % / 5 % most drug-sensitive
        for frac in (.01, .05):
            top = (f["sens"] >= f.loc[ok, "sens"].quantile(1 - frac)).values
            R[f"top{int(frac * 100)}pct"] = matched_null(f, top, col, n=2000)
        if f["hit"].any(): R["hits"] = matched_null(f, f["hit"].values, col, n=2000)
        if d == "li": R["li_specific"] = matched_null(f, f.index.isin(lispec), col, n=2000)
        # drug sensitivity by genetic association: nominal risk genes
        risk = (f[col] <= .01).values & ok.values
        R["risk_p01_sens"] = dict(n=int(risk.sum()), mean_risk=float(f["sens"][risk].mean()), mean_rest=float(f["sens"][~risk & ok.values].mean()))
        hh = f["hit"].values & ok.values
        R["fisher_hit_x_p05"] = dict(table=[[int((hh & (f[col] <= .05)).sum()), int((hh & (f[col] > .05)).sum())],
                                            [int((~hh & ok & (f[col] <= .05)).sum()), int((~hh & ok & (f[col] > .05)).sum())]])
        R["fisher_hit_x_p05"]["p"] = float(fisher_exact(R["fisher_hit_x_p05"]["table"], "greater")[1]) if hh.any() else None
        J["tests"][d][dc] = R
    print(d, {dc: (round(v["partial_spearman"], 3), f"{v['p_partial']:.2g}", v.get("hits", {}).get("obs"), v.get("hits", {}).get("p")) for dc, v in J["tests"][d].items()}, flush=True)

# the whole family of tests above, BH-adjusted together (drug x disease x test)
fam = [(f"{d} {dc} {k}", (r[k]["p"] if isinstance(r[k], dict) else None) if k != "partial" else r["p_partial"])
       for d, x in J["tests"].items() for dc, r in x.items() for k in ("partial", "hits", "li_specific", "top1pct", "top5pct") if k == "partial" or k in r]
fq = st.bh(np.array([p for _, p in fam]))
J["family"] = [dict(test=n, p=float(p), q=float(q)) for (n, p), q in zip(fam, fq)]
print("family:", len(fam), "tests; p<=.05:", sum(p <= .05 for _, p in fam), "; min q", float(np.min(fq)), flush=True)
# what drives the nominal set signals: the drug-sensitive proteins with p <= 0.05 for the disease
J["drivers"] = {}
for d, dc, frac in (("car", "bd", .01), ("li", "scz", .05), ("li", "bd", .05), ("top", "scz", .05)):
    f = frame(d); top = f[f["sens"] >= f.loc[f[f"{dc}_p"].notna(), "sens"].quantile(1 - frac)]
    J["drivers"][f"{d} {dc} top{int(frac * 100)}pct"] = top.loc[top[f"{dc}_p"] <= .05].sort_values(f"{dc}_p")[["gene", f"{dc}_p", "sens"]].round(4).values.tolist()
pd.DataFrame({**{f"sens_{d}": sens[d] for d in sens}, **{f"hit_{d}": hits[d] for d in hits}}).join(G[["scz_p", "scz_q", "bd_p"]]).join(meta) \
    .to_csv(f"{OUT}/fs05_per_protein.tsv", sep="\t")

# ---------------------------------------------------------------- 4. per-hit table
rows = []
for d in ("car", "lam", "top", "li"):
    for acc in hits[d].index[hits[d].values]:
        g = G.loc[acc] if acc in G.index else pd.Series(dtype=float)
        rows.append(dict(drug=NICE[d], gene=meta["gene"].get(acc), accession=acc, li_specific=acc in lispec, length=meta["length"].get(acc),
                         scz_p=g.get("scz_p"), scz_q=g.get("scz_q"), scz_or_ptv=g.get("scz_or_ptv"),
                         bd_p=g.get("bd_p"), bd_p_ptv=g.get("bd_p_ptv"), bd_p_dmis=g.get("bd_p_dmis")))
HT = pd.DataFrame(rows); HT.to_csv(f"{OUT}/fs05_hit_genetics.tsv", sep="\t", index=False)
J["hit_rows"] = HT.to_dict("records")
print(HT[(HT.scz_p <= .05) | (HT.bd_p <= .05)].to_string())

# ---------------------------------------------------------------- 5. established risk genes in the pull-downs
detected = set(D73.X.index) | set(D76.X.index)
R1 = G[(G["scz_q"] <= .05)].copy(); R1["set"] = "SCHEMA FDR 5%"
R2 = G[(G["bd_p"] <= 1e-3)].copy(); R2["set"] = "BipEx p <= 0.001"
RK = pd.concat([R1, R2]); RK = RK[~RK.index.duplicated()]
RK["gene"] = [h.set_index("ensembl_gene_id")["symbol"].get(e) for e in e2a.groupby("acc")["ens"].first().reindex(RK.index)]
RK["in_fs73"] = RK.index.isin(D73.X.index); RK["in_fs76"] = RK.index.isin(D76.X.index)
J["risk_genes"] = dict(schema_fdr=int(len(R1)), schema_fdr_detected=int(R1.index.isin(detected).sum()),
                       bipex=int(len(R2)), bipex_detected=int(R2.index.isin(detected).sum()),
                       schema_fdr_detected_genes=sorted(RK.loc[RK.index.isin(detected) & (RK["set"] == "SCHEMA FDR 5%"), "gene"].dropna()),
                       bipex_detected_genes=sorted(RK.loc[RK.index.isin(detected) & (RK["set"] == "BipEx p <= 0.001"), "gene"].dropna()))
T = pd.DataFrame(tval).reindex(sorted(detected & set(RK.index)))
for k in T: T[k] = tval[k].reindex(T.index)
T.index = [RK.loc[a, "gene"] for a in T.index]
T.to_csv(f"{OUT}/fs05_risk_gene_t.tsv", sep="\t")
J["risk_t_max"] = {g: dict(max_abs_t=float(T.loc[g].abs().max()), where=str(T.loc[g].abs().idxmax())) for g in T.index if T.loc[g].notna().any()}
print(J["risk_genes"])

# ---------------------------------------------------------------- figures
# A. QQ: disease p of drug hits against all detected proteins
f, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for a_, dc in zip(ax, ("scz", "bd")):
    sets = {"all detected proteins": None, "FS73 hits (21)": None, "Li hits (30)": None, "Li-specific (11)": None}
    alld = pd.Index(sorted(detected)); pv = {"all detected proteins": G[f"{dc}_p"].reindex(alld).dropna()}
    h73 = set().union(*[set(hits[d].index[hits[d].values]) for d in DRUG])
    pv["FS73 hits"] = G[f"{dc}_p"].reindex(sorted(h73)).dropna()
    pv["Li hits"] = G[f"{dc}_p"].reindex(hits["li"].index[hits["li"].values]).dropna()
    pv["Li-specific"] = G[f"{dc}_p"].reindex(sorted(lispec)).dropna()
    for (k, v), c, m in zip(pv.items(), [pl.BASE, "#e08e0b", pl.HIT, "#6a3d9a"], ["", "o", "s", "^"]):
        o = np.sort(v.values); e = (np.arange(1, len(o) + 1) - .5) / len(o)
        if m: a_.scatter(-np.log10(e), -np.log10(o), s=18, color=c, marker=m, label=f"{k} (n={len(o)})", zorder=3)
        else: a_.plot(-np.log10(e), -np.log10(o), color=c, lw=2, label=f"{k} (n={len(o)})")
        if m:
            nm = meta["gene"].reindex(v.sort_values().index).values
            for i in range(min(3, len(o))):
                if o[i] <= .05: a_.annotate(nm[i], (-np.log10(e[i]), -np.log10(o[i])), fontsize=6.5, xytext=(3, 0), textcoords="offset points")
    lim = max(a_.get_xlim()[1], 2.5); a_.plot([0, lim], [0, lim], "k:", lw=.8)
    a_.set_xlabel("expected -log10 p (uniform)"); a_.set_ylabel(f"observed -log10 p, {DIS[dc]}")
    a_.set_title(f"{'AB'[dc == 'bd']}  Do drug hits carry {DIS[dc].split(' (')[0]} association?"); a_.legend(fontsize=7, loc="upper left")
f.tight_layout(); f.savefig(f"{FIG}/qq_hits.png", dpi=150); plt.close(f)

# B. scatter: drug sensitivity against genetic association, every detected protein
f, ax = plt.subplots(2, 5, figsize=(19, 7.6))
for j, d in enumerate(("car", "lam", "top", "li", "mg")):
    fr = frame(d)
    for i, dc in enumerate(("scz", "bd")):
        ok = fr[f"{dc}_p"].notna(); x = -np.log10(fr.loc[ok, f"{dc}_p"]).values; y = fr.loc[ok, "sens"].values
        lab = np.where((fr.loc[ok, f"{dc}_p"] <= 1e-3).values | fr.loc[ok, "hit"].values & (fr.loc[ok, f"{dc}_p"] <= .05).values, fr.loc[ok, "gene"].values, "")
        pl.scatter_reg(ax[i, j], x, y, f"-log10 p, {DIS[dc]}", f"-log10 p, {NICE[d]}", f"{NICE[d]} vs {DIS[dc].split(' (')[0]}",
                       highlight=fr.loc[ok, "hit"].values | (lab != ""), labels=lab, robust_lim=False)
        tt = J["tests"][d][dc]; ax[i, j].text(.97, .97, f"partial rho = {tt['partial_spearman']:.3f}\n(p {tt['p_partial']:.2g})",
                                               transform=ax[i, j].transAxes, ha="right", va="top", fontsize=7)
f.tight_layout(); f.savefig(f"{FIG}/scatter_all.png", dpi=140); plt.close(f)

# C. established risk genes: t for every drug contrast
if len(T):
    Tt = T.dropna(how="all")
    f, a_ = plt.subplots(figsize=(7.5, .26 * len(Tt) + 1.6))
    im = a_.imshow(Tt.values, cmap="RdBu_r", vmin=-5, vmax=5, aspect="auto")
    a_.set_xticks(range(Tt.shape[1])); a_.set_xticklabels(Tt.columns, rotation=40, ha="right", fontsize=7)
    a_.set_yticks(range(len(Tt))); a_.set_yticklabels([f"{g}  ({RK.set_index('gene').loc[g, 'set'] if g in set(RK.gene) else ''})" for g in Tt.index], fontsize=6.5)
    for (i, k), v in np.ndenumerate(Tt.values):
        if np.isfinite(v) and abs(v) >= 3: a_.text(k, i, f"{v:.1f}", ha="center", va="center", fontsize=5.5)
    plt.colorbar(im, ax=a_, label="moderated t", shrink=.6); a_.set_title("Established risk genes found in the pull-downs")
    f.tight_layout(); f.savefig(f"{FIG}/risk_genes_t.png", dpi=150); plt.close(f)

# D. lithium dose curves of the risk genes in FS76 (SCHEMA exome-wide genes first, then the rest)
EW = ["SETD1A", "CUL1", "XPO7", "TRIO", "CACNA1G", "SP4", "GRIA3", "GRIN2A", "HERC1", "RB1CC1"]
sym2acc = {g: a for a, g in RK["gene"].items()}
acc = [sym2acc[g] for g in EW if g in sym2acc and sym2acc[g] in D76.X.index] + \
      [a for a in RK.index if a in D76.X.index and RK.loc[a, "gene"] not in EW and D76.X.loc[a].notna().sum() >= 24]
nc = 6; nr = int(np.ceil(len(acc) / nc))
f, ax = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.3 * nr), squeeze=False)
for k, (a_, ac) in enumerate(zip(ax.flat, acc)):
    g = RK.loc[ac, "gene"]; tt = t76.loc[ac, "trend Mg0.25:t"]
    pl.dose_curve(a_, D76, ac, "conc", "Mg", zero=0.09, title=f"{g}{' *' if g in EW else ''}  (Li t {tt:+.1f})")
    a_.set_xlabel("Li (mM)", fontsize=7); a_.legend().remove() if k else a_.legend(fontsize=6)
for a_ in ax.flat[len(acc):]: a_.axis("off")
f.suptitle("Schizophrenia / bipolar risk genes in the lithium screen (* = SCHEMA exome-wide)", fontsize=10)
f.tight_layout(); f.savefig(f"{FIG}/risk_genes_li.png", dpi=140); plt.close(f)

json.dump(J, open(f"{OUT}/fs05.json", "w"), indent=1, default=lambda o: None if o is None or (isinstance(o, float) and np.isnan(o)) else float(o))
print(json.dumps({d: {dc: {k: v for k, v in r.items() if k in ("partial_spearman", "p_partial", "hits", "li_specific", "top1pct", "top5pct", "fisher_hit_x_p05")}
                      for dc, r in x.items()} for d, x in J["tests"].items()}, indent=1, default=float))
print(J["risk_t_max"])
print("FS05_DONE")
