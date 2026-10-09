"""fs06: drug-sensitive HSPB1 pull-down proteins against common-variant (GWAS) schizophrenia and bipolar genes.

Follows fs05 (rare-variant exome genes), with the same drug-sensitivity values and tests (pdpipe.genetics).

Genetics
  PGC3 schizophrenia (Trubetskoy et al. 2022), European autosomal summary statistics.
  PGC bipolar 2024 (O'Connell et al. 2025), European, no 23andMe.
  Gene-level p from MAGMA v1.10 (slurm/magma_pgc.sbatch): SNP-wise mean model, 1000G EUR LD reference,
  35 kb up / 10 kb down. Significant genes: Bonferroni p < 0.05 / genes tested.
  PGC3 prioritised genes: the 120 genes of Extended Data Table 1 (FINEMAP, SMR or rare-variant support).

Gene p from MAGMA rises with the number of SNPs in a gene, so set tests draw random sets within deciles of
log(SNPs per gene) and the correlations control for SNP count, protein length and pull-down level.
Neighbouring genes share GWAS signal (LD); the tests treat genes as independent, which is close to right for
small scattered hit sets but makes the all-protein correlation p somewhat optimistic.

Checks that the genetics behaves as expected: MAGMA schizophrenia vs bipolar (genetic correlation ~0.7)
and MAGMA schizophrenia vs SCHEMA exome (known convergence).
"""
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, fisher_exact
from pdpipe import stats as st, plots as pl, io, genetics as gx

warnings.filterwarnings("ignore")
ROOT = io.ROOT; GW = f"{ROOT}/data/annot/gwas"
FIG = f"{ROOT}/reports/figures/psych"; OUT = f"{ROOT}/data/fs_data/psych"
rng = np.random.default_rng(0); J = {}

# ---------------------------------------------------------------- genetics keyed by UniProt
mg, raw = {}, {}
for t in ("scz", "bd"):
    m = pd.read_csv(f"{GW}/magma_{t}.genes.out", sep=r"\s+")
    m = m.rename(columns={"GENE": "entrez", "P": f"{t}_p", "ZSTAT": f"{t}_z", "NSNPS": f"{t}_nsnp", "CHR": f"{t}_chr", "START": f"{t}_start"})
    m["entrez"] = m["entrez"].astype(str)
    J[f"magma_{t}"] = dict(genes=int(len(m)), bonferroni=float(.05 / len(m)), n_sig=int((m[f"{t}_p"] < .05 / len(m)).sum()))
    m[f"{t}_sig"] = m[f"{t}_p"] < .05 / len(m)
    mg[t] = gx.by_accession(m[["entrez", f"{t}_p", f"{t}_z", f"{t}_nsnp", f"{t}_sig", f"{t}_chr", f"{t}_start"]], "entrez", f"{t}_p")
    raw[t] = m
G = mg["scz"].drop(columns=["entrez"]).join(mg["bd"].drop(columns=["symbol", "entrez"]), how="outer")
G["symbol"] = G["symbol"].fillna(mg["bd"]["symbol"])
G["chr"] = G["scz_chr"].fillna(G["bd_chr"]); G["start"] = G["scz_start"].fillna(G["bd_start"])
# extended MHC, build 37: chr6 25-34 Mb. Long-range LD there inflates every gene's MAGMA p.
G["mhc"] = (G["chr"].astype(str) == "6") & G["start"].between(25e6, 34e6)
for c in ("scz_sig", "bd_sig"): G[c] = G[c].fillna(False).astype(bool)
import openpyxl
wb = openpyxl.load_workbook(f"{GW}/scz2022_ED_table1.xlsx", read_only=True)
ed1 = pd.DataFrame(list(wb["Extended.Data.Table.1"].iter_rows(values_only=True))[1:], columns=list(next(wb["Extended.Data.Table.1"].iter_rows(values_only=True))))
ens2acc = gx.hgnc().dropna(subset=["ens"]).groupby("ens")["acc"].apply(list)
prio = {a for e in ed1["Ensembl.ID"] for a in ens2acc.get(e, [])}
J["pgc3_prioritised"] = dict(genes=int(len(ed1)), protein_coding=int((ed1["gene_biotype"] == "protein_coding").sum()), with_uniprot=len(prio))
exome = pd.read_csv(f"{OUT}/fs05_per_protein.tsv", sep="\t", index_col=0)[["scz_p", "bd_p"]].rename(columns=lambda c: "exome_" + c)

# genetics sanity checks across all genes
chk = G.join(exome, how="left")
J["checks"] = {"magma scz vs magma bd": spearmanr(chk["scz_z"], chk["bd_z"], nan_policy="omit").correlation,
               "magma scz vs SCHEMA exome": spearmanr(chk["scz_z"], -np.log10(chk["exome_scz_p"]), nan_policy="omit").correlation,
               "magma bd vs BipEx exome": spearmanr(chk["bd_z"], -np.log10(chk["exome_bd_p"]), nan_policy="omit").correlation}
print("magma", J["magma_scz"], J["magma_bd"], "checks", J["checks"], flush=True)

# ---------------------------------------------------------------- drug sensitivity
DS = gx.drug_sensitivity(); sens, hits, meta, lispec = DS["sens"], DS["hits"], DS["meta"], DS["lispec"]
DIS = {"scz": "schizophrenia GWAS (PGC3)", "bd": "bipolar GWAS (PGC 2024)"}
detected = set(DS["ds73"].X.index) | set(DS["ds76"].X.index)


def frame(d):
    f = pd.DataFrame({"sens": sens[d], "hit": hits[d].reindex(sens[d].index).fillna(False)}).join(G).join(meta)
    f["level"] = DS["level"][d].reindex(f.index); f["loglen"] = np.log10(f["length"])
    for t in ("scz", "bd"): f[f"{t}_lognsnp"] = np.log10(f[f"{t}_nsnp"] + 1)
    return f.dropna(subset=["sens"])


def run_tests(key, keep):
  J[key] = {}
  for d in sens:
    f = frame(d); f = f[keep(f)]; J[key][d] = {}
    for t in ("scz", "bd"):
        col = f"{t}_p"; ok = f[col].notna() & f["loglen"].notna()
        pr, pp_, n = gx.partial_spearman(f, col, covars=("loglen", "level", f"{t}_lognsnp"))
        R = dict(n_universe=int(ok.sum()), spearman=float(spearmanr(f.loc[ok, "sens"], -np.log10(f.loc[ok, col])).correlation),
                 partial_spearman=pr, p_partial=pp_)
        fo = f[ok]
        for frac in (.01, .05):
            R[f"top{int(frac * 100)}pct"] = gx.matched_null(fo, (fo["sens"] >= fo["sens"].quantile(1 - frac)).values, col, rng, strata=f"{t}_lognsnp")
        if fo["hit"].any():
            R["hits"] = gx.matched_null(fo, fo["hit"].values, col, rng, strata=f"{t}_lognsnp")
            R["hits_rank"] = gx.matched_null(fo, fo["hit"].values, col, rng, strata=f"{t}_lognsnp", rank=True)
        if d == "li":
            R["li_specific"] = gx.matched_null(fo, fo.index.isin(lispec), col, rng, strata=f"{t}_lognsnp")
            R["li_specific_rank"] = gx.matched_null(fo, fo.index.isin(lispec), col, rng, strata=f"{t}_lognsnp", rank=True)
        hh = fo["hit"].values; sg = fo[f"{t}_sig"].fillna(False).values.astype(bool)
        tab = [[int((hh & sg).sum()), int((hh & ~sg).sum())], [int((~hh & sg).sum()), int((~hh & ~sg).sum())]]
        R["fisher_hit_x_sig"] = dict(table=tab, p=float(fisher_exact(tab, "greater")[1]) if hh.any() else None)
        if t == "scz":
            pr_ = fo.index.isin(prio)
            tab = [[int((hh & pr_).sum()), int((hh & ~pr_).sum())], [int((~hh & pr_).sum()), int((~hh & ~pr_).sum())]]
            R["fisher_hit_x_prioritised"] = dict(table=tab, p=float(fisher_exact(tab, "greater")[1]) if hh.any() else None)
        J[key][d][t] = R
    print(key, d, {t: (round(v["partial_spearman"], 3), f"{v['p_partial']:.2g}", v.get("hits", {}).get("obs"), v.get("hits", {}).get("p"),
                  v["fisher_hit_x_sig"]["table"][0]) for t, v in J[key][d].items()}, flush=True)



run_tests("tests", lambda f: np.ones(len(f), bool))
run_tests("tests_noMHC", lambda f: ~f["mhc"].fillna(False).astype(bool).values)

fam = [(f"{key} {d} {t} {k}", r["p_partial"] if k == "partial" else r[k]["p"]) for key in ("tests", "tests_noMHC")
       for d, x in J[key].items() for t, r in x.items() for k in ("partial", "hits", "hits_rank", "li_specific", "top1pct", "top5pct") if k == "partial" or k in r]
fq = st.bh(np.array([p for _, p in fam]))
J["family"] = [dict(test=n, p=float(p), q=float(q)) for (n, p), q in zip(fam, fq)]
print("family:", len(fam), "tests; p<=.05:", sum(p <= .05 for _, p in fam), "; min q", float(np.min(fq)), flush=True)

# per-hit table and GWAS genes in the pull-downs
rows = []
for d in ("car", "lam", "top", "li"):
    for acc in hits[d].index[hits[d].values]:
        g = G.loc[acc] if acc in G.index else pd.Series(dtype=float)
        rows.append(dict(drug=gx.NICE[d], gene=meta["gene"].get(acc), accession=acc, li_specific=acc in lispec, scz_magma_p=g.get("scz_p"),
                         scz_sig=bool(g.get("scz_sig", False) == True), mhc=bool(g.get("mhc", False) == True), chr=g.get("chr"), start=g.get("start"), pgc3_prioritised=acc in prio, bd_magma_p=g.get("bd_p"),
                         bd_sig=bool(g.get("bd_sig", False) == True)))
HT = pd.DataFrame(rows); HT.to_csv(f"{OUT}/fs06_hit_gwas.tsv", sep="\t", index=False)
print(HT[(HT.scz_magma_p <= .01) | (HT.bd_magma_p <= .01) | HT.pgc3_prioritised].to_string())

# locus check: is each strongly associated hit the lead gene of its region (+-500 kb), or riding on a neighbour?
J["locus"] = []
for _, r in HT.iterrows():
    for t in ("scz", "bd"):
        pv = r[f"{t}_magma_p"]
        if pd.notna(pv) and pv <= 1e-3:
            g = G.loc[r["accession"]]; nb = G[(G["chr"].astype(str) == str(g["chr"])) & ((G["start"] - g["start"]).abs() <= 5e5)].dropna(subset=[f"{t}_p"])
            lead = nb.sort_values(f"{t}_p").iloc[0]
            J["locus"].append(dict(drug=r["drug"], gene=r["gene"], trait=t, p=float(pv), chr=str(g["chr"]), mb=round(float(g["start"]) / 1e6, 2),
                                   mhc=bool(g["mhc"]), genes_in_window=int(len(nb)), rank_in_window=int((nb[f"{t}_p"] < pv).sum()) + 1,
                                   lead=lead["symbol"], lead_p=float(lead[f"{t}_p"])))
print(pd.DataFrame(J["locus"]).to_string(), flush=True)

sets = {"PGC3 prioritised": prio, "SCZ MAGMA significant": set(G.index[G["scz_sig"].fillna(False).astype(bool)]),
        "BD MAGMA significant": set(G.index[G["bd_sig"].fillna(False).astype(bool)])}
J["gwas_genes_detected"] = {k: dict(total=len(v), detected=len(v & detected)) for k, v in sets.items()}
J["detection_rate_all"] = float(len(detected & set(G.index)) / len(G))
gw = sorted(set().union(*sets.values()) & detected)
T = pd.DataFrame({k: DS["tval"][k].reindex(gw) for k in DS["tval"]}, index=gw)
sym = gx.hgnc().groupby("acc")["symbol"].first()
hlab = [f"{sym.get(a, a)}  ({'P' if a in prio else ''}{'S' if a in sets['SCZ MAGMA significant'] else ''}{'B' if a in sets['BD MAGMA significant'] else ''})" for a in gw]
T.index = [sym.get(a, a) for a in gw]; T.to_csv(f"{OUT}/fs06_gwas_gene_t.tsv", sep="\t")
drug_cols = [c for c in T.columns if c != "Mg2+"]
J["gwas_gene_t"] = dict(n_genes=len(gw), n_drug_cells=int(T[drug_cols].notna().sum().sum()), n_abs_t3=int((T[drug_cols].abs() >= 3).sum().sum()),
                        cells_abs_t3=[(g, c, float(T.loc[g, c])) for g in T.index for c in drug_cols if abs(T.loc[g, c]) >= 3])
# q-values of those genes in their own drug test (from the screens)
c73 = pd.read_csv(f"{ROOT}/data/fs_data/FS73/contrasts.tsv", sep="\t", index_col=0); c73 = c73.loc[:, ~c73.columns.duplicated()]
qmap = {**{f"{d} {x}": c73[f"{d}_{x}:q"] for d in ("car", "lam", "top") for x in (10, 100)},
        "Li, Mg 0.25": DS["t76"]["trend Mg0.25:q"], "Li, Mg 2.5": DS["t76"]["trend Mg2.5:q"]}
s2a = dict(zip(T.index, gw))
J["gwas_gene_t"]["cells_abs_t3"] = [(g, c, v, float(qmap[c].get(s2a[g], np.nan))) for g, c, v in J["gwas_gene_t"]["cells_abs_t3"]]
print(J["gwas_genes_detected"], J["gwas_gene_t"], flush=True)

# ---------------------------------------------------------------- robustness of the lithium signal (MHC excluded throughout)
# 1. enrichment against the sensitivity cut-off, with within-screen controls: the Li trend at 2.5 mM Mg (same
#    screen, same noise structure, no Li effect) and the Mg effect (strong, reproducible, not lithium).
t76 = DS["t76"]
CUR = {"lithium, 0.25 mM Mg": sens["li"], "lithium, 2.5 mM Mg (control)": -np.log10(t76["trend Mg2.5:p"]),
       "Mg2+ (control)": sens["mg"], "lamotrigine": sens["lam"], "carbamazepine": sens["car"]}
FRAC = [.01, .02, .05, .1, .2, .5]


def tail_curve(s, t, n=2000):
    f = pd.DataFrame({"sens": s}).join(G).join(meta); f["level"] = DS["level"]["li"].reindex(f.index)
    f = f.dropna(subset=["sens", f"{t}_p"]); f = f[~f["mhc"].fillna(False).astype(bool)]; f[f"{t}_lognsnp"] = np.log10(f[f"{t}_nsnp"] + 1)
    out = []
    for fr in FRAC:
        r = gx.matched_null(f, (f["sens"] >= f["sens"].quantile(1 - fr)).values, f"{t}_p", rng, n=n, strata=f"{t}_lognsnp")
        out.append(dict(frac=fr, n=r["n"], excess=r["obs"] - r["null_mean"], null95=r["null95"] - r["null_mean"], p=r["p"]))
    return out


J["tail_curves"] = {t: {k: tail_curve(v, t) for k, v in CUR.items()} for t in ("scz", "bd")}
for t in J["tail_curves"]:
    for k, v in J["tail_curves"][t].items(): print("curve", t, f"{k:30}", " ".join(f"{r['frac']:.2f}:{r['excess']:+.2f}(p{r['p']:.3f})" for r in v), flush=True)

# 2. split halves: rank proteins by the Li trend tested (moderated t) on 2 replicates, test the top 5 % against
#    GWAS, and repeat with the other 2. A real signal should appear in both halves independently.
S76, X76 = DS["ds76"].samples, DS["ds76"].X
DOSES = ["0", "0.3", "1", "3", "5", "10"]
TRW = st.trend_weights([f"0.25|{d}" for d in DOSES], np.log10([0.09, .3, 1, 3, 5, 10]))
J["split_half"] = []
for half in st._splits(S76["BR"].unique()):
    for side, keep in (("A", S76["BR"].isin(half)), ("B", ~S76["BR"].isin(half))):
        sub = DS["ds76"].subset(keep.values); Dh, nh = st.design(sub.samples, block=True)
        fh = st.ebayes(st.lmfit(sub.X, Dh)); c = np.zeros(len(nh))
        for k_, w_ in TRW.items(): c[nh.index(k_)] += w_
        e = -np.log10(st.contrast(fh, c)["p"])          # moderated-t p of the Li trend from this half alone
        brs = sorted(int(b) for b in S76.loc[keep, "BR"].unique())
        for t in ("scz", "bd"):
            f = pd.DataFrame({"sens": e}).join(G); f = f.dropna(subset=["sens", f"{t}_p"]); f = f[~f["mhc"].fillna(False).astype(bool)]
            f[f"{t}_lognsnp"] = np.log10(f[f"{t}_nsnp"] + 1)
            r = gx.matched_null(f, (f["sens"] >= f["sens"].quantile(.95)).values, f"{t}_p", rng, strata=f"{t}_lognsnp")
            J["split_half"].append(dict(brs=brs, trait=t, excess=r["obs"] - r["null_mean"], p=r["p"]))
print(pd.DataFrame(J["split_half"]).to_string(), flush=True)

# ---------------------------------------------------------------- figures
# A. QQ of MAGMA p for the drug hits
f, ax = plt.subplots(1, 2, figsize=(11, 4.6))
h73 = set().union(*[set(hits[d].index[hits[d].values]) for d in gx.DRUG])
for a_, t in zip(ax, ("scz", "bd")):
    pv = {"all detected proteins": G[f"{t}_p"].reindex(sorted(detected)).dropna(), "FS73 hits": G[f"{t}_p"].reindex(sorted(h73)).dropna(),
          "Li hits": G[f"{t}_p"].reindex(hits["li"].index[hits["li"].values]).dropna(), "Li-specific": G[f"{t}_p"].reindex(sorted(lispec)).dropna()}
    for (k, v), c, mk in zip(pv.items(), [pl.BASE, "#e08e0b", pl.HIT, "#6a3d9a"], ["", "o", "s", "^"]):
        o = np.sort(v.values); e = (np.arange(1, len(o) + 1) - .5) / len(o)
        if mk:
            a_.scatter(-np.log10(e), -np.log10(o), s=18, color=c, marker=mk, label=f"{k} (n={len(o)})", zorder=3)
            nm = meta["gene"].reindex(v.sort_values().index).values
            for i in range(min(3, len(o))):
                if o[i] <= .01 and not (k == "Li-specific" and nm[i] in set(meta["gene"].reindex(pv["Li hits"].index))): a_.annotate(nm[i], (-np.log10(e[i]), -np.log10(o[i])), fontsize=6.5, xytext=(3, 0), textcoords="offset points")
        else: a_.plot(-np.log10(e), -np.log10(o), color=c, lw=2, label=f"{k} (n={len(o)})")
    a_.plot([0, 4.5], [0, 4.5], "k:", lw=.8); a_.set_ylim(0, min(a_.get_ylim()[1], 16))
    a_.set_xlabel("expected -log10 p (uniform)"); a_.set_ylabel(f"observed MAGMA -log10 p, {DIS[t]}")
    a_.set_title(f"{'AB'[t == 'bd']}  Do drug hits carry {DIS[t].split(' (')[0]} association?", fontsize=10); a_.legend(fontsize=7, loc="upper left")
f.tight_layout(); f.savefig(f"{FIG}/gwas_qq_hits.png", dpi=150); plt.close(f)

# B. scatter, every protein
f, ax = plt.subplots(2, 5, figsize=(19, 7.6))
for j, d in enumerate(("car", "lam", "top", "li", "mg")):
    fr = frame(d)
    for i, t in enumerate(("scz", "bd")):
        ok = fr[f"{t}_p"].notna(); x = -np.log10(fr.loc[ok, f"{t}_p"]).values; y = fr.loc[ok, "sens"].values
        sig = fr.loc[ok, f"{t}_sig"].fillna(False).values.astype(bool)
        lab = np.where((sig & (y > 1.3)) | (fr.loc[ok, "hit"].values & (x > 2)), fr.loc[ok, "gene"].values, "")
        pl.scatter_reg(ax[i, j], x, y, f"MAGMA -log10 p, {DIS[t]}", f"-log10 p, {gx.NICE[d]}", f"{gx.NICE[d]} vs {DIS[t].split(' GWAS')[0]} GWAS",
                       highlight=fr.loc[ok, "hit"].values | (lab != ""), labels=lab, robust_lim=False)
        ax[i, j].axvline(-np.log10(J[f"magma_{t}"]["bonferroni"]), color="k", ls="--", lw=.6)
        tt = J["tests"][d][t]; ax[i, j].text(.97, .97, f"partial rho = {tt['partial_spearman']:.3f}\n(p {tt['p_partial']:.2g})",
                                              transform=ax[i, j].transAxes, ha="right", va="top", fontsize=7)
f.tight_layout(); f.savefig(f"{FIG}/gwas_scatter_all.png", dpi=140); plt.close(f)

# C. GWAS genes in the pull-downs: PGC3 prioritised genes, and any MAGMA-significant gene with |t| >= 3 for a drug
keep = np.array([(a in prio) or bool((T.loc[T.index[i], drug_cols].abs() >= 3).any()) for i, a in enumerate(gw)])
Th = T[keep]; hl = [l for l, k in zip(hlab, keep) if k]
f, a_ = plt.subplots(figsize=(7.5, .2 * len(Th) + 1.6))
im = a_.imshow(Th.values, cmap="RdBu_r", vmin=-5, vmax=5, aspect="auto")
a_.set_xticks(range(Th.shape[1])); a_.set_xticklabels(Th.columns, rotation=40, ha="right", fontsize=7)
a_.set_yticks(range(len(Th))); a_.set_yticklabels(hl, fontsize=5.5)
for (i, k), v in np.ndenumerate(Th.values):
    if np.isfinite(v) and abs(v) >= 3: a_.text(k, i, f"{v:.1f}", ha="center", va="center", fontsize=5)
plt.colorbar(im, ax=a_, label="moderated t", shrink=.4)
a_.set_title("GWAS genes in the pull-downs\n(P = PGC3 prioritised; S / B = MAGMA significant for SCZ / BD)", fontsize=8)
f.tight_layout(); f.savefig(f"{FIG}/gwas_genes_t.png", dpi=150); plt.close(f)

# D. GWAS enrichment against the lithium-sensitivity cut-off, with controls
f, ax = plt.subplots(1, 2, figsize=(11, 4.3))
cols = {"lithium, 0.25 mM Mg": pl.HIT, "lithium, 2.5 mM Mg (control)": "#e08e0b", "Mg2+ (control)": pl.LINE, "lamotrigine": "#6a3d9a", "carbamazepine": pl.BASE}
for a_, t in zip(ax, ("scz", "bd")):
    for k, v in J["tail_curves"][t].items():
        x = np.arange(len(v)); a_.plot(x, [r["excess"] for r in v], "o-", color=cols[k], label=k, lw=1.6 if k.startswith("lithium, 0.25") else 1, ms=4)
        for i, r in enumerate(v):
            if k.startswith("lithium, 0.25") and r["p"] <= .01: a_.annotate("p < 0.001" if r["p"] < .001 else f"p {r['p']:.3f}", (i, r["excess"]), fontsize=6.5, xytext=(3, 4), textcoords="offset points")
    v = J["tail_curves"][t]["lithium, 0.25 mM Mg"]; a_.fill_between(np.arange(len(v)), 0, [r["null95"] for r in v], color="k", alpha=.08, lw=0, label="95% of random sets (Li, 0.25)")
    a_.axhline(0, color="k", lw=.5); a_.set_xticks(range(len(FRAC))); a_.set_xticklabels([f"top {int(fr * 100)}%" for fr in FRAC])
    a_.set_xlabel("most drug-sensitive proteins"); a_.set_ylabel("mean MAGMA -log10 p above random sets")
    a_.set_title(f"{'AB'[t == 'bd']}  {DIS[t]}, MHC excluded", fontsize=10)
ax[0].legend(fontsize=6.5)
f.tight_layout(); f.savefig(f"{FIG}/gwas_tail_curves.png", dpi=150); plt.close(f)

# E. split halves
sh = pd.DataFrame(J["split_half"])
f, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for a_, t in zip(ax, ("scz", "bd")):
    d = sh[sh.trait == t].reset_index(drop=True)
    a_.bar(range(len(d)), d["excess"], color=[pl.HIT if p <= .05 else pl.BASE for p in d["p"]])
    a_.set_xticks(range(len(d))); a_.set_xticklabels(["BR " + ",".join(map(str, b)) for b in d["brs"]], rotation=40, ha="right", fontsize=7)
    for i, r in d.iterrows(): a_.text(i, r["excess"], "p < 0.001" if r["p"] < .001 else f"p {r['p']:.3f}", ha="center", va="bottom", fontsize=6.5)
    a_.axhline(0, color="k", lw=.5); a_.set_ylabel("top 5% by Li trend:\nexcess mean -log10 p"); a_.set_title(f"{'CD'[t == 'bd']}  {DIS[t]}: each half on its own", fontsize=9)
f.tight_layout(); f.savefig(f"{FIG}/gwas_split_half.png", dpi=150); plt.close(f)

json.dump(J, open(f"{OUT}/fs06.json", "w"), indent=1, default=lambda o: None if o is None else (bool(o) if isinstance(o, np.bool_) else float(o)))
for key in ("tests", "tests_noMHC"):
  for d, x in J[key].items():
    for t, r in x.items():
        s = f"{key[-5:]:6}{d:4}{t:4} n{r['n_universe']} partial {r['partial_spearman']:+.3f} (p {r['p_partial']:.2g}) "
        for k in ("hits", "hits_rank", "li_specific", "li_specific_rank", "top1pct", "top5pct"):
            if k in r: s += f"| {k} n{r[k]['n']} {r[k]['obs']:.2f} vs {r[k]['null_mean']:.2f} (95% {r[k]['null95']:.2f}) p {r[k]['p']:.3f} "
        s += f"| hit x sig {r['fisher_hit_x_sig']['table'][0]}" + (f" | hit x prio {r['fisher_hit_x_prioritised']['table'][0]}" if t == "scz" else "")
        print(s)
for r in sorted(J["family"], key=lambda r: r["p"])[:6]: print(r)
print("FS06_DONE")
