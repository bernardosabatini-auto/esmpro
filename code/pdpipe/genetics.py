"""Cross-reference pull-down screens with disease genetics.

drug_sensitivity()   one -log10 p per protein per drug for FS73 and FS76, hit flags, t values, Li-specific set
hgnc()               HGNC table: Ensembl gene / Entrez gene -> UniProt accessions
by_accession(df)     re-key a gene-level table (column 'ens' or 'entrez') to UniProt, keeping each accession's best p
matched_null()       mean -log10 p of a protein set against random sets drawn within length deciles
partial_spearman()   rank correlation of drug sensitivity with genetic association, controlling for covariates
"""
import numpy as np, pandas as pd
from scipy.stats import rankdata, t as tdist
from . import stats as st, io

DRUG = {"car": "carbamazepine", "lam": "lamotrigene", "top": "topiramate"}
NICE = {"car": "carbamazepine", "lam": "lamotrigine", "top": "topiramate", "li": "lithium", "mg": "Mg2+ (comparison)"}


def hgnc():
    h = pd.read_csv(f"{io.ROOT}/data/annot/hgnc_complete_set.txt.gz", sep="\t",
                    usecols=["symbol", "ensembl_gene_id", "entrez_id", "uniprot_ids"], dtype=str)
    h = h.dropna(subset=["uniprot_ids"])
    return pd.DataFrame([(s, e, z, a) for s, e, z, u in zip(h.symbol, h.ensembl_gene_id, h.entrez_id, h.uniprot_ids)
                         for a in u.split("|")], columns=["symbol", "ens", "entrez", "acc"])


def by_accession(df, key, pcol):
    """df keyed by `key` ('ens' or 'entrez'); returns one row per UniProt accession, best (smallest) pcol kept."""
    m = hgnc().dropna(subset=[key])[[key, "acc", "symbol"]]
    return m.merge(df, on=key).sort_values(pcol).groupby("acc").first()


def drug_sensitivity():
    """Returns dict with sens, hits, tval (per contrast), lispec, meta (gene, length), level (control pull-down level),
    ds73, ds76, t76."""
    R = io.ROOT
    D73 = io.load_proteoda(io.load_config("fs73"))
    Dm, names = st.design(D73.samples, block=True); fit = st.ebayes(st.lmfit(D73.X, Dm))

    def cvec(w):
        c = np.zeros(len(names))
        for k, v in w.items(): c[names.index(k)] += v
        return c

    c73 = pd.read_csv(f"{R}/data/fs_data/FS73/contrasts.tsv", sep="\t", index_col=0)
    c73 = c73.loc[:, ~c73.columns.duplicated()]
    sens, hits, tval = {}, {}, {}
    for d, nm in DRUG.items():
        F = st.ftest(fit, np.stack([cvec({f"{nm}|{x}": 1, "ctl|0": -1}) for x in (10, 100)], 1))
        sens[d] = -np.log10(F["p"])
        hits[d] = (c73[f"{d}_10:q"] <= .05) | (c73[f"{d}_100:q"] <= .05)
        for x in (10, 100): tval[f"{d} {x}"] = c73[f"{d}_{x}:t"]
    t76 = pd.read_csv(f"{R}/data/fs_data/FS76/fs02_tests.tsv", sep="\t", index_col=0)
    sens["li"] = -np.log10(t76["trend Mg0.25:p"]); sens["mg"] = -np.log10(t76["Mg effect:p"])
    hits["li"] = (t76[["trend Mg0.25:q", "trend Mg2.5:q", "any dose Mg0.25:q", "any dose Mg2.5:q"]] <= .05).any(axis=1)
    hits["mg"] = pd.Series(False, index=t76.index)
    tval["Li, Mg 0.25"] = t76["trend Mg0.25:t"]; tval["Li, Mg 2.5"] = t76["trend Mg2.5:t"]; tval["Mg2+"] = t76["Mg effect:t"]
    f04 = pd.read_csv(f"{R}/data/fs_data/FS76/fs04_li_specific.tsv", sep="\t")
    lispec = set(f04.loc[f04["call"].fillna("").str.startswith("Li-specific"), "accession"]) - {"Q9ULD6"}   # INTU: one sample
    D76 = io.load_proteoda(io.load_config("fs76"))
    meta = pd.concat([D73.proteins[["gene", "length"]], D76.proteins[["gene", "length"]]]).groupby(level=0).first()
    level = {**{d: D73.X[D73.samples.loc[D73.samples["cond"] == "ctl|0", "column"]].mean(1) for d in DRUG},
             **{d: D76.X[D76.samples.loc[D76.samples["cond"] == "0.25|0", "column"]].mean(1) for d in ("li", "mg")}}
    return dict(sens=sens, hits=hits, tval=tval, lispec=lispec, meta=meta, level=level, ds73=D73, ds76=D76, t76=t76)


def matched_null(f, sel, col, rng, n=2000, strata="loglen", rank=False):
    """Mean -log10 p of `col` over the proteins in `sel` against random sets of the same size drawn within
    deciles of `strata` (protein length by default). rank=True replaces -log10 p by its percentile among all
    proteins, so a single extreme gene (e.g. in the MHC) cannot carry the set."""
    ok = f[col].notna(); z = -np.log10(f.loc[ok, col]).values
    if rank: z = rankdata(z) / len(z)
    dec = pd.qcut(f.loc[ok, strata], 10, labels=False, duplicates="drop").values
    s = np.asarray(sel)[ok.values]; obs = z[s].mean(); counts = np.bincount(dec[s], minlength=dec.max() + 1)
    pools = [np.nonzero(dec == k)[0] for k in range(dec.max() + 1)]
    null = np.zeros(n)
    for k, c in enumerate(counts):
        if c: null += z[pools[k][rng.random((n, len(pools[k]))).argsort(1)[:, :c]]].sum(1)   # c without replacement
    null /= s.sum()
    return dict(n=int(s.sum()), obs=float(obs), null_mean=float(null.mean()), null95=float(np.percentile(null, 95)),
                p=float((1 + (null >= obs).sum()) / (n + 1)), n_p05=int((f.loc[ok, col].values[s] <= .05).sum()),
                expected_p05=float(s.sum() * (f.loc[ok, col] <= .05).mean()))


def partial_spearman(f, col, covars=("loglen", "level")):
    ok = f[col].notna() & f[list(covars)].notna().all(axis=1)
    y = rankdata(f.loc[ok, "sens"]); x = rankdata(-np.log10(f.loc[ok, col]))
    Z = np.column_stack([np.ones(ok.sum())] + [rankdata(f.loc[ok, c]) for c in covars])
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]; rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    r = np.corrcoef(rx, ry)[0, 1]; n = int(ok.sum()); df = n - 2 - len(covars)
    tt = r * np.sqrt(df / (1 - r * r)); return float(r), float(2 * tdist.sf(abs(tt), df)), n
