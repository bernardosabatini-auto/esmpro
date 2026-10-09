"""pdpipe.io: load a pull-down screen into the standard Dataset.

A Dataset holds three tables that share indices:
  X         proteins x samples, log2 normalised intensity, NaN = not detected
  samples   one row per sample: column name, one column per design field, `cond` (joined condition key)
  proteins  one row per protein: uniprot_id, gene, name, length, plus any annotation columns
and `published`, the source workbook's own statistics, kept only for cross-checking.

Supported source: the proteoDA/limma workbook export ("Targets" sheet converted to TSV by
fs00_convert.py), whose per-sample columns are named normInt_<f1>_<f2>_..._BR<k>.
"""
import os, json
from dataclasses import dataclass, field
import numpy as np, pandas as pd

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
ANNOT = ["Reviewed", "Entry Name", "Gene Names (primary)", "Gene Ontology (biological process)",
         "Gene Ontology (cellular component)", "Gene Ontology (molecular function)", "Subcellular location [CC]",
         "Transmembrane", "Post-translational modification"]


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    samples: pd.DataFrame
    proteins: pd.DataFrame
    published: pd.DataFrame = None
    meta: dict = field(default_factory=dict)

    def cond_order(self):
        return list(dict.fromkeys(self.samples["cond"]))

    def subset(self, mask):
        s = self.samples[mask]
        return Dataset(self.name, self.X[s["column"]], s.reset_index(drop=True), self.proteins, self.published, self.meta)


def _header_row(path):
    with open(path) as fh:
        for i, l in enumerate(fh):
            if l.startswith("uniprot_id\t"): return i
    raise ValueError(f"no 'uniprot_id' header row in {path}")


def load_proteoda(cfg):
    """cfg: dict from a configs/*.yaml file. Keys: name, source, fields (names of the underscore-separated
    tokens after the prefix), numeric (fields to cast to float), condition (fields that define a condition)."""
    path = os.path.join(ROOT, cfg["source"])
    t = pd.read_csv(path, sep="\t", skiprows=_header_row(path), low_memory=False)
    t = t[t["uniprot_id"].notna() & (t["uniprot_id"].astype(str) != "")].reset_index(drop=True)
    pre = cfg.get("prefix", "normInt_")
    cols = [c for c in t.columns if c.startswith(pre)]
    rows = []
    for c in cols:
        tok = c[len(pre):].split("_")
        if len(tok) != len(cfg["fields"]):
            raise ValueError(f"{c}: {len(tok)} tokens, config expects {cfg['fields']}")
        r = dict(zip(cfg["fields"], tok)); r["column"] = c; rows.append(r)
    S = pd.DataFrame(rows)
    S["cond"] = S[cfg["condition"]].agg("|".join, axis=1)          # from the raw tokens, so "0.3" stays "0.3"
    for f in cfg.get("numeric", []): S[f] = S[f].astype(float)
    S["BR"] = S["BR"].str.replace("BR", "").astype(int)
    X = t[cols].apply(pd.to_numeric, errors="coerce")
    X.index = t["uniprot_id"].astype(str)
    P = pd.DataFrame({"uniprot_id": X.index, "gene": t["Genes"].astype(str).values,
                      "name": t["Protein names"].astype(str).values,
                      "length": pd.to_numeric(t.get("Length"), errors="coerce").values})
    for a in ANNOT:
        if a in t.columns: P[a] = t[a].values
    P.index = X.index
    stat = [c for c in t.columns if c not in cols and c not in ANNOT and c not in ("uniprot_id", "Genes", "Protein names", "Gene Names", "Length", "Proteotypic")]
    pub = t[stat].copy(); pub.index = X.index
    # order samples by condition fields, then BR, so plots and tables follow the design
    S = S.sort_values(cfg["condition"] + ["BR"]).reset_index(drop=True)
    X = X[S["column"]]
    return Dataset(cfg["name"], X, S, P, pub, dict(cfg))


def load_diann(cfg):
    """DIA-NN protein report (pd_data/*_report_out.tsv): one column per MS injection, raw log2, NaN = missing.

    cfg keys: name, source, pattern (regex with named groups for the design fields and BR; the TR token is
    matched but not captured), condition, numeric, exclude (sample ids to drop, "<cond fields>_<BR>").
    Technical injections of one biological replicate are averaged in log2 (over those detected). Each sample is
    then median-normalised as in ga01: the median of its log-ratios to each protein's across-sample mean, over
    proteins detected in >= 90 % of samples. The offsets are kept in meta["offsets"] so a global shift is not lost.
    Protein groups are keyed by their first accession; a duplicate key keeps the better-detected group.
    samples["inj_pos"] is each sample's mean position in the MS acquisition order (0-1), for drift checks."""
    import re
    path = os.path.join(ROOT, cfg["source"])
    t = pd.read_csv(path, sep="\t", low_memory=False)
    pat = re.compile(cfg["pattern"]); rows = []
    for c in t.columns:
        m = pat.search(c)
        if m: rows.append({**m.groupdict(), "injection": c})
    I = pd.DataFrame(rows)
    keys = cfg["condition"] + ["BR"]
    I["column"] = I[keys].agg("_".join, axis=1)
    V = t[I["injection"]].apply(pd.to_numeric, errors="coerce")
    acc = t["Protein.Group"].astype(str).str.split(";").str[0]
    order = V.notna().sum(1).sort_values(ascending=False).index
    keep = ~acc.loc[order].duplicated()
    keep = keep.reindex(t.index).values
    t, V, acc = t[keep].reset_index(drop=True), V[keep].reset_index(drop=True), acc[keep].reset_index(drop=True)
    X = V.T.groupby(I["column"].values).mean().T                      # log2 mean over technical injections
    X.index = acc.values
    S = I.drop_duplicates("column").drop(columns=["injection"]).reset_index(drop=True)
    S["n_injections"] = S["column"].map(I.groupby("column").size())
    inj = I["injection"].str.extract(r"_(\d+)$")[0].astype(float)      # MS acquisition index, last token of the name
    I["inj_pos"] = inj.rank(pct=True).values                            # position in the run, 0-1
    S["inj_pos"] = S["column"].map(I.groupby("column")["inj_pos"].mean())
    S = S[~S["column"].isin(cfg.get("exclude", []))]
    S["cond"] = S[cfg["condition"]].agg("|".join, axis=1)
    for f in cfg.get("numeric", []): S[f] = S[f].astype(float)
    S["BR"] = S["BR"].astype(int)
    S = S.sort_values(cfg["condition"] + ["BR"]).reset_index(drop=True)
    X = X[S["column"]]
    det = X.notna().mean(1) >= .9
    off = (X[det] - X[det].mean(1).values[:, None]).median(0)
    X = X - off
    P = pd.DataFrame({"uniprot_id": X.index, "gene": t["Genes"].astype(str).str.split(";").str[0].values,
                      "name": t["Protein.Names"].astype(str).values, "protein_group": t["Protein.Group"].astype(str).values},
                     index=X.index)
    P = P.join(uniprot_annotation(), how="left")
    meta = dict(cfg); meta["offsets"] = off.round(4).to_dict(); meta["n_norm_proteins"] = int(det.sum())
    return Dataset(cfg["name"], X, S, P, None, meta)


def uniprot_annotation():
    """Reviewed human UniProt: length, mass, GO MF/CC, keywords, cofactor, subcellular location, sequence
    (data/annot/uniprot_human_full.tsv.gz), indexed by accession."""
    p = f"{ROOT}/data/annot/uniprot_human_full.tsv.gz"
    if not os.path.exists(p): return pd.DataFrame()
    u = pd.read_csv(p, sep="\t", index_col=0)
    return u.rename(columns={"Length": "length", "Sequence": "sequence", "Gene Ontology (molecular function)": "Gene Ontology (molecular function)"}).drop(columns=["Gene Names (primary)"], errors="ignore")


def load(cfg):
    return load_diann(cfg) if cfg.get("format") == "diann" else load_proteoda(cfg)


def load_config(name):
    import yaml
    return yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", f"{name}.yaml")))


def save(ds, outdir):
    os.makedirs(outdir, exist_ok=True)
    ds.X.to_csv(f"{outdir}/matrix.tsv", sep="\t")
    ds.samples.to_csv(f"{outdir}/samples.tsv", sep="\t", index=False)
    ds.proteins.to_csv(f"{outdir}/proteins.tsv", sep="\t")
    json.dump(ds.meta, open(f"{outdir}/meta.json", "w"), indent=1, default=str)
