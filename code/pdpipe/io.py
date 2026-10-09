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


def load_config(name):
    import yaml
    return yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", f"{name}.yaml")))


def save(ds, outdir):
    os.makedirs(outdir, exist_ok=True)
    ds.X.to_csv(f"{outdir}/matrix.tsv", sep="\t")
    ds.samples.to_csv(f"{outdir}/samples.tsv", sep="\t", index=False)
    ds.proteins.to_csv(f"{outdir}/proteins.tsv", sep="\t")
    json.dump(ds.meta, open(f"{outdir}/meta.json", "w"), indent=1, default=str)
