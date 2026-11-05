"""export_tables: write every pull-down run out as an analysis-ready table.

One directory per run and fraction under data/export/levels/, each holding the SAME three files, so a
new user learns one format and reads them all. The preprocessing that the campaign applied is already
done; data/export/README.md says exactly what it was.

What is done to a DIA-NN run before it is written (pdpipe.io.load_diann):
  1. protein groups keyed by their first accession; a duplicate key keeps the better-detected group
  2. technical MS injections of one biological replicate averaged in log2, over those detected
  3. each biological-replicate sample median-normalised: the median of its log-ratios to each protein's
     across-sample mean, taken over proteins detected in >= 90 % of samples. Offsets kept in meta.json
  4. nothing imputed. A protein not detected in a sample stays NaN, never 0

GA_20 has no DIA-NN report in this repo; it is exported from the replicate blocks it was reduced to.
FS_73 / FS_76 arrive as proteoDA workbooks already log2 and normalised upstream.

Run:  python code/export_tables.py
"""
import os, json, re, shutil
import numpy as np, pandas as pd
from pdpipe import io

ROOT = io.ROOT
OUT = f"{ROOT}/data/export"
CAMP = f"{ROOT}/data/campaign"

# run key -> (pdpipe config, bait, fraction, one line on what the fraction is)
RUNS = [
    ("GA20_ip",     None,    "HSPB1",             "ip",    "pull-down, KCl 0/75/150 mM"),
    ("GA22_ip",     "ga22",  "HSPB1",             "ip",    "pull-down, KCl x staurosporine"),
    ("GA22_sup",    "ga22s", "HSPB1",             "sup",   "post-IP supernatant (unbound) for GA22_ip"),
    ("GA24_ip",     "ga24",  "HSPB1",             "ip",    "pull-down, KCl x NAD / NADP"),
    ("GA24_sup",    "ga24s", "HSPB1",             "sup",   "post-IP supernatant (unbound) for GA24_ip"),
    ("GA33_ip",     "ga33",  "DNAJA1 / DNAJB11",  "ip",    "pull-down, 35/37/43 C x staurosporine"),
    ("GA33_input",  "ga33p", "DNAJA1 / DNAJB11",  "input", "pre-IP lysate matched to GA33_ip"),
    ("FS73_ip",     "fs73",  "HSPB1",             "ip",    "pull-down, baseline screen"),
    ("FS76_ip",     "fs76",  "HSPB1",             "ip",    "pull-down, Mg2+ x lithium screen"),
]

CONTAM = re.compile(r"^(KRT\d|KRTAP|FLG|CALML5|CSTA|LYZ|IG[HKL][VCJ]|IGHG|IGHA|IGHM|IGKC|IGLC|DCD"
                    r"|SERPINB12|KPRP|LOR$|SBSN|DSC1|DSG1|JUP$|PKP1|HRNR|CASP14|CDSN)")


def is_contaminant(group, genes):
    return bool(str(group).startswith("contam") or
                any(CONTAM.match(g.strip()) for g in str(genes).split(";")))


def export_ga20(d):
    """GA_20 was reduced to biological-replicate blocks before this repo; export at that level."""
    b = np.load(f"{ROOT}/data/pd_data/blocks.npz", allow_pickle=True)
    acc = b["accession"].astype(str)
    cols, rows = {}, []
    for cond, key in (("0", "br_base"), ("75", "br_s75"), ("150", "br_s150")):
        M = b[key].astype(float)
        for k in range(M.shape[1]):
            name = f"{cond}_{k + 1}"
            cols[name] = M[:, k]
            rows.append(dict(column=name, KCl=cond, BR=k + 1, cond=cond, n_injections=2, inj_pos=np.nan))
    X = pd.DataFrame(cols, index=pd.Index(acc, name="accession"))
    S = pd.DataFrame(rows)
    P = pd.DataFrame({"accession": acc, "gene": b["gene"].astype(str),
                      "protein_group": b["accession_raw"].astype(str),
                      "n_peptides": b["npeptides"]}, index=pd.Index(acc, name="accession"))
    meta = dict(name="GA20", source="data/pd_data/blocks.npz",
                note="exported from replicate blocks; the DIA-NN report is not in this repo",
                condition=["KCl"], normalisation="median, applied upstream")
    _write(d, X, S, P, meta)


def _write(d, X, S, P, meta):
    os.makedirs(d, exist_ok=True)
    X.round(4).to_csv(f"{d}/matrix.tsv", sep="\t")
    S.to_csv(f"{d}/samples.tsv", sep="\t", index=False)
    P.to_csv(f"{d}/proteins.tsv", sep="\t", index=False)
    json.dump(meta, open(f"{d}/meta.json", "w"), indent=1, default=str)
    det = X.notna().mean(1)
    print(f"  {os.path.basename(d):<12} {X.shape[0]:>6} proteins x {X.shape[1]:>3} samples   "
          f"detected in all: {int((det == 1).sum()):>5}   any missing: {X.isna().mean().mean():.3f}", flush=True)


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(f"{OUT}/levels", exist_ok=True)
    print("levels (one directory each, same three files):", flush=True)
    index = []
    for key, cfg, bait, frac, what in RUNS:
        d = f"{OUT}/levels/{key}"
        if cfg is None:
            export_ga20(d)
        else:
            ds = io.load(io.load_config(cfg))
            P = ds.proteins.reset_index(drop=True)
            keep = [c for c in ("uniprot_id", "gene", "name", "protein_group", "length") if c in P]
            P = P[keep].rename(columns={"uniprot_id": "accession"})
            if "protein_group" in P:
                P["contaminant"] = [is_contaminant(g, n) for g, n in zip(P["protein_group"], P["gene"])]
            m = dict(ds.meta)
            m["normalisation"] = ("median over proteins detected in >=90% of samples; "
                                  "per-sample offsets in this file under 'offsets'")
            m["injections"] = "technical MS injections averaged in log2 before normalisation"
            m["missing"] = "NaN = not detected; nothing imputed"
            _write(d, ds.X, ds.samples, P, m)
        n_s = pd.read_csv(f"{d}/samples.tsv", sep="\t").shape[0]
        n_p = sum(1 for _ in open(f"{d}/matrix.tsv")) - 1
        index.append(dict(level=key, bait=bait, fraction=frac, what=what, proteins=n_p, samples=n_s))
    pd.DataFrame(index).to_csv(f"{OUT}/levels_index.tsv", sep="\t", index=False)

    # ---- the campaign's derived targets, with their reliability ceilings
    Z = np.load(f"{CAMP}/targets.npz", allow_pickle=True)
    names = list(Z["names"].astype(str))
    U = pd.read_csv(f"{CAMP}/union.tsv", sep="\t")["accession"].astype(str).values
    T = pd.DataFrame(Z["Y"], columns=names)
    T.insert(0, "accession", U)
    T.round(4).to_csv(f"{OUT}/targets.tsv", sep="\t", index=False)
    meta = json.load(open(f"{CAMP}/targets_meta.json"))
    M = pd.DataFrame([dict(target=t, run=v.get("run"), bait=v.get("bait"), kind=v.get("kind"),
                           description=v.get("what"), ceiling=v.get("ceiling"), n=v.get("n"))
                      for t, v in meta.items()])
    M["ceiling"] = M["ceiling"].round(3)
    M.to_csv(f"{OUT}/targets_meta.tsv", sep="\t", index=False)
    print(f"\ntargets.tsv            {T.shape[0]} proteins x {len(names)} targets", flush=True)

    # ---- protein annotation and the held-out blocks, shared by every table
    F = np.load(f"{CAMP}/folds.npz", allow_pickle=True)
    ann = io.uniprot_annotation()
    A = pd.DataFrame({"accession": U,
                      "cluster30": F["group"].astype(str),
                      "heldout_block": F["outer"].astype(int)})
    # gene symbols come from the runs themselves: uniprot_annotation() drops the primary-gene column.
    gmap = {}
    for key, *_ in RUNS:
        p = pd.read_csv(f"{OUT}/levels/{key}/proteins.tsv", sep="\t")
        if "gene" in p:
            for acc, g in zip(p["accession"].astype(str), p["gene"].astype(str)):
                if acc not in gmap and g and g not in ("nan", "None"):
                    gmap[acc] = g
    A.insert(1, "gene", [gmap.get(a, "") for a in U])
    print(f"  gene symbols resolved for {sum(bool(x) for x in A['gene'])} of {len(A)} proteins", flush=True)
    if len(ann):
        a = ann.reindex(U)
        for c in ("length", "Mass", "Keywords"):
            if c in a:
                A[c.lower()] = a[c].values
        seq = a["sequence"] if "sequence" in a else None
        if seq is not None:
            with open(f"{OUT}/sequences.fasta", "w") as fh:
                for acc, s in zip(U, seq.values):
                    if isinstance(s, str) and s:
                        fh.write(f">{acc}\n{s}\n")
    A.to_csv(f"{OUT}/proteins.tsv", sep="\t", index=False)
    print(f"proteins.tsv           {len(A)} proteins, with 30% identity cluster and held-out block", flush=True)

    # data/ is gitignored, so the documents live in the repo and are copied in beside the tables, where
    # whoever receives the tarball will actually look for them.
    doc = f"{ROOT}/docs/DATA_EXPORT.md"
    if os.path.exists(doc):
        shutil.copy(doc, f"{OUT}/README.md")
        print("README.md              copied from docs/DATA_EXPORT.md", flush=True)
    os.makedirs(f"{OUT}/docs", exist_ok=True)
    DOCS = ["docs/DATA_EXPORT.md", "docs/DATA_EXPORT.pdf",
            "docs/HSP_PULLDOWNS_OVERVIEW.md", "docs/HSP_PULLDOWNS_OVERVIEW.pdf",
            "docs/PIPELINE.md",
            "reports/campaign/hsp_mlp_campaign_report.md",
            "reports/campaign/hsp_mlp_campaign_report.pdf"]
    got = []
    for rel in DOCS:
        src = f"{ROOT}/{rel}"
        if os.path.exists(src):
            shutil.copy(src, f"{OUT}/docs/{os.path.basename(rel)}")
            got.append(os.path.basename(rel))
        else:
            print(f"  MISSING, not copied: {rel}", flush=True)
    print(f"docs/                  {len(got)} documents: {', '.join(got)}", flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
