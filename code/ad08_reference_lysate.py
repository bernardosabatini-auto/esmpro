"""ad08: reference input lysate for the GA_33 pull-downs.

All manipulations were made in lysate, the soluble proteins do not change across temperature
(western blot; confirmed in the GA_33 input, additional text 8), so one reference lysate describes the
input to every reaction of an experiment.

GA_33: pd_data/GA_33_Proteome_report_out.tsv, 12 samples (bait x staurosporine x temperature), each
  measured in 3 technical (MS) replicates, no biochemical replicates; log2, not normalised, not imputed.
  Two samples deviate from the rest (DNAJA1 / no drug / 35 C, and DNAJB11 / staurosporine / 35 C, the
  latter heavily contaminated) and are left out. Reference = mean over the other 10 samples.
GA_20 has no input of its own. GA_22 and GA_24 are separate experiments (salt changes the input), not
  input controls for GA_20, so no GA_20 reference is built.

Per run: median-centred over proteins detected in >= 90 % of that file's runs. Technical replicates
averaged within a sample, then samples averaged. Contaminants (keratins, filaggrins, immunoglobulins,
lysozyme, dermcidin, cornified-envelope proteins, "contam_" entries) are flagged and excluded downstream.
Output is aligned to the rows of the pull-down's blocks.npz; each clean sample's own (matched) input is
saved too, for a robustness check against the reference.
"""
import os, re, csv, itertools, json
import numpy as np, pandas as pd

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
CONTAM = re.compile(r"^(KRT\d|KRTAP|FLG|CALML5|CSTA|LYZ|IG[HKL][VCJ]|IGHG|IGHA|IGHM|IGKC|IGLC|DCD|SERPINB12|KPRP|LOR$|SBSN|DSC1|DSG1|JUP$|PKP1|HRNR|CASP14|CDSN)")


def load(fn, keep=None):
    df = pd.read_csv(fn, sep="\t", low_memory=False)
    runs = [c for c in df.columns[8:] if keep is None or keep(c)]
    V = df[runs].apply(pd.to_numeric, errors="coerce").values.astype(float)
    det = np.isfinite(V).mean(1) >= 0.9
    off = np.nanmedian(V[det] - np.nanmean(V[det], 1, keepdims=True), 0)
    return df, runs, V - off, off


def contaminant(pg, genes):
    return str(pg).startswith("contam") or any(CONTAM.match(g.strip()) for g in str(genes).split(";"))


def align(df_in, ref, blocks_dir):
    """Map input protein groups onto a pull-down's blocks rows: exact protein group, else first accession."""
    blk = np.load(f"{blocks_dir}/blocks.npz", allow_pickle=True)
    pg_in = df_in["Protein.Group"].astype(str).values
    by_pg = {p: i for i, p in enumerate(pg_in)}
    by_first = {}
    for i, p in enumerate(pg_in):
        by_first.setdefault(p.split(";")[0], i)
    raw = [str(a) for a in blk["accession_raw"]] if "accession_raw" in blk.files else [str(a) for a in blk["accession"]]
    first = [str(a) for a in blk["accession"]]
    idx = np.array([by_pg.get(r, by_first.get(f, -1)) for r, f in zip(raw, first)])
    out = np.full(len(idx), np.nan); ok = idx >= 0
    out[ok] = ref[idx[ok]]
    return out, idx


# ---------------- GA_33
df33, runs33, V33, off33 = load(f"{ROOT}/pd_data/GA_33_Proteome_report_out.tsv")
lab = [re.search(r"Proteome_(21A|24)_STAU_(\d+)_T_(\d+)_BR(\d+)_TR(\d+)", c).groups() for c in runs33]
OUT33 = {("21A", "0", "35"), ("24", "10", "35")}
samples = {}
for k in itertools.product(("21A", "24"), ("0", "10"), ("35", "37", "43")):
    js = [j for j, l in enumerate(lab) if l[:3] == k]
    with np.errstate(invalid="ignore"):
        samples[k] = np.nanmean(V33[:, js], 1)
use = [k for k in samples if k not in OUT33]
S = np.column_stack([samples[k] for k in use])
with np.errstate(invalid="ignore"):
    ref33 = np.nanmean(S, 1)
n33 = np.isfinite(S).sum(1)
con33 = np.array([contaminant(p, g) for p, g in zip(df33["Protein.Group"], df33["Genes"])])
lev33, idx33 = align(df33, ref33, f"{ROOT}/data/ga_data")
blk33 = np.load(f"{ROOT}/data/ga_data/blocks.npz", allow_pickle=True)
con33_rows = np.array([contaminant(p, g) for p, g in zip(blk33["accession_raw"], blk33["gene"])])
print(f"GA_33 input: {len(df33)} protein groups; reference from {len(use)} samples ({len(use)*3} runs), excluded {sorted(OUT33)}", flush=True)
print(f"  proteins in >= 5 of 10 samples: {(n33 >= 5).sum()}; contaminants flagged {con33.sum()}", flush=True)
print(f"  pull-down proteins with a reference level: {np.isfinite(lev33).sum()} of {len(lev33)} ({100*np.isfinite(lev33).mean():.1f}%)", flush=True)
# reproducibility of the reference: two halves of the 10 samples
h1 = np.nanmean(S[:, 0::2], 1); h2 = np.nanmean(S[:, 1::2], 1); ok = np.isfinite(h1) & np.isfinite(h2) & ~con33
r = np.corrcoef(h1[ok], h2[ok])[0, 1]
print(f"  reliability: two halves of the samples correlate r {r:.4f} -> {2*r/(1+r):.4f} for all ten", flush=True)
np.savez(f"{ROOT}/data/ga_data/input_reference.npz", level=lev33, input_index=idx33, contaminant=con33_rows,
         n_samples=np.where(idx33 >= 0, n33[np.maximum(idx33, 0)], 0), excluded=np.array(sorted("/".join(k) for k in OUT33)),
         samples=np.array(["/".join(k) for k in use]),
         matched=np.column_stack([align(df33, samples[k], f"{ROOT}/data/ga_data")[0] for k in use]))

print("\nAD08_DONE", flush=True)
