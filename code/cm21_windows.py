"""cm21: build the Boltz-2 co-folding jobs for the Design A pilot.

Design A: the HSPB1 alpha-crystallin domain (ACD) as a dimer -- the client-binding unit -- plus one
short window of a client. The ACD construct is residues 84-171, which is exactly the construct
crystallised in 9ef3 and 3q9p, so the prediction can be checked against experiment.

WHY WINDOWS AND NOT THE WHOLE CLIENT. HSPB1 binds non-native, locally unfolded segments, not folded
surfaces. A full-length co-fold would hand the model a folded client and ask it to dock a chaperone
onto a native surface -- the state we already know is wrong. A short window has no choice but to be
extended, which is the state the chaperone actually sees. It is also ~190 tokens instead of ~1000.

WHY UNIFORM TILING AND NOT THE AGGREGATION-PRONE WINDOWS. Choosing windows by hydrophobicity or APR
content would confound the structural score with the composition that selected them: the score would
then "predict" binding for the same reason a hydrophobicity scale does, which the sequence model
already has. Every residue is tiled exactly once and the model is left to choose.

CONTROLS, all fixed before any result is seen:
  pos  9ef3 is the ACD with its own C-terminal IxI motif peptide (EITIPVTFE) bound in the beta4/beta8
       groove. It was RELEASED 2026-06-24, after Boltz-2 was trained, so it is genuinely held out.
  pos  6gjh is the same ACD with an N-terminal-region peptide, released 2019, so it IS in training --
       it tests the plumbing, not generalisation. Labelled as such.
  neg  homopolymer windows (E, K, S, G): no groove-binding motif, should score at the floor.
  shuf one real window per protein, residues permuted. Same composition, no sequence order. This is
       the control that decides whether the co-fold sees anything a composition model does not.

Writes data/campaign/boltz/in/shard_*/ and data/campaign/boltz_jobs.tsv.
"""
import os, shutil
import numpy as np, pandas as pd

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
C = f"{ROOT}/data/campaign"
BDIR = f"{C}/boltz"
WIN, STRIDE = 16, 16
NSHARD = int(os.environ.get("CM21_SHARDS", "8"))
SEED = 0

HSPB1 = ("MTERRVPFSLLRGPSWDPFRDWYPHSRLFDQAFGLPRLPEEWSQWLGGSSWPGYVRPLPPAAIESPAVAAPAYSRALSRQ"
         "LSSGVSEIRHTADRWRVSLDVNHFAPDELTVKTKDGVVEITGKHEERQDEHGYISRCFTRKYTLPPGVDPTQVSSSLSPE"
         "GTLTVEAPMPKLATQSNEITIPVTFESRAQLGGPEAAKSDETAAK")
ACD_BEG, ACD_END = 84, 171                       # 1-based, inclusive: the 9ef3 / 3q9p construct
ACD = HSPB1[ACD_BEG - 1:ACD_END]
assert len(ACD) == 88 and ACD.startswith("GVSEI") and ACD.endswith("APMPK"), ACD

MSA = os.environ.get("CM21_MSA", "empty")        # "empty" = single sequence; or a path to an .a3m

CONTROLS = [
    ("CTRL_pos_9ef3_ixi9",   "EITIPVTFE",        "pos_heldout"),   # 9ef3 peptide, released 2026-06
    ("CTRL_pos_ixi16",       "ATQSNEITIPVTFESR", "pos_heldout"),   # 16-mer centred on the same motif
    ("CTRL_pos_6gjh_nt5",    "ALSRQ",            "pos_intrain"),   # 6gjh peptide, released 2019
    ("CTRL_pos_6gjh_nt16",   "PAYSRALSRQLSSGVS", "pos_intrain"),
    ("CTRL_neg_polyE",       "E" * WIN,          "neg"),
    ("CTRL_neg_polyK",       "K" * WIN,          "neg"),
    ("CTRL_neg_polyS",       "S" * WIN,          "neg"),
    ("CTRL_neg_polyG",       "G" * WIN,          "neg"),
    ("CTRL_neg_polyEK",      "EK" * (WIN // 2),  "neg"),
]

YAML = """version: 1
sequences:
  - protein:
      id: [A, B]
      sequence: {acd}
      msa: {msa}
  - protein:
      id: C
      sequence: {pep}
      msa: {msa}
"""


def windows(seq):
    """Non-overlapping WIN-mers covering every residue; the last is flushed to the C-terminus."""
    L = len(seq)
    starts = list(range(0, max(L - WIN, 0) + 1, STRIDE))
    if not starts:
        return [(0, seq)]
    if starts[-1] + WIN < L:
        starts.append(L - WIN)
    return [(s, seq[s:s + WIN]) for s in starts]


P = pd.read_csv(f"{C}/pilot_proteins.tsv", sep="\t")
rng = np.random.default_rng(SEED)
jobs = []
for _, r in P.iterrows():
    ws = windows(r.sequence)
    for s, w in ws:
        jobs.append(dict(name=f"{r.accession}_{s:04d}", accession=r.accession,
                         start=s + 1, end=s + len(w), pep=w, kind="window"))
    s, w = ws[rng.integers(len(ws))]                      # one shuffled twin per protein
    sh = "".join(rng.permutation(list(w)))
    jobs.append(dict(name=f"SHUF_{r.accession}_{s:04d}", accession=r.accession,
                     start=s + 1, end=s + len(w), pep=sh, kind="shuffled"))
for n, p, k in CONTROLS:
    jobs.append(dict(name=n, accession="CONTROL", start=0, end=0, pep=p, kind=k))
jobs.append(dict(name="APO_acd_dimer", accession="CONTROL", start=0, end=0, pep="", kind="apo"))

J = pd.DataFrame(jobs)
print(f"{len(J)} jobs: " + ", ".join(f"{k}={v}" for k, v in J.kind.value_counts().items()))
print(f"peptide length: median {J[J.kind=='window'].pep.str.len().median():.0f}, "
      f"tokens per job ~{2*len(ACD)+WIN}")

if os.path.isdir(f"{BDIR}/in"):
    shutil.rmtree(f"{BDIR}/in")
J["shard"] = np.arange(len(J)) % NSHARD            # round-robin: equal token load per shard
for sh in range(NSHARD):
    os.makedirs(f"{BDIR}/in/shard_{sh:02d}", exist_ok=True)
for _, r in J.iterrows():
    if r.kind == "apo":
        txt = YAML.format(acd=ACD, pep="", msa=MSA).split("  - protein:\n      id: C")[0]
    else:
        txt = YAML.format(acd=ACD, pep=r.pep, msa=MSA)
    with open(f"{BDIR}/in/shard_{r.shard:02d}/{r['name']}.yaml", "w") as fh:
        fh.write(txt)

J.to_csv(f"{C}/boltz_jobs.tsv", sep="\t", index=False)
print(f"wrote {len(J)} yaml files into {NSHARD} shards under {BDIR}/in")
print(f"  per shard: {len(J)//NSHARD}")
print("CM21_DONE")
