"""cm20: choose the ~200 proteins for the Design A chaperone-client co-folding pilot.

Tier 1 of the complex-structure plan. Tier 0 (cm17-cm19) showed that the native partner's occlusion
of a client's surface explains none of the pull-down residual, which rules out steric competition as
the explanation for the complex effect but says nothing about the chaperone side: no experimental
structure contains HSPB1 bound to a client. This tier predicts that interface instead.

READOUT. GA22_part_base and GA24_part_base -- the bound:free partition of each protein in an HSPB1
pull-down, no salt, no drug. These are the right targets because the partition divides out abundance
experimentally (bound over the matched supernatant from the same reaction), so a protein scores high
because it binds HSPB1, not because there is a lot of it. They are also two INDEPENDENT runs of the
same quantity, which gives the selection its own reproducibility check.

SELECTION. Stratified by decile of the two-run mean, 20 per decile. Stratifying on the outcome buys
power for a go/no-go but inflates any correlation computed on the sample, so cm22 reports both the
stratified estimate and a population estimate reweighted by the inverse sampling probability. One
protein per MMseqs2 30 % cluster, so that no fold is counted twice.

SCOPE LIMIT. Length <= MAXLEN, because the design tiles the whole sequence in non-overlapping windows
and cost is linear in length. This is a real restriction on what the pilot can conclude: it says
nothing about large multidomain proteins. Stated, not hidden.

Writes data/campaign/pilot_proteins.tsv and pilot.fasta.
"""
import os, json
import numpy as np, pandas as pd

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
C = f"{ROOT}/data/campaign"
BAIT = "P04792"                 # HSPB1 itself: the bait, not a client
MAXLEN, MINLEN = 600, 60
PER_DECILE, NDEC = 20, 10
SEED = 0

T = np.load(f"{C}/targets.npz", allow_pickle=True)
acc, Y, names = T["accession"], T["Y"], list(T["names"])
F = np.load(f"{C}/folds.npz", allow_pickle=True)
group = F["group"]
U = pd.read_csv(f"{C}/union.tsv", sep="\t").set_index("accession")

y22 = Y[:, names.index("GA22_part_base")].astype(float)
y24 = Y[:, names.index("GA24_part_base")].astype(float)
both = np.isfinite(y22) & np.isfinite(y24)
print(f"partition measured in both runs: {both.sum()} proteins")
r = np.corrcoef(y22[both], y24[both])[0, 1]
print(f"run-to-run agreement of the readout: r = {r:.3f}  (this is the ceiling the pilot can reach)")


def z(v, k):
    return (v - v[k].mean()) / v[k].std()


part = np.where(both, (z(y22, both) + z(y24, both)) / 2, np.nan)

ln = U["length"].reindex(acc).values
ok = both & np.isfinite(ln) & (ln >= MINLEN) & (ln <= MAXLEN) & (acc != BAIT)
ok &= U["has_seq"].reindex(acc).fillna(False).values.astype(bool)
print(f"eligible after length {MINLEN}-{MAXLEN}, sequence present, bait removed: {ok.sum()}")

# one protein per 30 % identity cluster, chosen at random within the cluster
rng = np.random.default_rng(SEED)
idx = np.where(ok)[0]
rng.shuffle(idx)
seen, uniq = set(), []
for i in idx:
    g = group[i]
    if g in seen:
        continue
    seen.add(g)
    uniq.append(i)
uniq = np.array(sorted(uniq))
print(f"after one-per-cluster: {len(uniq)}")

# decile strata on the eligible, de-duplicated set
q = np.quantile(part[uniq], np.linspace(0, 1, NDEC + 1))
q[0], q[-1] = -np.inf, np.inf
dec = np.searchsorted(q, part[uniq], side="right") - 1
dec = np.clip(dec, 0, NDEC - 1)

pick, nstrat = [], {}
for d in range(NDEC):
    cand = uniq[dec == d]
    nstrat[d] = len(cand)
    take = rng.choice(cand, size=min(PER_DECILE, len(cand)), replace=False)
    pick.extend(take.tolist())
pick = np.array(sorted(pick))
pdec = dec[np.searchsorted(uniq, pick)]
# inverse sampling probability, for the population-weighted estimate in cm22
w = np.array([nstrat[d] / min(PER_DECILE, nstrat[d]) for d in pdec])

seqs = {}
with open(f"{C}/union.fasta") as fh:
    a = None
    for line in fh:
        if line.startswith(">"):
            a = line[1:].strip()
            seqs[a] = []
        elif a:
            seqs[a].append(line.strip())
seqs = {k: "".join(v) for k, v in seqs.items()}

P = pd.DataFrame(dict(accession=acc[pick], length=ln[pick].astype(int), decile=pdec,
                      part_mean=part[pick], part_ga22=y22[pick], part_ga24=y24[pick],
                      weight=w, group=group[pick]))
P["sequence"] = [seqs.get(a, "") for a in P.accession]
missing = (P.sequence.str.len() == 0).sum()
bad = (P.sequence.str.len() != P.length).sum()
print(f"sequences: {missing} missing, {bad} with a length mismatch against union.tsv")
P = P[(P.sequence.str.len() == P.length)].reset_index(drop=True)

print(f"\npilot set: {len(P)} proteins, {P.length.sum():,} residues")
print(P.groupby("decile").agg(n=("accession", "size"), part=("part_mean", "mean"),
                              len=("length", "mean")).round(2).to_string())
print(f"\npartition range covered: {P.part_mean.min():.2f} to {P.part_mean.max():.2f} "
      f"(population sd 1.00 by construction)")

P.to_csv(f"{C}/pilot_proteins.tsv", sep="\t", index=False)
with open(f"{C}/pilot.fasta", "w") as fh:
    for a, s in zip(P.accession, P.sequence):
        fh.write(f">{a}\n{s}\n")
json.dump(dict(readout=["GA22_part_base", "GA24_part_base"], run_agreement=float(r),
               n=len(P), maxlen=MAXLEN, per_decile=PER_DECILE, seed=SEED,
               n_eligible=int(ok.sum()), n_clustered=int(len(uniq))),
          open(f"{C}/cm20_pilot.json", "w"), indent=1)
print(f"\nwrote {C}/pilot_proteins.tsv, pilot.fasta, cm20_pilot.json")
print("CM20_DONE")
