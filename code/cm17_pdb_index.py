"""cm17: which experimental PDB entries should we measure our proteins' native interfaces in?

Tier 0 of the complex-structure plan. The hypothesis under test is competition: a protein is pulled down
by HSPB1 only to the extent that the surface the chaperone wants -- exposed hydrophobic, aggregation-prone --
is not already occupied by its native partner. cm12/cm16 showed the sequence model's residual is shared
within curated complexes; this asks whether the shared quantity is an interface property.

Nothing is predicted here. We use experimental structures that already exist, so this tier costs CPU only.

Scope. Every entry containing AT LEAST ONE measured protein, not every entry containing two. Requiring two
would condition on having a measured partner, which is close to the variable under test, and would drop the
informative zero class -- a protein whose only structures are monomeric has an occlusion of 0, and that is a
measurement, not a missing value. Whether a partner chain was itself measured is irrelevant to how much
surface it buries, so partners are taken from the structure regardless.

Cost and bias control: at most MAX_ENTRIES structures per protein, ranked by UniProt coverage and then by
assembly simplicity. Without the cap the features would be dominated by how often a protein has been
crystallised, which is a proxy for abundance and stability -- the confound this analysis has to survive.
The count that survives the cap is carried through to cm19 as an explicit covariate anyway.

Accessions in union.tsv may be dotted protein groups (A0A024R1R8.Q9Y2S6), so each component is mapped
separately and a component that also exists as its own single-accession row is bound to that row in preference.

Writes data/campaign/pdb_manifest.tsv (one line per measured chain) and data/campaign/pdb_entries.txt.
"""
import os, gzip, collections
import numpy as np, pandas as pd

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
C = f"{ROOT}/data/campaign"
SIFTS = f"{ROOT}/data/external/sifts/pdb_chain_uniprot.tsv.gz"
MAX_ENTRIES = 15

U = pd.read_csv(f"{C}/union.tsv", sep="\t")
accs = U["accession"].astype(str).tolist(); LEN = U["length"].values

comp2row, ambiguous = {}, collections.Counter()
for i, a in enumerate(accs):
    parts = a.split(".")
    for p in parts:
        p = p.split("-")[0]                       # drop isoform suffix; SIFTS keys on the canonical accession
        if not p: continue
        if p in comp2row:
            ambiguous[p] += 1
            if len(parts) < len(accs[comp2row[p]].split(".")): comp2row[p] = i
        else:
            comp2row[p] = i
print(f"{len(accs)} measured rows -> {len(comp2row)} component accessions "
      f"({sum(ambiguous.values())} components seen in more than one row)", flush=True)

# ---- SIFTS: measured chains, and the total chain count of every entry (used to prefer simple assemblies)
seg = collections.defaultdict(list)
entry_chains = collections.Counter()
with gzip.open(SIFTS, "rt") as fh:
    for line in fh:
        if line.startswith("#") or line.startswith("PDB\t"): continue
        f = line.rstrip("\n").split("\t")
        entry_chains[(f[0], f[1])] = 1
        row = comp2row.get(f[2])
        if row is None: continue
        try: b, e = int(f[7]), int(f[8])
        except ValueError: continue
        seg[(f[0], f[1])].append((f[2], row, b, e))
nchain = collections.Counter()
for pdb, _ in entry_chains: nchain[pdb] += 1

rec = []
for (pdb, ch), ss in seg.items():
    if len({s[0] for s in ss}) > 1: continue      # chimeric chain: two UniProt entries on one chain, skip
    acc, row = ss[0][0], ss[0][1]
    rec.append(dict(pdb=pdb, chain=ch, accession=acc, row=row,
                    sp_beg=min(s[2] for s in ss), sp_end=max(s[3] for s in ss),
                    sp_cov=sum(s[3] - s[2] + 1 for s in ss), n_chains=nchain[pdb]))
M = pd.DataFrame(rec)
M["frac_seq"] = np.clip(M.sp_cov / np.maximum(LEN[M.row.values], 1), 0, 1.5)
print(f"{len(M)} measured chains in {M.pdb.nunique()} PDB entries, {M.row.nunique()} distinct proteins", flush=True)

keep = set()
for row, g in M.sort_values(["frac_seq", "n_chains"], ascending=[False, True]).groupby("row"):
    keep.update(g.pdb.drop_duplicates().head(MAX_ENTRIES).tolist())
M2 = M[M.pdb.isin(keep)].copy()

print(f"\nafter the <= {MAX_ENTRIES} entries/protein cap: {len(keep)} entries, {len(M2)} measured chains, "
      f"{M2.row.nunique()} proteins")
cc = M2.groupby("pdb").n_chains.first()
print("chains per kept entry: " + "  ".join(f"p{q}={np.percentile(cc, q):.0f}" for q in (50, 75, 90, 99))
      + f"  max={cc.max()}  (>300 chains, which cm18 skips: {(cc > 300).sum()})")
print("UniProt coverage of the modelled chain: "
      + "  ".join(f"p{q}={np.percentile(M2.frac_seq, q):.2f}" for q in (10, 25, 50, 75, 90))
      + f"  (< 50 % of their protein: {(M2.frac_seq < 0.5).mean():.1%})")
ne = M2.groupby("row").pdb.nunique()
print(f"structures per protein: median {ne.median():.0f}, "
      f"{(ne == 1).sum()} proteins with exactly one, {(ne >= MAX_ENTRIES).sum()} at the cap")

M2.sort_values(["pdb", "chain"]).to_csv(f"{C}/pdb_manifest.tsv", sep="\t", index=False)
with open(f"{C}/pdb_entries.txt", "w") as fh: fh.write("\n".join(sorted(keep)) + "\n")
print(f"\nwrote {C}/pdb_manifest.tsv and pdb_entries.txt")
print("CM17_DONE")
