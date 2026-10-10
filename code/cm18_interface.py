"""cm18: how much of each protein's chaperone-relevant surface is already occupied by a native partner?

Tier 0 of the complex-structure plan, and the only tier that needs no structure prediction at all. The
hypothesis is competition. HSPB1 binds exposed hydrophobic / aggregation-prone surface on non-native clients.
If a protein spends its life with that surface buried against a native partner, it should be a poor client; if
a perturbation (salt, heat, Mg) loosens the native interface, it should become a better one. cm12/cm16 found
that the sequence model's residual is shared within curated complexes -- this asks whether the shared quantity
is an interface property, using experimental structures that already exist.

Design
  For every PDB biological assembly holding >= 2 of our measured proteins (cm17), and for every measured chain
  A in it, compute the solvent accessible surface of A twice: alone, and in the presence of its partners. The
  difference, resolved by residue class, is what the partner occludes.

  Two cost controls, both of which also reduce bias:
    - a chain's SASA in context only needs partner atoms within 12 A (probe 1.4 A, max radius 1.8 A, so the
      influence radius is < 8 A). Trimming to that neighbourhood makes a ribosome no more expensive than a
      dimer, and is exact.
    - at most MAX_ENTRIES structures per protein, ranked by sequence coverage then by simplicity. Without this
      the features would be dominated by how often a protein has been crystallised, which is itself a proxy
      for abundance and stability -- the confound the whole analysis is trying to avoid.

Guards (this project loses weeks to silent misalignment, so each is reported, not assumed)
  - chain naming: RCSB assemblies name symmetry copies "A-2", while SIFTS knows them as "A". Matching on the
    raw name would silently discard exactly the symmetry-generated partners that matter most (a homodimer
    absent from the asymmetric unit). Names are reduced with base_chain().
  - sequence: every mapped chain is aligned against its UniProt sequence and the identity is recorded. Chains
    below MIN_IDENT are dropped and counted.
  - a chain with no partner atom within 5 A must come out with interface area ~0; the distribution is printed.

Writes data/campaign/iface_entry.tsv (one row per protein x entry) and data/campaign/interface.tsv (per protein).
"""
import os, sys, gzip, glob, json, time, collections
import numpy as np, pandas as pd
import gemmi, freesasa
from scipy.spatial import cKDTree
from multiprocessing import Pool

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
C = f"{ROOT}/data/campaign"; CIF = f"{ROOT}/data/external/pdb_assemblies/cif"
freesasa.setVerbosity(freesasa.silent)

MAX_ENTRIES = 25        # structures per protein USED IN THE AGGREGATION. Distinct from cm17's cap of 15,
                        # which bounds the entries selected FOR a protein; a protein also appears in entries
                        # selected for others, so it can reach hundreds of structures regardless.
MAX_CHAINS = 300        # skip ribosome / capsid scale assemblies
NEIGH = 12.0            # A, partner atoms that can affect chain A's SASA
MIN_IDENT = 0.80        # chain-to-UniProt identity below this is treated as a mapping failure
SB_CUT = 4.0            # A, carboxylate O to basic N

AA3 = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L",
           LYS="K", MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V",
           MSE="M", SEC="U", PYL="O")
RAD = dict(C=1.70, N=1.55, O=1.52, S=1.80, P=1.80, SE=1.90)
HYD = set("AVLIMFWYC")
POLAR = set("STNQYHDEKR")
POS, NEG = set("KR"), set("DE")
ACID_O, BASE_N = {"OD1", "OD2", "OE1", "OE2"}, {"NZ", "NE", "NH1", "NH2", "ND1", "NE2"}
NUC = {"A", "C", "G", "U", "DA", "DC", "DG", "DT", "DI", "DU"}


def base_chain(name):
    """'A-2' (RCSB symmetry copy) -> 'A' (what SIFTS calls it)"""
    return name.split("-")[0]


def apr_mask(seq):
    """aggregation-prone residues: a 7-window with >= 5 hydrophobics, no charge, at most one P/G.
    This is the surface a chaperone competes for, and the surface partial unfolding exposes."""
    s = np.array(list(seq)); n = len(s)
    h = np.isin(s, list(HYD)).astype(int); c = np.isin(s, list(POS | NEG)).astype(int)
    pg = np.isin(s, ["P", "G"]).astype(int)
    m = np.zeros(n, bool)
    if n < 7: return m
    ch, cc, cp = np.cumsum(np.r_[0, h]), np.cumsum(np.r_[0, c]), np.cumsum(np.r_[0, pg])
    for i in range(n - 6):
        if ch[i + 7] - ch[i] >= 5 and cc[i + 7] - cc[i] == 0 and cp[i + 7] - cp[i] <= 1:
            m[i:i + 7] = True
    return m


def model_atoms(mdl):
    """ONE pass over the assembly, not one per chain.

    The first version of this rebuilt the partner atom list inside the per-chain loop, which makes an entry
    cost O(n_chains x n_atoms) in Python -- a 50-chain assembly did fifty full passes for the same data. Here
    every atom is visited once and each chain is then a pair of array slices, because gemmi yields atoms
    chain by chain so a chain's atoms are contiguous.

    Returns flat arrays plus, per chain, the slice bounds, its one-letter sequence and the per-atom index of
    the protein residue within that chain (-1 for nucleic atoms).
    """
    xyz, rad, pres, isnuc, acid, base = [], [], [], [], [], []
    chains = []
    for ch in mdl:
        a0 = len(xyz); seq = []; k = 0
        for res in ch:
            one = AA3.get(res.name); nuc = res.name.strip() in NUC
            if one is None and not nuc: continue
            keep = [a for a in res if a.element.name != "H"]
            if not keep: continue
            for a in keep:
                xyz.append((a.pos.x, a.pos.y, a.pos.z))
                rad.append(RAD.get(a.element.name.upper(), 1.70))
                pres.append(-1 if nuc else k); isnuc.append(nuc)
                acid.append((not nuc) and a.name in ACID_O); base.append((not nuc) and a.name in BASE_N)
            if not nuc: seq.append(one); k += 1
        chains.append(dict(name=ch.name, a0=a0, a1=len(xyz), seq="".join(seq), nres=k))
    if not xyz: return None
    return (np.asarray(xyz, np.float64), np.asarray(rad, np.float64), np.asarray(pres, np.int32),
            np.asarray(isnuc, bool), np.asarray(acid, bool), np.asarray(base, bool), chains)


def sasa(xyz, rad, n_first=None):
    """freesasa on raw coordinates, one C call. Returns the areas of the first n_first atoms."""
    res = freesasa.calcCoord(xyz.ravel().tolist(), rad.tolist())
    m = len(rad) if n_first is None else n_first
    return np.fromiter((res.atomArea(i) for i in range(m)), np.float64, m)


def per_res(a, ridx, n):
    out = np.zeros(n); np.add.at(out, ridx, a); return out


def do_entry(args):
    pdb, rows = args                       # rows: list of dicts from the manifest for this entry
    try:
        st = gemmi.read_structure(f"{CIF}/{pdb}.cif.gz")
    except Exception as e:
        return [], dict(pdb=pdb, err=f"read:{type(e).__name__}")
    if len(st) == 0: return [], dict(pdb=pdb, err="nomodel")
    mdl = st[0]
    if len(mdl) > MAX_CHAINS: return [], dict(pdb=pdb, err="toobig", n_chains=len(mdl))

    M = model_atoms(mdl)
    if M is None: return [], dict(pdb=pdb, err="noatoms")
    XYZ, RADA, PRES, ISNUC, ACID, BASE, chains = M

    want = collections.defaultdict(list)
    for r in rows: want[r["chain"]].append(r)
    # Copies of the same chain in an assembly ("A", "A-2", ...) are related by the assembly's own symmetry
    # operators, so their environments are identical by construction. Measure the first and record how many
    # there were; doing all of them would double or quadruple the cost for an identical answer.
    ncopy = collections.Counter(base_chain(c["name"]) for c in chains)
    npart_chain = sum(1 for c in chains if c["a1"] - c["a0"] > 20) - 1
    seen, out, skipped = set(), [], []

    for ci, cinfo in enumerate(chains):
        b = base_chain(cinfo["name"])
        hits = want.get(b)
        if not hits or b in seen: continue
        seen.add(b)
        a0, a1, seq, nres = cinfo["a0"], cinfo["a1"], cinfo["seq"], cinfo["nres"]
        if nres < 20 or a1 - a0 < 50: continue
        own = slice(a0, a1)
        if (PRES[own] < 0).any(): continue                 # mixed protein/nucleic chain: not our case
        xyz, rad, ridx = XYZ[own], RADA[own], PRES[own]

        oth = np.r_[np.arange(a0), np.arange(a1, len(XYZ))]
        if len(oth):
            # neighbourhood trim: exact, because nothing beyond ~8 A can change an atom's accessibility.
            # One C call returning an array, rather than a list per query point.
            d, _ = cKDTree(xyz).query(XYZ[oth], k=1, distance_upper_bound=NEIGH)
            oth = oth[np.isfinite(d)]
        onuc = oth[ISNUC[oth]] if len(oth) else oth
        opro = oth[~ISNUC[oth]] if len(oth) else oth

        free_a = sasa(xyz, rad)
        bp_a = sasa(np.vstack([xyz, XYZ[opro]]), np.r_[rad, RADA[opro]], n_first=len(rad)) if len(opro) else free_a
        if len(onuc):
            idx = np.r_[opro, onuc]
            ba_a = sasa(np.vstack([xyz, XYZ[idx]]), np.r_[rad, RADA[idx]], n_first=len(rad))
        else:
            ba_a = bp_a

        sf, sb, sa = (per_res(x, ridx, nres) for x in (free_a, bp_a, ba_a))
        s = np.array(list(seq))
        hyd, apr, pol = np.isin(s, list(HYD)), apr_mask(seq), np.isin(s, list(POLAR))
        d_ = np.clip(sf - sb, 0, None)                     # occluded by protein partners
        dn = np.clip(sb - sa, 0, None)                     # additionally occluded by nucleic acid
        tot = sf.sum()
        if tot <= 0: continue

        # interface ion pairs: A's carboxylate O to a partner's basic N, and the reverse. count_neighbors
        # returns the pair count in one C call.
        nsb = 0
        if len(opro):
            oa, ob = XYZ[opro[ACID[opro]]], XYZ[opro[BASE[opro]]]
            ma, mb = XYZ[own][ACID[own]], XYZ[own][BASE[own]]
            if len(ma) and len(ob): nsb += int(cKDTree(ma).count_neighbors(cKDTree(ob), SB_CUT))
            if len(mb) and len(oa): nsb += int(cKDTree(mb).count_neighbors(cKDTree(oa), SB_CUT))

        ident = gemmi.align_string_sequences(list(seq), list(hits[0]["uniprot_seq"]), []).calculate_identity() / 100.0
        if ident < MIN_IDENT:
            skipped.append((pdb, cinfo["name"], hits[0]["accession"], round(ident, 3))); continue

        rec = dict(
            pdb=pdb, chain=cinfo["name"], accession=hits[0]["accession"], row=hits[0]["row"],
            ident=ident, n_res_model=nres, frac_seq=hits[0]["frac_seq"],
            n_chains=len(chains), n_partner_chains=npart_chain, n_copies=ncopy[b],
            has_nucleic=int(len(onuc) > 0),
            sasa_free_per_res=tot / nres,
            iface_area=d_.sum(), iface_frac=d_.sum() / tot, iface_area_per_res=d_.sum() / nres,
            hphob_free_per_res=sf[hyd].sum() / nres,
            hphob_occluded_frac=(d_[hyd].sum() / sf[hyd].sum()) if sf[hyd].sum() > 0 else np.nan,
            hphob_exposed_bound_per_res=sb[hyd].sum() / nres,
            apr_free_per_res=sf[apr].sum() / nres if apr.any() else 0.0,
            apr_occluded_frac=(d_[apr].sum() / sf[apr].sum()) if (apr.any() and sf[apr].sum() > 0) else np.nan,
            apr_exposed_bound_per_res=sb[apr].sum() / nres if apr.any() else 0.0,
            iface_hphob_frac=d_[hyd].sum() / d_.sum() if d_.sum() > 0 else np.nan,
            iface_polar_frac=d_[pol].sum() / d_.sum() if d_.sum() > 0 else np.nan,
            iface_pos_frac=d_[np.isin(s, list(POS))].sum() / d_.sum() if d_.sum() > 0 else np.nan,
            iface_neg_frac=d_[np.isin(s, list(NEG))].sum() / d_.sum() if d_.sum() > 0 else np.nan,
            iface_saltbridge_per_1000A2=1000.0 * nsb / d_.sum() if d_.sum() > 100 else np.nan,
            nucleic_occluded_frac=dn.sum() / tot,
        )
        out.append(rec)
    return out, dict(pdb=pdb, err="", n_skip=len(skipped), skipped=skipped[:3])


def main():
    if os.environ.get("CM18_AGG_ONLY"):          # rebuild interface.tsv from the saved per-chain records
        aggregate(pd.read_csv(f"{C}/iface_entry.tsv", sep="\t")); print("CM18_DONE"); return
    nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))
    M = pd.read_csv(f"{C}/pdb_manifest.tsv", sep="\t")
    have = {os.path.basename(f)[:-7] for f in glob.glob(f"{CIF}/*.cif.gz")}
    M = M[M.pdb.isin(have)].copy()
    print(f"{len(M)} measured chains in {M.pdb.nunique()} downloaded entries", flush=True)

    # UniProt sequences, for the identity guard
    seqs = {}
    for fa in ("union.fasta", "proteome_rest.fasta"):
        p = f"{C}/{fa}"
        if not os.path.exists(p): continue
        acc, buf = None, []
        for line in open(p):
            if line.startswith(">"):
                if acc and acc not in seqs: seqs[acc] = "".join(buf)
                acc = line[1:].split()[0].split("|")[-1] if "|" in line else line[1:].split()[0]
                buf = []
            else: buf.append(line.strip())
        if acc and acc not in seqs: seqs[acc] = "".join(buf)
    print(f"{len(seqs)} UniProt sequences available for the identity guard", flush=True)
    M["uniprot_seq"] = M.accession.map(seqs)
    nomiss = M.uniprot_seq.notna()
    print(f"  chains without a sequence (dropped): {(~nomiss).sum()}", flush=True)
    M = M[nomiss]

    # the <= 15 entries/protein cap is applied in cm17, which writes the manifest; nothing to re-select here
    ne = M.groupby("row").pdb.nunique()
    print(f"{M.pdb.nunique()} entries, {len(M)} chains, {M.row.nunique()} proteins "
          f"(structures per protein: median {ne.median():.0f}, max {ne.max()})", flush=True)

    jobs = [(pdb, g.to_dict("records")) for pdb, g in M.groupby("pdb")]
    lim = int(os.environ.get("CM18_LIMIT", 0))          # smoke test on a random subset; 0 = the real run
    if lim:
        jobs = [jobs[i] for i in np.random.default_rng(0).permutation(len(jobs))[:lim]]
        print(f"CM18_LIMIT={lim}: SMOKE TEST on {len(jobs)} random entries, results are not the real run",
              flush=True)
    t0 = time.time(); recs, errs, nskip = [], collections.Counter(), 0
    with Pool(nproc) as pool:
        for i, (out, info) in enumerate(pool.imap_unordered(do_entry, jobs, chunksize=8)):
            recs.extend(out); errs[info.get("err", "")] += 1; nskip += info.get("n_skip", 0)
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(jobs)} entries  {len(recs)} chains  {time.time()-t0:.0f}s", flush=True)
    print(f"done in {time.time()-t0:.0f}s on {nproc} cores; entry status {dict(errs)}; "
          f"chains failing the identity guard: {nskip}", flush=True)

    D = pd.DataFrame(recs)
    D.to_csv(f"{C}/iface_entry.tsv", sep="\t", index=False)
    aggregate(D)


def aggregate(D):
    print(f"\n{len(D)} (chain, entry) records, {D.row.nunique()} proteins")
    print(f"chain-to-UniProt identity: p05 {D.ident.quantile(.05):.3f}  median {D.ident.median():.3f}")
    lone = D[D.n_partner_chains == 0]
    print(f"GUARD chains with no partner chain (n={len(lone)}): mean iface_frac "
          f"{lone.iface_frac.mean() if len(lone) else float('nan'):.5f} (must be ~0)")
    print(f"iface_frac over all chains: median {D.iface_frac.median():.3f}, p90 {D.iface_frac.quantile(.9):.3f}")

    # The <= 15 entries/protein cap in cm17 bounds the structures chosen FOR a protein, but a protein also turns
    # up in entries chosen for other proteins -- ubiquitin and actin appear hundreds of times. The median is
    # insensitive to that, but a max over 337 structures is mechanically larger than a max over 3, which would
    # make the _max features a disguised measure of how well studied a protein is. So the aggregation draws a
    # fixed random subsample first, and the true count is kept as n_entries for cm19 to control on.
    n_all = D.groupby("row").pdb.nunique()
    print(f"structures per protein before subsampling: median {n_all.median():.0f}, max {n_all.max()}, "
          f"{(n_all > MAX_ENTRIES).sum()} proteins over the cap")
    D = (D.sample(frac=1.0, random_state=0)
           .groupby("row", group_keys=False).head(MAX_ENTRIES).sort_values(["row", "pdb"]))
    print(f"after subsampling to <= {MAX_ENTRIES}: {len(D)} records, "
          f"median {D.groupby('row').pdb.nunique().median():.0f} per protein")

    DEPTH = ["n_entries", "n_partner_chains", "n_partner_chains_max", "n_res_model", "sasa_free_per_res",
             "frac_seq", "n_chains", "n_copies", "has_nucleic", "ident"]
    iface = [c for c in D.columns if c not in ("pdb", "chain", "accession", "row") and c not in DEPTH]
    g = D.groupby("row")
    G = g[iface].median()
    G = G.join(g[iface].max().add_suffix("_max")).join(g[iface].min().add_suffix("_min"))
    for c in ("n_partner_chains", "n_res_model", "sasa_free_per_res", "frac_seq", "n_chains",
              "n_copies", "has_nucleic", "ident"):
        G[c] = g[c].median()
    G["n_entries"] = n_all.reindex(G.index)          # the TRUE count, not the subsampled one
    G["n_partner_chains_max"] = g["n_partner_chains"].max()
    G.to_csv(f"{C}/interface.tsv", sep="\t")
    print(f"\nwrote {C}/interface.tsv: {len(G)} proteins x {G.shape[1]} columns "
          f"({len(iface)} interface features as median/max/min, {len(DEPTH)} depth covariates)")
    print("CM18_DONE")


if __name__ == "__main__":
    main()
