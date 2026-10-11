"""cm22: score the Design A validation gate against experiment.

The pilot is only worth running if the co-folding model can place a peptide it has never seen into
the HSPB1 ACD groove. 9ef3 (ACD 84-171 + its own C-terminal IxI peptide EITIPVTFE) was released
2026-06-24, after Boltz-2 was trained, so it is a genuine held-out test. 6gjh (2019) is in training
and only tests that the pipeline runs.

Three things are measured, and all three have to pass before the full pilot is launched:
  1. fold      -- predicted ACD dimer vs the 9ef3 dimer, Ca RMSD. If the domain itself is wrong the
                  groove is wrong and nothing downstream means anything.
  2. placement -- the peptide's Ca RMSD to the experimental peptide after superposing on the ACD ONLY.
                  Superposing on the ACD and then measuring the peptide is the honest way round; it
                  does not let a good peptide fit paper over a bad domain.
  3. contrast  -- the held-out positive must score above the homopolymer negatives on interface
                  confidence. A model that is confident about everything is useless as a ranker.

The experimental groove is not asserted from the literature, it is measured here: the residues of
9ef3 chain A within 4.5 A of chain D.
"""
import os, json, glob
import numpy as np, gemmi
from scipy.spatial import cKDTree

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
C = f"{ROOT}/data/campaign"
OUT = f"{C}/boltz/val/out"
EXP = f"{ROOT}/data/external/pdb_assemblies/cif/9ef3.cif.gz"
CUT = 4.5


def one(n):
    r = gemmi.find_tabulated_residue(n)
    return r.one_letter_code.upper() if r and r.is_amino_acid() else None


def chains(path):
    st = gemmi.read_structure(path)
    st.setup_entities()
    out = {}
    for ch in st[0]:
        rs = [r for r in ch if one(r.name)]
        if rs:
            out[ch.name] = rs
    return out


def ca(rs):
    v = []
    for r in rs:
        a = r.find_atom("CA", "*")
        v.append([a.pos.x, a.pos.y, a.pos.z] if a else [np.nan] * 3)
    return np.array(v)


def allxyz(rs):
    return np.array([[a.pos.x, a.pos.y, a.pos.z] for r in rs for a in r])


def contacts(prs, lrs, cut=CUT):
    """local residue indices (1-based along prs) of prs within cut of any atom of lrs"""
    if not len(lrs):
        return []
    t = cKDTree(allxyz(lrs))
    hit = []
    for i, r in enumerate(prs, 1):
        x = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in r])
        if t.query(x, k=1, distance_upper_bound=cut)[0].min() < cut:
            hit.append(i)
    return hit


def kabsch(P, Q):
    """rotation+translation taking P onto Q; returns rmsd and the transform applied to P"""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    Pa = (P - pc) @ R.T + qc
    return float(np.sqrt(((Pa - Q) ** 2).sum(1).mean())), (R, pc, qc)


def apply(X, tr):
    R, pc, qc = tr
    return (X - pc) @ R.T + qc


# ---- experimental reference: 9ef3 ACD (chain A, local 1-88) + peptide (chain D) ---------------
E = chains(EXP)
eA, eD = E["A"], E["D"]
eB = E.get("B-2", [])
GROOVE = set(contacts(eA, eD))
print(f"experimental groove (9ef3 chain A residues within {CUT} A of the IxI peptide): "
      f"{len(GROOVE)} residues, local {sorted(GROOVE)}")
print("  as UniProt numbering: " + " ".join(f"{one(eA[i-1].name)}{i+83}" for i in sorted(GROOVE)))

# experimental ACD dimer, in the numbering our predictions use (local 1-88 per chain)
eAnum = {r.seqid.num: r for r in eA}
eBnum = {r.seqid.num: r for r in eB}
shared = sorted(set(eAnum) & set(eBnum))
print(f"experimental dimer: chain A {len(eAnum)} res, chain B-2 {len(eBnum)} res, "
      f"{len(shared)} numbered in both")

rows = []
for d in sorted(glob.glob(f"{OUT}/boltz_results_in/predictions/*")):
    name = os.path.basename(d)
    cif = f"{d}/{name}_model_0.cif"
    cjs = f"{d}/confidence_{name}_model_0.json"
    if not os.path.exists(cif):
        print(f"  MISSING {name}")
        continue
    P = chains(cif)
    cf = json.load(open(cjs)) if os.path.exists(cjs) else {}
    pep = P.get("C", [])
    acd = {k: v for k, v in P.items() if k in ("A", "B")}
    r = dict(name=name, n_pep=len(pep),
             iptm=cf.get("iptm"), ptm=cf.get("ptm"),
             plddt=cf.get("complex_plddt"), iplddt=cf.get("complex_iplddt"),
             conf=cf.get("confidence_score"), ipde=cf.get("complex_ipde"))
    pci = cf.get("pair_chains_iptm", {})
    try:
        r["iptm_AC"] = max(float(pci["0"]["2"]), float(pci["2"]["0"]))
        r["iptm_BC"] = max(float(pci["1"]["2"]), float(pci["2"]["1"]))
        r["iptm_pep"] = max(r["iptm_AC"], r["iptm_BC"])
    except Exception:
        r["iptm_AC"] = r["iptm_BC"] = r["iptm_pep"] = np.nan

    if pep:
        best = max(acd, key=lambda k: len(contacts(acd[k], pep)))
        hit = contacts(acd[best], pep)
        r["bound_chain"], r["n_contact"] = best, len(hit)
        r["groove_frac"] = len(set(hit) & GROOVE) / max(len(hit), 1)
        r["groove_recall"] = len(set(hit) & GROOVE) / len(GROOVE)
        r["contacts"] = " ".join(map(str, hit))
    # ---- fold accuracy against the experimental dimer -------------------------------------
    # Two symmetry points, both of which bit once already in this campaign (cm18's "A-2" naming):
    #  (a) the ACD dimer has a 2-fold axis, so its two grooves are equivalent BY CONSTRUCTION. The
    #      peptide binding the other one is the same answer, not a 43 A error -- that is just the
    #      distance between the two grooves. So the chain assignment is chosen to put the predicted
    #      bound chain onto experimental chain A, the one the experimental peptide occupies.
    #  (b) the fit is restricted to the ordered core (the residues resolved in BOTH experimental
    #      chains, 11-86). Residues 1-10 of this construct are a flexible tail with a
    #      crystal-specific conformation; including them moves the apparent monomer RMSD from
    #      1.8 A to 9.5 A and tilts the frame the peptide is then measured in.
    pAnum = {rr.seqid.num: rr for rr in acd.get("A", [])}
    pBnum = {rr.seqid.num: rr for rr in acd.get("B", [])}
    if pAnum and pBnum and shared:
        assign = [(pAnum, pBnum), (pBnum, pAnum)]
        if pep and r.get("bound_chain"):                 # (a): bound chain -> experimental chain A
            assign = [(pAnum, pBnum)] if r["bound_chain"] == "A" else [(pBnum, pAnum)]
        best_rms, best_tr, best_as = np.inf, None, None
        for a, b in assign:
            idx = [i for i in shared if i in a and i in b]
            Pm = np.vstack([ca([a[i] for i in idx]), ca([b[i] for i in idx])])
            Qm = np.vstack([ca([eAnum[i] for i in idx]), ca([eBnum[i] for i in idx])])
            k = np.isfinite(Pm).all(1) & np.isfinite(Qm).all(1)
            rms, tr = kabsch(Pm[k], Qm[k])
            if rms < best_rms:
                best_rms, best_tr, best_as = rms, tr, (a, b)
        r["dimer_rmsd"] = best_rms
        # monomer core only, to separate a wrong fold from a wrong dimer arrangement
        idx = [i for i in shared if i in best_as[0]]
        Pm, Qm = ca([best_as[0][i] for i in idx]), ca([eAnum[i] for i in idx])
        k = np.isfinite(Pm).all(1) & np.isfinite(Qm).all(1)
        r["monomer_rmsd"] = kabsch(Pm[k], Qm[k])[0]
        # full construct including the flexible tail, reported so the difference stays visible
        idxf = [i for i in sorted(eAnum) if i in best_as[0]]
        Pf, Qf = ca([best_as[0][i] for i in idxf]), ca([eAnum[i] for i in idxf])
        kf = np.isfinite(Pf).all(1) & np.isfinite(Qf).all(1)
        r["monomer_rmsd_full"] = kabsch(Pf[kf], Qf[kf])[0]
        # peptide placement. Superpose on the BOUND MONOMER's core and then measure the peptide --
        # the docking convention (align the receptor, score the ligand). The groove is on one
        # monomer, so using the dimer frame would charge the peptide for any error in the dimer
        # arrangement, which is a separate question.
        # Both threading directions are reported. A groove is an open channel and an extended
        # peptide can lie in it either way round; getting the site right and the polarity wrong is a
        # different (and, for ranking which SEGMENTS engage the groove, much milder) failure than
        # missing the site, so the two are not merged into one number.
        if pep and len(pep) == len(eD):
            idxb = [i for i in shared if i in best_as[0]]
            _, trb = kabsch(ca([best_as[0][i] for i in idxb]), ca([eAnum[i] for i in idxb]))
            Pp, Q = apply(ca(pep), trb), ca(eD)
            fwd = float(np.sqrt(((Pp - Q) ** 2).sum(1).mean()))
            rev = float(np.sqrt(((Pp - Q[::-1]) ** 2).sum(1).mean()))
            r["pep_rmsd_fwd"], r["pep_rmsd_rev"] = fwd, rev
            r["pep_rmsd_on_acd"] = min(fwd, rev)
            r["threading"] = "forward" if fwd <= rev else "REVERSED"
            r["pep_centroid_off"] = float(np.linalg.norm(Pp.mean(0) - Q.mean(0)))
    rows.append(r)

import pandas as pd
D = pd.DataFrame(rows)
cols = ["name", "n_pep", "iptm_pep", "iptm", "ptm", "plddt", "iplddt", "conf",
        "n_contact", "groove_frac", "groove_recall", "bound_chain",
        "monomer_rmsd", "dimer_rmsd", "pep_rmsd_fwd", "pep_rmsd_rev", "pep_centroid_off", "threading"]
cols = [c for c in cols if c in D.columns]
print("\n" + D[cols].round(3).to_string(index=False))
D.to_csv(f"{C}/cm22_val.tsv", sep="\t", index=False)

print("\n---- gate ----")
g = D.set_index("name")


def val(n, c):
    return float(g.loc[n, c]) if n in g.index and c in g.columns and np.isfinite(g.loc[n, c]) else np.nan


pos = val("VAL_pos_9ef3_ixi9", "iptm_pep")
negs = [val(n, "iptm_pep") for n in g.index if n.startswith("VAL_neg")]
negs = [x for x in negs if np.isfinite(x)]
print(f"1. fold       : apo monomer core RMSD {val('VAL_apo','monomer_rmsd'):.2f} A "
      f"(full construct incl. flexible tail {val('VAL_apo','monomer_rmsd_full'):.2f} A), "
      f"dimer core RMSD {val('VAL_apo','dimer_rmsd'):.2f} A   (pass < 2.5 / < 4)")
thr = g.loc['VAL_pos_9ef3_ixi9', 'threading'] if 'threading' in g.columns else '?'
print(f"2. placement  : held-out IxI peptide {val('VAL_pos_9ef3_ixi9','pep_rmsd_on_acd'):.2f} A on the bound-monomer frame "
      f"({thr}; forward {val('VAL_pos_9ef3_ixi9','pep_rmsd_fwd'):.1f} A, reverse {val('VAL_pos_9ef3_ixi9','pep_rmsd_rev'):.1f} A), "
      f"centroid off by {val('VAL_pos_9ef3_ixi9','pep_centroid_off'):.2f} A, "
      f"groove overlap {val('VAL_pos_9ef3_ixi9','groove_frac'):.2f}   (pass < 4 A and > 0.5)")
print(f"3. contrast   : held-out positive iptm {pos:.3f} vs homopolymer negatives "
      f"{np.mean(negs):.3f} (max {np.max(negs):.3f})   (pass: positive above every negative)")
json.dump({k: (None if not np.isfinite(v) else v) for k, v in
           dict(pos_iptm=pos, neg_iptm_max=float(np.max(negs)) if negs else np.nan,
                apo_monomer_rmsd=val("VAL_apo", "monomer_rmsd"),
                apo_dimer_rmsd=val("VAL_apo", "dimer_rmsd"),
                pep_rmsd=val("VAL_pos_9ef3_ixi9", "pep_rmsd_on_acd"),
                groove_frac=val("VAL_pos_9ef3_ixi9", "groove_frac")).items()},
          open(f"{C}/cm22_val.json", "w"), indent=1)
print("CM22_DONE")
