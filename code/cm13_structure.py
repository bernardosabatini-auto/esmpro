"""cm13: structural features from AlphaFold models, aimed at what mean-pooled sequence embeddings cannot express.

Motivation (cm11/cm12 and the charge probe): net charge is the strongest simple predictor of the salt response,
but the *capacity* to form ion pairs is not, and the sequence MLP has already absorbed all of the compositional
charge signal (residual correlations |rho| <= 0.07). Whatever is left that is a protein property must be spatial:
where the charge sits, whether an ion pair is buried or solvent-facing, how large a contiguous charged patch is.
None of that survives averaging over residues.

Reads the AlphaFold human proteome tar by streaming it once (the parallel filesystem makes 20k small files
expensive, so nothing is extracted to disk), parses the F1 PDB model in memory, and writes one row per accession
to data/campaign/struct.tsv.

Features, per protein, all normalised by length where they are counts:
  burial        frac_buried, sasa_per_res, rg_ratio (Rg against the 2.2 L^0.38 globule), contact_density,
                contact_order
  ion pairs     saltbridge_per_res and its buried / surface split (carboxylate O to K-NZ, R-NE/NH1/NH2,
                H-ND1/NE2 within 4.0 A)
  electrostatics surface_charge (charge weighted by relative SASA), buried_charge, charge_patch_pos /
                charge_patch_neg (the most charged 10 A surface neighbourhood -- the quantity a net-charge
                feature cannot see), dipole, and dh_d75 / dh_d150 / dh_dmg: the change in intramolecular
                Debye-Huckel energy between the two ionic strengths actually used in each experiment
                (kappa = 0.329 sqrt(I) /A; MgCl2 contributes I = 3c)
  divalent      carboxyl_cluster: carboxylate-oxygen pairs from different D/E residues within 6 A, the
                geometric signature of a Mg2+ site
  hydrophobics  exposed_hydrophobic_frac, hydrophobic_patch (largest exposed hydrophobic 10 A neighbourhood),
                buried_apr (longest buried hydrophobic run, which is what partial unfolding exposes)
  order         plddt_mean, plddt_frac_low / mid, longest_low_frac, n_domains
  shape         helix_frac, sheet_frac

Run: slurm/cm13.sbatch (kempner_rtx, 1 GPU held only because the CPU account cannot submit; the work is CPU).
"""
import os, sys, gzip, tarfile, json, time
import numpy as np, pandas as pd
import freesasa
from scipy.spatial import cKDTree

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae"); C = f"{ROOT}/data/campaign"
TAR = f"{ROOT}/data/external/afdb_human/UP000005640_9606_HUMAN_v6.tar"
freesasa.setVerbosity(freesasa.silent)

AA3 = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L",
           LYS="K", MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V")
MAXASA = dict(A=129, R=274, N=195, D=193, C=167, Q=225, E=223, G=104, H=224, I=197, L=201, K=236, M=224,
              F=240, P=159, S=155, T=172, W=285, Y=263, V=174)                      # Tien et al. 2013, theoretical
RAD = dict(C=1.70, N=1.55, O=1.52, S=1.80)
CHG = dict(D=-1.0, E=-1.0, K=1.0, R=1.0, H=0.1)                                      # at pH 7.4
HYD = set("AVLIMFWYC")
ACID_O, BASE_N = {"OD1", "OD2", "OE1", "OE2"}, {"NZ", "NE", "NH1", "NH2", "ND1", "NE2"}
KAPPA = {"d75": 0.329 * np.sqrt(0.075), "d150": 0.329 * np.sqrt(0.150),              # added KCl / NaCl, I = c
         "mg_lo": 0.329 * np.sqrt(3 * 0.00025), "mg_hi": 0.329 * np.sqrt(3 * 0.0025)}  # MgCl2, I = 3c


def parse_pdb(txt):
    """ATOM records of the single chain in an AlphaFold model -> arrays. B-factor is pLDDT."""
    an, rn, ri, xyz, bf = [], [], [], [], []
    for l in txt.splitlines():
        if not l.startswith("ATOM"): continue
        if l[76:78].strip() == "H" or l[12:16].strip().startswith("H"): continue     # AFDB has none, be safe
        an.append(l[12:16].strip()); rn.append(l[17:20].strip()); ri.append(int(l[22:26]))
        xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54]))); bf.append(float(l[60:66]))
    return np.array(an), np.array(rn), np.array(ri), np.array(xyz, dtype=np.float64), np.array(bf)


def features(an, rn, ri, xyz, bf):
    u, first = np.unique(ri, return_index=True); order = np.argsort(u); u = u[order]; first = first[order]
    seq = np.array([AA3.get(x, "X") for x in rn[first]]); L = len(u)
    if L < 20 or (seq == "X").mean() > .1: return None
    rmap = {r: i for i, r in enumerate(u)}; res_of = np.array([rmap[r] for r in ri])
    elem = np.array([a[0] if a[0] in RAD else "C" for a in an])
    radii = np.array([RAD[e] for e in elem])

    res_ = freesasa.calcCoord(xyz.ravel().tolist(), radii.tolist())
    sasa_atom = np.fromiter((res_.atomArea(i) for i in range(len(radii))), float, len(radii))
    sasa_res = np.bincount(res_of, weights=sasa_atom, minlength=L)
    rsa = sasa_res / np.array([MAXASA.get(s, 200) for s in seq])
    exposed = rsa > .25

    # representative point per residue: CB where present, else CA
    rep = np.zeros((L, 3)); have = np.zeros(L, bool)
    for nm in ("CA", "CB"):
        m = an == nm; rep[res_of[m]] = xyz[m]; have[res_of[m]] = True
    if not have.all(): return None
    plddt = np.bincount(res_of, weights=bf, minlength=L) / np.bincount(res_of, minlength=L)

    q = np.array([CHG.get(s, 0.0) for s in seq])
    T = cKDTree(rep)
    f = {}
    # --- burial / shape
    f["frac_buried"] = float((rsa < .25).mean()); f["sasa_per_res"] = float(sasa_res.mean())
    cen = rep.mean(0); rg = float(np.sqrt(((rep - cen) ** 2).sum(1).mean()))
    f["rg_ratio"] = rg / (2.2 * L ** .38)
    pr = np.array(sorted(T.query_pairs(8.0)))
    if len(pr) < 10: return None
    sep = np.abs(pr[:, 0] - pr[:, 1]); far = sep > 2
    f["contact_density"] = float(far.sum()) / L
    f["contact_order"] = float(sep[far].mean()) / L
    # --- ion pairs
    ai = np.isin(an, list(ACID_O)) & np.isin(rn, ["ASP", "GLU"])
    bi = np.isin(an, list(BASE_N)) & np.isin(rn, ["LYS", "ARG", "HIS"])
    sb = set()
    if ai.any() and bi.any():
        TA, TB = cKDTree(xyz[ai]), cKDTree(xyz[bi]); hits = TA.query_ball_tree(TB, 4.0)
        ra, rb = res_of[ai], res_of[bi]
        for k, lst in enumerate(hits):
            for m_ in lst: sb.add((ra[k], rb[m_]))
    sb = np.array(sorted(sb)).reshape(-1, 2)
    f["saltbridge_per_res"] = len(sb) / L
    if len(sb):
        bur = (rsa[sb[:, 0]] < .25) & (rsa[sb[:, 1]] < .25)
        f["saltbridge_buried_per_res"] = float(bur.sum()) / L; f["saltbridge_surface_per_res"] = float((~bur).sum()) / L
    else: f["saltbridge_buried_per_res"] = f["saltbridge_surface_per_res"] = 0.0
    # --- electrostatics: where the charge is, not how much there is
    f["surface_charge"] = float((q * rsa).sum()) / L
    f["buried_charge"] = float((q * (1 - rsa)).sum()) / L
    qs = q * rsa
    nb = T.query_ball_point(rep, 10.0)
    patch = np.array([qs[np.array(ix)].sum() for ix in nb])
    f["charge_patch_pos"] = float(patch[exposed].max()) if exposed.any() else 0.0
    f["charge_patch_neg"] = float(patch[exposed].min()) if exposed.any() else 0.0
    f["charge_patch_span"] = f["charge_patch_pos"] - f["charge_patch_neg"]
    f["dipole"] = float(np.linalg.norm((q[:, None] * (rep - cen)).sum(0))) / (L * rg)
    ch = np.nonzero(q != 0)[0]
    if len(ch) > 2:
        d = np.linalg.norm(rep[ch][:, None] - rep[ch][None], axis=-1); iu = np.triu_indices(len(ch), 1)
        qq = (q[ch][:, None] * q[ch][None])[iu]; dd = np.maximum(d[iu], 2.0)
        E = lambda k: float((qq * np.exp(-k * dd) / dd).sum()) / L
        f["dh_d75"] = E(KAPPA["d75"]) - E(0.0); f["dh_d150"] = E(KAPPA["d150"]) - E(0.0)
        f["dh_dmg"] = E(KAPPA["mg_hi"]) - E(KAPPA["mg_lo"]); f["dh_unscreened"] = E(0.0)
    else: f["dh_d75"] = f["dh_d150"] = f["dh_dmg"] = f["dh_unscreened"] = 0.0
    # --- divalent site proxy: carboxylate oxygens of different residues within 6 A
    if ai.sum() > 1:
        TAo = cKDTree(xyz[ai]); ra = res_of[ai]
        cc = sum(1 for i_, j_ in TAo.query_pairs(6.0) if ra[i_] != ra[j_])
        f["carboxyl_cluster"] = cc / L
    else: f["carboxyl_cluster"] = 0.0
    # --- hydrophobics
    hyd = np.array([s in HYD for s in seq], float); amax = np.array([MAXASA.get(s, 200) for s in seq])
    f["exposed_hydrophobic_frac"] = float((rsa * amax * hyd).sum() / max(sasa_res.sum(), 1.0))
    hs = rsa * amax * hyd
    f["hydrophobic_patch"] = float(max(hs[np.array(ix)].sum() for ix in nb)) if L else 0.0
    bap = (hyd > 0) & (rsa < .25); run = best = 0
    for v in bap:
        run = run + 1 if v else 0; best = max(best, run)
    f["buried_apr"] = best / L
    # --- order
    f["plddt_mean"] = float(plddt.mean()); f["plddt_frac_low"] = float((plddt < 50).mean())
    f["plddt_frac_mid"] = float(((plddt >= 50) & (plddt < 70)).mean())
    lo = plddt < 70; run = best = 0
    for v in lo:
        run = run + 1 if v else 0; best = max(best, run)
    f["longest_low_frac"] = best / L
    dom, run = 0, 0
    for v in ~lo:
        run = run + 1 if v else 0
        if run == 40: dom += 1
    f["n_domains"] = float(dom)
    # --- crude secondary structure from CA geometry
    ca = np.zeros((L, 3)); m = an == "CA"; ca[res_of[m]] = xyz[m]
    d3 = np.linalg.norm(ca[3:] - ca[:-3], axis=1) if L > 3 else np.array([9.])
    d2 = np.linalg.norm(ca[2:] - ca[:-2], axis=1) if L > 2 else np.array([9.])
    f["helix_frac"] = float(((d3 > 4.5) & (d3 < 6.4)).mean()); f["sheet_frac"] = float((d2 > 6.4).mean())
    f["n_res_model"] = float(L)
    return f


def main():
    want = set(pd.read_csv(f"{C}/union.tsv", sep="\t")["accession"].astype(str))
    rest = np.load(f"{C}/proteome_rest_features.npz", allow_pickle=True)["accession"].astype(str)
    want |= set(rest)
    print(f"{len(want)} accessions wanted; streaming {TAR}", flush=True)
    out, t0, seen, bad = {}, time.time(), 0, 0
    tf = tarfile.open(TAR, "r|")
    for m in tf:
        if not m.name.endswith("-model_v6.pdb.gz"): continue
        acc = m.name.split("-")[1]
        if acc not in want or acc in out: continue
        if "-F1-" not in m.name: continue                                            # fragment 1 only; see n_res_model
        seen += 1
        try:
            txt = gzip.decompress(tf.extractfile(m).read()).decode()
            f = features(*parse_pdb(txt))
            if f is not None: out[acc] = f
            else: bad += 1
        except Exception as e:
            bad += 1
            if bad < 5: print(f"  {acc}: {type(e).__name__} {e}", flush=True)
        if seen % 2000 == 0: print(f"  {seen} parsed, {len(out)} kept, {time.time() - t0:.0f} s", flush=True)
    T = pd.DataFrame(out).T; T.index.name = "accession"
    T.to_csv(f"{C}/struct.tsv", sep="\t")
    print(f"{len(T)} proteins with structural features, {bad} failed, {time.time() - t0:.0f} s", flush=True)
    print(T.describe().T[["mean", "std", "min", "max"]].round(3).to_string(), flush=True)
    print("CM13_DONE")


if __name__ == "__main__":
    main()
