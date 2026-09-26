"""Gate 22: benchmark sets for the GENERATIVE claims (EigenFold / SimpleFold protocol).

  apo/holo (Saldano 2022; EigenFold splits/apo.csv, 90 pairs) and fold-switch (CoDNaS, codnas.csv, 77
  pairs): one sequence (csv `seqres`), two experimental states. cameo2022.csv (183 single-chain targets)
  is built as an extra external folding benchmark (SimpleFold's CAMEO22).

For every target: download both PDB entries from RCSB, take the named chains, keep residues with
N/CA/C/O, align the observed residues to `seqres` (global alignment) so per-residue quantities are
comparable across states, PCA-canonicalise state A and encode it (so the standard scorer works),
and store:
  val/<name>: z, ca_coords (state A, canonical), backbone; attrs sequence (= seqres), partner,
              seqidx_A, ca_A (observed-residue Ca of A, canonical frame), seqidx_B, ca_B (state B, own frame)
Writes data/phase1_dataset/dataset_{apo,codnas,cameo22}.h5; ESMC embeddings via gate11 afterwards.
  BUILD_MAX_LEN=512 python gate22_build_ensembles.py
"""
import os, sys, csv, json, argparse, numpy as np, torch, h5py
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/code"); sys.path.insert(0, ROOT + "/ProteinAE_v1")
os.environ.setdefault("BUILD_MAX_LEN", "512")
from gate15_build_pdb import _fetch, parse_chain
from canonicalize import canonicalize_pyg_data
from torch_geometric.data import Data
E = f"{ROOT}/data/ensembles"; D = f"{ROOT}/data/phase1_dataset"
ap = argparse.ArgumentParser(); ap.add_argument("--sets", default="apo,codnas,cameo22"); ap.add_argument("--max-len", type=int, default=512); ap.add_argument("--device", default="cuda")
a = ap.parse_args()

def align(obs, ref):
    """Global alignment (match 2, mismatch -1, gap -1); returns ref index for each obs residue (-1 if unaligned)."""
    n, m = len(obs), len(ref); S = np.zeros((n + 1, m + 1), dtype=np.int32); S[:, 0] = -np.arange(n + 1); S[0, :] = -np.arange(m + 1)
    for i in range(1, n + 1):
        oi = obs[i - 1]; row = S[i - 1]
        for j in range(1, m + 1):
            S[i, j] = max(row[j - 1] + (2 if oi == ref[j - 1] else -1), row[j] - 1, S[i, j - 1] - 1)
    idx = np.full(n, -1, dtype=np.int64); i, j = n, m
    while i > 0 and j > 0:
        if S[i, j] == S[i - 1, j - 1] + (2 if obs[i - 1] == ref[j - 1] else -1):
            if obs[i - 1] == ref[j - 1]: idx[i - 1] = j - 1
            i -= 1; j -= 1
        elif S[i, j] == S[i - 1, j] - 1: i -= 1
        else: j -= 1
    return idx

def get_state(tag):
    pid, ch = tag.replace(".pdb", "").split(".")[:2]
    text = _fetch(pid)
    if text is None: return None
    seq, c = parse_chain(text, ch)
    return seq, c

os.chdir(ROOT + "/ProteinAE_v1")
import hydra
from einops import rearrange
from proteinfoundation.proteinflow.proteinae import ProteinAE
from proteinfoundation.autoencode import ProteinProcessor
from proteinfoundation.utils.dense_padding_data_loader import DensePaddingDataLoader
from proteinfoundation.flow_matching.r3n_fm import R3NFlowMatcher
from proteinfoundation.utils.coors_utils import ang_to_nm
dev = torch.device(a.device)
with hydra.initialize_config_dir(config_dir=f"{os.getcwd()}/configs/experiment_config", version_base=hydra.__version__):
    hydra.compose(config_name="inference_proteinae", return_hydra_config=True)
ae = ProteinAE.load_from_checkpoint("checkpoints/ae_r1_d8_v1.ckpt", strict=True, weights_only=False, map_location=dev).eval().to(dev)
fm = R3NFlowMatcher(zero_com=True, scale_ref=1.0); BB = ProteinProcessor.BACKBONE_ATOM_INDICES
@torch.no_grad()
def encode(coords):
    batch = next(iter(DensePaddingDataLoader([Data(coords=torch.from_numpy(coords), id="x")], batch_size=1)))
    x_1 = rearrange(batch["coords"][:, :, BB, :], "b n c d -> b (n c) d")
    cm = batch["mask_dict"]["coords"][..., BB, 0]; mask = cm[..., 1]; cmf = rearrange(cm, "b n c -> b (n c)")
    x_1 = fm._mask_and_zero_com(ang_to_nm(x_1), cmf)
    batch.update({"x_1": x_1, "mask": mask, "coords_mask": cmf, "nsamples": 1, "nres": int(x_1.shape[-2] // 4)})
    for k, v in batch.items():
        if isinstance(v, torch.Tensor): batch[k] = v.to(dev)
    return ae.encoder(batch)["single_repr"].float().cpu().numpy()[0]

summary = {}
for set_name in a.sets.split(","):
    csvf = {"apo": "apo.csv", "codnas": "codnas.csv", "cameo22": "cameo2022.csv"}[set_name]; partner_col = {"apo": "holo", "codnas": "other", "cameo22": None}[set_name]
    rows = list(csv.DictReader(open(f"{E}/{csvf}"))); out = h5py.File(f"{D}/dataset_{set_name}.h5", "w"); g = out.create_group("val"); kept = 0; skipped = {}
    for r in rows:
        name = r["name"].replace(".pdb", ""); seqres = r["seqres"].strip()
        if not (32 <= len(seqres) <= a.max_len) or set(seqres) - set("ACDEFGHIKLMNPQRSTVWY"): skipped[name] = "length/nonstandard"; continue
        A = get_state(r["name"])
        if A is None or len(A[0]) < 32: skipped[name] = "state A"; continue
        seqA, cA = A; idxA = align(seqA, seqres)
        if (idxA >= 0).sum() < 0.5 * len(seqres): skipped[name] = "alignment A"; continue
        try:
            item, _ = canonicalize_pyg_data(Data(coords=torch.from_numpy(cA))); ccA = item.coords.numpy().astype(np.float32)
        except Exception: skipped[name] = "canonicalise"; continue
        # the model works on the full seqres; store state A's Ca placed on seqres positions for the standard scorer (unobserved -> masked later via seqidx)
        grp = g.create_group(name)
        ca_full = np.zeros((len(seqres), 3), dtype=np.float32); ok = idxA >= 0; ca_full[idxA[ok]] = ccA[ok, 1]
        z_full = np.zeros((len(seqres), 8), dtype=np.float32); zA = encode(ccA); z_full[idxA[ok]] = zA[ok]
        grp.create_dataset("z", data=z_full); grp.create_dataset("ca_coords", data=ca_full)
        grp.create_dataset("seqidx_A", data=idxA); grp.create_dataset("ca_A", data=ccA[:, 1, :])
        grp.attrs["sequence"] = seqres; grp.attrs["n_residues"] = len(seqres); grp.attrs["n_obs_A"] = int(ok.sum()); grp.attrs["state_A"] = r["name"]
        if partner_col:
            B = get_state(r[partner_col])
            if B is None or len(B[0]) < 32: del g[name]; skipped[name] = "state B"; continue
            seqB, cB = B; idxB = align(seqB, seqres)
            if (idxB >= 0).sum() < 0.5 * len(seqres): del g[name]; skipped[name] = "alignment B"; continue
            grp.create_dataset("seqidx_B", data=idxB); grp.create_dataset("ca_B", data=cB[:, 1, :]); grp.attrs["state_B"] = r[partner_col]; grp.attrs["n_obs_B"] = int((idxB >= 0).sum())
        kept += 1
    out.attrs["source"] = f"EigenFold splits/{csvf}; RCSB PDB experimental coordinates; seqres as model input"; out.close()
    open(f"{ROOT}/notes/{set_name}_names.txt", "w").write("\n".join(sorted(g_ for g_ in h5py.File(f"{D}/dataset_{set_name}.h5", "r")["val"].keys())) + "\n")
    summary[set_name] = {"kept": kept, "skipped": skipped}; print(f"{set_name}: kept {kept}/{len(rows)}; skipped {json.dumps(skipped)[:300]}", flush=True)
json.dump(summary, open(f"{ROOT}/notes/gate22_build.json", "w"), indent=1)
