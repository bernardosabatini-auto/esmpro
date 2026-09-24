"""Feasibility for 500-residue proteins: does the frozen ProteinAE encoder/decoder round-trip
hold beyond the 256-residue window it has been used at? Takes experimental X-ray/EM chains of
--min-len..--max-len residues from pdb_seqres (same filters as gate15), downloads, parses,
canonicalises, encodes with the frozen encoder, decodes with the 3-step decoder and reports TM /
RMSD per length band (Foldseek TM-align, exhaustive). Also times the decoder per length.
  python gate17_long_roundtrip.py --n 240 --min-len 257 --max-len 640
"""
import os, sys, gzip, time, argparse, tempfile, shutil, collections
import numpy as np, torch, h5py
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/code"); sys.path.insert(0, ROOT + "/ProteinAE_v1")
from gate15_build_pdb import _fetch, parse_chain
import gate15_build_pdb as G15
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=240); ap.add_argument("--min-len", type=int, default=257); ap.add_argument("--max-len", type=int, default=640)
ap.add_argument("--max-res", type=float, default=2.5); ap.add_argument("--workers", type=int, default=12)
a = ap.parse_args(); D = f"{ROOT}/data/pdb"
method, res = {}, {}
for l in open(f"{D}/pdb_entry_type.txt"):
    f = l.split()
    if len(f) == 3: method[f[0].lower()] = (f[1], f[2])
for l in open(f"{D}/resolu.idx"):
    f = [x.strip() for x in l.split(";")]
    if len(f) == 2 and len(f[0]) == 4:
        try: res[f[0].lower()] = float(f[1])
        except ValueError: pass
AA = set("ACDEFGHIKLMNPQRSTVWY"); cands = []
rng = np.random.default_rng(0)
with gzip.open(f"{D}/pdb_seqres.txt.gz", "rt") as fh:
    hdr = None
    for l in fh:
        if l.startswith(">"): hdr = l; continue
        seq = l.strip(); f = hdr[1:].split(); pid, ch = f[0].split("_", 1); pid = pid.lower()
        if f[1] != "mol:protein" or not (a.min_len <= len(seq) <= a.max_len) or set(seq) - AA: continue
        m = method.get(pid); r = res.get(pid, -1.0)
        if m is None or m[0] not in ("prot", "prot-nuc") or m[1] != "diffraction" or not (0 < r <= a.max_res): continue
        cands.append((pid, ch, len(seq), r, m[1]))
print(f"{len(cands)} candidate chains of {a.min_len}-{a.max_len} residues", flush=True)
# stratify by length band, one chain per entry
bands = [(257, 320), (321, 384), (385, 448), (449, 512), (513, 640)]
pick, seen = [], set()
per = max(1, a.n // len(bands))
for lo, hi in bands:
    pool = [c for c in cands if lo <= c[2] <= hi and c[0] not in seen]; rng.shuffle(pool)
    for c in pool[:per * 3]:
        if c[0] in seen: continue
        pick.append(c); seen.add(c[0])
        if sum(1 for p in pick if lo <= p[2] <= hi) >= per: break
G15.MAX_LEN = a.max_len
from multiprocessing import Pool
items = []
with Pool(a.workers) as pool:
    for r in pool.imap_unordered(G15._one, pick, chunksize=2):
        if isinstance(r, dict): items.append(r)
print(f"{len(items)} chains parsed/canonicalised", flush=True)
os.chdir(ROOT + "/ProteinAE_v1"); sys.path.insert(0, ".")
import hydra
from einops import rearrange
from proteinfoundation.proteinflow.proteinae import ProteinAE
from proteinfoundation.autoencode import ProteinProcessor
from proteinfoundation.utils.dense_padding_data_loader import DensePaddingDataLoader
from proteinfoundation.flow_matching.r3n_fm import R3NFlowMatcher
from proteinfoundation.utils.coors_utils import ang_to_nm
from torch_geometric.data import Data
import gate7_latent_flow as G
from gate6_fape_train import _write_pseudo_backbone_pdb, _foldseek_tm
from gate6_corrected_eval import kabsch_rmsd
dev = torch.device("cuda")
with hydra.initialize_config_dir(config_dir=f"{os.getcwd()}/configs/experiment_config", version_base=hydra.__version__):
    hydra.compose(config_name="inference_proteinae", return_hydra_config=True)
ae = ProteinAE.load_from_checkpoint("checkpoints/ae_r1_d8_v1.ckpt", strict=True, weights_only=False, map_location=dev).eval().to(dev)
fm = R3NFlowMatcher(zero_com=True, scale_ref=1.0); BB = ProteinProcessor.BACKBONE_ATOM_INDICES
dec = G.load_decoder(dev)
@torch.no_grad()
def encode(batch_items):
    datas = [Data(coords=torch.from_numpy(it["coords"]), id=it["id"]) for it in batch_items]
    batch = next(iter(DensePaddingDataLoader(datas, batch_size=len(datas))))
    x_1 = rearrange(batch["coords"][:, :, BB, :], "b n c d -> b (n c) d")
    coords_mask = batch["mask_dict"]["coords"][..., BB, 0]; mask = coords_mask[..., 1]
    cmf = rearrange(coords_mask, "b n c -> b (n c)")
    x_1 = fm._mask_and_zero_com(ang_to_nm(x_1), cmf)
    batch.update({"x_1": x_1, "mask": mask, "coords_mask": cmf, "nsamples": 1, "nres": int(x_1.shape[-2] // 4)})
    for k, v in batch.items():
        if isinstance(v, torch.Tensor): batch[k] = v.to(dev)
    return ae.encoder(batch)["single_repr"].float(), batch["mask"]
work = tempfile.mkdtemp(prefix="longrt_"); os.makedirs(f"{work}/gt"); os.makedirs(f"{work}/pr")
items.sort(key=lambda it: it["coords"].shape[0]); rms, lens, tdec, z_norm = {}, {}, collections.defaultdict(list), []
with torch.no_grad():
    for s in range(0, len(items), 8):
        b = items[s:s+8]; z, m = encode(b)
        z_norm += (z.norm(dim=-1)[m]).tolist()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16): pred = dec(z, m).float().cpu()
        torch.cuda.synchronize(); tdec[b[-1]["coords"].shape[0] // 64 * 64].append((time.perf_counter() - t0) / len(b))
        ca = torch.zeros_like(pred)
        for j, it in enumerate(b): ca[j, :it["coords"].shape[0]] = torch.from_numpy(it["coords"][:, 1, :])
        r = kabsch_rmsd(pred, ca, m.cpu())
        for j, it in enumerate(b):
            L = it["coords"].shape[0]; rms[it["id"]] = r[j]; lens[it["id"]] = L
            _write_pseudo_backbone_pdb(ca[j, :L].numpy(), f"{work}/gt/{it['id']}.pdb"); _write_pseudo_backbone_pdb(pred[j, :L].numpy(), f"{work}/pr/{it['id']}.pdb")
tm = _foldseek_tm(f"{work}/pr", f"{work}/gt", f"{work}/tm"); shutil.rmtree(work, ignore_errors=True)
print(f"latent per-residue L2: mean {np.mean(z_norm):.3f} (256-window training data: 2.828)")
print(f"{'band':>9s} {'n':>4s} {'TM':>6s} {'TM>0.5':>7s} {'TM>0.9':>7s} {'RMSD':>6s} {'cov':>5s}")
for lo, hi in bands:
    ids = [i for i in rms if lo <= lens[i] <= hi]
    if not ids: continue
    v = np.array([tm.get(i, 0.0) for i in ids]); rr = np.array([rms[i] for i in ids])
    print(f"{lo:4d}-{hi:<4d} {len(ids):4d} {v.mean():6.3f} {(v > 0.5).mean():7.2f} {(v > 0.9).mean():7.2f} {np.nanmean(rr):6.2f} {sum(i in tm for i in ids)/len(ids):5.2f}")
print("decoder time per protein by length (3 ODE steps, bf16): " + ", ".join(f"L~{k}: {np.mean(v)*1000:.0f} ms" for k, v in sorted(tdec.items())))
