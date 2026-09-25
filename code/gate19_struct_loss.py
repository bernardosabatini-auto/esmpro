"""Gate 19: structural auxiliary loss through the frozen, differentiable decoder.

SimpleFold trains with flow matching + an LDDT loss on the one-step estimate and reports the
LDDT term is required for local accuracy. Our flow lives in the latent, but the decoder is
differentiable (gate6 FAPE head used it), so the same idea applies: decode the one-step latent
estimate x1_hat = x_t + (1 - t) v with gradient, decode the TRUE latent without gradient as the
target (round trip 0.999), and add an LDDT-style pairwise-distance loss (AlphaFold3 form,
cutoff 15 A, thresholds 0.5/1/2/4 A) on a random subset of --aux-n samples per step.
CLAUDE.md's warning concerns a latent-MSE term (pose-committed); this term is geometric and
pose-free. Cost: one decoder pass with grad + one without, on aux-n samples.

  python gate19_struct_loss.py --d-pair 64 --n-pair-blocks 6 --aux-w 1.0 --aux-n 16 ... (gate7/gate10 flags)
"""
import os, sys, torch, torch.nn.functional as F
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/code")
import gate7_latent_flow as G7, gate10_pair_flow as G10
from gate7_latent_flow import D_LAT

AUX = {"dec": None, "w": 1.0, "n": 16, "cutoff": 15.0}

def _decoder(device):
    if AUX["dec"] is None:
        cwd = os.getcwd(); os.chdir(ROOT + "/ProteinAE_v1")
        try: AUX["dec"] = G7.load_decoder(device)
        finally: os.chdir(cwd)
    return AUX["dec"]

def lddt_loss(ca_pred, ca_true, mask, cutoff=15.0):
    """1 - LDDT-like score over pairs within `cutoff` in the target. ca_* (B,L,3) A, mask (B,L)."""
    dp = torch.cdist(ca_pred.float(), ca_pred.float()); dt = torch.cdist(ca_true.float(), ca_true.float())
    pm = mask[:, :, None] & mask[:, None, :]
    pm = pm & (dt < cutoff) & ~torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[None]
    dd = (dp - dt).abs()
    score = sum(torch.sigmoid(c - dd) for c in (0.5, 1.0, 2.0, 4.0)) / 4
    return 1.0 - (score * pm).sum() / pm.sum().clamp(min=1)

def fm_loss_struct(net, z1, esm, mask, p_drop=0.1, p_sc=0.5, contact=None):
    raw = getattr(net, "module", net)
    R = G10.REPEAT_COPIES
    if R > 1:
        pair1 = raw.compute_pair(esm, mask, contact); pair = pair1.repeat_interleave(R, dim=0)
        z1, esm, mask = G10._repeat(z1, R), G10._repeat(esm, R), G10._repeat(mask, R)
    B = z1.shape[0]; dev = z1.device
    x0 = torch.randn_like(z1)
    t = G7.sample_t(B, dev); tt = t[:, None, None]
    x_t = (1 - tt) * x0 + tt * z1
    v_target = z1 - x0
    drop = torch.rand(B, device=dev) < p_drop
    if R == 1: pair = raw.compute_pair(esm, mask, contact)
    x_sc = None
    if raw.self_cond and torch.rand(()) < p_sc:
        with torch.no_grad():
            v0 = net(x_t, t, esm, mask, drop, None, pair=pair.detach())
            x_sc = (x_t + (1 - tt) * v0).detach()
    v = net(x_t, t, esm, mask, drop, x_sc, pair=pair)
    m = mask.unsqueeze(-1).float()
    fm = (((v - v_target) ** 2) * m).sum() / (m.sum() * D_LAT).clamp(min=1.0)
    if AUX["w"] <= 0: return fm
    # --- structural term on a random subset of samples ---
    n = min(AUX["n"], B); idx = torch.randperm(B, device=dev)[:n]
    dec = _decoder(dev)
    x1_hat = F.layer_norm((x_t + (1 - tt) * v)[idx], (D_LAT,)) * m[idx]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
        ca_pred = dec(x1_hat.float(), mask[idx])
        with torch.no_grad(): ca_true = dec(z1[idx].float(), mask[idx])
    ok = torch.isfinite(ca_pred).all(-1).all(-1) & torch.isfinite(ca_true).all(-1).all(-1)
    if ok.sum() == 0: return fm
    aux = lddt_loss(ca_pred[ok], ca_true[ok], mask[idx][ok], AUX["cutoff"])
    return fm + AUX["w"] * aux

def install():
    G10.install(); G7.fm_loss = fm_loss_struct

if __name__ == "__main__":
    install()
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--d-pair", type=int, default=64); p.add_argument("--n-pair-blocks", type=int, default=6)
    p.add_argument("--aux-w", type=float, default=1.0); p.add_argument("--aux-n", type=int, default=16); p.add_argument("--pair-unfused", action="store_true")
    pa, rest = p.parse_known_args(); sys.argv = [sys.argv[0]] + rest
    AUX["w"], AUX["n"] = pa.aux_w, pa.aux_n
    a = G7.parse_args()
    class _Net(G10.PairFlowNet):
        def __init__(self, **kw):
            super().__init__(**kw, d_pair=pa.d_pair, n_pair_blocks=pa.n_pair_blocks, pair_fused=not pa.pair_unfused)
            self.extra_arch = {"d_pair": pa.d_pair, "n_pair_blocks": pa.n_pair_blocks, "pair_contact": False, "pair_fused": not pa.pair_unfused}
    G7.LatentFlowNet = _Net
    print(f"[gate19] pair flow + structural LDDT loss: d_pair {pa.d_pair}, blocks {pa.n_pair_blocks}, aux_w {pa.aux_w}, aux_n {pa.aux_n}", flush=True)
    G7.main(a)
