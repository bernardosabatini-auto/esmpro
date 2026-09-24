"""Gate 16: structural self-conditioning ("recycling") for the pair-track latent flow.

The pair track so far sees only the sequence embedding. Here the model's own
current structure estimate is fed back into it as geometry: at a denoising
step the clean-latent estimate x1_hat = x_t + (1 - t) v is decoded through the
frozen ProteinAE decoder (3 Euler steps, no grad), its Ca distogram (16 bins,
2-20 A) plus the timestep is projected into the pair representation, and the
pair track is recomputed with it. Training uses the estimate from the
self-conditioning pass that already exists (one extra decoder pass and one
extra no-grad pair pass per step; the with-grad pair pass is unchanged), with
probability --p-rec, so the network also works without geometry (first
sampling step, CFG null branch). Sampling recomputes the pair with the decoded
estimate every --rec-every Euler steps.

This is the generative analogue of AlphaFold/ESMFold recycling: the pair
representation gets real distances to reason over, at the cost of a decoder
pass instead of a second trunk pass.

  python gate16_recycle_flow.py --d-pair 128 --n-pair-blocks 8 --p-rec 0.5 --rec-every 5 ... (gate7/gate10 flags)
"""
import os, sys, torch, torch.nn.functional as F
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/code")
import gate7_latent_flow as G7, gate10_pair_flow as G10
from gate7_latent_flow import D_LAT

N_BINS = 16
EDGES = torch.linspace(2.0, 20.0, N_BINS - 1)          # 15 edges -> 16 bins, last = > 20 A
REC = {"dec": None, "p_rec": 0.5, "rec_every": 5, "chunk": int(os.environ.get("REC_CHUNK", "96"))}

def _decoder(device):
    if REC["dec"] is None:
        cwd = os.getcwd(); os.chdir(ROOT + "/ProteinAE_v1")
        try: REC["dec"] = G7.load_decoder(device)
        finally: os.chdir(cwd)
    return REC["dec"]

@torch.no_grad()
def struct_feats(z_hat, mask, t):
    """Decode the latent estimate and return (B, L, L, N_BINS + 1) distogram one-hot + timestep channel."""
    dec = _decoder(z_hat.device); B, L = mask.shape
    z = F.layer_norm(z_hat.float(), (D_LAT,)) * mask.unsqueeze(-1)
    cas = []
    for s in range(0, B, REC["chunk"]):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=z.is_cuda):
            cas.append(dec(z[s:s + REC["chunk"]], mask[s:s + REC["chunk"]]).float())
    ca = torch.cat(cas, 0)
    bad = ~torch.isfinite(ca).all(-1).all(-1)                      # (B,)
    ca = torch.nan_to_num(ca, nan=0.0, posinf=0.0, neginf=0.0)
    d = torch.cdist(ca, ca)                                         # (B,L,L) in A
    idx = torch.bucketize(d, EDGES.to(d.device))                    # 0..N_BINS-1
    feats = F.one_hot(idx, N_BINS).float()
    pm = (mask[:, :, None] & mask[:, None, :]).unsqueeze(-1).float() * (~bad)[:, None, None, None].float()
    feats = feats * pm
    tch = t.float()[:, None, None, None].expand(B, L, L, 1) * pm
    return torch.cat([feats, tch], -1)


class RecFlowNet(G10.PairFlowNet):
    """PairFlowNet whose pair track accepts the decoded-estimate distogram (+ t)."""
    def __init__(self, *a, **kw):
        kw["pair_dist_bins"] = N_BINS + 1
        super().__init__(*a, **kw)


def fm_loss_rec(net, z1, esm, mask, p_drop=0.1, p_sc=0.5, contact=None):
    raw = getattr(net, "module", net)
    R = G10.REPEAT_COPIES
    if R > 1:   # repeated batching: R noisy copies per protein; the geometry-free pair is shared, the recycled one is per copy
        z1, esm, mask = G10._repeat(z1, R), G10._repeat(esm, R), G10._repeat(mask, R)
    B = z1.shape[0]; dev = z1.device
    x0 = torch.randn_like(z1)
    t = torch.sigmoid(torch.randn(B, device=dev)); tt = t[:, None, None]
    x_t = (1 - tt) * x0 + tt * z1
    v_target = z1 - x0
    drop = torch.rand(B, device=dev) < p_drop
    do_sc = raw.self_cond and torch.rand(()) < p_sc
    do_rec = do_sc and torch.rand(()) < REC["p_rec"]
    x_sc = None
    if do_rec:
        with torch.no_grad():
            pair0 = raw.compute_pair(esm, mask)                       # geometry-free pair for the estimate pass, no grad
            v0 = net(x_t, t, esm, mask, drop, None, pair=pair0)
            x1_hat = (x_t + (1 - tt) * v0).detach(); x_sc = x1_hat
        pair = raw.compute_pair(esm, mask, dist_feats=struct_feats(x1_hat, mask, t))   # with grad, sees the decoded estimate
    else:
        pair = raw.compute_pair(esm, mask)                            # with grad (as gate10)
        if do_sc:
            with torch.no_grad():
                v0 = net(x_t, t, esm, mask, drop, None, pair=pair.detach())
                x_sc = (x_t + (1 - tt) * v0).detach()
    v = net(x_t, t, esm, mask, drop, x_sc, pair=pair)
    m = mask.unsqueeze(-1).float()
    return (((v - v_target) ** 2) * m).sum() / (m.sum() * D_LAT).clamp(min=1.0)


@torch.no_grad()
def sample_rec(net, esm, mask, n_steps=25, cfg_w=2.0, gen=None, project=True, contact=None):
    raw = getattr(net, "module", net)
    B, L = mask.shape; dev = esm.device
    pair = raw.compute_pair(esm, mask)
    x = torch.randn(B, L, D_LAT, device=dev, generator=gen); x_sc = None
    ts = torch.linspace(0, 1, n_steps + 1, device=dev)
    ones = torch.ones(B, dtype=torch.bool, device=dev)
    for i in range(n_steps):
        t = ts[i].expand(B); dt = ts[i + 1] - ts[i]
        if i > 0 and REC["rec_every"] > 0 and i % REC["rec_every"] == 0 and x_sc is not None:
            pair = raw.compute_pair(esm, mask, dist_feats=struct_feats(x_sc, mask, t))
        v = net(x, t, esm, mask, None, x_sc, pair=pair)
        if cfg_w != 1.0:
            v_u = net(x, t, esm, mask, ones, x_sc, pair=pair)
            v = v_u + cfg_w * (v - v_u)
        if raw.self_cond: x_sc = x + (1 - ts[i]) * v
        x = x + v * dt
    if project: x = F.layer_norm(x, (D_LAT,))
    return x * mask.unsqueeze(-1)


def install():
    G10.install()
    G7.LatentFlowNet = RecFlowNet
    G7.fm_loss = fm_loss_rec
    G7.sample = sample_rec


if __name__ == "__main__":
    install()
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--d-pair", type=int, default=64); p.add_argument("--n-pair-blocks", type=int, default=6)
    p.add_argument("--p-rec", type=float, default=0.5, help="probability that a training step feeds the decoded estimate back")
    p.add_argument("--rec-every", type=int, default=5, help="sampling: recompute the pair with the decoded estimate every k Euler steps (0 = never)")
    p.add_argument("--pair-unfused", action="store_true")
    pa, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    REC["p_rec"], REC["rec_every"] = pa.p_rec, pa.rec_every
    a = G7.parse_args()
    class _Net(RecFlowNet):
        def __init__(self, **kw):
            super().__init__(**kw, d_pair=pa.d_pair, n_pair_blocks=pa.n_pair_blocks, pair_fused=not pa.pair_unfused)
            self.extra_arch = {"d_pair": pa.d_pair, "n_pair_blocks": pa.n_pair_blocks, "pair_contact": False, "pair_fused": not pa.pair_unfused,
                               "recycle": True, "p_rec": pa.p_rec, "rec_every": pa.rec_every}
    G7.LatentFlowNet = _Net
    print(f"[gate16] recycling pair flow: d_pair {pa.d_pair}, blocks {pa.n_pair_blocks}, p_rec {pa.p_rec}, rec_every {pa.rec_every}", flush=True)
    G7.main(a)
