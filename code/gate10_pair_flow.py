"""Gate 10: pair-aware latent flow (cheap pair track, not an ESMFold trunk).

Adds residue-pair reasoning to the Gate 7 latent flow model while keeping it
generative and cheap:

  * A small pair representation p_ij (d_pair, default 64) is built ONCE per
    protein from the single-sequence embedding (outer-sum of two linear
    projections) + a relative-position embedding, optionally + the ESM-2
    pretrained contact map (one channel) and optionally + a distogram of a
    previous sample's decoded structure (recycling hook).
  * It is refined by a few blocks of triangular multiplicative updates
    (outgoing + incoming) and a pair transition MLP. No triangle attention.
  * Every DiT layer reads the pair tensor through a tiny linear map to a
    per-head attention bias. That is the only pair -> single path.

Because the pair track depends on the conditioning only (not on x_t or t),
it is computed once and reused across self-conditioning passes in training
and across all Euler steps at sampling. Parameter cost: under 1M for
d_pair 64 x 6 blocks; compute ~2 L^3 d_pair per block per protein.

Everything else (flow matching, CFG, self-conditioning, EMA, DDP, online
ESM-2, RAM cache, TM evaluation) is inherited from gate7_latent_flow.py.
"""
import os, sys, math
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate7_latent_flow as G7
from gate7_latent_flow import (D_LAT, D_ESM, MAX_LEN, timestep_embedding, DiTBlock)


class PairAttention(nn.Module):
    """Self-attention with a learned relative-position bias PLUS an external
    per-head pair bias (B, H, L, L) from the pair track."""
    def __init__(self, d, n_heads, d_pair, rel_pos=32, dropout=0.0):
        super().__init__()
        self.h, self.dh, self.rel = n_heads, d // n_heads, rel_pos
        self.qkv = nn.Linear(d, 3 * d); self.out = nn.Linear(d, d)
        self.bias = nn.Embedding(2 * rel_pos + 1, n_heads); nn.init.zeros_(self.bias.weight)
        self.dropout = dropout

    def forward(self, x, mask, pb):
        """pb: (B,H,L,L) pair bias for this layer (precomputed once for all layers)."""
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        pos = torch.arange(L, device=x.device)
        rel = (pos[None, :] - pos[:, None]).clamp(-self.rel, self.rel) + self.rel
        bias = self.bias(rel).permute(2, 0, 1).unsqueeze(0)                       # (1,H,L,L)
        pad = torch.zeros(B, 1, 1, L, device=x.device, dtype=bias.dtype).masked_fill(~mask[:, None, None, :], float("-inf"))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=(bias.to(q.dtype) + pb.to(q.dtype) + pad.to(q.dtype)),
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(o.transpose(1, 2).reshape(B, L, D))


class PairDiTBlock(DiTBlock):
    def __init__(self, d, n_heads, rel_pos, dropout, d_pair):
        super().__init__(d, n_heads, rel_pos, dropout)
        self.attn = PairAttention(d, n_heads, d_pair, rel_pos, dropout)

    def forward(self, x, c, mask, pb):
        s1, b1, g1, s2, b2, g2 = self.ada(c).unsqueeze(1).chunk(6, dim=-1)
        x = x + g1 * self.attn(self.n1(x) * (1 + s1) + b1, mask, pb)
        x = x + g2 * self.mlp(self.n2(x) * (1 + s2) + b2)
        return x


class TriangleMultiply(nn.Module):
    """AlphaFold2 triangular multiplicative update, outgoing (mode 'out') or
    incoming ('in'), gated. p: (B,L,L,d). O(L^3 d)."""
    def __init__(self, d, mode):
        super().__init__()
        self.mode = mode
        self.norm_in = nn.LayerNorm(d)
        self.a = nn.Linear(d, d); self.b = nn.Linear(d, d)
        self.ga = nn.Linear(d, d); self.gb = nn.Linear(d, d)
        self.norm_out = nn.LayerNorm(d)
        self.out = nn.Linear(d, d); self.gate = nn.Linear(d, d)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # residual starts at identity

    def forward(self, p, pmask):
        z = self.norm_in(p)
        a = torch.sigmoid(self.ga(z)) * self.a(z) * pmask
        b = torch.sigmoid(self.gb(z)) * self.b(z) * pmask
        if self.mode == "out":   # sum_k a_ik b_jk
            x = torch.einsum("bikd,bjkd->bijd", a, b)
        else:                    # sum_k a_ki b_kj
            x = torch.einsum("bkid,bkjd->bijd", a, b)
        return torch.sigmoid(self.gate(z)) * self.out(self.norm_out(x))


class TriangleMultiplyFused(TriangleMultiply):
    """Same computation as TriangleMultiply with the five per-pair projections (a, b, their
    gates, the output gate) fused into one GEMM on the (B*L*L, d) pair matrix: 5 launches ->
    1, better tensor-core utilisation, same parameter count."""
    def __init__(self, d, mode):
        nn.Module.__init__(self)
        self.mode = mode
        self.norm_in = nn.LayerNorm(d)
        self.proj = nn.Linear(d, 5 * d)
        self.norm_out = nn.LayerNorm(d)
        self.out = nn.Linear(d, d)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, p, pmask):
        z = self.norm_in(p)
        a_, b_, ga, gb, g = self.proj(z).chunk(5, dim=-1)
        a = torch.sigmoid(ga) * a_ * pmask
        b = torch.sigmoid(gb) * b_ * pmask
        if self.mode == "out":
            x = torch.einsum("bikd,bjkd->bijd", a, b)
        else:
            x = torch.einsum("bkid,bkjd->bijd", a, b)
        return torch.sigmoid(g) * self.out(self.norm_out(x))



def fuse_triangle_state_dict(sd, target_keys):
    """Convert a checkpoint saved with the five-GEMM TriangleMultiply (a, b, ga, gb, gate) into the
    fused layout (proj = cat[a, b, ga, gb, gate]) when the target model is fused, and back when the
    target is unfused. Keys may carry torch.compile's '_orig_mod.' prefix on either side."""
    target = set(target_keys)
    if set(sd) == target: return sd
    def strip(k): return k.replace("._orig_mod.", ".").replace("_orig_mod.", "")
    src = {strip(k): v for k, v in sd.items()}; tmap = {strip(k): k for k in target}
    out = {}
    done = set()
    for k in list(src):
        if k in done: continue
        if k.endswith(".a.weight"):
            base = k[:-len("a.weight")]
            if base + "proj.weight" in tmap:
                for suf in ("weight", "bias"):
                    out[tmap[base + "proj." + suf]] = torch.cat([src[base + f"{n}.{suf}"] for n in ("a", "b", "ga", "gb", "gate")], 0)
                    done.update(base + f"{n}.{suf}" for n in ("a", "b", "ga", "gb", "gate"))
                continue
        if k.endswith(".proj.weight"):
            base = k[:-len("proj.weight")]
            if base + "a.weight" in tmap:
                for suf in ("weight", "bias"):
                    parts = src[base + "proj." + suf].chunk(5, 0)
                    for n, part in zip(("a", "b", "ga", "gb", "gate"), parts): out[tmap[base + f"{n}.{suf}"]] = part
                    done.add(base + "proj." + suf)
                continue
        if k not in done and k in tmap: out[tmap[k]] = src[k]; done.add(k)
    return out

class PairBlock(nn.Module):
    def __init__(self, d, dropout=0.0, fused=False):
        super().__init__()
        TM = TriangleMultiplyFused if fused else TriangleMultiply
        self.tri_out = TM(d, "out"); self.tri_in = TM(d, "in")
        self.norm = nn.LayerNorm(d)
        self.trans = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Dropout(dropout), nn.Linear(4 * d, d))
        nn.init.zeros_(self.trans[-1].weight); nn.init.zeros_(self.trans[-1].bias)

    def forward(self, p, pmask):
        p = p + self.tri_out(p, pmask)
        p = p + self.tri_in(p, pmask)
        p = p + self.trans(self.norm(p)) * pmask
        return p


class PairTrack(nn.Module):
    """Build and refine the pair representation from the conditioning."""
    def __init__(self, d_in, d_pair=64, n_blocks=6, rel_pos=32, use_contact=False, n_dist_bins=0, dropout=0.0, fused=False):
        super().__init__()
        self.rel = rel_pos
        self.norm_s = nn.LayerNorm(d_in)
        self.left = nn.Linear(d_in, d_pair); self.right = nn.Linear(d_in, d_pair)
        self.relpos = nn.Embedding(2 * rel_pos + 1, d_pair)
        self.use_contact = use_contact
        self.contact = nn.Linear(1, d_pair) if use_contact else None
        self.n_dist_bins = n_dist_bins
        self.dist = nn.Linear(n_dist_bins, d_pair) if n_dist_bins else None   # recycling hook
        self.blocks = nn.ModuleList([PairBlock(d_pair, dropout, fused) for _ in range(n_blocks)])
        if os.environ.get("PAIR_COMPILE", "1") == "1" and torch.cuda.is_available():
            # fuse the memory-bound elementwise chains (gates, norms, masks); the
            # triangle contraction itself is cheap
            self.blocks = nn.ModuleList([torch.compile(b, dynamic=True) for b in self.blocks])
        self.norm_out = nn.LayerNorm(d_pair)

    def forward(self, s, mask, contact=None, dist_feats=None):
        B, L, _ = s.shape
        z = self.norm_s(s.float())
        p = self.left(z).unsqueeze(2) + self.right(z).unsqueeze(1)               # (B,L,L,d)
        pos = torch.arange(L, device=s.device)
        rel = (pos[None, :] - pos[:, None]).clamp(-self.rel, self.rel) + self.rel
        p = p + self.relpos(rel).unsqueeze(0)
        if self.use_contact and contact is not None:
            p = p + self.contact(contact.float().unsqueeze(-1))
        if self.dist is not None:
            if dist_feats is not None:
                p = p + self.dist(dist_feats.float())
            else:   # no geometry this step: still touch the parameters so DDP sees them every iteration
                p = p + self.dist.bias + 0.0 * self.dist.weight.sum()
        p = p.to(torch.bfloat16) if p.is_cuda else p               # pair track in bf16 on GPU
        pmask = (mask[:, :, None] & mask[:, None, :]).unsqueeze(-1).to(p.dtype)
        p = p * pmask
        for blk in self.blocks:
            # activation checkpointing: keep only the block input, recompute the
            # ~10 L x L x d intermediates in backward (memory 10x smaller)
            if torch.is_grad_enabled() and os.environ.get("PAIR_CKPT", "1") == "1":
                p = checkpoint(blk, p, pmask, use_reentrant=False)
            else:
                p = blk(p, pmask)
        return self.norm_out(p.float())


class PairFlowNet(nn.Module):
    """Gate 7 LatentFlowNet + PairTrack; same forward signature plus optional
    `contact` and `dist_feats`, and a `pair` kwarg to reuse a precomputed pair
    tensor (self-conditioning pass, sampling steps)."""
    def __init__(self, d_model=768, n_layers=16, n_heads=12, dropout=0.0, rel_pos=32, self_cond=True,
                 d_cond=D_ESM, d_pair=64, n_pair_blocks=6, pair_contact=False, pair_dist_bins=0, pair_fused=False):
        super().__init__()
        self.self_cond = self_cond
        self.in_proj = nn.Linear(D_LAT * (2 if self_cond else 1), d_model)
        self.cond_norm = nn.LayerNorm(d_cond); self.cond_proj = nn.Linear(d_cond, d_model)
        self.null_cond = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Embedding(MAX_LEN, d_model)
        self.t_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.pair = PairTrack(d_cond, d_pair, n_pair_blocks, rel_pos, pair_contact, pair_dist_bins, dropout, fused=pair_fused)
        self.null_pair = nn.Parameter(torch.zeros(1, 1, 1, d_pair))
        self.pair_bias_norm = nn.LayerNorm(d_pair)
        self.pair_bias = nn.Linear(d_pair, n_layers * n_heads, bias=False); nn.init.zeros_(self.pair_bias.weight)
        self.n_heads = n_heads
        self.blocks = nn.ModuleList([PairDiTBlock(d_model, n_heads, rel_pos, dropout, d_pair) for _ in range(n_layers)])
        if G7.DIT_COMPILE and torch.cuda.is_available():
            self.blocks = nn.ModuleList([torch.compile(b, dynamic=True) for b in self.blocks])
        self.out_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.out_ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.out_proj = nn.Linear(d_model, D_LAT)
        nn.init.zeros_(self.out_ada[1].weight); nn.init.zeros_(self.out_ada[1].bias)
        nn.init.zeros_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)
        self.d_model = d_model

    def compute_pair(self, esm, mask, contact=None, dist_feats=None):
        return self.pair(esm, mask, contact, dist_feats)

    def forward(self, x_t, t, esm, mask, cond_drop=None, x_sc=None, pair=None, contact=None, dist_feats=None):
        B, L, _ = x_t.shape
        if self.self_cond:
            x_sc = torch.zeros_like(x_t) if x_sc is None else x_sc
            x_in = torch.cat([x_t, x_sc], dim=-1)
        else:
            x_in = x_t
        if cond_drop is None:
            cond_drop = torch.zeros(B, dtype=torch.bool, device=x_t.device)
        c_tok = self.cond_proj(self.cond_norm(esm.float()))
        c_tok = torch.where(cond_drop[:, None, None], self.null_cond.expand(B, L, -1), c_tok)
        if pair is None:
            pair = self.compute_pair(esm, mask, contact, dist_feats)
        # CFG: dropped samples also lose the pair track
        pair = torch.where(cond_drop[:, None, None, None], self.null_pair.expand(B, L, L, -1), pair)
        h = self.in_proj(x_in) + c_tok + self.pos.weight[:L][None]
        m = mask.unsqueeze(-1).float()
        c_pool = (c_tok * m).sum(1) / m.sum(1).clamp(min=1.0)
        c = self.t_mlp(timestep_embedding(t, self.d_model)) + c_pool
        pb_all = self.pair_bias(self.pair_bias_norm(pair)).permute(0, 3, 1, 2)   # (B, n_layers*H, L, L), once
        pbs = pb_all.to(torch.bfloat16 if pb_all.is_cuda else pb_all.dtype).contiguous().split(self.n_heads, dim=1)   # bf16: the residual stream h is fp32 and .to(h.dtype) doubled the largest tensor of the model
        for blk, pb in zip(self.blocks, pbs):
            h = blk(h, c, mask, pb)
        s, b = self.out_ada(c).unsqueeze(1).chunk(2, dim=-1)
        return self.out_proj(self.out_norm(h) * (1 + s) + b)


# ---------------------------------------------------------------------------
# Flow-matching loss and sampler that compute the pair track once
# ---------------------------------------------------------------------------
REPEAT_COPIES = int(os.environ.get("REPEAT_COPIES", "1"))   # copies of each protein per step sharing one pair computation

def _repeat(x, r):
    return x if r == 1 or x is None else x.repeat_interleave(r, dim=0)


def fm_loss_pair(net, z1, esm, mask, p_drop=0.1, p_sc=0.5, contact=None):
    raw = getattr(net, "module", net)
    R = REPEAT_COPIES
    if R > 1:   # pair once per protein, then R noisy copies (different t, x0, cond-drop) through the trunk
        pair1 = raw.compute_pair(esm, mask, contact)
        pair = pair1.repeat_interleave(R, dim=0)
        z1, esm, mask = _repeat(z1, R), _repeat(esm, R), _repeat(mask, R)
    B = z1.shape[0]; dev = z1.device
    x0 = torch.randn_like(z1)
    t = G7.sample_t(B, dev); tt = t[:, None, None]
    x_t = (1 - tt) * x0 + tt * z1
    v_target = z1 - x0
    drop = torch.rand(B, device=dev) < p_drop
    if R == 1: pair = raw.compute_pair(esm, mask, contact)          # once, with grad
    x_sc = None
    if raw.self_cond and torch.rand(()) < p_sc:
        with torch.no_grad():
            v0 = net(x_t, t, esm, mask, drop, None, pair=pair.detach())
            x_sc = (x_t + (1 - tt) * v0).detach()
    v = net(x_t, t, esm, mask, drop, x_sc, pair=pair)
    m = mask.unsqueeze(-1).float()
    return (((v - v_target) ** 2) * m).sum() / (m.sum() * D_LAT).clamp(min=1.0)


@torch.no_grad()
def sample_pair(net, esm, mask, n_steps=25, cfg_w=2.0, gen=None, project=True, contact=None):
    raw = getattr(net, "module", net)
    B, L = mask.shape; dev = esm.device
    pair = raw.compute_pair(esm, mask, contact)          # once per protein, shared by all steps
    x = torch.randn(B, L, D_LAT, device=dev, generator=gen); x_sc = None
    ts = torch.linspace(0, 1, n_steps + 1, device=dev)
    ones = torch.ones(B, dtype=torch.bool, device=dev)
    for i in range(n_steps):
        t = ts[i].expand(B); dt = ts[i + 1] - ts[i]
        v = net(x, t, esm, mask, None, x_sc, pair=pair)
        if cfg_w != 1.0:
            v_u = net(x, t, esm, mask, ones, x_sc, pair=pair)
            v = v_u + cfg_w * (v - v_u)
        if raw.self_cond: x_sc = x + (1 - ts[i]) * v
        x = x + v * dt
    if project: x = F.layer_norm(x, (D_LAT,))
    return x * mask.unsqueeze(-1)


def install():
    """Route gate7's trainer through the pair model: monkey-patch the pieces
    gate7.main() uses so the whole training/eval/DDP/resume machinery is
    reused unchanged."""
    G7.LatentFlowNet = PairFlowNet
    G7.fm_loss = fm_loss_pair
    G7.sample = sample_pair


if __name__ == "__main__":
    install()
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--d-pair", type=int, default=64); p.add_argument("--n-pair-blocks", type=int, default=6)
    p.add_argument("--pair-contact", action="store_true", help="add the ESM-2 pretrained contact map as a pair input (online mode)")
    p.add_argument("--pair-unfused", action="store_true", help="use the original five-GEMM triangle update (old checkpoints); default is the fused one")
    pa, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    a = G7.parse_args()
    # inject the pair hyper-parameters into gate7's arch dict via a wrapper class
    class _Net(PairFlowNet):
        def __init__(self, **kw):
            super().__init__(**kw, d_pair=pa.d_pair, n_pair_blocks=pa.n_pair_blocks, pair_contact=pa.pair_contact, pair_fused=not pa.pair_unfused)
            self.extra_arch = {"d_pair": pa.d_pair, "n_pair_blocks": pa.n_pair_blocks, "pair_contact": pa.pair_contact, "pair_fused": not pa.pair_unfused}
    _Net.__name__ = "PairFlowNet"
    G7.LatentFlowNet = _Net
    if pa.pair_contact:
        raise SystemExit("--pair-contact: contact-map plumbing lands in the next revision")
    a.label = a.label
    print(f"[gate10] pair track: d_pair {pa.d_pair}, blocks {pa.n_pair_blocks}, contact {pa.pair_contact}", flush=True)
    G7.main(a)
