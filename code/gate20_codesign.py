"""Gate 20: joint sequence + structure (latent) generation — SimpleDesign in the ProteinAE latent.

One network, two time axes:
  sequence  t_seq ~ U(0,1): each residue is MASKED with probability 1 - t_seq; the frozen ESMC-6B embeds
            the masked sequence (online, once per protein per step, shared by the R repeated copies);
            a sequence head on the trunk output predicts the masked amino acids (cross-entropy weighted by
            t_seq, normalised by the number of masked positions, as in SimpleDesign eq. 1).
  structure t_str (late-t resampled) per copy: the usual latent flow-matching loss.
Because the trunk sees the noised latent AND the (partially masked) sequence, the same weights do
folding (t_seq = 1), inverse folding (all masked, clean latent at t_str = 1) and co-design (both axes
moving). Classifier-free dropout of the conditioning also gives unconditional structure generation.

Runs through gate7's trainer in ONLINE-ESM mode (`--esm online --esm-kind esmc`): the embedder is
replaced by a masking embedder that remembers the mask and the true tokens for the loss.
  python gate20_codesign.py --d-pair 64 --n-pair-blocks 6 --seq-w 1.0 --esm online --esm-kind esmc --esm-path .../data/esmc6b ...
"""
import os, sys, math, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/code")
import gate7_latent_flow as G7, gate10_pair_flow as G10
from gate7_latent_flow import D_LAT

AA = "ACDEFGHIKLMNPQRSTVWY"; AA_IDX = {c: i for i, c in enumerate(AA)}
SEQ = {"w": 1.0, "p_fold": 0.25}   # p_fold: fraction of steps with t_seq = 1 (pure folding), so folding quality is not diluted


class MaskingESM(G7.OnlineESM):
    """OnlineESM that, when training (grad enabled), masks each protein's sequence at rate 1 - t_seq and keeps
    (t_seq, masked positions, true tokens) in self.last for the loss. In eval (no grad) it embeds the plain sequence."""
    def forward(self, seqs, L=None):
        L = L or G7.MAX_LEN; B = len(seqs); dev = self.device
        if not torch.is_grad_enabled():
            self.last = None; return super().forward(seqs, L)
        t_seq = torch.rand(B); t_seq[torch.rand(B) < SEQ["p_fold"]] = 1.0
        tokens = torch.full((B, L), -100, dtype=torch.long); mpos = torch.zeros(B, L, dtype=torch.bool); masked = []
        for i, s in enumerate(seqs):
            s = s[:L]; m = torch.rand(len(s)) < (1 - t_seq[i])
            ids = torch.tensor([AA_IDX.get(c, 0) for c in s])
            tokens[i, :len(s)][m] = ids[m]; mpos[i, :len(s)] = m
            masked.append("".join(self.tok.mask_token if mm else c for c, mm in zip(s, m.tolist())))
        self.last = {"t_seq": t_seq.to(dev), "tokens": tokens.to(dev), "mpos": mpos.to(dev)}
        return super().forward(masked, L)


class CodesignNet(G10.PairFlowNet):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.seq_head = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, len(AA)))

    def forward(self, x_t, t, esm, mask, cond_drop=None, x_sc=None, pair=None, contact=None, dist_feats=None, return_logits=False):
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
        pair = torch.where(cond_drop[:, None, None, None], self.null_pair.expand(B, L, L, -1), pair)
        h = self.in_proj(x_in) + c_tok + self.pos.weight[:L][None]
        m = mask.unsqueeze(-1).float()
        c_pool = (c_tok * m).sum(1) / m.sum(1).clamp(min=1.0)
        c = self.t_mlp(G10.timestep_embedding(t, self.d_model)) + c_pool
        pb_all = self.pair_bias(self.pair_bias_norm(pair)).permute(0, 3, 1, 2)
        pbs = pb_all.to(torch.bfloat16 if pb_all.is_cuda else pb_all.dtype).contiguous().split(self.n_heads, dim=1)
        for blk, pb in zip(self.blocks, pbs):
            h = blk(h, c, mask, pb)
        s, b = self.out_ada(c).unsqueeze(1).chunk(2, dim=-1)
        v = self.out_proj(self.out_norm(h) * (1 + s) + b)
        if return_logits:
            return v, self.seq_head(h)
        return v


def fm_loss_codesign(net, z1, esm, mask, p_drop=0.1, p_sc=0.5, contact=None):
    """Latent flow loss + masked-sequence cross-entropy. Needs the MaskingESM side channel."""
    raw = getattr(net, "module", net); emb = fm_loss_codesign.embed
    R = G10.REPEAT_COPIES; last = emb.last
    if R > 1:
        pair1 = raw.compute_pair(esm, mask, contact); pair = pair1.repeat_interleave(R, dim=0)
        z1, esm, mask = G10._repeat(z1, R), G10._repeat(esm, R), G10._repeat(mask, R)
    B = z1.shape[0]; dev = z1.device
    x0 = torch.randn_like(z1); t = G7.sample_t(B, dev); tt = t[:, None, None]
    x_t = (1 - tt) * x0 + tt * z1; v_target = z1 - x0
    drop = torch.rand(B, device=dev) < p_drop
    if R == 1: pair = raw.compute_pair(esm, mask, contact)
    x_sc = None
    if raw.self_cond and torch.rand(()) < p_sc:
        with torch.no_grad():
            v0 = net(x_t, t, esm, mask, drop, None, pair=pair.detach())
            x_sc = (x_t + (1 - tt) * v0).detach()
    v, logits = net(x_t, t, esm, mask, drop, x_sc, pair=pair, return_logits=True)
    m = mask.unsqueeze(-1).float()
    fm = (((v - v_target) ** 2) * m).sum() / (m.sum() * D_LAT).clamp(min=1.0)
    if last is None or SEQ["w"] <= 0: return fm
    tokens, mpos, t_seq = G10._repeat(last["tokens"], R), G10._repeat(last["mpos"], R), G10._repeat(last["t_seq"], R)
    L = mask.shape[1]; tokens, mpos = tokens[:, :L], mpos[:, :L]
    keep = mpos & mask & ~drop[:, None]          # no sequence loss on condition-dropped copies
    if keep.sum() == 0: return fm
    ce = F.cross_entropy(logits.float().reshape(-1, len(AA)), torch.where(keep, tokens, torch.full_like(tokens, -100)).reshape(-1), ignore_index=-100, reduction="none").view(B, L)
    per = (ce * keep).sum(1) / keep.sum(1).clamp(min=1)          # mean over masked positions per copy
    seq_loss = (t_seq * per).sum() / (keep.sum(1) > 0).sum().clamp(min=1)   # beta(t) = t weighting (SimpleDesign)
    fm_loss_codesign.last_parts = (fm.item(), seq_loss.item())
    return fm + SEQ["w"] * seq_loss


def install():
    G10.install()
    G7.LatentFlowNet = CodesignNet
    G7.OnlineESM = MaskingESM
    G7.fm_loss = fm_loss_codesign


if __name__ == "__main__":
    install()
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--d-pair", type=int, default=64); p.add_argument("--n-pair-blocks", type=int, default=6)
    p.add_argument("--seq-w", type=float, default=1.0); p.add_argument("--p-fold", type=float, default=0.25); p.add_argument("--pair-unfused", action="store_true")
    pa, rest = p.parse_known_args(); sys.argv = [sys.argv[0]] + rest
    SEQ["w"], SEQ["p_fold"] = pa.seq_w, pa.p_fold
    a = G7.parse_args()
    assert a.esm == "online" and a.esm_kind == "esmc", "co-design needs --esm online --esm-kind esmc (masked sequences are embedded on the fly)"
    class _Net(CodesignNet):
        def __init__(self, **kw):
            super().__init__(**kw, d_pair=pa.d_pair, n_pair_blocks=pa.n_pair_blocks, pair_fused=not pa.pair_unfused)
            self.extra_arch = {"d_pair": pa.d_pair, "n_pair_blocks": pa.n_pair_blocks, "pair_contact": False, "pair_fused": not pa.pair_unfused,
                               "codesign": True, "seq_w": pa.seq_w, "p_fold": pa.p_fold}
    G7.LatentFlowNet = _Net
    # the loss needs the embedder: gate7.main builds it as `embed`; expose it through OnlineESM construction
    _orig_init = MaskingESM.__init__
    def _init(self, *args, **kw):
        _orig_init(self, *args, **kw); fm_loss_codesign.embed = self
    MaskingESM.__init__ = _init
    print(f"[gate20] co-design: pair {pa.d_pair}x{pa.n_pair_blocks}, seq loss weight {pa.seq_w}, p_fold {pa.p_fold}", flush=True)
    G7.main(a)
