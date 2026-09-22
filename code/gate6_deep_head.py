"""
Gate 6: Deeper head for long-range reasoning.

Architecture: 10-layer d256 transformer with increased dropout,
using the 100k dataset. The hypothesis is that 2 layers of self-attention
can't model the long-range contacts encoded in z, and depth is needed
for the residue-residue interaction reasoning that maps sequence context
to 3D spatial encoding.

Also tests a variant with 1D conv preprocessing to give the transformer
richer local features (secondary structure patterns).
"""
import os as _os
# Portable root. Set ESM_PROAE_ROOT to relocate the project; set
# FOLDSEEK_BIN if foldseek is not on PATH.
ROOT = _os.environ.get("ESM_PROAE_ROOT", "/home/guest/projects/esm_proae")
_FOLDSEEK = _os.environ.get("FOLDSEEK_BIN", "foldseek")
_os.makedirs(_os.path.join(ROOT, "notes"), exist_ok=True)  # results dir may not exist in a fresh checkout
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

PROJECT = Path(ROOT)
H5_PATH = PROJECT / "data" / "phase1_dataset" / "dataset_100k.h5"


class ProteinDataset(Dataset):
    def __init__(self, h5_path, split, max_len=256):
        self.h5 = h5py.File(h5_path, "r")
        self.split = split
        self.names = list(self.h5[split].keys())
        self.max_len = max_len

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        grp = self.h5[self.split][self.names[idx]]
        esm = torch.from_numpy(grp["esm2_emb"][:].astype(np.float32))
        z = torch.from_numpy(grp["z"][:])
        n = esm.shape[0]
        if n < self.max_len:
            esm = torch.cat([esm, torch.zeros(self.max_len - n, 1280)], 0)
            z = torch.cat([z, torch.zeros(self.max_len - n, 8)], 0)
        else:
            esm, z = esm[:self.max_len], z[:self.max_len]
            n = self.max_len
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[:n] = True
        return esm, z, mask, n


class DeepHead(nn.Module):
    """10-layer d256 transformer with conv preprocessing."""
    def __init__(self, d_in=1280, d_model=256, d_out=8, n_layers=10, n_heads=8,
                 dropout=0.15, use_conv=False, normalize_out=False):
        super().__init__()
        self.use_conv = use_conv
        # The ProteinAE encoder LayerNorms every latent, so true z satisfies
        # layer_norm(z, (d_out,)) exactly (per-residue zero mean, unit variance,
        # hence L2 = sqrt(d_out)). Projecting the head output onto that manifold
        # keeps the decoder in-distribution. Off by default so existing
        # checkpoints and callers are unaffected.
        self.normalize_out = normalize_out
        self.d_out = d_out

        if use_conv:
            # 1D conv to capture local secondary structure patterns
            self.conv = nn.Sequential(
                nn.Conv1d(d_in, d_model, kernel_size=7, padding=3),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=11, padding=5),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.input_proj = nn.Identity()  # conv already projects
        else:
            self.input_proj = nn.Linear(d_in, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, d_out)

    def forward(self, x, mask):
        if self.use_conv:
            # x: (B, L, 1280) -> conv expects (B, C, L)
            h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, L, d_model)
        else:
            h = self.input_proj(x)
        h = self.encoder(h, src_key_padding_mask=~mask)
        out = self.output_proj(h)
        if self.normalize_out:
            out = F.layer_norm(out, (self.d_out,))
        return out


def train_model(model, train_loader, val_loader, test_loader, device,
                lr=1e-4, weight_decay=0.05, n_epochs=40, patience=10,
                baseline_mse=None, label=""):
    n_params = sum(p.numel() for p in model.parameters())
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val, best_epoch = float('inf'), 0
    history = []

    print(f"\n  {label}: {n_params:,} params, lr={lr}")
    print(f"  {'Ep':>4s}  {'Train':>10s}  {'Val':>10s}  {'Ratio':>8s}  {'Time':>6s}")

    for epoch in range(n_epochs):
        t0 = time.perf_counter()
        model.train()
        tl, nb = 0.0, 0
        for esm, z, mask, _ in train_loader:
            esm, z, mask = esm.to(device), z.to(device), mask.to(device)
            pred = model(esm, mask)
            diff = ((pred - z) ** 2).sum(-1) * mask.float()
            loss = diff.sum() / (mask.float().sum() * 8)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item(); nb += 1
        train_mse = tl / nb

        model.eval()
        vs, vn = 0.0, 0
        with torch.no_grad():
            for esm, z, mask, _ in val_loader:
                esm, z, mask = esm.to(device), z.to(device), mask.to(device)
                diff = ((model(esm, mask) - z) ** 2).sum(-1) * mask.float()
                vs += diff.sum().item(); vn += mask.float().sum().item() * 8
        val_mse = vs / vn
        scheduler.step()
        ratio = val_mse / baseline_mse if baseline_mse else 0
        elapsed = time.perf_counter() - t0
        history.append({"epoch": epoch+1, "train": train_mse, "val": val_mse, "ratio": ratio})

        print(f"  {epoch+1:4d}  {train_mse:10.6f}  {val_mse:10.6f}  {ratio:8.4f}  {elapsed:6.1f}s")

        if val_mse < best_val:
            best_val = val_mse; best_epoch = epoch + 1
            torch.save(model.state_dict(), str(PROJECT / "data" / "phase1_dataset" / f"best_{label}.pt"))
        if epoch + 1 - best_epoch >= patience:
            print(f"  Early stopping (best: {best_epoch})")
            break

    # Test
    model.load_state_dict(torch.load(
        str(PROJECT / "data" / "phase1_dataset" / f"best_{label}.pt"), weights_only=True))
    model.eval()
    ts, tn = 0.0, 0
    with torch.no_grad():
        for esm, z, mask, _ in test_loader:
            esm, z, mask = esm.to(device), z.to(device), mask.to(device)
            diff = ((model(esm, mask) - z) ** 2).sum(-1) * mask.float()
            ts += diff.sum().item(); tn += mask.float().sum().item() * 8
    test_mse = ts / tn
    test_sigma = float(np.sqrt(test_mse))

    return {
        "label": label, "n_params": n_params, "best_epoch": best_epoch,
        "test_mse": test_mse, "test_sigma": test_sigma,
        "ratio": test_mse / baseline_mse if baseline_mse else 0,
        "history": history,
    }


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    print("=" * 60)
    print("Deep head experiments on 100k data")
    print("=" * 60)

    train_ds = ProteinDataset(H5_PATH, "train")
    val_ds = ProteinDataset(H5_PATH, "val")
    test_ds = ProteinDataset(H5_PATH, "test")

    # Smaller batch for deeper models
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Baseline
    h5 = h5py.File(str(H5_PATH), "r")
    z_sum, z_n = np.zeros(8, np.float64), 0
    for name in h5["train"]:
        z = h5["train"][name]["z"][:]
        z_sum += z.sum(0); z_n += z.shape[0]
    mean_z = z_sum / z_n
    bl_sum, bl_n = 0.0, 0
    for name in h5["test"]:
        z = h5["test"][name]["z"][:]
        bl_sum += ((z - mean_z) ** 2).sum(); bl_n += z.shape[0] * 8
    baseline_mse = bl_sum / bl_n
    baseline_sigma = float(np.sqrt(baseline_mse))
    h5.close()
    print(f"Baseline sigma: {baseline_sigma:.4f}")

    results = []

    # Model 1: 10-layer d256 (deep transformer)
    print(f"\n{'='*60}")
    print("Model 1: 10-layer d256 deep transformer")
    m1 = DeepHead(n_layers=10, d_model=256, dropout=0.15, use_conv=False)
    r1 = train_model(m1, train_loader, val_loader, test_loader, device,
                     lr=1e-4, baseline_mse=baseline_mse, label="deep10L")
    results.append(r1)
    print(f"  Result: sigma={r1['test_sigma']:.4f}")
    del m1; torch.cuda.empty_cache()

    # Model 2: 10-layer d256 with conv preprocessing
    print(f"\n{'='*60}")
    print("Model 2: 10-layer d256 with conv preprocessing")
    m2 = DeepHead(n_layers=10, d_model=256, dropout=0.15, use_conv=True)
    r2 = train_model(m2, train_loader, val_loader, test_loader, device,
                     lr=1e-4, baseline_mse=baseline_mse, label="deep10L_conv")
    results.append(r2)
    print(f"  Result: sigma={r2['test_sigma']:.4f}")
    del m2; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  {'Model':>25s}  {'Params':>10s}  {'Sigma':>8s}  {'Ratio':>8s}  {'Best ep':>8s}")
    print(f"  {'Mean-z baseline':>25s}  {'—':>10s}  {baseline_sigma:8.4f}  {'1.0000':>8s}  {'—':>8s}")
    print(f"  {'2L-d128 (prev best)':>25s}  {'562k':>10s}  {'0.6206':>8s}  {'0.8142':>8s}  {'—':>8s}")
    print(f"  {'6L-d256 (100k prev)':>25s}  {'5M':>10s}  {'0.6249':>8s}  {'0.8508':>8s}  {'—':>8s}")
    for r in results:
        print(f"  {r['label']:>25s}  {r['n_params']:>10,}  {r['test_sigma']:8.4f}  "
              f"{r['ratio']:8.4f}  {r['best_epoch']:>8d}")
    print(f"  {'ESMFold':>25s}  {'3B':>10s}  {'0.4866':>8s}  {'—':>8s}  {'—':>8s}")

    with open(PROJECT / "notes" / "gate6_deep_head_results.json", "w") as f:
        json.dump({"deep_head": results}, f, indent=2)
    print(f"\nSaved to notes/gate6_deep_head_results.json")


if __name__ == "__main__":
    main()
