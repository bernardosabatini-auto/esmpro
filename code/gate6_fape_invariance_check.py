"""Bug-1 guard (CLAUDE.md section 6): FAPE must read ~0 for a true structure
under rigid rotation/translation, and must NOT read ~0 for 1 A noise.
Runs on CPU or GPU in seconds. Exit code 1 on failure."""
import os, sys, math
import numpy as np, torch
ROOT = os.environ["ESM_PROAE_ROOT"]
sys.path.insert(0, ROOT + "/code")
import h5py
from gate6_fape_train import fape_loss, H5_PATH

torch.manual_seed(0)
h5 = h5py.File(H5_PATH, "r")
names = list(h5["val"].keys())[:8]
L = 256
ca = torch.zeros(len(names), L, 3); mask = torch.zeros(len(names), L, dtype=torch.bool)
for i, n in enumerate(names):
    c = torch.from_numpy(h5["val"][n]["ca_coords"][:])[:L]
    ca[i, :len(c)] = c; mask[i, :len(c)] = True

def rot(axis, deg):
    a = torch.tensor(axis, dtype=torch.float32); a = a / a.norm()
    t = math.radians(deg); K = torch.tensor([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return torch.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * (K @ K)

ok = True
def check(label, pred, expect_zero):
    global ok
    v = fape_loss(pred, ca, mask, clamp_distance=10.0).item()
    good = (v < 1e-3) if expect_zero else (v > 0.05)
    ok &= good
    print(f"  {label:<28s} FAPE={v:.5f}   {'OK' if good else 'FAIL'}")

print("FAPE invariance guard")
check("identity", ca.clone(), True)
for deg in (30, 90, 180):
    check(f"rotation {deg} deg", ca @ rot([1, 2, 3], deg).T, True)
for t in (1, 5, 20):
    check(f"translation {t} A", ca + torch.tensor([t, -t, 0.5 * t]), True)
check("rot 90 + trans 20", (ca @ rot([0, 1, 0], 90).T) + 20.0, True)
check("1 A gaussian noise (nonzero)", ca + torch.randn_like(ca), False)
check("mirror image (nonzero)", ca * torch.tensor([1., 1., -1.]), False)
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
