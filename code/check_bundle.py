"""Self-check: is this bundle complete and runnable?

Catches the failure that hit twice — a shipped script importing a local module
that was never copied (gate6_100k.py, then canonicalize.py). Both would have
been caught here rather than on the cluster.

    python check_bundle.py            # structure only; works with no env built
    python check_bundle.py --deps     # also check third-party packages

A local module is one with no sibling .py and no entry in THIRD_PARTY — that is
the real bug class. Uninstalled packages are reported separately under --deps,
so a bare machine does not drown the signal.

Exit 0 = structurally complete, 1 = something is missing.
"""
import argparse, ast, importlib.util, os, pathlib, sys

CODE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(os.environ.get("ESM_PROAE_ROOT", CODE.parent))

# Everything this project pulls from the environment rather than the bundle.
THIRD_PARTY = {
    "torch", "numpy", "scipy", "h5py", "einops", "pandas", "tqdm", "loguru",
    "lightning", "hydra", "omegaconf", "transformers", "esm", "Bio", "biotite",
    "matplotlib", "sklearn", "jaxtyping", "beartype", "graphein", "yaml",
    # ProteinAE_v1 is a shipped package, imported as an installed one would be
    "proteinfoundation", "openfold",
}
REQUIRED_DEPS = ["torch", "lightning", "hydra", "h5py", "numpy", "einops", "scipy"]

problems = []


def mark(good, msg, fatal=True):
    if not good and fatal:
        problems.append(msg)
    print(f"  [{'ok ' if good else 'FAIL'}] {msg}")


def imported_names(path):
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as e:
        problems.append(f"{path.name}: syntax error line {e.lineno}")
        print(f"  [FAIL] {path.name}: syntax error line {e.lineno}")
        return set()
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            names.add(n.module.split(".")[0])
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deps", action="store_true",
                    help="also verify third-party packages and foldseek")
    args = ap.parse_args()

    scripts = sorted(p for p in CODE.glob("*.py") if p.stem != "check_bundle")
    shipped = {p.stem for p in scripts}
    print(f"bundle : {CODE}")
    print(f"root   : {ROOT}")
    print(f"modules: {len(scripts)}\n")

    print("local module closure")
    unresolved = {}
    for f in scripts:
        for m in sorted(imported_names(f)):
            if m in shipped or m in THIRD_PARTY or m in sys.stdlib_module_names:
                continue
            if importlib.util.find_spec(m) is not None:
                continue          # installed after all
            unresolved.setdefault(m, []).append(f.name)
    for m, users in sorted(unresolved.items()):
        mark(False, f"'{m}' not shipped and not installed — needed by "
                    f"{', '.join(sorted(users))}")
    if not unresolved:
        mark(True, f"every local import resolves across {len(scripts)} modules")

    print("\nrequired files")
    for rel, why in [("data/phase1_dataset/dataset_100k.h5", "training + scoring"),
                     ("ProteinAE_v1/checkpoints/ae_r1_d8_v1.ckpt", "frozen decoder")]:
        mark((ROOT / rel).exists(), f"{rel}  ({why})")

    print("\noptional files")
    for rel, why in [("data/phase1_dataset/smallscale_final_40k.pt", "warm-start"),
                     ("data/phase1_dataset/backbone_100k.h5", "true-FAPE, off"),
                     ("data/phase1_dataset/structures.tar.gz", "dataset rebuild")]:
        p = ROOT / rel
        print(f"  [{'ok ' if p.exists() else '-- '}] {rel}  ({why})")

    if args.deps:
        print("\nthird-party packages")
        for m in REQUIRED_DEPS:
            mark(importlib.util.find_spec(m) is not None, m)
        from shutil import which
        fs = os.environ.get("FOLDSEEK_BIN", "foldseek")
        mark(which(fs) is not None or pathlib.Path(fs).exists(),
             f"foldseek ({fs}) — required for TM-score")
    else:
        print("\n  (run with --deps once the conda env is built)")

    print()
    if problems:
        print(f"INCOMPLETE — {len(problems)} problem(s)")
        return 1
    print("BUNDLE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
