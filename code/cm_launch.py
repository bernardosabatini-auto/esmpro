"""cm_launch: write one sweep's specs and submit them as up to N RTX jobs (cm03.sbatch), never more than
--max-gpus of this campaign's jobs at once (counts only jobs named cm03* of this user).

  python cm_launch.py --sweep s1 [--max-gpus 8] [--dry]

Each sweep is a list of groups (architecture shapes); groups that share inputs go into the same job so the
inputs are loaded once; jobs are balanced by an estimated cost (members x epochs x input width).
"""
import os, json, argparse, subprocess, itertools, copy

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
ap = argparse.ArgumentParser(); ap.add_argument("--sweep", required=True); ap.add_argument("--max-gpus", type=int, default=8)
ap.add_argument("--dry", action="store_true"); ap.add_argument("--extra", default="")
a = ap.parse_args()
HP = {"lr": [3e-4, 1e-3, 3e-3], "wd": [1e-3, 1e-2, 1e-1], "drop": [0.1, 0.3], "in_drop": [0.2, 0.5], "seeds": 2, "epochs": 150, "batch": 512}
ALL = list(range(0, 81, 10))


def sweep(name):
    """returns list of (input_key, spec_inputs, groups)"""
    if name == "s1":   # inputs x architecture
        inputs = {"L50": dict(layers=[50], sae=False), "L80": dict(layers=[80], sae=False), "L50_80": dict(layers=[50, 80], sae=False),
                  "Lall": dict(layers=ALL, sae=False), "SAE": dict(layers=[], sae=True), "L50_SAE": dict(layers=[50], sae=True),
                  "Lall_SAE": dict(layers=ALL, sae=True), "simple": dict(layers=[], sae=False)}
        out = []
        for key, inp in inputs.items():
            groups = []
            for p, s_, h, k in itertools.product([32, 128], [128, 512], [256, 1024], [16, 64]):
                if not inp["layers"] and p == 128: continue
                if not inp["sae"] and s_ == 512: continue
                modes = ["proj", "mix"] if len(inp["layers"]) > 1 else ["proj"]
                for mode in modes:
                    groups.append(dict(name=f"{key}_{mode}_p{p}_s{s_}_h{h}_k{k}", mode=mode, p=p, s=s_, h=h, k=k))
            out.append((key, dict(**inp, simple=True), groups))
        return out
    if name == "s2":   # refine the best input (layers 50+80 and all layers, projection), single- vs multi-task, LORO trunks
        meta = json.load(open(f"{ROOT}/data/campaign/targets_meta.json")); targets = list(meta)
        best = dict(mode="proj", p=128, s=128, h=1024, k=64)
        out = []
        for key, lay in (("L50_80", [50, 80]), ("Lall", ALL)):
            groups = [dict(name=f"{key}_p{p}_h{h}_k{k}_e{e}", mode="proj", p=p, s=128, h=h, k=k, epochs=e)
                      for p, h, k, e in itertools.product([128, 256], [512, 2048], [64, 128, 256], [150, 300])]
            out.append(dict(key=f"refine_{key}", spec=dict(layers=lay, sae=False, simple=True), groups=groups))
        # single-task: same architecture, one target per spec
        for t in targets:
            out.append(dict(key=f"single_{t}", spec=dict(layers=[50, 80], sae=False, simple=True, targets=[t]), groups=[dict(name=f"single_{t}", **best)]))
        # leave-one-run-out trunks (latents saved for the transfer read-out) and the full model (latent + parameters saved)
        runs = sorted({m["run"] for m in meta.values()})
        for r in runs:
            out.append(dict(key=f"loro_{r}", spec=dict(layers=[50, 80], sae=False, simple=True, targets=[t for t in targets if meta[t]["run"] != r]),
                            groups=[dict(name=f"loro_{r}", **best)], extra="--save-latent"))
        out.append(dict(key="full", spec=dict(layers=[50, 80], sae=False, simple=True), groups=[dict(name="full", **best, seeds=4)], extra="--save-latent --save-model"))
        out.append(dict(key="full_sae", spec=dict(layers=[], sae=True, simple=True), groups=[dict(name="full_sae", **{**best, "s": 512}, seeds=4)], extra="--save-latent --save-model"))
        return out
    raise SystemExit(f"unknown sweep {name}")


def cost(inp, g):
    g = {**HP, **g}
    w = (len(inp["layers"]) * 2560 * g["p"] + (16384 * g["s"] if inp["sae"] else 0) + g["h"] * (g["p"] + g["s"] + g["k"])) * g.get("epochs", 150) / 150 * (g.get("seeds", 2) / 2)
    return w


jobs = []
items = [it if isinstance(it, dict) else dict(key=it[0], spec=it[1], groups=it[2]) for it in sweep(a.sweep)]
for it in items:
    key, inp, groups = it["key"], it["spec"], it["groups"]; inp = dict(inp, _extra=it.get("extra", ""))
    # split each input's groups into chunks so no single job dominates
    groups = sorted(groups, key=lambda g: -cost(inp, g)); per = max(1, len(groups) // 2 if len(groups) > 8 else len(groups))
    for i in range(0, len(groups), per): jobs.append((key, inp, groups[i:i + per], sum(cost(inp, g) for g in groups[i:i + per])))
# merge into at most max-gpus jobs, greedy by cost, keeping one input per job where possible
jobs.sort(key=lambda j: -j[3]); bins = [[] for _ in range(min(a.max_gpus, len(jobs)))]; load = [0] * len(bins)
for j in jobs: b = load.index(min(load)); bins[b].append(j); load[b] += j[3]
spec_dir = f"{ROOT}/code/specs/{a.sweep}"; os.makedirs(spec_dir, exist_ok=True)
for bi, b in enumerate(bins):
    for ji, (key, inp, groups, _) in enumerate(b):
        spec = {k: v for k, v in inp.items() if k != "_extra"}; spec.update(defaults=HP, groups=groups); fn = f"{spec_dir}/job{bi}_{ji}_{key}.json"
        json.dump(spec, open(fn, "w"), indent=1)
    # one sbatch per bin: run its specs one after another
    cmds = " && ".join(f"code/cm03_mlp.py --spec code/specs/{a.sweep}/job{bi}_{ji}_{key}.json --out data/campaign/runs/{a.sweep} {inp['_extra']} {a.extra}".strip()
                       for ji, (key, inp, groups, _) in enumerate(b))
    script = f"{ROOT}/slurm/cm_{a.sweep}_job{bi}.sbatch"
    base = open(f"{ROOT}/slurm/cm03.sbatch").read()
    body = base.replace('/n/home08/bsabatini/.conda/envs/proteinae/bin/python -u code/cm03_mlp.py "$@"',
                        " && ".join(f"/n/home08/bsabatini/.conda/envs/proteinae/bin/python -u {c}" for c in cmds.split(" && ")))
    body = body.replace("#SBATCH -J cm03", f"#SBATCH -J cm03_{a.sweep}_{bi}").replace("#SBATCH -t 0-08:00", "#SBATCH -t 0-12:00")
    open(script, "w").write(body)
    n_groups = sum(len(g) for _, _, g, _ in b)
    print(f"job {bi}: {len(b)} specs, {n_groups} groups, cost {load[bi]/1e6:.0f}M -> {script}")
    if not a.dry: print(subprocess.run(["sbatch", script], capture_output=True, text=True).stdout.strip())
