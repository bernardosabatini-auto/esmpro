"""Draft a run report (markdown) from a training log plus the results JSON.
Handles both trainers (gate6 FAPE: 'esm_proae_train_<job>.out' /
notes/gate6_<label>_results.json; gate7 flow: 'latent_flow*_<job>.out' /
notes/gate7_<label>_results.json). The draft has the config and the curve;
the interpretation sections are left as TODO for a human/agent pass.

  python make_run_report.py --label h200_ca_bond --job 47785117 [--heldout notes/tm_best_h200_ca_bond_off1000.json]
"""
import os, re, json, argparse, glob
from pathlib import Path
ROOT = Path(os.environ["ESM_PROAE_ROOT"])
ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True); ap.add_argument("--job", required=True)
ap.add_argument("--heldout", default=None, help="json from gate6_score_checkpoint --offset 1000")
ap.add_argument("--gate", default=None, help="json from gate6_score_checkpoint (offset 0)")
ap.add_argument("--out", default=None)
a = ap.parse_args()

logs = glob.glob(str(ROOT / "logs" / f"*_{a.job}.out"))
assert logs, f"no log for job {a.job}"
log = open(logs[0]).read()
flow = "Gate 7" in log
res_path = ROOT / "notes" / (f"gate7_{a.label}_results.json" if flow else f"gate6_{a.label}_results.json")
res = json.load(open(res_path)) if res_path.exists() else {}
if not res:   # the FAPE trainer writes its json only at the end; fall back to the resumable checkpoint
    import torch
    ck = ROOT / "data" / "phase1_dataset" / f"last_{a.label}.ckpt"
    if ck.exists():
        st = torch.load(str(ck), weights_only=False, map_location="cpu")
        res = {"history": st.get("history", []), "arch": st.get("arch"), "best_tm": st.get("best_tm"),
               "best_tm_epoch": st.get("best_tm_epoch"), "best_ep": st.get("best_ep"),
               "best_val": st.get("best_val"), "best_epoch": st.get("best_epoch"),
               "config": {**(st.get("arch") or {}), **(st.get("loss_cfg") or {})}}
cfg = res.get("config", {})
hist = res.get("history", [])

lines = [f"# {a.label} — {'latent flow' if flow else 'FAPE head'} run", "",
         f"**Job** {a.job}. Log `logs/{os.path.basename(logs[0])}`.", ""]
if flow:
    lines += ["## Configuration",
              f"`code/gate7_latent_flow.py`; arch {res.get('arch')}; {res.get('n_params', 0)/1e6:.1f}M params; "
              f"batch {cfg.get('batch_size')} per GPU; lr {cfg.get('lr')}; warmup {cfg.get('warmup')}; "
              f"EMA {cfg.get('ema')}; p_drop {cfg.get('p_drop')}; sample steps {cfg.get('sample_steps')}; "
              f"eval every {cfg.get('eval_every')} on {cfg.get('eval_n')} proteins.", ""]
    m = re.search(r"world (\d+), batch (\d+)/GPU = (\d+) effective", log)
    if m: lines.append(f"Data-parallel world {m.group(1)}, effective batch {m.group(3)}.")
    m = re.search(r"peak GPU=([0-9.]+) GB", log)
    if m: lines.append(f"Peak GPU {m.group(1)} GB.")
    lines += ["", "## Curve", "| epoch | train | val | TM (best w) | TM>0.5 | RMSD | coverage |", "|---|---|---|---|---|---|---|"]
    for r in hist:
        ev = [(k, v) for k, v in r.items() if k.startswith("eval_w")]
        if ev:
            k, v = max(ev, key=lambda kv: kv[1].get("tm", -1))
            lines.append(f"| {r['epoch']} | {r['train']:.4f} | {r['val']:.4f} | {v['tm']:.3f} ({k[6:]}) | "
                         f"{v['tm_frac']*100:.0f}% | {v.get('rmsd', float('nan')):.2f} | {v['coverage']*100:.0f}% |")
        else:
            lines.append(f"| {r['epoch']} | {r['train']:.4f} | {r['val']:.4f} | | | | |")
    lines += ["", f"Best TM {res.get('best_tm')} at epoch {res.get('best_ep')}."]
else:
    lines += ["## Configuration",
              f"`code/gate6_fape_train.py`; head {cfg.get('n_layers')}L d{cfg.get('d_model')}; batch {cfg.get('batch_size')}; "
              f"ODE steps {cfg.get('ode_steps')}; lr {cfg.get('lr')}; clamp {cfg.get('clamp')}; "
              f"frac_unclamped {cfg.get('frac_unclamped')}; normalize_out {cfg.get('normalize_out')}; "
              f"loss {cfg.get('loss', 'ca')}; bb_points {cfg.get('bb_points')}; bond_weight {cfg.get('bond_weight', 0)}; "
              f"fix_end_frames {cfg.get('fix_end_frames', False)}; warm-start {cfg.get('warm_start')}; "
              f"select on {cfg.get('select_on')}, patience {cfg.get('patience')}.", ""]
    m = re.search(r"peak GPU=([0-9.]+) GB", log)
    if m: lines.append(f"Peak GPU {m.group(1)} GB.")
    ep_t = re.findall(r"^\s+\d+\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)s", log, re.M)
    if ep_t: lines.append(f"Epoch time {float(ep_t[-1]):.0f} s.")
    lines += ["", "## Curve (TM on 200 val proteins each epoch)",
              "| epoch | train | val FAPE | TM | TM>0.5 | coverage |" + (" N-CA dev |" if any('val_nca_dev' in r for r in hist) else ""),
              "|---|---|---|---|---|---|" + ("---|" if any('val_nca_dev' in r for r in hist) else "")]
    for r in hist:
        row = (f"| {r['epoch']} | {r['train']:.4f} | {r['val']:.4f} | {r.get('tm_mean', float('nan')):.3f} | "
               f"{r.get('tm_frac_above_0.5', float('nan'))*100:.0f}% | 100% |")
        if 'val_nca_dev' in r: row += f" {r['val_nca_dev']:.3f} |"
        lines.append(row)
    lines += ["", f"Best TM {res.get('best_tm')} at epoch {res.get('best_tm_epoch')}; best val FAPE "
                  f"{res.get('best_val')} at epoch {res.get('best_epoch')}."]
    stop = re.search(r"Early stopping.*", log)
    lines.append(stop.group(0).strip() if stop else "Run did not early-stop (time limit, cancel, or still running).")

def score_block(title, path):
    if not path or not Path(path).exists(): return []
    d = json.load(open(path))
    return ["", f"## {title}",
            f"TM {d['tm_mean']:.3f}, TM>0.5 {d['tm_gt_0.5']*100:.0f}%, TM>0.3 {d['tm_gt_0.3']*100:.0f}%, "
            f"RMSD {d['rmsd_mean']:.2f} A (100 proteins, coverage reported in the log)."]
lines += score_block("Gate set (section 4 proteins; overlaps the selection set)", a.gate)
lines += score_block("Held-out slice (offset 1000; selection-free)", a.heldout)
lines += ["", "## Outcome", "TODO", "", "## What it changed", "TODO", ""]
out = a.out or str(ROOT / "reports" / f"{a.label}.md")
open(out, "w").write("\n".join(lines)); print(f"wrote {out}")
