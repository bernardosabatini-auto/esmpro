"""Write reports/STATUS.md: one line per training job (running or recently
ended) with its latest evaluation. Run any time; safe alongside training."""
import os, re, glob, subprocess, datetime
from pathlib import Path
ROOT = Path(os.environ["ESM_PROAE_ROOT"])
q = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%i|%j|%P|%T|%M|%b"], capture_output=True, text=True).stdout
state = {l.split("|")[0]: l.split("|") for l in q.strip().splitlines() if l}
rows = []
for log in sorted(glob.glob(str(ROOT / "logs" / "*.out")), key=os.path.getmtime, reverse=True):
    job = re.search(r"_(\d+)\.out$", log).group(1)
    txt = open(log, errors="ignore").read()
    if "esm_proae_train" in log:
        label = re.search(r"(h200_[a-z0-9_]+):", txt); kind = "FAPE head"
        evals = re.findall(r"TM ([0-9.]+)\s+TM>0.5 ([0-9.]+)\s+coverage ([0-9.]+)", txt)
        ep = len(re.findall(r"^\s+\d+\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+s", txt, re.M))
        if not evals: continue
        tm, frac, cov = evals[-1]; rmsd = ""
    elif "latent_flow" in log:
        label = re.search(r"(lf_[A-Za-z0-9_]+):", txt); kind = "latent flow"
        evals = re.findall(r"w=([0-9.]+): TM ([0-9.]+)\s+TM>0.5 ([0-9.]+)\s+RMSD ([0-9.]+).*?coverage ([0-9.]+)", txt)
        ep = len(re.findall(r"^\s+\d+\s+train", txt, re.M))
        if not evals: continue
        w, tm, frac, rmsd, cov = evals[-1]; tm = f"{tm} (w={w})"
    else:
        continue
    if not label: continue
    st = state.get(job)
    status = f"{st[3].lower()} {st[4]} on {st[2]}" if st else "ended"
    if not st and (datetime.datetime.now().timestamp() - os.path.getmtime(log)) > 6 * 3600: continue
    rows.append(f"| {label.group(1)} | {kind} | {job} | {status} | {ep} | {tm} | {float(frac)*100:.0f}% | {rmsd} | {float(cov)*100:.0f}% |")
out = ["# Live status", "", f"Updated {datetime.datetime.now():%Y-%m-%d %H:%M} (cluster time). Latest per-epoch evaluation of each job; "
       "these are selection-set numbers (100 or 200 val proteins), see README for the held-out caveat. "
       "Reference: inherited checkpoint 0.427 / 32% / 11.56 A.", "",
       "| label | kind | job | status | epochs | TM | TM>0.5 | RMSD | coverage |", "|---|---|---|---|---|---|---|---|---|"] + rows
notes = ROOT / "reports" / "status_notes.md"
if notes.exists():
    out += ["", "## Notes (held-out checks of running models)", "", notes.read_text().rstrip()]
(ROOT / "reports" / "STATUS.md").write_text("\n".join(out) + "\n"); print("\n".join(rows))
