"""pdpipe.study: run every level of the HSP campaign's analysis on a new lysate.

    python -m pdpipe.study --study ga22_24 --dry-run      # show the plan, run nothing
    python -m pdpipe.study --study mynew --check          # what the data can and cannot support
    python -m pdpipe.study --study mynew                  # run every level the data support
    python -m pdpipe.study --study mynew --levels targets,ceilings,complex
    python -m pdpipe.study --study mynew --skip structure --resume

A *study* is one or more MS fractions of the same set of reactions. Everything the campaign could do
follows from which fractions exist, so the pipeline detects that first and gates the levels on it:

    ip      the pull-down itself                               REQUIRED
    sup     the post-IP supernatant (what stayed unbound)      enables the bound:free partition
    input   the pre-IP lysate                                  enables enrichment and a true abundance axis
    mock    a BAIT-FREE IP (beads only, or untagged control)   enables specific binding, and the direct
                                                               capture control the campaign never had

Each extra fraction buys a specific thing, and the pipeline says so rather than failing late:

  * with `sup`   -> partition targets, and the measurement-stage gradient of report section 3g, which is
                    what showed the complex effect is mostly a lysate property rather than a capture
                    artefact. Two stages is the minimum for that comparison.
  * with `input` -> enrichment over a matched input, and abundance measured in THIS lysate instead of
                    borrowed from PaxDb. Section 4 showed abundance is the strongest single predictor of a
                    baseline pull-down, so a borrowed abundance is a real loss of power.
  * with `mock`  -> (a) a specificity target, IP minus bait-free IP, which separates genuine binding from
                    bead carry-over in a way neither partition nor enrichment can; (b) a DIRECT test of the
                    capture-nonlinearity hypothesis. A bait-free IP passes over the same beads with no
                    bait, so any complex-level structure it shows is created by capture itself. The
                    campaign had to infer this indirectly from the supernatant/pull-down ordering (3g);
                    one bait-free IP would settle it outright.

Levels, in order. "generic" means it makes no assumption about which experiment this is.

  load        parse each fraction, QC, replicate structure, drift            generic
  targets     per-fraction baselines and contrasts, plus every derived       generic
              fraction the capabilities allow; split-half ceiling for each
  embed       ESMC-6B layer states + SAE for any protein not already cached  generic, GPU
  folds       MMseqs2 30 % identity clustering -> 5 held-out blocks          generic
  predict     out-of-fold ridge on the embedding; optional multi-task MLP    generic, GPU
  properties  named-property baseline (abundance, Tm, keywords) and what     generic
              sequence adds over it
  complex     the complex effect: do complex-mates miss the prediction       generic
              together? (report 3c), with permutation null and bootstrap
  decompose   what the complex effect is made of (report 3g): the            needs >= 2 fractions
              measurement-stage gradient, a flexible abundance control,
              abundance-discordant partners, the co-capture asymmetry
  structure   the three structural tiers (3d AlphaFold, 3e experimental      opt-in; all three were
              interfaces, 3f Boltz-2 co-folding)                             null in the campaign
  report      assemble markdown + figures, render PDF and HTML               generic

`structure` is OFF by default on purpose. All three tiers returned nothing on 33 targets, and 3g then
explained why: they were looking for an interface property, which could only ever have accounted for the
smallest of the four components the complex effect is made of. Turn it on only to test a new hypothesis,
not as routine analysis; `--levels structure` prints the cost before it starts.

Study configs live in configs/studies/<name>.yaml. See configs/studies/TEMPLATE.yaml.
"""
import os, sys, json, argparse, time

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.join(HERE, "configs", "studies")

# fraction key -> (what it is, what it enables)
FRACTIONS = {
    "ip":    "the pull-down (required)",
    "sup":   "post-IP supernatant, the unbound pool",
    "input": "pre-IP lysate",
    "mock":  "bait-free IP (beads only / untagged control)",
}

# derived fraction -> (minuend, subtrahend, capability flag, description)
DERIVED = {
    "part":   ("ip", "sup",   "has_sup",   "bound : free partition (IP - supernatant, same tube)"),
    "enrich": ("ip", "input", "has_input", "enrichment over the matched pre-IP lysate (IP - input)"),
    "spec":   ("ip", "mock",  "has_mock",  "specific binding (IP - bait-free IP, same beads)"),
}

# level -> (requires, description, rough cost)
LEVELS = {
    "load":       ([],                    "parse each fraction, QC, replicate structure", "minutes, CPU"),
    "targets":    (["load"],              "baselines, contrasts, derived fractions, ceilings", "minutes, CPU"),
    "embed":      ([],                    "ESMC-6B + SAE for proteins not already cached", "~1 min/1600 proteins, GPU"),
    "folds":      ([],                    "MMseqs2 30 % clustering into 5 held-out blocks", "minutes, CPU"),
    "predict":    (["targets", "embed", "folds"], "out-of-fold ridge; optional multi-task MLP", "minutes (ridge) to ~10 GPU-h (MLP sweep)"),
    "properties": (["targets", "folds"],  "named-property baseline and what sequence adds", "minutes, CPU"),
    "complex":    (["targets", "predict"], "the complex effect (3c), null and bootstrap", "~10 min, CPU"),
    "decompose":  (["complex"],           "what the complex effect is made of (3g)", "~5 min, CPU"),
    "structure":  (["predict"],           "AlphaFold / experimental interfaces / co-folding (3d-3f)", "HOURS of GPU; all null in the campaign"),
    "report":     (["targets"],           "assemble markdown, figures, PDF and HTML", "minutes, CPU"),
}
DEFAULT_LEVELS = [k for k in LEVELS if k != "structure"]


def load_study(name):
    import yaml
    p = name if os.path.exists(name) else os.path.join(STUDY_DIR, f"{name}.yaml")
    if not os.path.exists(p):
        have = sorted(f[:-5] for f in os.listdir(STUDY_DIR) if f.endswith(".yaml"))
        raise SystemExit(f"no study config {name!r}. Available: {', '.join(have)}")
    cfg = yaml.safe_load(open(p))
    cfg.setdefault("options", {})
    cfg["_path"] = p
    fr = cfg.get("fractions") or {}
    cfg["fractions"] = {k: v for k, v in fr.items() if v}          # drop nulls so absence is unambiguous
    if "ip" not in cfg["fractions"]:
        raise SystemExit(f"{p}: fractions.ip is required (the pull-down itself)")
    unknown = set(cfg["fractions"]) - set(FRACTIONS)
    if unknown:
        raise SystemExit(f"{p}: unknown fraction(s) {sorted(unknown)}; allowed: {sorted(FRACTIONS)}")
    return cfg


def capabilities(cfg):
    """What the declared data can support. Every gate downstream reads this and nothing else."""
    f = cfg["fractions"]
    c = {f"has_{k}": (k in f) for k in FRACTIONS}
    c["n_fractions"] = len(f)
    c["has_abundance"] = c["has_input"] or c["has_sup"]
    c["abundance_source"] = ("input (matched, this lysate)" if c["has_input"] else
                             "supernatant (~ lysate, this experiment)" if c["has_sup"] else
                             "PaxDb (borrowed from other cells -- weaker; see report section 4)")
    c["has_partition"] = c["has_sup"]
    c["has_enrichment"] = c["has_input"]
    c["has_specificity"] = c["has_mock"]
    c["has_stage_gradient"] = c["n_fractions"] >= 2
    c["has_direct_capture_control"] = c["has_mock"]
    c["contrasts"] = list(cfg.get("contrasts") or [])
    c["has_perturbation"] = len(c["contrasts"]) > 0
    c["derived"] = [d for d, (_, _, flag, _) in DERIVED.items() if c[flag]]
    return c


def gate(level, caps, cfg):
    """(runnable, reason). Reason is printed whether or not the level runs."""
    if level == "decompose" and not caps["has_stage_gradient"]:
        return False, ("needs >= 2 fractions of the same reactions; only `ip` is declared. "
                       "Add `sup` (cheapest) or `mock` to enable it.")
    if level == "structure" and not cfg["options"].get("structure", False):
        return False, "opt-in: set options.structure: true, or pass --levels structure"
    if level == "predict" and not caps["has_perturbation"] and caps["n_fractions"] == 1:
        return True, "only baseline targets available (no contrasts, one fraction)"
    return True, ""


def plan(cfg, requested=None, skip=()):
    caps = capabilities(cfg)
    want = list(requested or DEFAULT_LEVELS)
    for lv in want:
        if lv not in LEVELS:
            raise SystemExit(f"unknown level {lv!r}; known: {', '.join(LEVELS)}")
    # pull in prerequisites of anything explicitly requested, preserving the canonical order
    need = set(want)
    changed = True
    while changed:
        changed = False
        for lv in list(need):
            for r in LEVELS[lv][0]:
                if r not in need:
                    need.add(r); changed = True
    rows = []
    for lv in LEVELS:
        if lv not in need:
            rows.append((lv, "not requested", "")); continue
        if lv in skip:
            rows.append((lv, "skipped", "--skip")); continue
        ok, why = gate(lv, caps, cfg)
        rows.append((lv, "run" if ok else "GATED", why))
    return caps, rows


def _print_caps(cfg, caps):
    print(f"\nstudy: {cfg['name']}   ({cfg.get('title', '')})")
    print(f"config: {cfg['_path']}")
    print("\nfractions declared:")
    for k, v in FRACTIONS.items():
        got = cfg["fractions"].get(k)
        print(f"  {k:<6} {'YES  ' + str(got) if got else 'no':<28} {v}")
    print("\nwhat that supports:")
    print(f"  abundance axis          {caps['abundance_source']}")
    print(f"  derived targets         {', '.join(caps['derived']) if caps['derived'] else 'none (ip only)'}")
    print(f"  contrasts               {', '.join(caps['contrasts']) if caps['contrasts'] else 'none (baseline only)'}")
    print(f"  stage gradient (3g)     {'yes, ' + str(caps['n_fractions']) + ' stages' if caps['has_stage_gradient'] else 'NO -- needs a second fraction'}")
    print(f"  direct capture control  {'yes (bait-free IP)' if caps['has_direct_capture_control'] else 'NO -- would need a bait-free IP'}")
    if not caps["has_mock"]:
        print("\n  note: a bait-free IP is the single most informative fraction you do not have.")
        print("        It separates binding from bead carry-over, and it tests the capture-nonlinearity")
        print("        hypothesis directly instead of by the indirect argument of report section 3g.")
    if not caps["has_input"] and caps["has_sup"]:
        print("\n  note: there is no pre-IP lysate, so abundance is taken from the post-IP supernatant.")
        print("        That is the same tube and is a good proxy, but it is the lysate MINUS whatever the")
        print("        bait removed, so it is slightly depleted exactly for the best binders. Enrichment")
        print("        over a true input is unavailable.")
    elif not caps["has_abundance"]:
        print("\n  note: neither a pre-IP lysate nor a supernatant was measured, so abundance is borrowed")
        print("        from PaxDb (other cells, other preparation). Report section 4 found abundance to be")
        print("        the strongest single predictor of a baseline pull-down, so this weakens every")
        print("        baseline model, and both enrichment and the stage gradient are unavailable.")


def _print_plan(rows):
    print(f"\n{'level':<12} {'status':<14} {'cost':<42} note")
    print("-" * 110)
    for lv, status, why in rows:
        cost = LEVELS[lv][2] if status == "run" else ""
        print(f"{lv:<12} {status:<14} {cost:<42} {why}")
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the HSP campaign analysis on a study.")
    ap.add_argument("--study", required=True, help="name under configs/studies/, or a path to a yaml")
    ap.add_argument("--levels", default=None, help="comma-separated; prerequisites are pulled in automatically")
    ap.add_argument("--skip", default="", help="comma-separated levels to skip even if requested")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--check", action="store_true", help="print capabilities and exit")
    ap.add_argument("--resume", action="store_true", help="skip levels whose output already exists")
    ap.add_argument("--list", action="store_true", help="list known levels and exit")
    a = ap.parse_args(argv)

    if a.list:
        for lv, (req, desc, cost) in LEVELS.items():
            print(f"{lv:<12} {desc:<58} [{cost}]" + (f"  requires: {','.join(req)}" if req else ""))
        return 0

    cfg = load_study(a.study)
    want = [x.strip() for x in a.levels.split(",")] if a.levels else None
    skip = {x.strip() for x in a.skip.split(",") if x.strip()}
    caps, rows = plan(cfg, want, skip)

    _print_caps(cfg, caps)
    _print_plan(rows)
    if a.check:
        return 0
    if a.dry_run:
        print("dry run: nothing executed.")
        return 0

    from . import levels as L
    out = os.path.join(ROOT, "data", "study", cfg["name"])
    os.makedirs(out, exist_ok=True)
    json.dump(dict(study=cfg["name"], capabilities={k: v for k, v in caps.items()},
                   plan=[dict(level=lv, status=s, note=w) for lv, s, w in rows]),
              open(os.path.join(out, "plan.json"), "w"), indent=1, default=str)

    state = {}
    for lv, status, why in rows:
        if status != "run":
            continue
        fn = getattr(L, f"level_{lv}", None)
        if fn is None:
            print(f"[{lv}] NOT IMPLEMENTED -- declared in the plan but no level_{lv} in pdpipe/levels.py")
            continue
        marker = os.path.join(out, f".done_{lv}")
        if a.resume and os.path.exists(marker):
            print(f"[{lv}] already done (--resume); delete {marker} to force")
            continue
        print(f"\n{'=' * 100}\n[{lv}] {LEVELS[lv][1]}\n{'=' * 100}", flush=True)
        t0 = time.time()
        state = fn(cfg, caps, out, state) or state
        open(marker, "w").write(f"{time.time()}\n")
        print(f"[{lv}] done in {time.time() - t0:.0f}s", flush=True)
    print(f"\nall requested levels finished. Outputs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
