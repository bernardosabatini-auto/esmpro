"""cm02: one target matrix over every HSP pull-down run, for the campaign models (cm03+).

Rows: the campaign proteins (data/campaign/union.tsv). Columns: per-protein quantities that do NOT involve a
drug, each with its split-half ceiling (Spearman-Brown reliability across proteins, biological-replicate
halves), so a model's R2 can be read against what the measurement allows.

Baselines (no perturbation; log2, each run median-normalised):
  GA20_ip_base              HSPB1 pull-down, no added salt
  GA22_ip_base, GA24_ip_base, FS73_ip_base, FS76_ip_base   HSPB1 pull-downs at the reference condition
  GA33_A1_ip_base, GA33_B11_ip_base                        DNAJA1 / DNAJB11 pull-downs, 35 C
  GA22_sup_base, GA24_sup_base                             post-IP supernatants (~ lysate abundance)
  GA33_input                                               pre-IP lysate (abundance)
  GA22_part_base, GA24_part_base                           bound : free partition (IP - sup, same tube)
  GA33_A1_enrich, GA33_B11_enrich                          35 C pull-down minus input
Effects:
  GA20_salt75, GA20_salt150                                HSPB1, salt vs none
  GA22_/GA24_{ip,sup,part}_salt60 / _salt120               HSPB1, KCl vs none, drug-free arms only
  GA33_{A1,B11}_heat37 / _heat43                           temperature vs 35 C, no staurosporine
  FS76_mg                                                  2.5 vs 0.25 mM Mg2+ (averaged over Li doses: Li has no
                                                           proteome-wide effect, reliability 0.14, so this gains
                                                           precision without importing a drug effect)
Drug arms (staurosporine, NAD/NADP, Li, anticonvulsants) are left out entirely. GA_22 / GA_24 use only the
drug-free arms (staurosporine 0; no cofactor), because the drug effects there are large on their targets.
GA_22 / GA_24 pull-down and partition models carry the quadratic injection-drift term (fs07).
"""
import os, json, warnings
import numpy as np, pandas as pd
from pdpipe import io, stats as st, partition as pt

warnings.filterwarnings("ignore")
ROOT = io.ROOT; OUT = f"{ROOT}/data/campaign"
U = pd.read_csv(f"{OUT}/union.tsv", sep="\t")["accession"].astype(str).tolist(); idx = {a: i for i, a in enumerate(U)}
Y, META = {}, {}


def put(name, s, meta):
    """s: Series indexed by accession."""
    v = np.full(len(U), np.nan)
    s = s.dropna(); s.index = [str(a).split(";")[0] for a in s.index]; s = s[~s.index.duplicated()]
    for a, x in s.items():
        if a in idx: v[idx[a]] = x
    Y[name] = v; META[name] = {**meta, "n": int(np.isfinite(v).sum())}
    print(f"{name:22} n {META[name]['n']:5}  sd {np.nanstd(v):.3f}  ceiling {meta.get('ceiling', np.nan):.3f}", flush=True)


def sb(r): return 2 * r / (1 + r) if r > -1 else np.nan


# ---------------------------------------------------------------- GA_20 (replicate blocks, 5 BR)
b = np.load(f"{ROOT}/data/pd_data/blocks.npz", allow_pickle=True); acc = pd.Index(b["accession"].astype(str))


def rel_blocks(f, mats, min_real=3):
    """Split-half reliability of f(*[mean over BR subset of each matrix]) across proteins."""
    nb = mats[0].shape[1]; rs = []
    for A in st._splits(list(range(nb))):
        A = sorted(A); B = [k for k in range(nb) if k not in A]
        a = f(*[np.nanmean(m[:, A], 1) for m in mats]); c = f(*[np.nanmean(m[:, B], 1) for m in mats])
        ok = np.isfinite(a) & np.isfinite(c); rs.append(np.corrcoef(a[ok], c[ok])[0, 1])
    return float(sb(np.mean(rs)))


def enough(*mats, k=3): return np.all([np.isfinite(m).sum(1) >= k for m in mats], 0)


ok = enough(b["br_base"])
put("GA20_ip_base", pd.Series(np.where(ok, np.nanmean(b["br_base"], 1), np.nan), index=acc),
    dict(run="GA20", bait="HSPB1", kind="baseline", what="pull-down level, no salt", ceiling=rel_blocks(lambda x: x, [b["br_base"]])))
for k in ("75", "150"):
    put(f"GA20_salt{k}", pd.Series(np.where(b[f"keep_s{k}"], b[f"y_s{k}"], np.nan), index=acc),
        dict(run="GA20", bait="HSPB1", kind="salt", what=f"salt {k} mM vs none", ceiling=rel_blocks(lambda x, y: x - y, [b[f"br_s{k}"], b["br_base"]])))

# ---------------------------------------------------------------- GA_33 (6 BR; input reference)
g = np.load(f"{ROOT}/data/ga_data/blocks.npz", allow_pickle=True); gacc = pd.Index(g["accession"].astype(str))
inp = np.load(f"{ROOT}/data/ga_data/input_reference.npz", allow_pickle=True); en = np.load(f"{ROOT}/data/ga_data/enrichment.npz", allow_pickle=True)
cont = inp["contaminant"]
lvl = np.where(cont, np.nan, inp["level"])
put("GA33_input", pd.Series(lvl, index=gacc), dict(run="GA33", bait="none", kind="abundance", what="pre-IP lysate level (reference input)",
                                                   ceiling=float(sb(np.nanmean([np.corrcoef(*[np.nan_to_num(inp['matched'][:, j]) for j in pair])[0, 1] for pair in [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]])))))
for bait, nm in (("21A", "A1"), ("24", "B11")):
    m35 = g[f"br_{bait}_0_35"]; ok = enough(m35) & ~cont
    put(f"GA33_{nm}_ip_base", pd.Series(np.where(ok, np.nanmean(m35, 1), np.nan), index=gacc),
        dict(run="GA33", bait="DNAJA1" if bait == "21A" else "DNAJB11", kind="baseline", what="pull-down level, 35 C", ceiling=rel_blocks(lambda x: x, [m35])))
    e = np.where(ok & np.isfinite(lvl), en[f"e_{bait}_0_35"], np.nan)
    put(f"GA33_{nm}_enrich", pd.Series(e, index=gacc), dict(run="GA33", bait="DNAJA1" if bait == "21A" else "DNAJB11", kind="enrichment",
                                                           what="35 C pull-down minus input", ceiling=rel_blocks(lambda x: x - lvl, [m35])))
    for t in ("37", "43"):
        y = np.where(g[f"keep_temp_{bait}_{t}v35"] & ~cont, g[f"y_temp_{bait}_{t}v35"], np.nan)
        put(f"GA33_{nm}_heat{t}", pd.Series(y, index=gacc), dict(run="GA33", bait="DNAJA1" if bait == "21A" else "DNAJB11", kind="temperature",
                                                                what=f"{t} vs 35 C", ceiling=rel_blocks(lambda x, z: x - z, [g[f"br_{bait}_0_{t}"], m35])))

# ---------------------------------------------------------------- pdpipe runs: FS73, FS76, GA22, GA24


def drift_fit(ds, drift):
    cov = None
    if drift:
        cov = st.drift_basis(ds.samples["inj_pos"], 2)
    D, names = st.design(ds.samples, block=True, covars=cov)
    f = st.ebayes(st.lmfit(ds.X, D, covar=[names.index(c) for c in cov] if cov is not None else None))
    adj = None
    if drift:
        H = cov.copy(); H.index = ds.samples["column"].values; adj = st.drift_adjuster(H)
    return f, names, adj


def cell(f, names, c): return pd.Series(f["beta"][:, names.index(c)], index=f["index"])


def contrast(f, names, w):
    c = np.zeros(len(names))
    for k, v in w.items(): c[names.index(k)] += v
    return st.contrast(f, c)["logFC"]


def detected(ds, conds, k=3):
    """proteins with >= k values in every listed condition"""
    S = ds.samples; return np.all([ds.X[S.loc[S["cond"] == c, "column"]].notna().sum(1) >= k for c in conds], 0)


fs73 = io.load(io.load_config("fs73")); f, nm, _ = drift_fit(fs73, False)
ok = detected(fs73, ["ctl|0"])
put("FS73_ip_base", cell(f, nm, "ctl|0").where(ok), dict(run="FS73", bait="HSPB1", kind="baseline", what="pull-down level, no drug",
                                                         ceiling=sb(st.split_half(fs73.X, fs73.samples, {"ctl|0": 1})["r_half"])))
fs76 = io.load(io.load_config("fs76")); f, nm, _ = drift_fit(fs76, False)
ok = detected(fs76, ["0.25|0"])
put("FS76_ip_base", cell(f, nm, "0.25|0").where(ok), dict(run="FS76", bait="HSPB1", kind="baseline", what="pull-down level, 0.25 mM Mg, no Li",
                                                          ceiling=sb(st.split_half(fs76.X, fs76.samples, {"0.25|0": 1})["r_half"])))
D = ["0", "0.3", "1", "3", "5", "10"]; MG = {**{f"2.5|{d}": 1 / 6 for d in D}, **{f"0.25|{d}": -1 / 6 for d in D}}
ok = detected(fs76, [f"{m}|{d}" for m in ("0.25", "2.5") for d in D], k=2)
put("FS76_mg", contrast(f, nm, MG).where(ok), dict(run="FS76", bait="HSPB1", kind="Mg", what="2.5 vs 0.25 mM Mg2+",
                                                    ceiling=sb(st.split_half(fs76.X, fs76.samples, MG)["r_half"])))

for e, ref in (("GA22", "0"), ("GA24", "CTRL|0")):
    ip, sup = io.load(io.load_config(e.lower())), io.load(io.load_config(e.lower() + "s"))
    keep = lambda ds: ds.subset((ds.samples["cond"].str.endswith(f"|{ref}") if e == "GA22" else ds.samples["cond"].str.startswith(f"{ref}|")).values)
    D3 = {"ip": keep(ip), "sup": keep(sup)}; D3["part"] = pt.pair(D3["ip"], D3["sup"])
    c = (lambda k: f"{k}|{ref}") if e == "GA22" else (lambda k: f"{ref}|{k}")
    for kind, ds in D3.items():
        f, nm, adj = drift_fit(ds, kind != "sup" or e == "GA24")
        ok = detected(ds, [c("0")])
        what = {"ip": "pull-down level", "sup": "supernatant level (~ lysate)", "part": "bound : free partition"}[kind]
        put(f"{e}_{kind}_base", cell(f, nm, c("0")).where(ok),
            dict(run=e, bait="HSPB1", kind={"ip": "baseline", "sup": "abundance", "part": "enrichment"}[kind], what=f"{what}, no KCl, no drug",
                 ceiling=sb(st.split_half(ds.X, ds.samples, {c("0"): 1}, adjust=adj)["r_half"])))
        for k in ("60", "120"):
            w = {c(k): 1, c("0"): -1}; ok = detected(ds, [c(k), c("0")])
            put(f"{e}_{kind}_salt{k}", contrast(f, nm, w).where(ok),
                dict(run=e, bait="HSPB1", kind="salt", what=f"{what}, KCl {k} vs 0", ceiling=sb(st.split_half(ds.X, ds.samples, w, adjust=adj)["r_half"])))

names = list(Y)
np.savez(f"{OUT}/targets.npz", accession=np.array(U), Y=np.stack([Y[n] for n in names], 1).astype(np.float32), names=np.array(names))
json.dump(META, open(f"{OUT}/targets_meta.json", "w"), indent=1, default=float)
print(f"{len(names)} targets x {len(U)} proteins -> data/campaign/targets.npz\nCM02_DONE")
