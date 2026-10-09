"""ad15b: is the sub-proportional capture a property of assembly?

Under mass action a protein's capture is linear in its own concentration unless the capturable form is not.
If a co-chaperone binds the free (unassembled) form, and a homo-n-mer assembles with a dissociation constant
below its concentration, the free form grows as concentration^(1/n): 0.5 for dimers, 0.33 for trimers, 0.25
for tetramers. Prediction: monomers are captured proportionally (slope near 1), homo-oligomers less,
falling with order. Classes from UniProt "Subunit structure" (first statement), reviewed human entries.
Slope of the control pull-down level (average of the two baits) on input, by class, 95 % cluster bootstrap.
"""
import os, re, csv, gzip, json
import numpy as np

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
GA = f"{ROOT}/data/ga_data"
blk = np.load(f"{GA}/blocks.npz", allow_pickle=True); G = np.load(f"{GA}/folds_ga05.npz", allow_pickle=True)["groups"]
EN = np.load(f"{GA}/enrichment.npz", allow_pickle=True); inp, con = EN["input"], EN["contaminant"].astype(bool)
st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{GA}/sequences_status.tsv"), delimiter="\t")}
acc = np.array([st.get(str(a), str(a)) for a in blk["accession"]])
rng = np.random.default_rng(0)
sub = {}
with gzip.open(f"{ROOT}/data/annot/uniprot_subunit.tsv.gz", "rt") as fh:
    next(fh)
    for l in fh:
        a, _, t = l.rstrip("\n").partition("\t"); sub[a] = t
ORDER = {"dimer": 2, "trimer": 3, "tetramer": 4, "pentamer": 5, "hexamer": 6, "heptamer": 7, "octamer": 8, "decamer": 10, "dodecamer": 12}


def klass(a):
    t = sub.get(a, "")
    if not t: return "no annotation"
    first = re.split(r"(?<=[a-z\)])\. ", t.replace("SUBUNIT: ", ""), maxsplit=1)[0].lower()
    m = re.search(r"homo(di|tri|tetra|penta|hexa|hepta|octa|deca|dodeca)mer", first)
    if m and "heterodimer" not in first: return f"homo-{m.group(1)}mer"
    if re.search(r"homo-?oligomer|homomultimer", first): return "homo-oligomer (order unstated)"
    if re.search(r"^monomer|\bmonomer\b", first) and "homo" not in first: return "monomer"
    if re.search(r"heterodimer|heterotrimer|heterotetramer|component of|part of|subunit of", first): return "hetero-complex"
    return "other annotation"


cls = np.array([klass(a) for a in acc])


def lev(c, T):
    L = blk[f"br_{c}_0_{T}"]
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(L).sum(1) >= 4, np.nanmean(L, 1), np.nan)


res = {}
for T in ("35", "43"):
    y = (lev("21A", T) + lev("24", T)) / 2
    base = np.isfinite(y) & np.isfinite(inp) & ~con
    print(f"\n{T} C: slope of pull-down level (both baits) on input, by assembly class [95 % cluster bootstrap]", flush=True)
    for k in ["monomer", "homo-dimer", "homo-trimer", "homo-tetramer", "homo-hexamer", "homo-oligomer (order unstated)", "hetero-complex", "other annotation", "no annotation", "all"]:
        m = base & ((cls == k) if k != "all" else True)
        if m.sum() < 30: continue
        b = np.polyfit(inp[m], y[m], 1)[0]
        ug, inv = np.unique(G[m], return_inverse=True); mem = [np.nonzero(inv == q)[0] for q in range(len(ug))]
        xm, ym = inp[m], y[m]; bs = []
        for _ in range(1000):
            s = np.concatenate([mem[q] for q in rng.integers(0, len(mem), len(mem))]); bs.append(np.polyfit(xm[s], ym[s], 1)[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  {k:<32} n {m.sum():>5}  input median {np.median(xm) - np.median(inp[base]):+.2f}  slope {b:+.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)
        res[f"{T}_{k}"] = dict(n=int(m.sum()), slope=[float(b), float(lo), float(hi)], input_median=float(np.median(xm) - np.median(inp[base])))
json.dump(res, open(f"{GA}/ad15b_results.json", "w"), indent=1)
print("\nAD15B_DONE", flush=True)
