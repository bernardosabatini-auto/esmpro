"""cm10: predicted HSP pull-down behaviour for the whole reviewed human proteome.

Measured proteins (the campaign union) keep their out-of-fold predictions from the multi-task MLP (s2 "full":
ESMC-6B layers 50 + 80 mean-pooled, sequence covariates; each protein predicted by the fold model that never saw
it). The 8,836 reviewed human proteins that no run measured are embedded the same way (cm01) and predicted by the
mean of the five fold models' best members, each trained on 80 % of the measured proteins, so they are as
"held-out" as the measured proteins' out-of-fold predictions. Expected accuracy per target = its out-of-fold R2.

Writes data/campaign/proteome_predictions.tsv (accession, gene, measured flag, 33 predicted targets) and a
figure comparing predicted distributions of measured and unmeasured proteins.
"""
import os, json
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from pdpipe import partition as pt

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae"); C = f"{ROOT}/data/campaign"
FIG = f"{ROOT}/reports/figures/campaign"
meta = json.load(open(f"{C}/targets_meta.json")); names = list(meta)
Mz = np.load(f"{C}/runs/s2/full_model.npz"); folds = sorted({int(k.split("_")[0][1:]) for k in Mz.files})
new = dict(np.load(f"{C}/proteome_rest_features.npz", allow_pickle=True)); acc_new = new["accession"].astype(str)
XL = new["pool"][:, [5, 8]].astype(np.float32)                         # layers 50 and 80
seq, a = {}, None
for l in open(f"{C}/proteome_rest.fasta"):
    if l.startswith(">"): a = l[1:].strip(); seq[a] = []
    else: seq[a].append(l.strip())
seq = {k: "".join(v) for k, v in seq.items()}
# the same sequence covariates as cm02b, in the same column order
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
cols = pd.read_csv(f"{C}/simple.tsv", sep="\t", index_col=0, nrows=1).columns
rows = []
for a_ in acc_new:
    s = seq[a_]; L = len(s); q, pI = pt.charge(s)
    r = dict(log10_length=np.log10(L), charge_per_res=q / L, pI=pI, gravy=np.mean([KD.get(x, 0) for x in s]), disorder_promoting=sum(s.count(x) for x in "ARGQSPEK") / L)
    r.update({f"aa_{x}": s.count(x) / L for x in "ACDEFGHIKLMNPQRSTVWY"}); rows.append(r)
SIM = pd.DataFrame(rows)[cols].values.astype(np.float32)


def predict(f, xl, xp):
    P = {k.split("_", 1)[1]: torch.tensor(Mz[k]) for k in Mz.files if k.startswith(f"f{f}_")}
    xl = (torch.tensor(xl) - P["norm_muL"]) / P["norm_sdL"]; xp = (torch.tensor(xp) - P["norm_muP"]) / P["norm_sdP"]
    hl = torch.einsum("bld,ldp->blp", xl, P["WL"]).reshape(len(xl), -1) + P["bL"]
    h = torch.nn.functional.gelu(torch.cat([hl, xp @ P["WP"] + P["bP"]], -1)); h = torch.nn.functional.gelu(h @ P["W1"] + P["b1"])
    y = (h @ P["W2"] + P["b2"]) @ P["WH"] + P["bH"]
    return (y * P["ys"] + P["ym"]).numpy()


with torch.no_grad():
    Pn = np.mean([predict(f, XL, SIM) for f in folds], 0)
# measured proteins: out-of-fold predictions; check the fold-model forward reproduces them on a few measured proteins
U = pd.read_csv(f"{C}/union.tsv", sep="\t")["accession"].astype(str).tolist()
oof = np.load(f"{C}/runs/s2/full_oof.npy"); r_ = np.load(f"{C}/runs/s2/full_rows.npy") if os.path.exists(f"{C}/runs/s2/full_rows.npy") else None
if r_ is None:
    F = np.load(f"{C}/folds.npz"); Y = np.load(f"{C}/targets.npz", allow_pickle=True)["Y"]; r_ = np.nonzero(F["has_features"] & np.isfinite(Y).any(1))[0]
outer = np.load(f"{C}/folds.npz")["outer"]
Lm = np.load(f"{C}/layers.npy", mmap_mode="r"); Sm = pd.read_csv(f"{C}/simple.tsv", sep="\t", index_col=0).fillna(0).values.astype(np.float32)
chk = r_[:400]; f0 = outer[chk]
with torch.no_grad():
    rep = np.concatenate([predict(f, Lm[chk[f0 == f]][:, [5, 8]].astype(np.float32), Sm[chk[f0 == f]]) for f in folds if (f0 == f).any()])
order = np.concatenate([np.nonzero(f0 == f)[0] for f in folds if (f0 == f).any()])
agree = np.corrcoef(rep.ravel(), oof[:400][order].ravel())[0, 1]
print(f"fold-model forward vs stored out-of-fold predictions (single best member vs top-8 mean): r = {agree:.3f}", flush=True)

meas = pd.DataFrame(oof, columns=names); meas.insert(0, "accession", [U[i] for i in r_]); meas.insert(1, "measured", True)
un = pd.DataFrame(Pn, columns=names); un.insert(0, "accession", acc_new); un.insert(1, "measured", False)
T = pd.concat([meas, un], ignore_index=True)
g = pd.read_csv(f"{ROOT}/data/annot/uniprot_human_full.tsv.gz", sep="\t", index_col=0, usecols=["Entry", "Gene Names (primary)"])["Gene Names (primary)"]
T.insert(1, "gene", T["accession"].map(g)); T.to_csv(f"{C}/proteome_predictions.tsv", sep="\t", index=False)
print(f"{len(T)} proteins ({int(T.measured.sum())} measured, {int((~T.measured).sum())} predicted only)", flush=True)

pick = ["GA22_sup_base", "GA33_A1_enrich", "GA22_part_base", "GA24_ip_salt120", "GA33_A1_heat43", "FS76_mg"]
f, ax = plt.subplots(1, len(pick), figsize=(3.6 * len(pick), 3.2))
for a_, t in zip(ax, pick):
    a_.hist(T.loc[T.measured, t], 60, density=True, alpha=.5, label="measured proteins"); a_.hist(T.loc[~T.measured, t], 60, density=True, alpha=.5, label="never measured")
    a_.set_title(f"{t}\n(expected R2 {json.load(open(f'{C}/runs/s2/full.json'))['r2'][t]:.2f})", fontsize=8); a_.set_xlabel("predicted (log2)", fontsize=8)
ax[0].legend(fontsize=7); f.tight_layout(); f.savefig(f"{FIG}/proteome.png", dpi=140); plt.close(f)
json.dump(dict(n_measured=int(T.measured.sum()), n_unmeasured=int((~T.measured).sum()), forward_check_r=float(agree)), open(f"{C}/cm10.json", "w"))
print("CM10_DONE")
