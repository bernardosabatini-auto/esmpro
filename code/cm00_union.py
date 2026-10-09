"""cm00: the protein universe of the campaign: every protein measured in any HSP pull-down run
(GA_20, GA_22, GA_24, GA_33, FS73, FS76; pull-downs, supernatants, inputs), with its sequence, and which
of them still need ESMC-6B features (pooled layers 0-80 and the layer-60 SAE).

Sequences: reviewed human UniProt (data/annot/uniprot_human_full.tsv.gz); anything else (isoforms,
unreviewed entries) is fetched from the UniProt REST API once and cached. Writes data/campaign/union.fasta,
missing.fasta and union.tsv (accession, runs it appears in, has_sae, has_layers).
"""
import os, json, gzip, io as _io, urllib.request
import numpy as np, pandas as pd

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
OUT = f"{ROOT}/data/campaign"; os.makedirs(OUT, exist_ok=True)
runs = {}
for n in ["FS73", "FS76", "GA22_24/GA22_ip", "GA22_24/GA22_sup", "GA22_24/GA24_ip", "GA22_24/GA24_sup"]:
    runs[n.split("/")[-1]] = pd.read_csv(f"{ROOT}/data/fs_data/{n}/proteins.tsv", sep="\t", index_col=0).index.astype(str)
runs["GA20"] = pd.Index(pd.read_csv(f"{ROOT}/data/pd_data/targets.tsv", sep="\t")["accession"].astype(str))
runs["GA33"] = pd.Index(pd.Series(np.load(f"{ROOT}/data/ga_data/blocks.npz", allow_pickle=True)["accession"].astype(str)).str.split(";").str[0])
# the GA_33 input (data/ga_data/input_reference.npz) is row-aligned with blocks.npz, so it adds no proteins
U = sorted(set().union(*[set(v) for v in runs.values()]))
print({k: len(v) for k, v in runs.items()}, "union", len(U), flush=True)

up = pd.read_csv(f"{ROOT}/data/annot/uniprot_human_full.tsv.gz", sep="\t", index_col=0, usecols=["Entry", "Sequence"])["Sequence"]
seq = {a: up[a] for a in U if a in up.index}
cache = f"{OUT}/extra_sequences.json"
extra = json.load(open(cache)) if os.path.exists(cache) else {}
need = [a for a in U if a not in seq and a not in extra]
for i in range(0, len(need), 100):
    q = "+OR+".join(f"accession:{a}" for a in need[i:i + 100])
    url = f"https://rest.uniprot.org/uniprotkb/search?query={q}&fields=accession,sequence&format=tsv&size=500&includeIsoform=true"
    try:
        t = pd.read_csv(_io.StringIO(urllib.request.urlopen(url, timeout=60).read().decode()), sep="\t")
        for a, s in zip(t["Entry"], t["Sequence"]): extra[a] = s
    except Exception as ex: print("fetch failed", ex, flush=True)
json.dump(extra, open(cache, "w"))
seq.update({a: extra[a] for a in U if a in extra and a not in seq})
print(f"sequences for {len(seq)} of {len(U)}; missing: {[a for a in U if a not in seq][:20]}", flush=True)

sae = set(np.load(f"{ROOT}/data/sae/sae_l60.npz", allow_pickle=True)["accession"].astype(str))
e2a = pd.read_csv(f"{ROOT}/data/pd_data/targets.tsv", sep="\t").set_index("entry")["accession"].astype(str).to_dict()   # GA_20 file is keyed by entry name
lay = set(e2a.get(x, x) for x in np.load(f"{ROOT}/data/pd_data/embeddings_layers.npz", allow_pickle=True)["accession"].astype(str)) | \
      set(np.load(f"{ROOT}/data/ga_data/embeddings_layers_new.npz", allow_pickle=True)["accession"].astype(str))
T = pd.DataFrame({"accession": U, "has_seq": [a in seq for a in U], "has_sae": [a in sae for a in U], "has_layers": [a in lay for a in U],
                  "length": [len(seq.get(a, "")) for a in U]})
for k, v in runs.items(): T[k] = T["accession"].isin(set(v))
T.to_csv(f"{OUT}/union.tsv", sep="\t", index=False)
with open(f"{OUT}/union.fasta", "w") as f:
    for a in U:
        if a in seq: f.write(f">{a}\n{seq[a]}\n")
miss = T[T.has_seq & ~(T.has_sae & T.has_layers)]
with open(f"{OUT}/missing.fasta", "w") as f:
    for a in miss["accession"]: f.write(f">{a}\n{seq[a]}\n")
print(f"need ESMC features: {len(miss)} proteins, {miss['length'].sum()/1e6:.2f}M residues; no sequence: {int((~T.has_seq).sum())}", flush=True)
