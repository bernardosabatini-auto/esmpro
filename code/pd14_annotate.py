"""pd14: fetch the Biohub agent-generated annotations for the SAE features that matter.

GET https://biohub.ai/esm/protein/api/v1alpha1/features/{index} (public, no token; marked alpha).
Only the features pd13 selected, plus a random 500 as a baseline for category frequencies, are
fetched: at most 4 requests in flight, retried with backoff, and cached in feature_annot.jsonl so a
rerun never refetches. Descriptions are LLM-generated hypotheses from activation patterns across
millions of proteins, not curated annotations, and are reported as such.
"""
import os, json, time, sys
import requests
from concurrent.futures import ThreadPoolExecutor

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
SD = f"{ROOT}/data/sae"
CACHE = f"{SD}/feature_annot.jsonl"
URL = "https://biohub.ai/esm/protein/api/v1alpha1/features/{}"
KEEP = ("feature_index", "label", "summary", "category", "activation_pattern", "exemplar_protein_families",
        "threshold", "uniref90_idf", "uniref90_max_activation", "uniref90_frequency", "decoder_nearest_neighbors")

have = {}
if os.path.exists(CACHE):
    for line in open(CACHE):
        d = json.loads(line); have[d["feature_index"]] = d
want = set()
for fn in sys.argv[1:] or (f"{SD}/features_to_annotate.txt", f"{SD}/features_baseline_sample.txt"):
    want |= {int(x) for x in open(fn).read().split()}
todo = sorted(want - set(have))
print(f"{len(want)} wanted, {len(want) - len(todo)} cached, fetching {len(todo)}", flush=True)
S = requests.Session()


def get(i):
    for t in range(6):
        try:
            r = S.get(URL.format(i), timeout=60)
            if r.status_code == 200:
                d = r.json()
                out = {k: d.get(k) for k in KEEP}
                out["top_swissprot"] = [x["uniprot_id"] for x in (d.get("top_swissprot_activations") or [])[:20]]
                return out
            if r.status_code == 404:
                return {"feature_index": i, "label": None, "missing": True}
        except Exception:
            pass
        time.sleep(2 ** t)
    return {"feature_index": i, "label": None, "error": True}


t0 = time.time()
with open(CACHE, "a") as fh, ThreadPoolExecutor(max_workers=4) as ex:
    for n, d in enumerate(ex.map(get, todo), 1):
        fh.write(json.dumps(d) + "\n"); fh.flush()
        if n % 100 == 0:
            print(f"  {n}/{len(todo)}  {n/(time.time()-t0):.1f}/s", flush=True)
bad = sum(1 for line in open(CACHE) if json.loads(line).get("error"))
print(f"done in {(time.time()-t0)/60:.1f} min; cache {sum(1 for _ in open(CACHE))} features, {bad} failed\nPD14_DONE", flush=True)
