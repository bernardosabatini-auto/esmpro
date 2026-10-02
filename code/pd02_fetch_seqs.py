"""pd02: fetch and cache UniProt sequences for the proteins in the HSPB1 salt experiment.

Column A of the workbook holds UniProt accessions, but the same find/replace that turned
missing values into zeros also ate "NA" out of eight accessions. The repair is deterministic:
put "NA" back where the "0" is and keep the candidate UniProt recognises. For example
`Q8072` -> `Q8NA72`, `Q9608` -> `Q96NA8`. Each repair is confirmed against the gene symbol
returned by UniProt, not assumed.

Everything is cached, so re-runs cost nothing and the mapping is auditable.

  python pd02_fetch_seqs.py
  python pd02_fetch_seqs.py --force      # ignore the cache
"""
import os, sys, csv, time, json, random, argparse, re
import requests

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
ap = argparse.ArgumentParser()
ap.add_argument("--targets", default=f"{ROOT}/data/pd_data/targets.tsv")
ap.add_argument("--out-dir", default=f"{ROOT}/data/pd_data")
ap.add_argument("--chunk", type=int, default=250)
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
FASTA = f"{a.out_dir}/sequences.fasta"
STATUS = f"{a.out_dir}/sequences_status.tsv"
ACC_RE = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")

S = requests.Session()
S.headers["User-Agent"] = "esm_proae-pd02 (research)"


def get(url, params=None, tries=5):
    """UniProt with exponential backoff; the pattern mirrors code/gate15_build_pdb.py."""
    for k in range(tries):
        try:
            r = S.get(url, params=params, timeout=120)
            if r.status_code == 200:
                return r.text
            if r.status_code in (400, 404):
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 ** k + random.random())
                continue
            return None
        except requests.RequestException:
            if k == tries - 1:
                return None
            time.sleep(1.5 ** k + random.random())
    return None


def parse_fasta(text):
    """UniProt FASTA -> {accession: (sequence, gene, entry_name)}."""
    out, acc, gene, entry, buf = {}, None, "", "", []
    for line in (text or "").splitlines():
        if line.startswith(">"):
            if acc:
                out[acc] = ("".join(buf), gene, entry)
            buf = []
            # Two header forms reach here. UniProt's own:  >sp|P24752|THIL_HUMAN ... GN=ACAT1
            # and this script's cache, written below:     >P24752|THIL_HUMAN|GN=ACAT1
            # Reading the cache as UniProt form keys every sequence by its entry name, so no
            # cached sequence ever matches and the whole set is silently re-fetched.
            parts = line[1:].split("|")
            if parts[0] in ("sp", "tr") and len(parts) > 2:
                acc, rest = parts[1], parts[2]
            else:
                acc, rest = parts[0].split()[0], (parts[1] if len(parts) > 1 else "")
            entry = rest.split()[0] if rest else ""
            m = re.search(r"\bGN=([^\s]+)", line)
            gene = m.group(1) if m else ""
        else:
            buf.append(line.strip())
    if acc:
        out[acc] = ("".join(buf), gene, entry)
    return out


def repair_candidates(acc):
    """The replace deleted 'NA' and left '0'. Reinsert it at each 0 and keep plausible forms."""
    out = []
    for i, ch in enumerate(acc):
        if ch == "0":
            cand = acc[:i] + "NA" + acc[i + 1:]
            if ACC_RE.match(cand):
                out.append(cand)
    return out


# ---- read the accession list -------------------------------------------------
rows = list(csv.DictReader(open(a.targets), delimiter="\t"))
wanted, raw_of, gene_of = [], {}, {}
for r in rows:
    acc = r["accession"]
    wanted.append(acc)
    raw_of[acc] = r["accession_raw"]
    gene_of[acc] = r["gene"]
print(f"{len(wanted)} accessions from {os.path.basename(a.targets)} "
      f"({sum(1 for v in raw_of.values() if ';' in v)} protein groups, first accession used)", flush=True)

cache = {}
if os.path.exists(FASTA) and not a.force:
    cache = parse_fasta(open(FASTA).read())
    print(f"  cache holds {len(cache)} sequences", flush=True)

need = [acc for acc in wanted if acc not in cache]
malformed = [acc for acc in need if not ACC_RE.match(acc)]
print(f"  {len(need)} to fetch, of which {len(malformed)} are malformed: {malformed}", flush=True)

# ---- batch fetch the well-formed ones ---------------------------------------
got, resolved, status = dict(cache), {}, {}
todo = [x for x in need if x not in malformed]
for s in range(0, len(todo), a.chunk):
    batch = todo[s:s + a.chunk]
    txt = get("https://rest.uniprot.org/uniprotkb/accessions",
              {"accessions": ",".join(batch), "format": "fasta"})
    found = parse_fasta(txt)
    got.update(found)
    for acc in batch:
        if acc in found:
            resolved[acc] = acc
            status[acc] = "direct"
    print(f"  batch {s//a.chunk + 1}: asked {len(batch)}, got {len(found)}, "
          f"running total {len(got)}", flush=True)
    time.sleep(0.2)

# ---- repair the malformed accessions ----------------------------------------
print("\n-- repairing accessions damaged by the NA->0 replace", flush=True)
for acc in malformed:
    cands = repair_candidates(acc)
    fixed = None
    for c in cands:
        txt = get(f"https://rest.uniprot.org/uniprotkb/{c}.fasta")
        if not txt:
            continue
        d = parse_fasta(txt)
        if c not in d:
            continue
        ug = d[c][1]
        sheet_gene = gene_of.get(acc, "")
        # the sheet's gene symbol is damaged the same way, so compare with NA reinserted too
        gene_ok = (ug == sheet_gene) or any(ug == sheet_gene[:i] + "NA" + sheet_gene[i + 1:]
                                            for i, ch in enumerate(sheet_gene) if ch == "0")
        print(f"   {acc} -> {c}: UniProt gene {ug!r}, sheet gene {sheet_gene!r} "
              f"{'CONFIRMED' if gene_ok else 'gene mismatch, rejected'}", flush=True)
        if gene_ok:
            got[c] = d[c]
            resolved[acc] = c
            status[acc] = "repaired"
            fixed = c
            break
    if not fixed:
        status[acc] = "unrepaired"
        print(f"   {acc}: no candidate confirmed (tried {cands})", flush=True)

# ---- anything still missing: try the search endpoint by gene symbol ---------
still = [acc for acc in wanted if acc not in resolved and acc not in got]
if still:
    print(f"\n-- {len(still)} accessions not found directly; trying secondary/obsolete lookup", flush=True)
    for acc in still:
        txt = get(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta")
        if txt:
            d = parse_fasta(txt)
            if d:
                k = next(iter(d))
                got[k] = d[k]
                resolved[acc] = k
                status[acc] = "redirected" if k != acc else "direct"
                continue
        status[acc] = "notfound"
    nf = [x for x in still if status.get(x) == "notfound"]
    print(f"   recovered {len(still)-len(nf)}, still missing {len(nf)}: {nf[:20]}", flush=True)

for acc in wanted:
    if acc in got and acc not in resolved:
        resolved[acc] = acc
        status.setdefault(acc, "cached")

# ---- write -------------------------------------------------------------------
with open(FASTA, "w") as fh:
    for acc in sorted(got):
        seq, gene, entry = got[acc]
        fh.write(f">{acc}|{entry}|GN={gene}\n")
        for i in range(0, len(seq), 60):
            fh.write(seq[i:i + 60] + "\n")

with open(STATUS, "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["sheet_accession", "resolved_accession", "status", "length", "gene_uniprot", "gene_sheet"])
    for acc in wanted:
        r = resolved.get(acc, "")
        seq, gene, _entry = got.get(r, ("", "", ""))
        w.writerow([acc, r, status.get(acc, "notfound"), len(seq), gene, gene_of.get(acc, "")])

have = [acc for acc in wanted if resolved.get(acc) and got.get(resolved[acc], ("",))[0]]
lens = sorted(len(got[resolved[acc]][0]) for acc in have)
from collections import Counter
print(f"\n-- coverage: {len(have)}/{len(wanted)} ({100*len(have)/len(wanted):.1f}%) have a sequence", flush=True)
print(f"   status: {dict(Counter(status.get(x,'notfound') for x in wanted))}", flush=True)
if lens:
    import numpy as np
    print(f"   length: median {int(np.median(lens))}, 90th pct {int(np.percentile(lens,90))}, max {lens[-1]}", flush=True)
    print(f"   over 2048 residues (the ESMC window): {sum(1 for L in lens if L > 2048)}", flush=True)
print(f"\nwrote {FASTA}\nwrote {STATUS}", flush=True)
print("PD02_DONE")
