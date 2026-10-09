"""md_tidy: make a pandoc-2.0 Markdown report render cleanly in PDF.
  * a blank line before every list that follows a paragraph line (pandoc 2.0 otherwise runs it inline)
  * table separator rows with dashes proportional to each column's widest cell, so pandoc wraps wide tables
    into relative-width columns instead of letting them run off the page
  python md_tidy.py report.md      (edits in place)
"""
import re, sys
p = sys.argv[1]; L = open(p).read().split("\n"); out = []
for l in L:
    if re.match(r"^(- |\d+\. )", l) and out and out[-1].strip() and not re.match(r"^(\s*- |\s*\d+\. |\s+\S|---|\|)", out[-1]): out.append("")
    out.append(l)
L = out; i = 0
while i < len(L):
    if L[i].startswith("|") and i + 1 < len(L) and re.match(r"^\|[\s\-:|]+\|$", L[i + 1]):
        j = i + 2
        while j < len(L) and L[j].startswith("|"): j += 1
        rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in [L[i]] + L[i + 2:j]]
        nc = len(rows[0]); w = [max(len(r[k]) if k < len(r) else 0 for r in rows) for k in range(nc)]
        tot = sum(w)
        if tot > 70:                                          # only wide tables need relative widths
            d = [max(3, round(60 * x / tot)) for x in w]
            L[i + 1] = "|" + "|".join("-" * k for k in d) + "|"
        i = j
    else: i += 1
open(p, "w").write("\n".join(L)); print("tidied", p)
