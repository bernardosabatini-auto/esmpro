"""pdpipe.report: render a markdown write-up to PDF (pdflatex) and self-contained HTML in the house style.

Conventions that keep pandoc 2.0 + pdflatex happy: no Unicode math symbols in the markdown (write
<=, >=, +/-), give wide pipe tables proportional separator dashes, and size figures with {width=..}.
"""
import os, subprocess
import numpy as np

STYLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_style.html")


def render(md, columns=110):
    d, f = os.path.split(os.path.abspath(md)); stem = f[:-3]
    subprocess.run(["pandoc", f, "-o", f"{stem}.pdf", "--pdf-engine=pdflatex", f"--columns={columns}"], cwd=d, check=True)
    subprocess.run(["pandoc", f, "-o", f"{stem}.html", "-s", "--self-contained", "--resource-path=.", "-H", STYLE], cwd=d, check=True)
    return f"{d}/{stem}.pdf", f"{d}/{stem}.html"


def fmt(x, d=2, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)): return "--"
    return f"{x:.{d}%}" if pct else f"{x:.{d}f}"


def table(header, rows, widths=None):
    """Pipe table. widths: relative column widths, turned into separator dash counts so pandoc wraps wide tables."""
    widths = widths or [max(len(str(h)), 3) for h in header]
    sep = "|" + "|".join("-" * max(3, int(w)) for w in widths) + "|"
    out = ["| " + " | ".join(map(str, header)) + " |", sep]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)
