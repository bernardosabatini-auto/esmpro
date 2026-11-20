"""render_report: one markdown report -> PDF (pdflatex) + self-contained HTML.

pdflatex here has no unicode-math support, so the PDF is built from a transliterated copy: superscript
digits and ion charges become real math or pandoc superscripts, and the handful of other non-ASCII
characters are mapped to LaTeX equivalents. The markdown itself keeps the proper characters, and the HTML
is built from it unchanged. The script fails loudly if any non-ASCII character survives the mapping,
because the alternative is a LaTeX error three hundred lines into a log.

  python code/render_report.py reports/additional/additional_text_12.md
"""
import os, re, sys, subprocess, tempfile

SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5",
       "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁺": "+", "⁻": "-"}
PLAIN = [("°C", r"$^\circ$C"), ("µM", r"$\mu$M"), ("µ", r"$\mu$"), ("ρ", r"$\rho$"), ("σ", r"$\sigma$"), ("Δ", r"$\Delta$"),
         ("×", r"$\times$"), ("≥", r"$\ge$"), ("≤", r"$\le$"), ("≈", r"$\approx$"), ("−", "-"),
         ("–", "--"), ("—", "---"), ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"),
         ("…", "..."), ("·", r"$\cdot$"), ("→", r"$\rightarrow$"), ("§", r"\S{}"), ("±", r"$\pm$"), ("≠", r"$\ne$")]


def for_latex(t):
    t = re.sub(r"10([⁻]?)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
               lambda m: "$10^{" + ("-" if m.group(1) else "") + "".join(SUP[c] for c in m.group(2)) + "}$", t)
    t = re.sub(r"\b(Mg|Li|Na|K|Ca|Zn|Mn)([²³]?)([⁺⁻])",
               lambda m: m.group(1) + "^" + ("".join(SUP[c] for c in m.group(2)) if m.group(2) else "")
                         + SUP[m.group(3)] + "^", t)
    for a, b in PLAIN: t = t.replace(a, b)
    for c, v in SUP.items(): t = t.replace(c, v)
    return t


def main(src):
    root = os.environ.get("ESM_PROAE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(src))))
    base = os.path.splitext(src)[0]
    raw = open(src).read()
    nofig = re.sub(r"!\[\*\*Figure \d+\.\*\* ", "![", raw)          # pandoc numbers figures itself
    tex = for_latex(nofig)
    left = sorted({c for c in tex if ord(c) > 127})
    if left:
        sys.exit(f"render_report: no LaTeX mapping for {left} -- add them to PLAIN or SUP")
    rp = f"{os.path.dirname(os.path.abspath(src))}:{root}/reports:{root}"
    with tempfile.TemporaryDirectory() as d:
        a, b = f"{d}/tex.md", f"{d}/html.md"
        open(a, "w").write(tex); open(b, "w").write(nofig)
        subprocess.run(["pandoc", a, "-o", f"{base}.pdf", "--pdf-engine=pdflatex", "--resource-path", rp,
                        "-V", "geometry:margin=1in", "-V", "fontsize=10pt", "-V", "colorlinks=true",
                        "--toc", "--toc-depth=2"], check=True)
        subprocess.run(["pandoc", b, "-o", f"{base}.html", "--self-contained", "--standalone",
                        "--resource-path", rp, "--toc", "--toc-depth=2"], check=True)
    for f in (f"{base}.pdf", f"{base}.html"):
        print(f"  {f}  {os.path.getsize(f):,} bytes")


if __name__ == "__main__":
    main(sys.argv[1])
