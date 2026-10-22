"""Read-only audit of paper/ for the Elsevier Ad Hoc Networks submission report."""
import re
import glob
import os
import struct

PAPER = os.path.join(os.path.dirname(__file__), "..", "paper")

def read(p):
    with open(os.path.join(PAPER, p), encoding="utf-8") as f:
        return f.read()

def latex_words(t):
    t = re.sub(r"(?<!\\)%.*", "", t)  # strip comments but keep \% (escaped percent)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    t = re.sub(r"[{}~]", " ", t)
    t = re.sub(r"[$\\]", " ", t)
    return [w for w in t.split() if any(c.isalnum() for c in w)]

abs_t = read("sections/abstract.tex")
print("abstract words:", len(latex_words(abs_t)))

main = read("main.tex")
kw = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", main, re.S)
if kw:
    kws = [k.strip() for k in kw.group(1).split(",") if k.strip()]
    print("keywords count:", len(kws))

n_cites = 0
sec_words = 0
for f in sorted(glob.glob(os.path.join(PAPER, "sections", "*.tex"))):
    t = open(f, encoding="utf-8").read()
    nc = len(re.findall(r"\\cite\b", t))
    n_cites += nc
    sec_words += len(latex_words(t))
    name = os.path.basename(f)
    print("  %-20s words=%5d cites=%d" % (name, len(latex_words(t)), nc))
print("total section words (excl. abstract):", sec_words)
print("total \\cite commands:", n_cites + len(re.findall(r"\\cite\b", main)))

bib = read("bibliography.bib")
entries = re.findall(r"^@(\w+)\s*\{", bib, re.M)
from collections import Counter
print("bib entries:", len(entries), dict(Counter(e.lower() for e in entries)))

# figure formats + resolutions
png = re.findall(r"\\includegraphics[^{]*\{([^}]+)\}", main + "".join(open(f, encoding="utf-8").read() for f in glob.glob(os.path.join(PAPER, "sections", "*.tex"))))
print("included graphics:", len(set(png)))
for g in sorted(set(png)):
    for cand in (g, os.path.join(PAPER, g)):
        if os.path.exists(cand) or os.path.exists(cand + ".png"):
            path = cand if os.path.exists(cand) else cand + ".png"
            info = ""
            if path.lower().endswith(".png"):
                with open(path, "rb") as fh:
                    data = fh.read()
                if data[:8] == b"\x89PNG\r\n\x1a\n":
                    w, h = struct.unpack(">II", data[16:24])
                    dpi = None
                    # find pHYs chunk
                    pos = 8
                    while pos + 8 <= len(data):
                        ln = struct.unpack(">I", data[pos:pos + 4])[0]
                        typ = data[pos + 4:pos + 8]
                        if typ == b"pHYs" and ln == 9:
                            pu, pa, unit = struct.unpack(">IIB", data[pos + 8:pos + 17])
                            dpi = round(pu * 0.0254) if unit == 1 else None
                            break
                        pos += 12 + ln
                    info = "PNG %dx%d%s" % (w, h, (" @%sdpi" % dpi) if dpi else " (no pHYs dpi)")
            print(f"  {g} -> {info}")
            break

# declarations present?
for pat in ["CRediT", "credit", "Declaration of interests", "declaration of interests",
            "Funding", "funding", "generative AI", "Generative AI", "AI-assisted",
            "Highlights", "Highlights", "Cover letter"]:
    hits = [os.path.basename(p) for p in glob.glob(os.path.join(PAPER, "**", "*.tex"), recursive=True)
            if pat in open(p, encoding="utf-8", errors="ignore").read()]
    if hits:
        print(f"contains '{pat}':", hits)
