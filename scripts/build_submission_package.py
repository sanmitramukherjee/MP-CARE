#!/usr/bin/env python3
"""Assemble the ready-to-upload Ad Hoc Networks submission bundle in
paper/submission/.

Outputs (everything Editorial Manager asks for at the attach-files step):
  MP-CARE_latex_source.zip   LaTeX source FLATTENED (Elsevier's LaTeX instructions
                             warn EM cannot process bundles with subfolders: the
                             sections/*.tex and figures/*.png files are copied to the
                             zip root and \\input / \\includegraphics paths rewritten).
                             Includes main.bbl so EM need not run BibTeX.
  MP-CARE_manuscript.pdf     copy of the freshly built main.pdf (upload as Manuscript)
  MP-CARE_highlights.docx    5 bullets, <=85 chars each (filename contains "highlights")
  MP-CARE_cover_letter.docx  cover letter
  MP-CARE_biology.docx       <=100-word author bio (guide: editable format)
  MP-CARE_data_zenodo.zip    dataset for the Zenodo deposit (Option C data policy)

Usage:
  python scripts/build_submission_package.py          # full run (recompiles PDF)
  python scripts/build_submission_package.py --skip-build   # reuse existing main.pdf

Requires: tectonic on PATH or at %LOCALAPPDATA%\\tectonic-bin\\tectonic.exe.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = PAPER / "submission"

TITLE = ("MP-CARE: Mobility-Predictive, Communication-Aware Routing with "
         "Event-Triggered Re-planning for Multi-UAV Swarms")

HIGHLIGHTS = [
    "MP-CARE couples flight planning and FANET routing in one feedback loop",
    "Composite route cost with predicted ETX and link expiration time",
    "Event-triggered re-planning with formal dwell-time stability bound",
    "+14.6 to +20.6 pp PDR over A*/GPSR; +29.7 pp over AODV on node failure",
    "Validated in ROS 2/Gazebo; scales to 50 UAVs at 0.082 ms/node/tick",
]

COVER_LEAD = "Dear Editor,"
COVER_BODY = [
    "We are pleased to submit our manuscript \"" + TITLE + "\" to Ad Hoc Networks "
    "for consideration as a research article.",
    "Multi-UAV swarms suffer when flight planning and network routing are decoupled: "
    "planners choose geometrically short paths through radio-damaged regions, and "
    "routing cannot influence trajectories when links degrade. MP-CARE closes this "
    "feedback loop with a composite route cost (predicted ETX, link expiration time), "
    "mobility-predictive link metrics, and event-triggered re-planning with a formal "
    "dwell-time stability bound. In 30-trial randomized campaigns it improves packet "
    "delivery ratio by 14.6-20.6 percentage points over distance-only A* and GPSR/OLSR "
    "under 5-10% injected packet loss, and by 29.7 points over AODV under mid-flight "
    "relay failure, with bounded per-node compute validated on ROS 2 and Gazebo up to "
    "50 UAVs.",
    "The manuscript is original, is not under consideration elsewhere, and all authors "
    "approve its submission. A preprint of this work is posted on SSRN "
    "(https://www.ssrn.com), in line with the journal's preprint policy. The datasets "
    "supporting the results will be deposited in a public archive (Zenodo) with the "
    "DOI given in the Data Availability statement, and the simulation code will be "
    "publicly released upon publication.",
    "Thank you for your consideration.",
]
COVER_SIGNOFF = "Sincerely,\nSanmitra Mukherjee\nCorresponding author\nsanmitra.mukherjee15@gmail.com"

BIO = ("Sanmitra Mukherjee holds a B.Tech. degree in Electronics and Communication "
       "Engineering and is currently an independent researcher. His research interests "
       "include unmanned aerial vehicle (UAV) swarms, flying ad-hoc networks (FANETs), "
       "decentralized communication-aware routing, and dynamic multi-agent system "
       "control.")


# --------------------------------------------------------------- tiny docx
def make_docx(path: Path, paragraphs: list[str]) -> None:
    """Write a minimal but valid .docx (no external deps)."""
    paras = "\n".join(
        f'<w:p><w:r><w:t xml:space="preserve">{escape(p)}</w:t></w:r></w:p>'
        for p in paragraphs
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/'
        'wordprocessingml/2006/main"><w:body>'
        + paras +
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/>'
        '</w:sectPr></w:body></w:document>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType='
        '"application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType='
        '"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '</Types>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/'
        'officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        '</Relationships>'
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)


# ------------------------------------------------------------- flat source
def build_flat_source_zip(zip_path: Path) -> None:
    """Copy main.tex + sections + figures + bib + bbl into one flat zip,
    rewriting \\input{sections/x} -> \\input{x} and
    \\includegraphics{figures/y} -> \\includegraphics{y}."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        main = (PAPER / "main.tex").read_text(encoding="utf-8")
        main = re.sub(r"\\input\{sections/", r"\\input{", main)
        main = re.sub(r"\\includegraphics(\[[^\]]*\])?\{figures/",
                      r"\\includegraphics\1{", main)
        z.writestr("main.tex", main)

        def flatten(tex_src: str) -> str:
            tex_src = re.sub(r"\\input\{sections/", r"\\input{", tex_src)
            return re.sub(r"\\includegraphics(\[[^\]]*\])?\{figures/",
                          r"\\includegraphics\1{", tex_src)

        for tex in sorted((PAPER / "sections").glob("*.tex")):
            # section files may \includegraphics{figures/...} and \input{sections/...}
            z.writestr(tex.name, flatten(tex.read_text(encoding="utf-8")))
        for fig in sorted((PAPER / "figures").glob("*.png")):
            z.write(fig, fig.name)
        z.write(PAPER / "bibliography.bib", "bibliography.bib")
        bbl = PAPER / "main.bbl"
        if bbl.exists():
            z.write(bbl, "main.bbl")


def verify_flat_build(zip_path: Path) -> bool:
    """Extract the flat zip into a temp dir and compile it — proves the exact
    bundle we hand to Editorial Manager is self-contained."""
    tectonic = Path.home() / "AppData/Local/tectonic-bin/tectonic.exe"
    exe = str(tectonic if tectonic.exists() else "tectonic")
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(td)
        try:
            r = subprocess.run([exe, "main.tex"], cwd=td, capture_output=True,
                               text=True, timeout=420)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"  [skip] flat-build verification unavailable: {e}")
            return False
        ok = (Path(td) / "main.pdf").exists() and "error: main.tex" not in (r.stdout + r.stderr)
        print(f"  [{'ok' if ok else 'FAIL'}] flat bundle compiles standalone")
        if not ok:
            tail = (r.stdout + r.stderr).strip().splitlines()[-8:]
            print("\n".join("    " + l for l in tail))
        return ok


def compile_manuscript(skip: bool) -> bool:
    pdf = PAPER / "main.pdf"
    if skip and pdf.exists():
        print(f"[skip-build] reusing {pdf}")
        return True
    tectonic = Path.home() / "AppData/Local/tectonic-bin/tectonic.exe"
    exe = str(tectonic if tectonic.exists() else "tectonic")
    # keep-intermediates so main.bbl exists for the source zip
    for _ in range(2):
        r = subprocess.run([exe, "--keep-intermediates", "main.tex"],
                           cwd=PAPER, capture_output=True, text=True, timeout=420)
        out = r.stdout + r.stderr
        if "error:" in out:
            print(out[-2000:])
            return False
    ok = pdf.exists()
    print(f"[{'ok' if ok else 'FAIL'}] manuscript PDF rebuilt ({pdf.stat().st_size // 1024} KiB)"
          if ok else "[FAIL] no PDF produced")
    return ok


def build_data_zip(zip_path: Path) -> None:
    """Dataset zip for the Zenodo deposit: per-trial CSVs + summary JSONs
    (+ the paper's plot PNGs), no raw ROS/Gazebo logs."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        n = 0
        for pat in ("*.csv", "*.json", "*.png"):
            for f in sorted((ROOT / "data" / "processed").glob(pat)):
                z.write(f, f"data/processed/{f.name}")
                n += 1
        for f in sorted((ROOT / "data" / "raw").glob("*.csv")):
            z.write(f, f"data/raw/{f.name}")
            n += 1
    mb = zip_path.stat().st_size / 1e6
    print(f"[ok] dataset zip: {zip_path.name} ({mb:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true",
                    help="reuse existing paper/main.pdf")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if not compile_manuscript(args.skip_build):
        return 1

    # flat LaTeX source + standalone verification
    src_zip = OUT / "MP-CARE_latex_source.zip"
    build_flat_source_zip(src_zip)
    print(f"[ok] flat source zip: {src_zip.name} "
          f"({src_zip.stat().st_size // 1024} KiB)")
    verify_flat_build(src_zip)

    # manuscript PDF copy
    shutil.copy2(PAPER / "main.pdf", OUT / "MP-CARE_manuscript.pdf")
    print("[ok] MP-CARE_manuscript.pdf")

    # highlights / cover letter / bio as real .docx
    for h in HIGHLIGHTS:
        assert len(h) <= 85, f"highlight too long ({len(h)}): {h}"
    make_docx(OUT / "MP-CARE_highlights.docx",
              [f"\u2022 {h}" for h in HIGHLIGHTS])
    print("[ok] MP-CARE_highlights.docx (5 bullets, all <=85 chars)")

    make_docx(OUT / "MP-CARE_cover_letter.docx",
              [COVER_LEAD, ""] + COVER_BODY + ["", COVER_SIGNOFF])
    print("[ok] MP-CARE_cover_letter.docx")

    bio_words = len(BIO.split())
    assert bio_words <= 100, f"bio is {bio_words} words"
    make_docx(OUT / "MP-CARE_biology.docx", ["Sanmitra Mukherjee", BIO])
    print(f"[ok] MP-CARE_biology.docx ({bio_words} words; passport photo still "
          f"needed from you)")

    build_data_zip(OUT / "MP-CARE_data_zenodo.zip")

    print("\nBundle in paper/submission/:")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:34s} {f.stat().st_size // 1024:>6d} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
