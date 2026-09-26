"""Fill the release identifiers and rebuild the PDFs.

  python paper/release_fill.py --repo https://github.com/<account>/cs2rb         # any time
  python paper/release_fill.py --doi 10.5281/zenodo.1234567                    # after reserving the DOI

Replaces {{REPO_URL}}, {{ZENODO_DOI}} (as a https://doi.org link),
{{ZENODO_DOI_BARE}} and {{RELEASE_DATE}} in the public text files and rebuilds
paper/CS2RB.pdf and paper/SSAC27_abstract.pdf. It lists any {{...}} token still
left in the repository and exits non-zero while one remains, so a clean exit
means the release text is complete.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["README.md", "DATASET_CARD.md", "LICENSE-DATA.md", "CITATION.cff", "paper/CS2RB.md", "paper/SSAC27_abstract.md"]
TOKEN = re.compile(r"\{\{[A-Z_]+\}\}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doi", help="bare DOI, e.g. 10.5281/zenodo.1234567")
    ap.add_argument("--repo", help="https URL of the public repository")
    ap.add_argument("--date", default=dt.date.today().isoformat(), help="release date (filled together with --doi)")
    a = ap.parse_args()
    repl = {}
    if a.doi:
        if not re.fullmatch(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", a.doi):
            raise SystemExit(f"not a bare DOI: {a.doi}")
        repl.update({"{{ZENODO_DOI}}": f"https://doi.org/{a.doi}", "{{ZENODO_DOI_BARE}}": a.doi, "{{RELEASE_DATE}}": a.date})
    if a.repo:
        if not re.fullmatch(r"https://[^\s{}]+", a.repo):
            raise SystemExit(f"not an https URL: {a.repo}")
        repl["{{REPO_URL}}"] = a.repo.rstrip("/")
    if not repl:
        ap.error("give --doi and/or --repo")
    for rel in FILES:
        p = ROOT / rel
        if not p.exists():
            continue
        s = p.read_text(encoding="utf-8")
        for k, v in repl.items():
            s = s.replace(k, v)
        p.write_text(s, encoding="utf-8")
    left = []
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in {".md", ".cff", ".txt", ".json"} and ".git" not in p.parts:
            hits = TOKEN.findall(p.read_text(encoding="utf-8", errors="ignore"))
            if hits:
                left.append((str(p.relative_to(ROOT)), sorted(set(hits))))
    for src, out in [("paper/CS2RB.md", "paper/CS2RB.pdf"), ("paper/SSAC27_abstract.md", "paper/SSAC27_abstract.pdf")]:
        if (ROOT / src).exists():
            cmd = [sys.executable, str(ROOT / "paper/build_pdf.py"), "--src", str(ROOT / src), "--out", str(ROOT / out)]
            subprocess.run(cmd + (["--allow-placeholders"] if left else []), check=True)
    if left:
        raise SystemExit(f"release fields still open: {left}")
    print("all release fields filled; PDFs rebuilt")


if __name__ == "__main__":
    main()
