"""Fail if the paper quotes a decimal number that no verified source contains.

Every decimal number in paper/sections/*.md (Indonesian decimal comma, e.g. 28,78)
must appear in NUMBERS.md (our results) or CITATIONS.md (verified external
claims), where it is written with a decimal point. Integers are not checked.
Also fails on PLACEHOLDER keys and [CITATION NEEDED] markers.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = [ROOT / "NUMBERS.md", ROOT / "CITATIONS.md"]

# A decimal written the Indonesian way: digits, comma, digits; not part of a
# list like "1,2,3" (no digit-comma-digit right after).
PAPER_DECIMAL = re.compile(r"(?<![\d,.])(\d+),(\d+)(?![\d]|,\d)")
SOURCE_DECIMAL = re.compile(r"(?<![\d.])\d+\.\d+(?![\d])")


def main() -> int:
    known = set()
    for src in SOURCES:
        if src.exists():
            known.update(SOURCE_DECIMAL.findall(src.read_text(encoding="utf-8")))
    problems = []
    for md in sorted((ROOT / "sections").glob("*.md")):
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            for m in PAPER_DECIMAL.finditer(line):
                value = f"{m.group(1)}.{m.group(2)}"
                if value not in known:
                    problems.append(f"{md.name}:{lineno}: {m.group(0)} not in NUMBERS.md/CITATIONS.md")
            for marker in ("PLACEHOLDER", "[CITATION NEEDED]", "TODO"):
                if marker in line:
                    problems.append(f"{md.name}:{lineno}: contains {marker}")
    print("\n".join(problems) or "check_numbers: OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
