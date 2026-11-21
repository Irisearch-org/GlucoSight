"""Parse TKPI 2017 (Tabel Komposisi Pangan Indonesia) food rows from the PDF text layer.

Input : pdftotext -layout output of the TKPI 2017 PDF (kept OUTSIDE the repo).
Output: CSV of every food row with per-100 g macros (kept OUTSIDE the repo).

Only the first five numeric columns are used: Air (g), Energi (kkal), Protein (g),
Lemak (g), KH/Karbohidrat (g). Column order was confirmed against the header.
Code letter 2 gives kind: 'P' = processed/composite (OLAHAN), 'R' = raw/single (TUNGGAL).

Names that wrap onto neighbouring lines are reassembled: a non-numeric line that
starts lower-case is a continuation of the previous row; one that starts upper-case
is a prefix of the next row. Heuristic, so rows must be spot-checked.
"""

import csv
import re
import sys
from pathlib import Path

CODE_RE = re.compile(r"^\s{0,3}([A-Z]{2}\d{3})(\s|$)")
NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")
SOURCE_RE = re.compile(r"^[A-Z][A-Z0-9]*-?\s*\d{0,4}$")
HEADER_WORDS = ("TUNGGAL", "OLAHAN", "KODE", "NAMA", "MASAKAN", "(g)", "(Kal)", "(mg)", "(mcg)")

# Food tables span these 1-based line numbers in pdftotext -layout output.
TABLE_START, TABLE_END = 470, 3440


def _is_header(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if s.isupper() or any(w in s for w in HEADER_WORDS):
        return True
    if re.fullmatch(r"[\d.]+", s):  # page numbers and wrapped decimals ("11.")
        return True
    if "(%)" in s or "Welfare" in s:  # running footer text
        return True
    return False


def parse(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()[TABLE_START - 1:TABLE_END]
    rows = []
    pending_prefix = []  # capital-start fragments waiting for their row
    last_row = None
    for raw in lines:
        m = CODE_RE.match(raw)
        if m:
            code = m.group(1)
            rest = raw[m.end():].strip()
            chunks = [c for c in re.split(r"\s{2,}", rest) if c]
            k = next((i for i, c in enumerate(chunks) if NUM_RE.match(c) or c in ("-", "tr")), None)
            name_chunks = chunks[:k] if k is not None else chunks
            nums = chunks[k:] if k is not None else []
            source = None
            if name_chunks and SOURCE_RE.match(name_chunks[-1]) and name_chunks[-1][:1].isupper():
                source = name_chunks[-1]
                name_chunks = name_chunks[:-1]
            name = " ".join(pending_prefix + name_chunks).strip()
            pending_prefix = []
            last_row = {"code": code, "name_parts": [name] if name else [],
                        "numbers": nums, "source": source}
            rows.append(last_row)
            continue
        if _is_header(raw):
            continue
        text = raw.strip()
        if text[0].islower() and last_row is not None:
            last_row["name_parts"].append(text)
        else:
            pending_prefix.append(text)
    return rows


def to_float(s):
    if s is None or s in ("-", "tr"):
        return None
    return float(s)


def main(pdf_text: Path, out_csv: Path):
    rows = parse(pdf_text)
    fields = ["code", "kind", "name", "water_g", "energy_kcal", "protein_g", "fat_g",
              "carbs_g", "source", "parsed_ok"]
    n_ok = 0
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            nums = r["numbers"]
            ok = len(nums) >= 5 and all(NUM_RE.match(x) for x in nums[:5])
            n_ok += ok
            w.writerow({
                "code": r["code"],
                "kind": "dish" if r["code"][1] == "P" else "raw",
                "name": " ".join(r["name_parts"]).strip(),
                "water_g": to_float(nums[0]) if ok else None,
                "energy_kcal": to_float(nums[1]) if ok else None,
                "protein_g": to_float(nums[2]) if ok else None,
                "fat_g": to_float(nums[3]) if ok else None,
                "carbs_g": to_float(nums[4]) if ok else None,
                "source": r["source"] or "",
                "parsed_ok": int(ok),
            })
    print(f"rows={len(rows)} parsed_ok={n_ok} -> {out_csv}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
