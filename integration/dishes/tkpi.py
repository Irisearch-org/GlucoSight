"""Parse TKPI 2017 (Tabel Komposisi Pangan Indonesia) food rows from the PDF.

Input : the TKPI 2017 PDF, converted with ``pdftotext -layout`` (kept OUTSIDE the repo).
Output: ``TkpiEntry`` rows with per-100 g Energi, Protein, Lemak, KH and Serat.

The repo is public and TKPI has no open licence, so no value parsed here is ever
committed; the dish-table build writes them to a gitignored file.

Column order (header of the food tables): Air, Energi, Protein, Lemak, KH, Serat, Abu, ...
Code letter 2: ``P`` = processed/composite (OLAHAN), ``R`` = raw/single (TUNGGAL).

Two layout problems are handled here:

* Names wrap over several lines, with the first line above the code line and the
  continuation below it. Fragments between two code lines are split between the
  row above (continuation) and the row below (prefix) by ``_assign_fragments``.
* An empty Serat cell leaves no token, so Abu would be read as Serat. The gap between
  the right edges of the KH and Serat tokens tells them apart (a missing cell leaves a
  gap of two column pitches).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

CODE_RE = re.compile(r"^\s{0,3}([A-Z]{2}\d{3})(\s|$)")
NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")
SOURCE_RE = re.compile(r"^[A-Z][A-Z0-9]*-?\s*\d{0,4}$")
CHUNK_RE = re.compile(r"\S+(?: \S+)*")
JUNK_TOKEN_RE = re.compile(r"^(KZG[A-Z]*-?|DABM-?|BKP|l?\d{3,4}|\d+/\d+|\d+\.\d*|\d+|-)$")
HEADER_WORDS = ("TUNGGAL", "OLAHAN", "KODE", "NAMA", "MASAKAN", "(g)", "(Kal)", "(mg)", "(mcg)")

# The food tables span these 1-based line numbers of the pdftotext -layout output.
TABLE_START, TABLE_END = 470, 3440
# Distance between right edges of adjacent numeric columns is 6-10 characters;
# a skipped (empty) cell makes it 14-16.
MAX_COLUMN_PITCH = 11

# Names the line-based reassembly cannot recover (wrapped over junk columns).
NAME_FIXES = {
    "CP006": "Kacang kedelai, goreng",
    "CP007": "Kacang kedelai, rebus",
    "CP024": "Emping (kerupuk melinjo)",
    "CP037": "Kembang tahu",
    "CP071": "Tempe gembus P3G",
    "GP085": "Pindang",
}


@dataclass(frozen=True)
class TkpiEntry:
    code: str
    name: str
    energy_kcal: float
    protein_g: float
    fat_g: float
    carbs_g: float
    fibre_g: float | None

    @property
    def is_processed(self) -> bool:
        return self.code[1] == "P"

    def atwater_ok(self, tol: float = 0.30) -> bool:
        """Energy agrees with 4P + 4C + 9F within +-tol (same rule as the CGMacros cleaning)."""
        calc = 4 * self.protein_g + 4 * self.carbs_g + 9 * self.fat_g
        return abs(calc - self.energy_kcal) <= max(tol * self.energy_kcal, 10.0)


def pdf_to_layout_text(pdf: Path) -> str:
    return subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"], check=True, capture_output=True, text=True
    ).stdout


def _is_header(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if s.isupper() or any(w in s for w in HEADER_WORDS):
        return True
    if re.fullmatch(r"[\d.]+", s):  # page numbers and wrapped decimals
        return True
    if "(%)" in s or "Welfare" in s:  # running footer text
        return True
    return False


def _fragment_text(line: str) -> str:
    """The name part of a wrapped-name line (source and junk cells removed)."""
    parts = []
    for chunk in re.split(r"\s{2,}", line.strip()):
        words = [w for w in chunk.split() if not JUNK_TOKEN_RE.match(w)]
        if words:
            parts.append(" ".join(words))
    return " ".join(parts)


def _parse_code_line(raw: str):
    m = CODE_RE.match(raw)
    chunks = [(c.start() + 0, c.end(), c.group()) for c in CHUNK_RE.finditer(raw, m.end())]
    k = next((i for i, c in enumerate(chunks) if NUM_RE.match(c[2]) or c[2] in ("-", "tr")), None)
    name_chunks = [c[2] for c in (chunks[:k] if k is not None else chunks)]
    nums = chunks[k:] if k is not None else []
    if name_chunks and SOURCE_RE.match(name_chunks[-1]) and name_chunks[-1][:1].isupper():
        name_chunks = name_chunks[:-1]
    return m.group(1), " ".join(name_chunks).strip(), nums


def _to_float(s: str) -> float | None:
    return None if s in ("-", "tr") else float(s)


def _fibre(nums) -> float | None:
    if len(nums) < 6:
        return None
    if nums[5][1] - nums[4][1] > MAX_COLUMN_PITCH:  # Serat cell empty: token is Abu
        return None
    return _to_float(nums[5][2])


def _assign_fragments(rows: list[dict], gaps: list[list[str]]) -> None:
    """Distribute name fragments found between code lines to the rows around them.

    A row whose code line has no name started above it; its continuation is the first
    fragment below. A named row only takes leading lower-case fragments. Whatever is
    left is the prefix of the next row.
    """
    prefix: list[str] = []
    for i, row in enumerate(rows):
        row["parts"] = prefix + ([row["name"]] if row["name"] else [])
        frags = gaps[i]
        prefix = []
        if not row["name"]:
            if frags:
                row["parts"].append(frags[0])
                frags = frags[1:]
        else:
            while frags and frags[0][:1].islower():
                row["parts"].append(frags[0])
                frags = frags[1:]
        prefix = frags


def parse_layout(text: str) -> list[TkpiEntry]:
    lines = text.splitlines()[TABLE_START - 1:TABLE_END]
    rows: list[dict] = []
    gaps: list[list[str]] = []
    lead: list[str] = []
    for raw in lines:
        if CODE_RE.match(raw):
            code, name, nums = _parse_code_line(raw)
            rows.append({"code": code, "name": name, "nums": nums})
            gaps.append([])
            continue
        if _is_header(raw):
            continue
        frag = _fragment_text(raw)
        if not frag:
            continue
        (gaps[-1] if rows else lead).append(frag)

    _assign_fragments(rows, gaps)

    entries = []
    for row in rows:
        nums = row["nums"]
        if len(nums) < 5 or not all(NUM_RE.match(n[2]) for n in nums[:5]):
            continue  # not parsed: dropped, never guessed
        water, energy, protein, fat, carbs = (float(n[2]) for n in nums[:5])
        name = NAME_FIXES.get(row["code"]) or re.sub(r"\s+", " ", " ".join(row["parts"])).strip()
        fibre = _fibre(nums)
        if fibre is not None and fibre > carbs:
            fibre = None
        entries.append(TkpiEntry(row["code"], name, energy, protein, fat, carbs, fibre))
    return entries
