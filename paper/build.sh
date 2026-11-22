#!/usr/bin/env bash
# Build paper/GlucoSight_paper.docx from paper/sections/*.md (in file-name order).
# Fails on unresolved citations or unverified decimal numbers.
set -euo pipefail
cd "$(dirname "$0")"

python3 check_numbers.py

log=$(mktemp)
pandoc metadata.yaml sections/*.md \
  --from markdown+pipe_tables+implicit_figures \
  --citeproc --bibliography references.bib --csl apa.csl \
  --reference-doc reference.docx \
  --resource-path .:figures \
  -o GlucoSight_paper.docx 2>"$log" || { cat "$log"; exit 1; }
if grep -qi "warning" "$log"; then cat "$log"; exit 1; fi
echo "built GlucoSight_paper.docx"
