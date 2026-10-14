#!/usr/bin/env bash
# validate -> LibreOffice -> pdf -> pdftoppm -> <out>/slide-NN.jpg
#
#   bash render.sh <deck_allvisible.pptx> <renders dir>
#
# Render the *_allvisible.pptx (LibreOffice does not export hidden slides); the deliverable
# with hidden backups sits next to it and is validated here too.
set -euo pipefail
DECK=$(readlink -f "$1"); OUT=$2
PY=${PY:-/home/itay/miniforge3/envs/openmc-env/bin/python}
SKILL=${PPTX_SKILL:-$HOME/.claude/skills/pptx/scripts/office}
DELIV=${DECK/_allvisible.pptx/.pptx}

[ -f "$DELIV" ] && "$PY" "$SKILL/validate.py" "$DELIV"
"$PY" "$SKILL/validate.py" "$DECK" > /dev/null

rm -rf "$OUT" && mkdir -p "$OUT"
"$PY" "$SKILL/soffice.py" --headless --convert-to pdf --outdir "$OUT" "$DECK" > /dev/null
PDF="$OUT/$(basename "${DECK%.pptx}").pdf"
pdftoppm -jpeg -r 150 "$PDF" "$OUT/slide"
ls -1 "$OUT"/slide-*.jpg
