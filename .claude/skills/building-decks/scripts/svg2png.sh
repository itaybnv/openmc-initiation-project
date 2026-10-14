#!/usr/bin/env bash
# Rasterise a hand-written SVG for a slide.   bash svg2png.sh in.svg out.png [dpi=300]
set -euo pipefail
IN=$1; OUT=$2; DPI=${3:-300}
CAIRO=${CAIROSVG:-$HOME/.local/bin/cairosvg}
"$CAIRO" "$IN" -o "$OUT" -d "$DPI" -b white
python3 - "$OUT" <<'EOF'
import struct, sys
with open(sys.argv[1], "rb") as f:
    f.seek(16); w, h = struct.unpack(">II", f.read(8))
print(f"{sys.argv[1]}: {w} x {h} px")
EOF
