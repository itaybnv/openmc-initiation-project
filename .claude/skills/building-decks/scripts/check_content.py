#!/usr/bin/env python
"""Text dump of the built deck checked against slides.json.

    python check_content.py <slides.json> <deck.pptx>

Every headline and bullet must be present and in deck order; every slide with notes must
have them. Markup is flattened the way PowerPoint shows it (F_{i} -> Fi).
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

json_path, deck_path = Path(sys.argv[1]), Path(sys.argv[2])
MD = shutil.which("markitdown") or "/home/itay/miniforge3/envs/openmc-env/bin/markitdown"
dump = subprocess.run([MD, str(deck_path)], capture_output=True, text=True, check=True).stdout
deck = json.loads(json_path.read_text(encoding="utf-8"))


def flat(s):
    s = s.replace("**", "")
    s = re.sub(r"[_^]\{([^}]*)\}", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


dump_flat = re.sub(r"\s+", " ", dump)
pos, bad = 0, []
expected = [("title", flat(deck["title"]["title"]))]
if deck["title"].get("name_line"):
    expected.append(("title", deck["title"]["name_line"]))
for s in deck["slides"]:
    expected.append((s["id"], flat(s["headline"])))
    expected.extend((s["id"], flat(b)) for b in s["bullets"])
for sid, t in expected:
    i = dump_flat.find(t, pos)
    if i < 0:
        bad.append((sid, "missing or out of order", t[:80]))
    else:
        pos = i + len(t)
for s in deck["slides"]:
    if s.get("notes"):
        first = re.sub(r"\s+", " ", s["notes"]).strip()[:60]
        if first not in dump_flat:
            bad.append((s["id"], "notes missing", first))
if bad:
    for b in bad:
        print("FAIL", *b)
    sys.exit(1)
print(f"check_content: PASS ({len(expected)} on-slide strings in order, notes present)")
