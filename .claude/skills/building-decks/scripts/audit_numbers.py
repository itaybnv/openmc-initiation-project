#!/usr/bin/env python
"""Every numeral on a slide must appear in the outline, and multi-digit ones in its appendix.

    python audit_numbers.py <outline.md> <slides.json>

Undoes the deck notation (± -> +/-, proper minus -> hyphen, × -> x, sub/superscript markup,
10^-n -> 1e-n) so tokens read as the outline writes them, then checks each numeric token
in headlines, bullets and tables against the whitespace-collapsed outline. Exit 1 and list
every miss. Multi-digit numbers absent from the Numbers appendix are listed as FAIL too:
a number may be quoted on a slide only if its source is recorded.
"""
import json
import re
import sys
from pathlib import Path

outline_path, json_path = Path(sys.argv[1]), Path(sys.argv[2])
outline = outline_path.read_text(encoding="utf-8")
outline_flat = re.sub(r"\s+", " ", outline)
appendix_flat = outline_flat[outline_flat.find("## Numbers appendix"):] if "## Numbers appendix" in outline_flat else ""
deck = json.loads(json_path.read_text(encoding="utf-8"))


def to_outline_ascii(s: str) -> str:
    s = s.replace("**", "")
    s = re.sub(r"[_^]\{([^}]*)\}", r"\1", s)
    s = s.replace("±", "+/-").replace("−", "-").replace("×", "x")
    s = re.sub(r"(\d+(?:\.\d+)?) x 10-(\d+)", r"\1e-\2", s)
    s = re.sub(r"(?<![\d.])10-(\d+)\b", r"1e-\1", s)
    s = s.replace("≥", ">=").replace("≤", "<=")
    return s


ISOTOPE = re.compile(r"\b[A-Z][a-z]?-\d{1,3}\b")   # U-235, B-10, He-3: names, not numbers
NUM = re.compile(r"(?:\+/-\s?)?[-+]?\d[\d ]*(?:\.\d+)?(?:e-?\d+)?(?:\s?%)?")

misses, not_in_appendix, n_tokens = [], [], 0
texts = [("title", deck["title"]["title"]), ("title", deck["title"].get("name_line", ""))]
for s in deck["slides"]:
    texts.append((s["id"], s["headline"]))
    texts.extend((s["id"], b) for b in s["bullets"])
    if s.get("table"):
        texts.extend((s["id"], c) for row in [s["table"]["header"]] + s["table"]["rows"] for c in row)
for sid, t in texts:
    flat = ISOTOPE.sub("", to_outline_ascii(t))
    for m in NUM.finditer(flat):
        tok = re.sub(r"\s+", " ", m.group(0).strip())
        if not re.search(r"\d", tok):
            continue
        n_tokens += 1
        if tok not in outline_flat:
            misses.append((sid, tok, flat[max(0, m.start() - 25):m.end() + 25]))
        elif sid != "title" and tok not in appendix_flat and re.search(r"\d\d|\.", tok):
            not_in_appendix.append((sid, tok))

print(f"{n_tokens} numeric tokens checked against {outline_path.name}")
if not_in_appendix:
    print("FAIL: numbers on slides that have no row in the Numbers appendix:")
    for sid, tok in sorted(set(not_in_appendix)):
        print(f"  slide {sid}: {tok}")
if misses:
    print("FAIL: numbers on slides that do not appear in the outline at all:")
    for sid, tok, ctx in misses:
        print(f"  slide {sid}: '{tok}'   ...{ctx}...")
if misses or not_in_appendix:
    sys.exit(1)
print("audit_numbers: PASS")
