#!/usr/bin/env python
"""Audit an outline.md against the format and the honesty rules.

    python audit_outline.py <outline.md> [--repo <repo root>]

Prints the ghost deck (headlines in order), then FAIL/WARN lines. Exit 1 on any FAIL.

FAIL: a slide missing Point / On-slide / Talk track (main slides also Likely question);
      a number on a slide that has no row in the Numbers appendix; an appendix source path
      that does not exist; a banned pattern in on-slide text or a talk track.
WARN: more than four bullets, a bullet over 170 characters, on-slide text over 600
      characters, a talk track outside 60 to 150 words, a NEW visual without a spec line
      under "Visuals to make", a slide with no visual.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BANNED = [
    (r"—", "em dash"), (r"!", "exclamation mark"),
    (r"\b(key|crucial|pivotal|vital|robust|novel|seamless|cutting[- ]edge|groundbreaking)\b", "puff word"),
    (r"\b(takeaway|insight|insights|landscape|journey|testament|delve|leverage)\b", "AI vocabulary"),
    (r"\b(it is worth noting|it's worth noting|in summary|in conclusion|importantly|notably)\b", "filler"),
    (r"\b(excited|thrilled|proud|happy to)\b", "sentiment"),
]
ISOTOPE = re.compile(r"\b[A-Z][a-z]?-\d{1,3}\b")   # U-235, B-10, He-3: names, not numbers
NUM = re.compile(r"(?:\+/-\s?)?[-+]?\d[\d ]*(?:\.\d+)?(?:e-?\d+)?(?:\s?%)?")
SLIDE_RE = re.compile(r"^## (Slide (\d+)|Backup (B\d+)) · (.+?)\s*$", re.M)


def parse(text: str):
    heads = list(SLIDE_RE.finditer(text))
    slides = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        end = min(end, _section_start(text, "## Visuals to make", end), _section_start(text, "## Numbers appendix", end))
        body = text[m.end():end]
        sid = m.group(2) or m.group(3)
        fields = {}
        cur = None
        for line in body.splitlines():
            fm = re.match(r"^(Point|On-slide|Visual|Table|Layout|Talk track|Likely question):\s*(.*)$", line)
            if fm:
                cur = fm.group(1)
                fields[cur] = fm.group(2).strip()
            elif cur and line.strip():
                fields[cur] = (fields[cur] + "\n" + line.rstrip()).strip()
        bullets = [b.strip()[2:].strip() for b in fields.get("On-slide", "").splitlines() if b.strip().startswith("- ")]
        slides.append({"id": sid, "backup": sid.startswith("B"), "headline": m.group(4).strip(),
                       "fields": fields, "bullets": bullets})
    return slides


def _section_start(text: str, marker: str, default: int) -> int:
    i = text.find(marker)
    return i if 0 <= i < default else default


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("outline")
    ap.add_argument("--repo", default=None)
    a = ap.parse_args(argv)
    path = Path(a.outline).resolve()
    repo = Path(a.repo).resolve() if a.repo else _find_repo(path)
    text = path.read_text(encoding="utf-8")
    slides = parse(text)
    fails, warns = [], []

    print("Ghost deck:")
    for s in slides:
        print(f"  {s['id']:>3}  {s['headline']}")
    print()

    appendix = text[text.find("## Numbers appendix"):] if "## Numbers appendix" in text else ""
    if not appendix:
        fails.append("no '## Numbers appendix' section")
    appendix_flat = re.sub(r"\s+", " ", appendix)
    visuals_to_make = text[text.find("## Visuals to make"):text.find("## Numbers appendix")] if "## Visuals to make" in text else ""

    for s in slides:
        f = s["fields"]
        for req in ("Point", "On-slide", "Talk track"):
            if not f.get(req):
                fails.append(f"slide {s['id']}: missing {req}")
        if not s["backup"] and not f.get("Likely question"):
            fails.append(f"slide {s['id']}: missing Likely question")
        if not f.get("Visual") and not f.get("Table"):
            warns.append(f"slide {s['id']}: no Visual and no Table")
        if len(s["bullets"]) > 4:
            warns.append(f"slide {s['id']}: {len(s['bullets'])} bullets (max four)")
        for i, b in enumerate(s["bullets"], 1):
            if len(b) > 170:
                warns.append(f"slide {s['id']}: bullet {i} is {len(b)} characters")
        total = sum(len(b) for b in s["bullets"]) + len(s["headline"])
        if total > 600:
            warns.append(f"slide {s['id']}: on-slide text {total} characters (budget 600)")
        words = len(f.get("Talk track", "").split())
        if words and not 60 <= words <= 150:
            warns.append(f"slide {s['id']}: talk track {words} words (60 to 150)")
        vis = f.get("Visual", "")
        if vis.startswith("NEW:") and f"{s['id']}" not in visuals_to_make:
            warns.append(f"slide {s['id']}: NEW visual has no entry under 'Visuals to make'")
        for piece in [s["headline"]] + s["bullets"] + [c for c in (f.get("Table") or "").split("|")]:
            for m in NUM.finditer(ISOTOPE.sub("", piece)):
                tok = re.sub(r"\s+", " ", m.group(0).strip())
                if not re.search(r"\d", tok) or re.fullmatch(r"[-+]?\d", tok) and not re.search(r"\d\s?%", tok):
                    continue   # single digits (slide refs, counts like "two") are not audited
                if tok not in appendix_flat:
                    fails.append(f"slide {s['id']}: number '{tok}' not in the Numbers appendix  ...{piece[max(0, m.start()-20):m.end()+20]}...")
        for where, txt in (("on-slide", " ".join(s["bullets"]) + " " + s["headline"]), ("talk track", f.get("Talk track", ""))):
            for pat, why in BANNED:
                for m in re.finditer(pat, txt, re.I):
                    fails.append(f"slide {s['id']}: {why} in {where}: '{txt[max(0, m.start()-30):m.end()+30].strip()}'")

    for row in re.finditer(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$", appendix, re.M):
        num, used, src = row.groups()
        if num.lower() in ("number", "---") or set(num) <= set("-"):
            continue
        p = re.split(r"[:#]", src.strip(), 1)[0].strip()
        if not p or p.startswith("<"):
            fails.append(f"appendix: '{num}' has no source path")
            continue
        cand = [repo / p, repo / "project" / p, repo / "project" / "analysis" / "outputs" / "presentation" / p,
                repo / "project" / "analysis" / p, path.parent / p]
        if not any(c.exists() for c in cand):
            fails.append(f"appendix: source for '{num}' not found: {p}")

    for w in warns:
        print("WARN", w)
    for f_ in fails:
        print("FAIL", f_)
    print(f"\n{len(slides)} slides, {len(warns)} warnings, {len(fails)} failures")
    sys.exit(1 if fails else 0)


def _find_repo(p: Path) -> Path:
    for q in [p] + list(p.parents):
        if (q / ".git").exists():
            return q
    return p.parent


if __name__ == "__main__":
    main()
