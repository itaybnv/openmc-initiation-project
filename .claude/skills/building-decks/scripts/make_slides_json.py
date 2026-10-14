#!/usr/bin/env python
"""outline.md -> slides.json (+ figures/ with slide-numbered copies).

    python make_slides_json.py <folder>/outline.md

Parses the outline format of writing-talk-outlines/references/outline-format.md, converts
the ASCII notation to deck notation (±, −, ×, Greek, sub/superscript runs), copies every
figure a slide uses into <folder>/figures/ as s<id>_<k>_<basename>, and writes
<folder>/slides.json for build_deck.js. Stops on a slide whose Visual is still a NEW: spec.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

SLIDE_RE = re.compile(r"^## (Slide (\d+)|Backup (B\d+)) · (.+?)\s*$", re.M)
FIELD_RE = re.compile(r"^(Point|On-slide|Visual|Table|Layout|Stage|Talk track|Likely question):\s*(.*)$")


def convert(s: str) -> str:
    """Outline ASCII -> deck notation. Order matters."""
    s = s.replace("`", "")
    rules = [
        (r"\b1e-(\d+)\b", r"10^{−\1}"),
        (r"(\d+(?:\.\d+)?)e-(\d+)\b", r"\1 × 10^{−\2}"),
        (r"\+/-", "±"),
        (r"(?<=[\s(\[=:/,])-(?=\d)", "−"),
        (r"^-(?=\d)", "−"),
        (r"(?<=\s)-(?=\s)", "−"),
        (r"\^-1\b", "^{−1}"),
        (r"(?<=\d)x\b", "×"),
        (r"(?<=\d) x (?=\d)", " × "),
        (r"\bsigma_alpha\b", "σ_{α}"), (r"\bsigma_f\b", "σ_{f}"), (r"\bsigma_t\b", "σ_{t}"),
        (r"\bsigma_s\b", "σ_{s}"), (r"\bsigma_a\b", "σ_{a}"), (r"\bnu_bar\b", "ν̄"),
        (r"\bnu_t\b", "ν_{t}"), (r"\bbeta_eff\b", "β_{eff}"), (r"\balpha_PK\b", "α_{PK}"),
        (r"\bk_inf\b", "k_{inf}"), (r"\bk_eff\b", "k_{eff}"), (r"\bM_sim\b", "M_{sim}"),
        (r"\bM_emp\b", "M_{emp}"), (r"\bF_tot\b", "F_{tot}"), (r"\bF_i\b", "F_{i}"),
        (r"\bg_i\b", "g_{i}"), (r"\bG_i\b", "G_{i}"), (r"\bsum_i\b", "Σ_{i}"), (r"\bsum_t\b", "Σ_{t}"),
        (r"<nu\(nu [-−] 1\)>", "⟨ν(ν − 1)⟩"), (r"\bchi2\b", "χ^{2}"),
        (r"\balpha\b", "α"), (r"\btau\b", "τ"), (r"\bsigma\b", "σ"), (r"\bmu\b", "μ"),
        (r"\bnu\b", "ν"), (r"\bxi\b", "ξ"), (r"\btheta\b", "θ"), (r"\bLambda\b", "Λ"),
        (r"\bepsilon\b", "ε"), (r"\beps\b", "ε"),
        (r"\^\(1/3\)", "^{1/3}"), (r"\^(\d)\b", r"^{\1}"), (r"\bcm3\b", "cm^{3}"),
        (r">=", "≥"), (r"<=", "≤"), (r"(?<=\d) us\b", " μs"), (r"\s->\s", " → "),
        # no line break between a number and its unit or sign
        (r"(?<=\d) (?=%|keV|MeV|eV|ms|μs|cm|mm|atom|histories)", "\u00a0"), (r" ± ", "\u00a0±\u00a0"),
    ]
    for pat, rep in rules:
        s = re.sub(pat, rep, s)
    return s


def parse(text: str):
    heads = list(SLIDE_RE.finditer(text))
    stop_markers = [i for i in (text.find("## Visuals to make"), text.find("## Numbers appendix")) if i >= 0]
    hard_stop = min(stop_markers) if stop_markers else len(text)
    slides = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        end = min(end, hard_stop) if m.end() < hard_stop else end
        body = text[m.end():end]
        fields, cur = {}, None
        for line in body.splitlines():
            fm = FIELD_RE.match(line)
            if fm:
                cur = fm.group(1)
                fields[cur] = fm.group(2).strip()
            elif cur and line.strip():
                fields[cur] = (fields[cur] + "\n" + line.rstrip()).strip()
        slides.append({"id": m.group(2) or m.group(3), "headline": m.group(4).strip(), "fields": fields})
    return slides


def header(text: str, key: str) -> str:
    m = re.search(rf"^\*\*{re.escape(key)}:\*\*\s*(.+?)\s*$", text, re.M)
    return m.group(1).strip() if m else ""


def resolve(p: str, repo: Path, folder: Path) -> Path | None:
    p = p.strip().strip("`")
    for base in (Path("/"), repo, repo / "project", repo / "project" / "analysis",
                 repo / "project" / "analysis" / "outputs", repo / "project" / "analysis" / "outputs" / "presentation", folder):
        c = base / p
        if c.exists():
            return c
    return None


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        sys.exit(__doc__)
    outline = Path(argv[0]).resolve()
    folder = outline.parent
    repo = next((q for q in [folder] + list(folder.parents) if (q / ".git").exists()), folder)
    text = outline.read_text(encoding="utf-8")
    title = next((l[2:].strip() for l in text.splitlines() if l.startswith("# ")), "Untitled")
    stages = [s.strip() for s in header(text, "Navigator").split(",") if s.strip()] or None
    figdir = folder / "figures"
    figdir.mkdir(exist_ok=True)
    out, problems = [], []
    n_main = 0
    for s in parse(text):
        f = s["fields"]
        sid = s["id"]
        hidden = sid.startswith("B")
        if not hidden:
            n_main += 1
        bullets = [convert(b.strip()[2:].strip()) for b in f.get("On-slide", "").splitlines() if b.strip().startswith("- ")]
        figures = []
        vis = f.get("Visual", "")
        if vis.startswith("NEW:"):
            problems.append(f"slide {sid}: Visual is still a NEW: spec; make the figure first")
        elif vis and vis.lower() not in ("none", "table"):
            for k, item in enumerate([v for v in re.split(r"[;|]", vis) if v.strip()], 1):
                src = resolve(item, repo, folder)
                if src is None:
                    problems.append(f"slide {sid}: figure not found: {item.strip()}")
                    continue
                dst = figdir / f"s{sid}_{k}_{src.name}"
                shutil.copyfile(src, dst)
                figures.append(dst.name)
        table = None
        if f.get("Table"):
            rows = [[convert(c.strip()) for c in r.strip().strip("|").split("|")] for r in f["Table"].splitlines() if r.strip() and not set(r.strip()) <= set("|-: ")]
            if rows:
                table = {"header": rows[0], "rows": rows[1:]}
        notes = f.get("Talk track", "")
        if f.get("Likely question"):
            notes += "\n\nLikely question: " + f["Likely question"]
        if f.get("Point"):
            notes = "Point: " + f["Point"] + "\n\n" + notes
        out.append({"id": sid, "number": (f"B{sid[1:]}" if hidden else str(n_main)), "hidden": hidden,
                    "headline": convert(s["headline"]), "bullets": bullets, "figures": figures,
                    "table": table, "layout": f.get("Layout", "").strip().lower() or None,
                    "stage": f.get("Stage", "").strip() or None, "notes": notes})
    deck = {"title": {"title": convert(title), "name_line": header(text, "Name line") or ""},
            "stages": stages, "slides": out}
    (folder / "slides.json").write_text(json.dumps(deck, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"slides.json: {n_main} main + {len(out) - n_main} backup slides, {sum(len(s['figures']) for s in out)} figures copied to {figdir}")
    for p in problems:
        print("FAIL", p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
