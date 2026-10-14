---
name: building-decks
description: Use when an approved outline.md exists for a talk in this project and the slides themselves need to be made, rendered, checked and delivered as a PowerPoint file. Also use when a built deck needs figures regenerated, a layout fixed, or a visual QA pass.
---

# Building decks

Turns an approved outline (see `writing-talk-outlines`) into a native, editable `.pptx`,
then looks at every rendered slide and fixes what is wrong before the user sees it. The
outline is the contract: the builder decides layout, never words or numbers.

Prerequisite, once per machine: `cd .claude/skills/building-decks/scripts && npm install`.
Tools assumed: Node 22, LibreOffice, poppler (`pdftoppm`), the `pptx` skill at
`~/.claude/skills/pptx` (its validator and LibreOffice wrapper), `markitdown` and
`cairosvg` on the path, Python from `openmc-env`.

## Steps

Work in the deck folder `<folder>` that holds `outline.md`. Scripts live in
`.claude/skills/building-decks/scripts/`; call them with paths, never copy them.

1. **Visuals.** For every `NEW:` line in the outline, make the figure following
   `references/figures.md` (data figures through `slide_style.py`; schematics as hand-
   written SVG rasterised with `svg2png.sh`). Put the script under
   `project/analysis/outputs/presentation/shared/` and the output under `shared/cartoons/`.
   Then replace the `NEW:` line in the outline with the file path. No slide is built with a
   missing visual.
2. **Slide data.** `python scripts/make_slides_json.py <folder>/outline.md`. It copies each
   figure into `<folder>/figures/` with a slide-numbered name, converts the outline's ASCII
   notation to proper symbols, and writes `<folder>/slides.json`. Fix any error it reports
   in the outline, not in the JSON.
3. **Number audit.** `python scripts/audit_numbers.py <folder>/outline.md <folder>/slides.json`
   must pass. A number on a slide that is not in the outline is a stop, not a fix.
4. **Build.** `node scripts/build_deck.js <folder>/slides.json <folder>/<name>.pptx`. Read
   its report: every line marked TEXT-dominant or "text estimate > box" needs a layout
   change or a shorter bullet; shorter bullets are an outline change, so go back to the
   outline's author for those.
5. **Content check.** `python scripts/check_content.py <folder>/slides.json <folder>/<name>.pptx`.
6. **Render.** `bash scripts/render.sh <folder>/<name>_allvisible.pptx <folder>/renders`.
   The validator runs first and must print "All validations PASSED!".
7. **Visual QA by a fresh subagent.** Dispatch the QA prompt from `references/prompts.md`
   with the renders folder and `references/visual-qa.md`. It returns a findings table.
   Fix in the generator or the layout, re-render, repeat. Four rounds at most; if a finding
   survives four rounds, say so to the user instead of hiding it.
8. **Look yourself.** Read every image in `<folder>/renders/`. Do not skip this because the
   subagent said clean.
9. **User review.** Hand over the renders and the `.pptx`. Record every correction: layout
   and taste corrections go to `references/house-style.md` under Taste rules; wording
   corrections go to `writing-talk-outlines/references/voice.md` through that skill's step 7.
10. **Handoff.** Write `<folder>/HANDOFF.md`: what was built, which figure scripts made which
    files, what is open. The deck folder is git-ignored, so this note is the only record.

## What the builder decides, and what it does not

- Decides: which layout each slide gets, figure sizes within the grid, when a table beats
  bullets, whether a backup needs a divider.
- Does not decide: words on a slide, numbers, which figure a slide shows, slide order. Those
  come from the outline; if one of them looks wrong, stop and say so.
- Never shrinks text below the house minimum to make it fit. Remove or move content
  instead, through the outline.

## Common mistakes

- Treating the subagent's "clean" as the end. The renders are the deliverable; look at them.
- Captioning figures or adding labels on the slide that the figure already carries.
- Letting one slide drift from the grid because its figure is an odd shape. Adjust the
  figure, not the grid.
- Regenerating a figure with different data than the number audit saw. Figures and
  appendix come from the same files; if a figure is rebuilt, re-run the audit.
