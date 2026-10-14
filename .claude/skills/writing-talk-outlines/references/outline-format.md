# Outline format

One Markdown file, `outline.md`, parsed by `building-decks/scripts/make_slides_json.py`.
Field names and the heading forms below are fixed; the parser depends on them. Notation in
the outline is plain ASCII (`+/-`, `alpha`, `tau`, `F_i`, `sum_i`, `1e-4`, `x` for times);
the build converts it to proper symbols.

```markdown
# <Talk title, sentence case, a statement or a question>

**Audience:** <profile name from audiences.md, one line on what they already know>
**Occasion:** <mentor update | group seminar | expert talk | conference>, <date>
**Format:** <minutes> min, <N> main slides, <M> backups
**Name line:** <Name> · <Month Year>
**Through-line:** <two to four sentences, said at the start and at the end>

## Slide 1 · <Headline: one full sentence stating the slide's claim>
Point: <one sentence: what the audience must leave this slide knowing>
On-slide:
- <fact with its number and error, under 170 characters>
- <...>
- <...>
Visual: <repo-relative path to an existing figure> | NEW: <what to draw, data source, panels>
Table: <optional; header row, then rows, cells separated by " | ">
Table source: <optional; one file the whole table is read from; then its cells need no appendix rows>
Layout: <optional: A figure-right, D figure-left, B band-and-row, stacked, text>
Talk track: <what is said, in sentences; includes the why; two to six sentences>
Likely question: <the question> -> <the answer, one or two sentences; name a backup if one exists>

## Slide 2 · ...

## Backup B1 · <Headline>
Point: <which likely question this answers>
On-slide:
- ...
Visual: ...
Talk track: ...

## Visuals to make
- <slide number>: <NEW spec repeated: what, data source file, panel list, size>

## Numbers appendix
| number | used on | source |
|---|---|---|
| 0.999 +/- 0.007 | slide 1 | reports/GATE_K090_REPORT.md |
| 2.66 | slides 3, 4 | analysis/outputs/multiplicity_toy/summary.json: prediction_from_collision_spectrum.M_sim_pred |
```

## What each field is for

- **Headline.** The claim. Reading only the headlines in order (the ghost deck) must give
  the whole argument. "Results" is not a headline; "Assembled from first moments, it lands
  on the pair histogram" is.
- **Point.** The one sentence the speaker would say if the projector died.
- **On-slide.** Three or four facts. Each carries its number and its error or its softness
  ("no result at 100M", "single-point, indicative"). No adjectives doing the work of a
  number. No sentence that is true of any project.
- **Visual.** One per slide. An existing path, or `NEW:` with enough spec for
  `building-decks` to make it without asking: data file, what is on each axis, panels,
  what is annotated. Tables are native, not images.
- **Talk track.** Spoken voice. It carries the why, the mechanism, the one caveat. It is
  not a longer copy of the bullets.
- **Likely question.** One per main slide. If the answer needs a figure, that is a backup.
- **Numbers appendix.** Every number that appears in a headline, bullet or table, with the
  file it came from. The source cell starts with the repo-relative path, then a comma and
  the key, line or report section when the file is long. A table read from one file cites
  it once with `Table source:` instead. After a file's first full path in the appendix, its
  bare name is enough. Rows for pending numbers start with "pending:". If a number has no
  file, it does not go on a slide.

## Budgets

- Main slides: one per two to four minutes of talk. Backups: one per likely question.
- Headline: under 120 characters (two lines at 24 pt). Bullets: under 600 characters in total.
- Talk track per slide: 60 to 150 words.
- Through-line: under 80 words.

## Softness vocabulary

Use these phrases, not synonyms: "retracted" (a published number withdrawn), "no result"
(an estimator that returned nothing at the available budget), "indicative" (a single-point
estimate without a split-sample error), "estimate, not measured" (a transfer factor or a
model, not data), "pending" (a run that has not finished).
