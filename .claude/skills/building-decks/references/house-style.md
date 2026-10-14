# House style

A paper, not a pitch. White page, one typeface, text in ink, colour only where it means
something, figures doing the work. The reader should feel they are looking at a well-set
article page, not at a template.

## Canvas and grid

- 16:9, 10.0 × 5.625 in, white. Margins 0.5 in left and right, 0.25 in top, 0.3 in bottom.
- Headline: x 0.5, y 0.66, w 9.0, h 0.70, bold 24 pt, one or two lines, bottom-anchored.
- Content area: y 1.55 to 5.3.
- Slide number: 9 pt grey, bottom right. No footer, no logo, no date on content slides.
- Title slide: title at y 1.9, bold 32 pt; one grey 14 pt line with name and month. Nothing
  else. Backups follow a plain divider slide ("Backups") and are hidden in the deliverable.
- Optional navigator strip at y 0.25 (five rounded pills, active one filled blue) when the
  outline declares `**Navigator:**` stages. Off by default.

## Layouts (chosen by the builder per slide)

- **A, figure right**: bullets in a 3.2 in column (3.4 to 3.6 for long bullets), figure in
  the remaining width. Default for one figure.
- **D, figure left**: mirror of A, for a slide whose figure must be read before the text.
- **B, band and row**: bullets in a full-width band on top (one to three lines), figures in
  a row below at one height. Default for two or three figures.
- **Stacked**: bullets in a 4.6 in column, figures stacked on the right. For two figures of
  very different aspect.
- **Table and row**: native table on top, a short band of bullets, figures below. For a
  results slide with numbers that must be exact.
- **Text**: bullets full width at 14 pt. Only for a proposals or questions slide.

## Type

Calibri (Carlito on Linux renders with the same metrics). Bullets 12 to 13 pt, never below
11.5. Tables 11 to 11.5 pt. Notes carry the talk track and the likely question, never shown.
Sub- and superscripts are text runs (`F_{i}`, `d^{4}`); formulas are text, not objects.

## Colour

Ink `30363D` for text. Grey `8A939B` for secondary text and lines. Blue `1F77B4` for the
assembly, the detector and the active navigator pill. Crimson `C0392B` for pairs, fission
and brute-force truth. Orange `C4620E` and rose `C2558B` appear in figures only. Colour is
never decorative: the only coloured things on a slide are the active pill, a figure, and at
most one word in a bullet when it names the assembly or the pairs.

## Figures on the slide

Only files from the deck's `figures/` folder, at native aspect, `contain` sizing, never
stretched, never cropped, never captioned (the figure carries its own labels and legend).
On most slides the figure area exceeds the text area; the build report prints this per slide.

## Text on the slide

Three or four bullets, each a fact with its number and error, under about 170 characters.
Sentence case. Proper minus signs, multiplication signs and Greek letters. No em dashes, no
exclamation marks. Nothing on the slide that is not in the outline, nothing in the
outline's on-slide text that is missing from the slide.

## Not allowed

Icons, stat tiles, cards, gradients, stripes, accent lines under titles, drop shadows,
background tints, section-number circles, stock images, agenda slides, summary slides,
thank-you slides, a subtitle blurb on the title slide, "key takeaway" labels, captions
under figures, a second typeface, italics.

## Taste rules

Recorded from the user's reviews. Each line is one rule with the deck it came from.

- Slide bullets are edited down to three or four short facts; outline prose is never
  copied onto a slide. (September 2026 expert deck: "1500 characters in a slide is insane.")
- Figures carry no prose annotations; the slide text explains the figure. (September 2026.)
- Every error bar shown on a slide is explained on that slide, in a bullet or the table
  header. (September 2026.)
- Numbers that must be read exactly go in a native table, not in bullets. (September 2026.)
- A navigator strip lights monotonically through the talk if used at all. (September 2026.)
