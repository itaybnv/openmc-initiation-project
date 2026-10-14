# Visual QA checklist

Run by a fresh subagent on the rendered slide images, then again by the main agent. One
pass per slide, in order. Report every finding as a row: slide, severity (block / fix /
note), what is wrong, where (quadrant or element), the likely fix. "Clean" is a valid
result only after every item below was checked on every slide.

## Blockers (must be fixed before the user sees the deck)

1. Text overflow: a bullet, headline or table cut at the box edge, running under a figure,
   or past the bottom margin. Look at the last line of every text box.
2. Overlap: any two elements touching or crossing (text over figure, figure over figure,
   figure over slide number).
3. Illegible figure text: axis labels, tick labels or legend entries smaller than the slide
   number's 9 pt at projection size (in a 1500 px wide render, below about 11 px tall).
4. A stretched or cropped figure (aspect ratio differs from the source file).
5. A missing element: a slide with no figure where the outline named one; a bullet absent.
6. A symbol defect: hyphen where a minus sign belongs, `+/-` instead of ±, `_{i}` markup
   visible, a Greek word spelled out, a bullet that lost its bullet mark.

## Fixes (should be fixed; say so if not)

7. Headline wrapping to a third line, or wrapping with a single orphan word.
8. Text column too narrow: more than two bullets wrap to four or more lines.
9. Figures in a row at different heights, or a row not centred under the band.
10. Inconsistent top edge of the content area across slides.
11. A figure with a title inside it that repeats the headline.
12. Empty space: more than a third of the content area blank on a slide that has a figure
    (the figure should grow, or the layout change).

## Notes (report, do not fix without asking)

13. Anything on the slide that is not in the outline's on-slide text.
14. A slide where the text area is larger than the figure area.
15. A number that looks inconsistent with another slide (same quantity, different value).
16. Colour used decoratively; an em dash; an exclamation mark; title case.

## How to look

Open each image at full size. Check the four edges first, then every box's last line,
then the figure's smallest text, then the whole slide at a glance for balance. Do not infer
from the generator or the JSON; the image is the evidence.
