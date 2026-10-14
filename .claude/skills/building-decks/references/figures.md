# Figures

Two kinds, two tools. Data figures come from the project's own code through
`scripts/slide_style.py`. Schematics, cartoons and diagrams are hand-written SVG. Native
pptxgenjs shapes are used only for structural elements (tables, the navigator, a divider).
No charts drawn by PowerPoint; no HTML-to-image conversions.

## Where things live

- Scripts: `project/analysis/outputs/presentation/shared/make_<name>.py` (one per figure or
  one per family), importing `slide_style` from this skill's `scripts/` folder.
- Outputs: `project/analysis/outputs/presentation/shared/cartoons/<name>.png`, plus the
  `.svg` next to it for hand-drawn ones. `make_slides_json.py` copies what a slide uses into
  the deck's `figures/`.
- The outputs folder is git-ignored. The deck's `HANDOFF.md` lists which script made which
  file so everything can be rebuilt.

## Data figures (matplotlib)

- Design for the box it will fill. A layout-A figure is 5.6 in wide and up to 3.75 in tall;
  a band-and-row figure is about 4.4 in wide and 2.5 in tall. Pass `figsize` in those
  inches so the type is correct at 1:1.
- `slide_style.use()` sets the palette, 11 pt base type, no top or right spines, proper
  minus signs, and 200 dpi on save. Do not override the type size downward.
- Legend on every data plot, no frame. No title inside the figure unless the figure is the
  only labelled thing on the slide. Axis labels with units.
- Fit bands translucent behind data points. Error bars with small caps. Data as points,
  models as lines.
- Colour by meaning: blue for the assembly and detector, crimson for pairs, fission and
  brute-force truth, grey for references and point kinetics, ink for everything else.
- Minimal annotation. If a number must be on the figure, it is one short label near the
  thing it describes, in ink, no box, no arrow unless the target is ambiguous.
- Log axes where the data span decades; mark "no result" points explicitly rather than
  omitting them.
- Save with `slide_style.save(fig, path)` (tight bounding box, 200 dpi, white background).

## Schematics and cartoons (SVG)

- Write the SVG by hand, one file per figure, viewBox in pixels at 96 per inch of the
  target box (a 5.6 in wide box is `viewBox="0 0 538 ..."`).
- Font `Carlito, Calibri, sans-serif`, 12 to 14 px labels, ink for text, 1.5 px strokes,
  the palette hex codes, no gradients, no drop shadows, no clip art.
- Geometry drawn to scale when it is a physical system (the sphere, the detector, a mesh);
  schematic only for flows and pipelines.
- Rasterise: `bash scripts/svg2png.sh in.svg out.png 300`. Keep the SVG; it is the source.
- When a cartoon needs real numbers (a mesh with the measured map inside it), draw it in
  matplotlib instead and keep the same visual rules.

## Tables

Native (`Table:` in the outline). Header bold, cells centred, light grey borders, 11 pt.
Units in the header, not in the cells. One table per slide.

## Reuse

Prefer regenerating an existing figure with `slide_style` over copying a notebook figure.
Notebook figures carry titles, grids and type sizes meant for a page, not a projector.
