// build_deck.js: slides.json -> <out>.pptx (backups hidden) and <out>_allvisible.pptx
// (same deck, nothing hidden; the render loop uses it because LibreOffice skips hidden
// slides when exporting). Layout rules: ../references/house-style.md.
//
//   node build_deck.js <folder>/slides.json <folder>/<name>.pptx
//
// Figures are read from <folder>/figures/ (make_slides_json.py puts them there).
'use strict';
const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');
const sizeOf = require('image-size');

const [, , jsonPath, outPath] = process.argv;
if (!jsonPath || !outPath) { console.error('usage: node build_deck.js <slides.json> <out.pptx>'); process.exit(2); }
const FOLDER = path.dirname(path.resolve(jsonPath));
const FIG = path.join(FOLDER, 'figures');
const deck = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));

const INK = '30363D', GREY = '8A939B', GREY_SOFT = 'C4CACF', BLUE = '1F77B4', WHITE = 'FFFFFF';
const FONT = 'Calibri';
const TOP = 1.55, BOT = 5.3, CH = BOT - TOP, L = 0.5, W = 9.0, R = L + W;
const MIN_PT = 11.5;

// ------------------------------------------------------------------ helpers
const aspectCache = {};
function aspect(file) {
  if (!aspectCache[file]) { const d = sizeOf(path.join(FIG, file)); aspectCache[file] = d.width / d.height; }
  return aspectCache[file];
}
function fit(file, box, anchor = 'tl') {
  const a = aspect(file);
  let w = box.w, h = w / a;
  if (h > box.h) { h = box.h; w = h * a; }
  let x = box.x, y = box.y;
  if (anchor === 'tc' || anchor === 'c') x = box.x + (box.w - w) / 2;
  if (anchor === 'tr') x = box.x + box.w - w;
  if (anchor === 'c') y = box.y + (box.h - h) / 2;
  return { x, y, w, h, path: path.join(FIG, file) };
}
const r3 = v => Math.round(v * 1000) / 1000;

// `_{..}` subscript, `^{..}` superscript, `**..**` bold -> pptxgenjs runs
function runs(text, base) {
  const out = [];
  const re = /(\*\*.+?\*\*|_\{[^}]*\}|\^\{[^}]*\})/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push({ text: text.slice(last, m.index), options: { ...base } });
    const tok = m[0];
    if (tok.startsWith('**')) out.push(...runs(tok.slice(2, -2), { ...base, bold: true }));
    else if (tok.startsWith('_')) out.push({ text: tok.slice(2, -1), options: { ...base, baseline: -500 } });
    else out.push({ text: tok.slice(2, -1), options: { ...base, baseline: 600 } });
    last = m.index + tok.length;
  }
  if (last < text.length) out.push({ text: text.slice(last), options: { ...base } });
  return out;
}
function plain(text) { return text.replace(/\*\*/g, '').replace(/[_^]\{([^}]*)\}/g, '$1'); }

// Calibri fit estimate (average glyph 0.46 em): used to pick layouts and to warn
function estimateHeight(bullets, w, pt, spaceAfter = 6) {
  const cpl = w / (0.0064 * pt);
  let lines = 0;
  for (const b of bullets) lines += Math.max(1, Math.ceil(plain(b).length / (cpl - 3)));
  return lines * pt * 1.2 / 72 + Math.max(0, bullets.length - 1) * spaceAfter / 72;
}

// ------------------------------------------------------------------ layouts
function layoutA(s, pt = 13, tw = 3.2) {
  return { text: { x: L, y: TOP, w: tw, h: CH, pt }, figs: [fit(s.figures[0], { x: L + tw + 0.5, y: TOP, w: W - tw - 0.5, h: CH })] };
}
function layoutD(s, pt = 13, tw = 3.2) {
  const fw = W - tw - 0.5;
  return { text: { x: L + fw + 0.5, y: TOP, w: tw, h: CH, pt }, figs: [fit(s.figures[0], { x: L, y: TOP, w: fw, h: CH })] };
}
function rowOfFigs(files, top, h, gap = 0.3) {
  const ws = files.map(f => h * aspect(f));
  const sumW = ws.reduce((a, b) => a + b, 0);
  const scale = Math.min(1, (W - gap * (files.length - 1)) / sumW);
  const rowW = sumW * scale + gap * (files.length - 1), rowH = h * scale;
  let x = L + (W - rowW) / 2;
  const y = top + (h - rowH) / 2;
  return files.map((f, i) => { const g = fit(f, { x, y, w: ws[i] * scale, h: rowH }); x += ws[i] * scale + gap; return g; });
}
function bandAndRow(s, pt = 12.5) {
  const th = s.bullets.length ? Math.min(1.6, estimateHeight(s.bullets, W, pt) + 0.1) : 0;
  const top = TOP + (th > 0 ? th + 0.1 : 0), h = BOT - top;
  return { text: th > 0 ? { x: L, y: TOP, w: W, h: th, pt } : null, figs: rowOfFigs(s.figures, top, h) };
}
function stacked(s, pt = 12, textW = 4.6) {
  const x = L + textW + 0.2, w = R - x;
  const needs = s.figures.map(f => w / aspect(f));
  const total = needs.reduce((a, b) => a + b, 0) + 0.25 * (s.figures.length - 1);
  const scale = Math.min(1, CH / total);
  const figs = []; let y = TOP;
  s.figures.forEach((f, i) => { const h = needs[i] * scale; figs.push(fit(f, { x, y, w, h }, 'tc')); y += h + 0.25; });
  return { text: { x: L, y: TOP, w: textW, h: CH, pt }, figs };
}
const TABLE_ROW_H = 0.27;
function tableAndRow(s, pt = 11.5) {
  const nrows = 1 + s.table.rows.length;
  const colw = tableColumns(s.table);
  const tw = colw.reduce((a, b) => a + b, 0);
  const ty = TOP + nrows * TABLE_ROW_H + 0.17;
  const th = s.bullets.length ? Math.min(1.3, estimateHeight(s.bullets, W, pt) + 0.1) : 0;
  const top = ty + th + 0.18, h = BOT - top;
  return { table: { x: L, y: TOP, w: tw, colw }, text: th > 0 ? { x: L, y: ty, w: W, h: th, pt } : null,
           figs: s.figures.length ? rowOfFigs(s.figures, top, h) : [] };
}
function tableColumns(t) {
  // width per column from the longest cell (0.075 in per character + padding), capped to the page
  const n = t.header.length;
  const w = [];
  for (let c = 0; c < n; c++) {
    const longest = Math.max(...[t.header[c], ...t.rows.map(r => r[c] || '')].map(x => plain(x).length));
    w.push(Math.max(0.55, Math.min(2.2, 0.075 * longest + 0.3)));
  }
  const sum = w.reduce((a, b) => a + b, 0);
  return sum > W ? w.map(x => x * W / sum) : w;
}
function textOnly(s) { return { text: { x: L, y: TOP, w: W, h: CH, pt: 14 }, figs: [] }; }

function choose(s) {
  const n = s.figures.length;
  const want = s.layout;
  if (s.table) return tableAndRow(s);
  if (want === 'text' || n === 0) return textOnly(s);
  if (want === 'd') return layoutD(s);
  if (want === 'stacked') return stacked(s);
  if (want === 'b' || want === 'band') return bandAndRow(s);
  if (n === 1 && aspect(s.figures[0]) > 2.0 && want !== 'a' && want !== 'd') return bandAndRow(s);   // wide figure: full width
  if (n === 1 || want === 'a') {
    // widen the column when the bullets need it; move to a band when they still don't fit
    for (const tw of [3.2, 3.4, 3.6]) if (estimateHeight(s.bullets, tw, 13) <= CH) return layoutA(s, 13, tw);
    for (const tw of [3.4, 3.6]) if (estimateHeight(s.bullets, tw, 12) <= CH) return layoutA(s, 12, tw);
    return bandAndRow(s);
  }
  return bandAndRow(s);
}

// ------------------------------------------------------------------ slide pieces
function addNavigator(slide, stage) {
  if (!deck.stages || !stage) return;
  const n = deck.stages.length, gap = 0.12, w = (W - gap * (n - 1)) / n;
  deck.stages.forEach((name, i) => {
    const active = name === stage;
    slide.addText(name, { shape: 'roundRect', rectRadius: 0.13, x: r3(L + i * (w + gap)), y: 0.25, w: r3(w), h: 0.26,
      fill: active ? { color: BLUE } : { type: 'none' }, line: { color: active ? BLUE : GREY, width: 0.75 },
      fontFace: FONT, fontSize: 10, color: active ? WHITE : GREY, align: 'center', valign: 'middle', margin: 0 });
  });
}
function addHeadline(slide, text) {
  const rs = runs(text, { fontFace: FONT, fontSize: 24, bold: true, color: INK });
  rs[0].options.lineSpacing = 25;
  slide.addText(rs, { x: L, y: 0.66, w: W, h: 0.70, margin: 0, align: 'left', valign: 'bottom', fit: 'none' });
}
function addBullets(slide, bullets, box, report) {
  if (box.pt < MIN_PT) throw new Error(`text at ${box.pt} pt is below the house minimum ${MIN_PT}`);
  const base = { fontFace: FONT, fontSize: box.pt, color: INK };
  const items = bullets.flatMap((b, i) => {
    const rs = runs(b, base);
    rs[0].options = { ...rs[0].options, bullet: { indent: 12 }, paraSpaceAfter: 6, lineSpacing: Math.round(box.pt * 12) / 10 };
    if (i < bullets.length - 1) rs[rs.length - 1].options = { ...rs[rs.length - 1].options, breakLine: true };
    return rs;
  });
  slide.addText(items, { x: r3(box.x), y: r3(box.y), w: r3(box.w), h: r3(box.h), margin: 0, valign: 'top', align: 'left', fit: 'none' });
  const est = estimateHeight(bullets, box.w, box.pt);
  if (est > box.h) report.push(`   text estimate ${est.toFixed(2)} in > box ${box.h.toFixed(2)} in`);
}
function addTable(slide, t, box) {
  const cell = (txt, bold) => ({ text: runs(txt, { fontFace: FONT, fontSize: bold ? 11 : 11.5, bold, color: INK }),
                                 options: { align: 'center', valign: 'middle', margin: [0.02, 0.04, 0.02, 0.04] } });
  const rows = [t.header.map(h => cell(h, true)), ...t.rows.map(r => r.map(c => cell(c, false)))];
  slide.addTable(rows, { x: box.x, y: box.y, w: box.w, colW: box.colw, rowH: TABLE_ROW_H, autoPage: false,
                         border: { type: 'solid', pt: 0.5, color: GREY_SOFT }, fill: { color: WHITE } });
}
function addNumber(slide, label) {
  slide.addText(String(label), { x: 9.1, y: 5.33, w: 0.4, h: 0.22, fontFace: FONT, fontSize: 9, color: GREY, align: 'right', valign: 'top', margin: 0 });
}

function build(allVisible) {
  const pres = new pptxgen();
  pres.layout = 'LAYOUT_16x9';
  pres.title = plain(deck.title.title);
  const t = pres.addSlide();
  t.background = { color: WHITE };
  const tr = runs(deck.title.title, { fontFace: FONT, fontSize: 32, bold: true, color: INK });
  tr[0].options.lineSpacing = 36;
  t.addText(tr, { x: L, y: 1.9, w: W, h: 1.2, margin: 0, align: 'left', valign: 'top' });
  if (deck.title.name_line) t.addText(deck.title.name_line, { x: L, y: 3.2, w: W, h: 0.4, margin: 0, fontFace: FONT, fontSize: 14, color: GREY, align: 'left', valign: 'top' });

  const report = [];
  let dividerDone = false;
  for (const s of deck.slides) {
    if (s.hidden && !dividerDone) {
      const d = pres.addSlide(); d.background = { color: WHITE };
      if (!allVisible) d.hidden = true;
      addHeadline(d, 'Backups');
      dividerDone = true;
    }
    const slide = pres.addSlide();
    slide.background = { color: WHITE };
    if (s.hidden && !allVisible) slide.hidden = true;
    const g = choose(s);
    addNavigator(slide, s.stage);
    addHeadline(slide, s.headline);
    if (g.table) addTable(slide, s.table, g.table);
    if (g.text && s.bullets.length) addBullets(slide, s.bullets, g.text, report);
    for (const f of g.figs) slide.addImage({ path: f.path, x: r3(f.x), y: r3(f.y), w: r3(f.w), h: r3(f.h) });
    addNumber(slide, s.number);
    if (s.notes) slide.addNotes(s.notes);
    const figArea = g.figs.reduce((a, f) => a + f.w * f.h, 0);
    const textArea = g.text && s.bullets.length ? g.text.w * g.text.h : 0;
    const kind = g.table ? 'table' : g.figs.length === 0 ? 'text' : g.figs.length === 1 && g.text && g.text.w < 5 ? 'A/D' : 'band/stack';
    report.push(`${String(s.id).padEnd(4)} ${kind.padEnd(10)} figs ${figArea.toFixed(1).padStart(5)} in2  text ${textArea.toFixed(1).padStart(5)} in2  ${figArea > textArea ? 'figure-dominant' : 'TEXT-dominant'}`);
  }
  return { pres, report };
}

// pptxgenjs writes an <a:pPr> for EVERY run of a paragraph; the copies after the first run
// carry <a:buNone/>, so a bulleted paragraph with a sub/superscript or bold run loses its
// bullet in LibreOffice and PowerPoint. Strip the stray ones from the packed XML.
async function stripStrayParagraphProps(file) {
  const JSZip = require('jszip');
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  let removed = 0;
  for (const name of Object.keys(zip.files)) {
    if (!/^ppt\/slides\/slide\d+\.xml$/.test(name)) continue;
    let xml = await zip.file(name).async('string');
    xml = xml.replace(/(<\/a:r>)<a:pPr[^>]*>(?:(?!<\/a:pPr>).)*<\/a:pPr>/g, (m0, g1) => { removed++; return g1; });
    zip.file(name, xml);
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' }));
  return removed;
}

(async () => {
  const out = path.resolve(outPath);
  const outAll = out.replace(/\.pptx$/, '_allvisible.pptx');
  const a = build(false);
  await a.pres.writeFile({ fileName: out });
  const b = build(true);
  await b.pres.writeFile({ fileName: outAll });
  const n1 = await stripStrayParagraphProps(out);
  await stripStrayParagraphProps(outAll);
  console.log(a.report.join('\n'));
  console.log(`stripped ${n1} stray paragraph properties; wrote ${path.basename(out)} and ${path.basename(outAll)}`);
})().catch(e => { console.error(e); process.exit(1); });
