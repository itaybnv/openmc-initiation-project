---
name: writing-talk-outlines
description: Use when a talk, seminar, mentor update, or slide deck for this project is requested, before any slide is built. Also use when an outline, content contract, or Numbers appendix needs writing or revising, or when the user reviews one.
---

# Writing talk outlines

The outline is the content contract for a deck. It fixes the claims, the numbers with
their sources, the visuals and the spoken track before anything is built. Slides are then
assembled from it by `building-decks` without re-deciding content. The outline is written by
a fresh subagent from files, never from this conversation, so the talk cannot inherit chat
phrasing, sycophancy or half-remembered numbers.

## Steps

1. **Brief.** Write `brief.md` in a new dated folder under
   `project/analysis/outputs/presentation/<YYYY-MM-DD>_<slug>/`: audience (pick a profile
   from `references/audiences.md`), occasion, time slot, what is new since the last talk,
   what must be retracted or flagged soft, and the source list: reports, JSON summaries,
   notebooks, figure files, earlier outlines. Paths only; no numbers in the brief. The user
   can correct the brief before step 2.
2. **Draft.** Dispatch a fresh general-purpose subagent with the writer prompt from
   `references/prompts.md`. It reads `references/outline-format.md`, `references/voice.md`,
   the audience profile, the brief and the sources, and writes `outline.md`. It must not be
   told anything about this conversation.
3. **Audit.** `python .claude/skills/writing-talk-outlines/scripts/audit_outline.py <folder>/outline.md`.
   Fix every FAIL (a number without a source, a missing field, a banned pattern) by sending
   the findings back to the writer subagent, not by editing numbers yourself.
4. **Edit.** Dispatch a second fresh subagent with the editor prompt. It applies
   writing-clearly-and-concisely, then humanizer, then checks every rule and pair in
   `references/voice.md`, on the on-slide text and talk tracks only. Numbers, sources and
   headlines' claims do not change in this pass. Re-run the audit.
5. **Read the ghost deck.** The audit prints the headlines in order. If they do not tell the
   whole argument on their own, the outline is not done.
6. **User review gate.** Hand the outline to the user. Nothing is built before they approve.
7. **Capture the review.** For every correction the user makes, append to
   `references/voice.md` a before/after pair under the matching slide role, or a one-line
   rule with one example. Sort: voice goes in `voice.md`; taste (what belongs on a slide)
   goes in `building-decks/references/house-style.md` under Taste rules; content decisions
   go in the outline itself. The next deck must not repeat a correction already recorded.

## Rules the writer must keep

- One claim per slide, stated as a full sentence in the slide heading. Topic labels are not
  headlines.
- Every number on a slide carries its error or its softness and appears in the Numbers
  appendix with the file it came from. No number is re-derived while writing.
- Retractions and soft results are said on the slide where the number lives, not in a
  closing caveat.
- Three to four on-slide facts, each under about 170 characters. The detail goes in the
  talk track.
- One visual per slide, existing file or a spec. A slide with no visual needs a reason.
- Backups exist only to answer a likely question that is written down.
- The deck ends on the last result or the proposal. No agenda, summary or thank-you slide.

## Common mistakes

- Pasting report prose into on-slide text. The on-slide text is written for a slide read
  in ten seconds; the report sentence goes in the talk track if anywhere.
- Quoting a number the audit cannot trace. Either find the file or drop the number.
- Softening with adjectives instead of error bars. "Strong agreement" says nothing;
  "0.999 ± 0.007" does.
- Writing the talk track as a second copy of the bullets. It is what is said out loud,
  including the one sentence that explains why the number matters.
- Letting the editor pass change a claim. If a headline reads better but says less, revert.
