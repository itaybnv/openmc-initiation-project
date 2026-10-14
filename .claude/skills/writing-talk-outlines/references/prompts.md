# Subagent prompts

Both subagents are dispatched fresh (general-purpose, no conversation context). Give them
file paths, not pasted content, and nothing about the conversation that led here. Replace
the angle-bracket fields. Keep the prompts as written otherwise; they are part of what keeps
the chat out of the deck.

## Writer

```
You are writing the content contract (outline.md) for a talk. Work only from the files
named here. Do not invent a number, a result or a claim that is not in a source file; if
something you need is missing, write "SOURCE MISSING: <what>" in its place and continue.

Read, in this order:
1. <repo>/.claude/skills/writing-talk-outlines/references/outline-format.md  (the format; follow it exactly)
2. <repo>/.claude/skills/writing-talk-outlines/references/voice.md  (how the text sounds)
3. <repo>/.claude/skills/writing-talk-outlines/references/audiences.md, profile "<profile>"
4. <folder>/brief.md  (audience, occasion, what is new, the source list)
5. Every source listed in brief.md.

Then write <folder>/outline.md. Rules that are not negotiable:
- Each slide heading is one full sentence stating the slide's claim, under 120 characters,
  carrying the softness word of any number in it. Read your headings in order when you
  finish; they must tell the whole argument alone, and each must say exactly what its
  source says, in the source's direction and quantity.
- Every number on a slide has its error or its softness word, and a row in the Numbers
  appendix naming the file it came from.
- Three or four on-slide facts per slide, under 170 characters each. Detail goes in the
  talk track.
- Retractions and soft numbers are stated on the slide where they live.
- No agenda, summary or thank-you slide. No adjectives in place of numbers. No em dashes.
- One spelling per number across the outline. SOURCE MISSING placeholders go in the
  appendix only; on the slide, pose the question in plain words.

Write the file, then reply with: the ghost deck (headings in order), the list of SOURCE
MISSING items, and nothing else.
```

## Editor

```
You are editing the on-slide text and the talk tracks of <folder>/outline.md. Do not change
any number, any source, the Numbers appendix, the Visual lines, or what a headline claims.

Read, in this order:
1. ~/.claude/skills/writing-clearly-and-concisely/elements-of-style.md
2. ~/.claude/skills/humanizer/SKILL.md
3. <repo>/.claude/skills/writing-talk-outlines/references/voice.md

Pass 1: apply the Elements of Style rules to every bullet and talk track (active voice,
definite concrete language, omit needless words, emphatic words at the end).
Pass 2: apply the humanizer to the same text. The register is technical and plain; do not
add opinions, first person, or personality.
Pass 3: check every rule and every before/after pair in voice.md against the text. Where
the text matches a "before", rewrite it the way the "after" is written.

Edit <folder>/outline.md in place. Reply with a list of the changes you made, one line
each ("slide 3, bullet 2: <before> -> <after>"), and the voice.md rules you applied.
```
