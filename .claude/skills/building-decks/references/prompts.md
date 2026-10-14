# Subagent prompts

Fresh general-purpose subagents, file paths only, nothing from the conversation.

## Builder (optional; the main agent may build directly)

```
Build the slide deck for the outline at <folder>/outline.md. Follow
<repo>/.claude/skills/building-decks/SKILL.md steps 2 to 6 exactly, with the scripts in
<repo>/.claude/skills/building-decks/scripts/ and the rules in
<repo>/.claude/skills/building-decks/references/house-style.md.

You decide layouts and figure sizes. You do not change any word, number, figure choice or
slide order; if one looks wrong, stop and report it. Never shrink text below 11.5 pt.

When the renders exist, reply with: the build report, the validator and content-check
results, and the path of the renders folder. Nothing else.
```

## Visual QA

```
You are checking rendered slide images for a scientific talk. Read
<repo>/.claude/skills/building-decks/references/visual-qa.md, then open every image in
<folder>/renders/ in order, at full size, and check each item on every slide. The image is
the evidence; do not read the generator or the JSON.

Reply with one table: slide | severity (block / fix / note) | finding | where | likely fix.
Then one line: the count of blockers and fixes. If a slide is clean, do not list it. Do not
praise the deck or comment on its content.
```
