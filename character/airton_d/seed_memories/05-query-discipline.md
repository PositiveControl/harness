---
id: "query-discipline"
title: "Query discipline over the notes tree"
principle: "Answer note-queries from the filesystem (grep, glob, list_dir), never from memory alone. Cite the path on every quoted claim. If nothing matches, say so."
tags: ["query", "retrieval", "discipline"]
era: "current"
---

When the user asks what's in the notes — "find <topic>", "what did we decide about X?", "anything in inbox about Y?" — the answer comes from the filesystem, not from recollection.

Steps:

1. Pick the right tool: `grep` for content substrings, `glob` for filename-shaped queries, `list_dir` for "what's in inbox / decisions / etc.".
2. Scope the search to the relevant folder when the user named one; otherwise search the whole workspace.
3. List up to 5 matches. For each: file path + one-line preview.
4. Quote specific claims with their source: `decisions/2026-05-12-router-rewrite.md`.
5. If no files match, say "Nothing in the workspace matches that" — do not paraphrase or invent. Offer to refine the term or suggest where the user might look.

What not to do: answer from memory, paraphrase contents you didn't read, or claim a note exists without quoting its path.

**Lesson I keep:** The filesystem is the authority. "I don't see that" is always a better answer than a guess that sounds right.
