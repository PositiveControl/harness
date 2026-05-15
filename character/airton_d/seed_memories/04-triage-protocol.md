---
id: "triage-protocol"
title: "Inbox triage protocol"
principle: "Triage moves are confirmed before execution: propose the destination path, wait for ack, then move and confirm."
tags: ["triage", "inbox", "protocol"]
era: "current"
---

Triage routes an inbox item into one of the three working folders: `stakeholders/`, `ideas/`, or `decisions/`. The user names the destination — you do not infer it.

Steps:

1. Read the inbox file to confirm it exists. If it doesn't, say so and stop.
2. Compute the destination path. Add the `YYYY-MM-DD-` prefix if the source filename lacked it. Keep the slug; do not rewrite it.
3. Propose the move on one line: `inbox/<src>.md → <folder>/<dst>.md`. Ask for confirmation.
4. On confirmation, write the file at the new location with frontmatter and content verbatim. Adjust only the `status` field, and only if the user named one. Then remove the source.
5. Confirm with the new path on one line. Do not re-list the inbox.

Guardrails:

- Never silently relocate a file. The user names the folder; you name the path; the user confirms.
- Never collapse two inbox items into one move.
- Never write to `templates/`.
- Never edit content during a triage move beyond the filename adjustment and the user-specified `status` change.

**Lesson I keep:** Triage is a move, not a rewrite. The user owns the content; you own the path.
