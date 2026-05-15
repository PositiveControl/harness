---
id: "workspace-layout"
title: "Notes workspace layout"
principle: "The workspace has four working folders (inbox, stakeholders, ideas, decisions) and one reference folder (templates)."
tags: ["layout", "workspace", "convention"]
era: "current"
---

The notes workspace lives at `character/airton_d/workspace/` (override with `HARNESS_AIRTON_D_WORKSPACE`). Five top-level folders:

- `inbox/` — Quick captures, unsorted. New items land here.
- `stakeholders/` — Meeting notes, review notes, planning discussions about specific people or teams.
- `ideas/` — Expanded concepts and definitions. Ideas that need room to breathe.
- `decisions/` — Decision records. The *why* behind a chosen path.
- `templates/` — Note templates. Reference-only; never written to.

Captures default to `inbox/`. Triage moves them into one of the three working folders when the user names the destination. `templates/` is read-only.

**Lesson I keep:** Inbox is the safe default. When in doubt about which working folder, file in inbox and let the user route.
