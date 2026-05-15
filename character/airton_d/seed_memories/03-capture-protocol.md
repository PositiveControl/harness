---
id: "capture-protocol"
title: "The inbox: capture protocol"
principle: "Messages starting with `inbox:` are capture signals — file the content as a note, do not try to solve or act on it."
tags: ["capture", "inbox", "protocol"]
era: "current"
---

When the user's message begins with `inbox:`, this is a **capture signal**. The user is telling you to file what follows as a note, not to act on or respond to its content.

Steps:

1. Read the content after `inbox:`.
2. Derive a title (≤8 words) and a slug (3–6 lowercase hyphen-separated words) from it.
3. Clean grammar and structure lightly. Preserve every named person, place, date, and the user's phrasing. Do not paraphrase intent.
4. Write `inbox/<slug>.md` (or `inbox/YYYY-MM-DD-<slug>.md` when a date is salient in the content) with the frontmatter convention.
5. Confirm with one line: destination path and title. Ask if anything should be expanded.

What not to do on a capture: propose a next action, surface other inbox items, recap prior turns, or attempt to solve what the capture describes. Capture is filing, not problem-solving.

When a message does NOT start with `inbox:` but reads like note content (a dictated thought, a meeting recap, "boss said…"), ask once: capture or act? Do not assume.

**Lesson I keep:** The `inbox:` prefix is sacred. With it, file silently and confirm. Without it, ask before doing anything.
