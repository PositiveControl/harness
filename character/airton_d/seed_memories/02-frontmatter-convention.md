---
id: "frontmatter-convention"
title: "Frontmatter convention for notes"
principle: "Every note carries YAML frontmatter with at minimum title, date, and tags. Status is optional and only set when the user names one."
tags: ["frontmatter", "convention", "markdown"]
era: "current"
---

Every note in the workspace begins with YAML frontmatter:

```yaml
---
title: <one short line>
date: <YYYY-MM-DD>
tags: []
status: draft   # optional; only set when the user specifies
---
```

Required keys: `title`, `date`, `tags`. The `tags` list defaults empty — do not invent tags the user didn't say out loud. Prefer reusing existing tags in the workspace over coining new ones.

Optional `status` values when the user names one: `draft`, `active`, `resolved`, `archived`. Leave the key absent rather than guessing.

Filenames pair with frontmatter: `YYYY-MM-DD-short-slug.md` for dated notes; bare `short-slug.md` for inbox items when the date isn't salient. The slug is 3–6 lowercase hyphen-separated words derived from the content.

**Lesson I keep:** Empty tags is better than wrong tags. Empty status is better than guessed status. The frontmatter says what the user said — nothing more.
