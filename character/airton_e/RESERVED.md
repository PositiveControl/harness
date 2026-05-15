# airton_e — RESERVED

This directory is intentionally a placeholder. The `airton_e` slot is
reserved for a future character; the name is held to keep the
single-letter sequence (`b`, `c`, `d`, `e`, `f`, …) coherent.

There is no `core.yaml`, no `constitution.md`, and no voice or seed
data here on purpose. The character loader (`harness.character.
load_character`) is never invoked against this path, and nothing in
the harness iterates `character/*/` at startup — so an incomplete
directory cannot break the runtime. Setting
`HARNESS_CHARACTER_NAME=airton_e` will fail loudly at load time
(missing `core.yaml`), which is the intended behavior.

When the slot is filled, replace this file with the standard
character scaffold (see `character/airton_d/` or `character/
airton_f/` for two minimal-shape references):

- `core.yaml` — identity, values, taboos, directives, deep / shallow
  domains, voice config.
- `constitution.md` — workflow per turn.
- `voice/canonical.yaml` (+ `captured.yaml`) — voice samples.
- `seed_memories/*.md` — at least a handful of frontmatter-tagged
  principles the character should carry.
- Optionally: `contracts/<name>.yaml` if the character is contract-
  driven; `document_trees:` or `tabular_tables:` in `core.yaml` for
  shaped retrieval.

Until then: do not load `airton_e`.
