"""Character scaffold templates.

Each subdirectory is one archetype (e.g. `atc/` — FAA-documentation
expert). `scripts/character_from_template.py` copies a chosen
archetype into `character/<new_name>/` and substitutes the
`{{CHARACTER_NAME}}` placeholder throughout.

Extending: add a new archetype as `character_templates/<name>/` with
the same file layout (core.yaml, constitution.md, voice/canonical.yaml,
voice/captured.yaml, atc_eval.yaml, router_eval.yaml, seed_memories/,
README.md). The copier discovers archetypes by directory listing.

Related: harness-ouo (interactive `harness character init` CLI sits
on top of this).
"""
