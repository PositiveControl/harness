# atc template

Starter scaffold for FAA-documentation-expert characters (airton_c
family). Copy into `character/<new_name>/` via:

    uv run python scripts/character_from_template.py <new_name> --from atc

The script substitutes `{{CHARACTER_NAME}}` across all files and
writes a minimal persona directory. Post-instantiation, hand-edit:

- `core.yaml` — narrow `premise`, `deep_domains`, `shallow_domains`
  to your variant's document scope. The template ships the generalist
  shape (CFR + AIM + JO 7110.65 + PCG + PHAK).
- `constitution.md` — same; scope rules reference all publications
  by default.
- `voice/canonical.yaml` — empty. Write 4-6 voice samples that
  exemplify the citation style you want. Start with the highest-
  traffic topics for your variant.
- `atc_eval.yaml` — empty. Write 5-10 Q&A cases with expected
  citations and keywords, drawn from the sections your ingested
  corpus will cover.

The gitignored directories (`bd/`, `data/`, `corpus/`, `workspace/`)
are auto-created by `config.py` on first use (harness-xbk.2).

Related beads: harness-ouo (Phase-1 interactive interview, to be
built on top of this template).
