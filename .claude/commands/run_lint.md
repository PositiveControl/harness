# Run Lint

Lint and type-check the working tree. `ruff` and `mypy` are repo-wide and fast; there's no reason to narrow them to changed files.

```bash
uv run ruff check . && uv run ruff format . && uv run mypy src tests
```

- Lint errors → fix them. `ruff check --fix .` handles the mechanical ones.
- Not auto-fixable → halt. Summarize what's left and propose the manual fix.
- Never silence with a blanket `# noqa` / `# type: ignore`. Fix it, or add a targeted per-file ignore in `pyproject.toml` with a comment explaining why.
- `ruff format` owns layout. Formatter and rule disagree → change the rule, don't hand-format.
