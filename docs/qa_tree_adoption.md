# QA checklist — tree adoption (harness-gav3)

Verified 2026-05-14. All sections passed. Use this when re-validating
the data-retrieval-primitives tree-adoption surface after dependency
bumps, embedder rotations, or refactors to `cli.py` / `cli_classic.py`
session bootstrap.

Each section: what to verify, then the exact command. All commands
run from repo root.

## 1. Static gates (must all be green)

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

```bash
# Full test suite — 1,987 expected, 3 env-gated skips
uv run pytest -q
```

```bash
# Tree-adoption-specific subset — 43 tests across the new surface
# (15 document_tree_store + 14 document_tree_ingest + 6
# document_tree_bootstrap + 8 returns_handler_character)
uv run pytest \
  tests/test_document_tree_store.py \
  tests/test_document_tree_ingest.py \
  tests/test_document_tree_bootstrap.py \
  tests/test_returns_handler_character.py -v
```

## 2. Generic ingest module — `harness-2zf4`

**Markdown adapter:** positional paths, embed-when-body, structural-only
headers, skipped-depth tolerance, empty file, frontmatter dropped,
sibling ordinals.

```bash
uv run pytest tests/test_document_tree_ingest.py -v -k markdown
```

**JSONL adapter:** ATC-shape grouping, body-by-chunk-index,
missing-hierarchy rows skipped, empty depth_fields rejected.

```bash
uv run pytest tests/test_document_tree_ingest.py -v -k jsonl
```

**Driver:** writes the tree, idempotent re-run, parent-before-child
enforcement, search reaches embedded leaves.

```bash
uv run pytest tests/test_document_tree_ingest.py -v -k ingest_blueprints
```

**Behavior parity with old script:** rebuild ATC tree and confirm
`13 / 91 / 675` counts.

```bash
uv run python scripts/atc_ingest_tree.py --rebuild --output /tmp/atc_qa.sqlite 2>&1 | tail -3
# expect: "done: 13 chapters, 91 section groups, 675 sections (embedded). Total embedded: 675"
rm -f /tmp/atc_qa.sqlite /tmp/atc_qa.sqlite-shm /tmp/atc_qa.sqlite-wal
```

## 3. Character spec — `harness-ejxn`

**Backward compat:** every existing character still loads with
`document_trees == ()`.

```bash
uv run pytest tests/test_character.py::test_document_trees_default_empty_for_existing_characters -v
```

**Spec parsing:** markdown spec resolves path, defaults format to
`markdown`, accepts `jsonl`, rejects unknown formats, missing fields,
duplicate names, non-list types.

```bash
uv run pytest tests/test_character.py -v -k document_trees
```

**Manual smoke:** confirm returns_handler loads with its new spec.

```bash
HARNESS_CHARACTER_NAME=returns_handler uv run python -c "
from harness.config import settings
from harness.character import load_character
c = load_character(settings.character_path)
assert len(c.document_trees) == 1, f'expected 1 tree spec, got {len(c.document_trees)}'
spec = c.document_trees[0]
assert spec.name == 'returns_workflow'
assert spec.source_format == 'markdown'
assert spec.source_path.exists(), f'source missing: {spec.source_path}'
print(f'OK: {spec.name} -> {spec.source_path.name} ({spec.source_format})')
"
```

## 4. CLI wiring — `harness-px7k`

**Bootstrap helpers + StoreBundle:** None when empty, populates from
markdown, idempotent, JSONL raises clearly, dim-swap rebuild, end-to-end
tree-slot contract.

```bash
uv run pytest tests/test_document_tree_bootstrap.py -v
```

**`memory rebuild-embeddings` extension:** the command surfaces a tree
mismatched-count line and rebuilds tree nodes.

```bash
# Clear stale dim mismatches first by removing any old tree DB
rm -f character/returns_handler/data/document_tree.sqlite*
# Fresh bootstrap + rebuild — should print "tree: 0 mismatched" after first run
HARNESS_CHARACTER_NAME=returns_handler uv run harness memory rebuild-embeddings 2>&1 | tail -5
```

**Live session smoke (classic REPL path, the default):** chat boots,
tree DB lands on disk.

```bash
rm -f character/returns_handler/data/document_tree.sqlite*
HARNESS_CHARACTER_NAME=returns_handler uv run harness chat --model echo --tools <<< $'hi\n/exit\n' 2>&1 | tail -3
ls -la character/returns_handler/data/document_tree.sqlite
# expect: file exists, ~90 KB
```

**Live session smoke (TUI path):** same smoke through the alt path.
TUI emits ANSI escapes; pipe the output and check the file lands.

```bash
rm -f character/returns_handler/data/document_tree.sqlite*
HARNESS_CHARACTER_NAME=returns_handler timeout 5 uv run harness chat --model echo --tools --tui 2>&1 > /dev/null
ls -la character/returns_handler/data/document_tree.sqlite
```

**Idempotence on re-launch:** second boot should not duplicate nodes.

```bash
COUNT_BEFORE=$(uv run python -c "
import sqlite3
c = sqlite3.connect('character/returns_handler/data/document_tree.sqlite')
print(c.execute('SELECT COUNT(*) FROM tree_nodes').fetchone()[0])
")
HARNESS_CHARACTER_NAME=returns_handler uv run harness chat --model echo --tools <<< $'hi\n/exit\n' > /dev/null 2>&1
COUNT_AFTER=$(uv run python -c "
import sqlite3
c = sqlite3.connect('character/returns_handler/data/document_tree.sqlite')
print(c.execute('SELECT COUNT(*) FROM tree_nodes').fetchone()[0])
")
echo "before=$COUNT_BEFORE after=$COUNT_AFTER"
test "$COUNT_BEFORE" = "$COUNT_AFTER" && echo OK || echo MISMATCH
```

## 5. Worked example — `harness-zae0`

**All three stores fire on one contract:**

```bash
uv run pytest tests/test_returns_handler_character.py::test_returns_handler_contract_fans_out_to_all_three_stores -v
```

**Existing tabular + episodic contract paths still resolve** (regression
check against the tree-slot addition):

```bash
uv run pytest tests/test_returns_handler_character.py -v
```

**Manual contract render** — confirms the contract orchestrator
resolves end-to-end against real character stores and returns hits
from all three shapes:

```bash
HARNESS_CHARACTER_NAME=returns_handler uv run python -c "
from pathlib import Path
from harness.character import load_character
from harness.config import settings
from harness.retrieval.context_package import AccessPolicy
from harness.retrieval.contract import StoreBundle, load_contract, assemble_package
from harness.store.episodic import EpisodicStore
from harness.store.tabular import build_tabular_store_for_character
from harness.store.document_tree import build_document_tree_store_for_character
from harness.retrieval.st_embedder import SentenceTransformersEmbedder

char = load_character(settings.character_path)
emb = SentenceTransformersEmbedder()
ep = EpisodicStore(db_path=settings.character_db_path, embedder=emb)
tab = build_tabular_store_for_character(settings.character_path, emb, char.tabular_tables)
tree = build_document_tree_store_for_character(settings.character_path, emb, char.document_trees)

contract = load_contract(settings.character_path / 'contracts' / 'returns_handler.yaml')
pkg = assemble_package(
    contract,
    variables={'customer_id': 'C9148', 'request_summary': 'damaged on arrival'},
    access=AccessPolicy(user_id='C9148', role='returns_handler'),
    stores=StoreBundle(episodic=ep, tabular=tab, tree=tree),
)
print(f'complete={pkg.is_complete} stores={sorted({h.provenance.store for h in pkg.hits})}')
print(f'tokens={pkg.tokens_used}/{pkg.budget.max_tokens} hits={len(pkg.hits)}')
for slot_name in ('customer_history', 'refund_policy', 'matching_orders', 'workflow_step'):
    hits = pkg.hits_for_slot(slot_name)
    print(f'  {slot_name}: {len(hits)} hits')
"
# expect: complete=True stores=['episodic', 'tabular', 'tree']
# each slot fills with >=1 hit (workflow_step is optional but should populate)
```

## 6. Deferred-feature surface

**JSONL via character spec raises a clear error** (not silently
misbehaves):

```bash
uv run pytest tests/test_document_tree_bootstrap.py::test_builder_raises_on_jsonl_spec -v
```

## 7. Embedder-swap robustness

**Dim-mismatch detection + rebuild path:**

```bash
uv run pytest tests/test_document_tree_bootstrap.py::test_rebuild_embeddings_recovers_mismatched_dim -v
```

**Live swap drill** — only run if you actually want to exercise an
embedder rotation. Skip unless you want the model download.

```bash
# Default embedder: bge-small-en-v1.5 (384-dim). Swap to a different-dim
# model. nomic-ai/nomic-embed-text-v1.5 is 768-dim.
HARNESS_CHARACTER_NAME=returns_handler \
HARNESS_EMBEDDER_REPO=nomic-ai/nomic-embed-text-v1.5 \
  uv run harness memory rebuild-embeddings 2>&1 | tail -10
# expect: tree: N mismatched (pre-rebuild); rebuilt N tree (post)
```

## 8. Pre-existing-character regression

**Each character still loads without the new spec affecting it:**

```bash
for c in airton airton_b airton_c airton_c1; do
  HARNESS_CHARACTER_NAME=$c uv run harness describe 2>&1 | head -1
done
# expect: each prints its premise line, no tracebacks
```

## 9. Bd ledger sanity

```bash
bd show harness-gav3      # epic should be closed, all 4 subtasks closed
bd dep tree harness-gav3  # confirm dep graph is clean
```

## Sign-off conditions

All static gates green, sections 2–6 tests pass, section 4 live-session
smoke creates `document_tree.sqlite`, section 5 contract call returns
`stores=['episodic', 'tabular', 'tree']`, section 8 prints all four
character premises.

## 2026-05-14 run summary

| Section | Result |
| --- | --- |
| 1. Static gates | 1987 passed / 3 skipped in 22s; 43 tests in tree-adoption subset |
| 2. Ingest module | markdown 6/6, jsonl 4/4, driver 5/5; ATC parity exact (13/91/675) |
| 3. Character spec | 7 spec tests pass; existing chars empty default |
| 4. CLI wiring | bootstrap 6/6; rebuild reports `rebuilt … 11 tree embeddings`; TUI + classic REPL both create `document_tree.sqlite` (~90 KB); idempotent (11→11 nodes) |
| 5. Worked example | 8/8 returns_handler tests pass; live render `complete=True stores=['episodic', 'tabular', 'tree']`, 8 hits, 1182/1500 tokens |
| 6. Deferred | JSONL-via-spec raises `NotImplementedError` |
| 7. Embedder swap | dim-swap unit test passes; live drill not run (needs model download) |
| 8. Char regression | airton / airton_b / airton_c / airton_c1 all describe cleanly |
| 9. Bd ledger | epic + 4 subtasks closed, dep tree clean |
