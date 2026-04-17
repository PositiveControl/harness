# Using Airton

This is the daily-use guide. For architecture, see `CLAUDE.md`. For phase sequencing and what's next, see `docs/roadmap.md`.

## Start a conversation

One command:

```
uv run harness chat --model mlx --persona --memories 3 --facts 5
```

What the flags do:

- `--model mlx` — use Qwen 2.5 7B on MLX (default for local dev speed; pass `--model-repo mlx-community/Qwen2.5-32B-Instruct-4bit` for the bigger model). Omit `--model mlx` to use the echo adapter (no model, just wiring — useful if MLX is down).
- `--persona` — run the two-pass voice rewriter. Pass 1 generates substance; pass 2 rewrites in Airton's register. Without it, Airton will drift toward generic-assistant prose.
- `--memories 3` — retrieve up to 3 episodic memories per turn that clear the 0.5 similarity floor.
- `--facts 5` — retrieve up to 5 semantic facts per turn that clear the 0.45 floor.

Other useful options:

- `--session NAME` — name the session. Defaults to `local`. Helpful when you want the scribe to pull from a specific conversation later.
- `--speaker NAME` — who you are. Defaults to `mark`. Determines whose relationship memory this session belongs to.
- `--chain-rewrites` — add a second concrete-substitution rewrite pass. More Airton, 1.5× latency.
- `--memories-threshold 0.5` / `--facts-threshold 0.45` — tune the similarity floors. Lower = more permissive.

## What's happening behind each turn

1. Your message is logged to the transcript.
2. Retrieval picks the 6 voice samples most similar to your message and puts them in the system prompt as few-shot examples.
3. Episodic memory is searched; hits that clear the floor are appended to the system prompt as "things I remember."
4. Semantic facts are searched; hits that clear the floor are appended as "relevant facts I know."
5. Qwen produces a draft (pass 1).
6. Qwen rewrites the draft in Airton's register (pass 2).
7. The final reply is logged to the transcript and shown.

If Airton doesn't know anything relevant, those floors keep the prompt clean — nothing irrelevant gets injected. That's by design.

## When Airton says something wrong

Two paths.

### Airton's substance is right but voice is off

You've just landed in voice-coverage territory: the prompt is near no existing voice sample, so Qwen is falling back to generic register.

Fix it by teaching Airton what it should have said. Two paths:

**In-chat (easiest):** type `/edit` at the `you ›` prompt. Your `$EDITOR` opens with Airton's last reply pre-loaded — rewrite it in the voice you want, save, and exit. The edited text becomes a new voice sample paired with the preceding user message. Saving without changes is a no-op.

**From the shell:**

```
uv run harness voice capture \
  --session <same session id you just chatted in> \
  --gold "What Airton should have said, in Airton's voice."
```

Either way writes a new sample to `character/airton/voice/captured.yaml`. On your next chat startup, retrieval will pick it up for similar future prompts.

To confirm the capture landed:

```
uv run harness voice list-captured
```

### Airton's substance is wrong

That's a factual error, not a voice drift. The scribe will eventually record whatever happened, but if you want to correct the record:

```
uv run harness memory fact-add SUBJECT PREDICATE OBJECT --confidence 0.95
```

Shared fact (no `--user`) if it's general; scoped (`--user mark`) if it's about you.

To supersede an existing fact, run `fact-add` with the new version then let the consolidator merge them next time you consolidate — the newer, higher-confidence one wins.

## Memory that compounds through use

Right now memory does NOT auto-update during chat. The transcript logs everything, but extraction is a deliberate step:

```
uv run harness memory scribe --session <session> --user mark --model mlx
```

The scribe reads unprocessed transcript turns for that session, asks Qwen to extract episodic summaries and atomic facts, writes them to the working tier. Watermark-tracked — reruns are incremental.

After a few scribe runs, the working tier accumulates. To clean it up (merge near-duplicates, pick canonical versions):

```
uv run harness memory consolidate
```

Consolidator clusters similar episodes and merges fact triples, promotes the winners to `consolidated` tier, retires the losers.

**Suggested cadence** (manual for now; scheduled consolidation is a pending phase):

- Scribe: after each meaningful chat session.
- Consolidate: weekly or when the working tier feels cluttered.

Inspect what's in memory any time:

```
uv run harness memory list                              # episodic
uv run harness memory list --tier consolidated          # only promoted
uv run harness memory fact-list                         # semantic facts
uv run harness memory search "some question"            # semantic search
uv run harness memory fact-search "some question"
```

## When things feel off

### Airton responds generically with no Airton register

- Is `--persona` on? Without it, you'll get raw Qwen output.
- Is `--model mlx` on? Without it, you're running the echo adapter.
- Did you accidentally leave `--memories 0 --facts 0`? Memory can help anchor voice.
- Check whether relevant voice samples exist: `uv run harness eval voice --model mlx --persona` — the suite runs against known prompts. If gold for your prompt shape doesn't exist, capture it.

### Airton cites wrong facts

- Check what facts are retrieved for that prompt: `uv run harness memory fact-search "your prompt"`. If wrong ones are ranking above right ones, raise `--facts-threshold` in chat (e.g., 0.55 instead of 0.45).
- Consider whether the scribe wrote noisy facts you should manually retire: find them with `fact-list` and supersede with a `fact-add` + consolidate.

### Airton doesn't remember something we discussed

- Did you run the scribe? Transcripts aren't automatically memory-ized.
- If scribed but not retrieved: check similarity with `memory search "query"`. Might be below the 0.5 floor. Lower `--memories-threshold` in chat.

### Embedding-related errors after switching models

- Run `uv run harness memory rebuild-embeddings` to re-vectorize all active rows with the current embedder.
- Verify with `memory search` — retrieval should work immediately after.

## Paths on disk

- `data/harness.sqlite` — transcripts, episodic, semantic, scribe watermarks. Back this up.
- `character/airton/voice/canonical.yaml` — the curated voice suite. Treat as source of truth; edit with care.
- `character/airton/voice/captured.yaml` — your captured corrections. Appends-only in practice; safe to edit.
- `character/airton/seed_memories/*.md` — Airton's formative narratives. Editable; reloaded on next character load.
- `character/airton/core.yaml` — identity: pronouns, premise, values, taboos. Changes are structural.
- `character/airton/constitution.md` — principles the critic enforces. Plain prose.

## Known limits

- Chat is interactive-only. No web, Slack, or Matrix yet.
- Memory doesn't auto-scribe; you run `memory scribe` manually.
- Consolidator ignores user scoping — fine with one user, needs fixing before inviting another.
- Voice only covers prompts near existing samples. Off-piste prompts regress to Qwen's default register until you capture them.
- No backup destination set — local only, no off-box copies yet.

## Suggested first session

1. `uv run harness chat --model mlx --persona --memories 3 --facts 5`
2. Talk to Airton about something real you're working on — actual engineering, not a test prompt.
3. When a reply lands wrong, capture the correction before moving on: `uv run harness voice capture --session local --gold "…"` in another terminal.
4. After the session, scribe it: `uv run harness memory scribe --session local --user mark --model mlx`.
5. Inspect what it wrote: `uv run harness memory list --tier working` and `fact-list --tier working`.
6. Optionally consolidate: `uv run harness memory consolidate`.

Repeat. The corpus compounds.
