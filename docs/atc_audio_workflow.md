# ATC audio workflow

End-to-end how-to for the cite-grounded ATC transmission verifier's audio mode (`harness-o92o`). Takes raw LiveATC mp3 clips, produces a labeled corpus, scores the lint pipeline against it. For architecture, see `CLAUDE.md`. For why this exists, see bead `harness-0pte`.

The pipeline is four scripts run in order, plus one eval subcommand. Each stage is idempotent — re-running picks up where it stopped, no double work. All artifacts land under `character/airton_c1/atc_audio/`.

## Prereqs

- ffmpeg + ffplay on PATH (`brew install ffmpeg`).
- `uv sync --extra all --extra asr` — adds mlx-whisper on top of the full core set. (Run additively with `all`, not `--extra asr` alone — a bare `--extra asr` prunes the rest of the env.)
- Source mp3s somewhere on disk. Default ingest dir is `../atc-audio` relative to the repo root; override with `--source`.
- ATC family character set: `export HARNESS_CHARACTER_NAME=airton_c1`.

## Stage 1: ingest

Walks the source dir, parses each filename for embedded metadata (event tag, ICAO, position, recorded-at UTC), normalizes audio to 16k mono FLAC via ffmpeg, and writes idempotent meta records keyed by `clip_id = sha256(source-bytes)[:12]`.

```
uv run python scripts/atc_audio_ingest.py
uv run python scripts/atc_audio_ingest.py --source /elsewhere/audio
uv run python scripts/atc_audio_ingest.py --dry-run     # parse filenames only, no ffmpeg
uv run python scripts/atc_audio_ingest.py --force       # re-ingest existing clips
```

What lands:

- `raw/<original-filename>.mp3` — copy of source for audit.
- `norm/<clip_id>.flac` — 16k mono FLAC, what STT consumes.
- `meta/<clip_id>.json` — parsed filename + duration + provenance.
- `index.jsonl` — one row per clip, sorted by recorded_at.

Filenames that don't match the dominant `<event><ICAO>-<position>...-Mon-DD-YYYY-HHMMZ` shape (free-form names like `DAL1082.mp3`) still get ingested with `parsed.parse_failed=true`. The labeling step triages those into the right bucket.

The whole `atc_audio/` dir is gitignored except `utt/` (next stage's output) — audio bytes are bulky, copyright-encumbered, and regenerable from source.

## Stage 2: transcribe

Runs mlx-whisper over each `norm/<clip_id>.flac`, writes `stt/<clip_id>.json` with segments + word timestamps + per-segment `avg_logprob` / `no_speech_prob`. Idempotent on `stt/<clip_id>.json`.

```
uv run python scripts/atc_audio_transcribe.py
uv run python scripts/atc_audio_transcribe.py --model mlx-community/whisper-tiny  # fast iteration
uv run python scripts/atc_audio_transcribe.py --only 04dd0a5b26d4 --force          # one clip
uv run python scripts/atc_audio_transcribe.py --limit 3                            # smoke run
```

Default model is `mlx-community/whisper-large-v3-mlx`. Switching models without `--force` keeps the older transcripts (idempotency). Pass `--force` when iterating on model choice.

Whisper's accuracy on radio audio is poor on its own (callsigns, alphanumerics, fast speech). That's expected — the labeling step fixes transcripts before they become eval ground truth.

## Stage 3: label

Sequential CLI that walks each whisper-segmented utterance and lets you write a labeled JSONL row. This is the bottleneck — every other stage is mechanical.

```
uv run python scripts/atc_audio_label.py                    # all clips
uv run python scripts/atc_audio_label.py --auto-play        # auto-play each utterance
uv run python scripts/atc_audio_label.py --only 8cf1e5c1f6c6
uv run python scripts/atc_audio_label.py --limit 3          # cap to N clips
```

### Per-utterance keys

Each utterance shows clip context + the whisper segment + this prompt:

```
(o)os  (l)abel  (p)lay  (s)kip-clip  (q)uit:
```

| Key | Action | Use for |
|---|---|---|
| `o` | Write an `out_of_scope` row using the whisper transcript as-is, advance | Pilot side, ground chatter, garbled or non-rule speech — **the fast path** |
| `l` | Enter the field-by-field label flow | Real controller phraseology you want to score |
| `p` | Play this utterance via ffplay (kills any prior playback) | When the transcript alone isn't enough |
| `s` | Save current labels, skip rest of clip, jump to next clip | Whole clip is uninteresting |
| `q` | Save current labels and exit | End of session |

Single letter, hit ENTER.

### The `l` flow

```
transcript [whisper text]:                    # ENTER keeps seed, type to correct
speaker (p)ilot/(c)ontroller/(u)nknown [c]:   # one letter
verdict (o)k/(w)rong/(i)ncomplete/(x)oos [x]: # one letter
section (e.g. 3-9-10):                        # only if verdict ≠ oos
phraseology (canonical):                      # only if verdict in {ok, wrong}
mismatch (one-line reason):                   # only if verdict in {wrong, incomplete}
notes:                                        # free-form, ENTER for blank
```

Defaults are in brackets; ENTER accepts. Validation catches forgotten fields (verdict=wrong without phraseology, oos with a section, etc.) and offers a retry instead of writing a malformed row.

### Speaker default heuristic (auto-seeded, you can override)

| Position field on the clip | Default |
|---|---|
| `CTAF` | pilot (no controller on frequency) |
| `Twr` / `Gnd` / `App` / `Dep` / `Ctr` | controller |
| `Guard` (121.5) | unknown |
| Empty / unrecognized | unknown |

### Output

Each labeled row appends one line to `character/airton_c1/atc_audio/utt/<clip_id>.jsonl`. The full schema:

```
clip_id, utt_index, start_s, end_s, speaker_role,
facility, position, frequency, recorded_at, event_tag,
transcript_text (human), transcript_seed (whisper), transcript_confidence,
expected_verdict, expected_section, expected_phraseology, mismatch, notes,
model_repo, human_verified, labeled_at
```

`utt/` is the only directory in `atc_audio/` that's tracked in git — these are the human-signed ground truth labels the eval scores against.

### Resuming + redoing

Re-running `atc_audio_label.py` reads existing `utt/*.jsonl` rows, skips any `(clip_id, utt_index)` already in there, and prompts only on the rest. Resume = just re-run.

Need to redo a specific row? Delete that line from `utt/<clip_id>.jsonl` and re-run; the (clip_id, utt_index) reopens for editing.

## Stage 4: eval

Scores the lint pipeline on the labeled corpus. Two passes per row — clean human transcript and noisy whisper seed — so you can tell whether accuracy loss came from STT or the lint tool.

```
harness eval atc-audio                          # all labeled clips
harness eval atc-audio --skip-noisy             # half the model load, clean pass only
harness eval atc-audio --only 8cf1e5c1f6c6      # pre-push gate subset
harness eval atc-audio --json                   # machine-readable
harness eval atc-audio --save-baseline          # snapshot current accuracy
harness eval atc-audio --compare-baseline       # diff vs saved
```

(All examples assume `harness` on PATH. Otherwise prepend `uv run`.)

Headline metrics:

- **clean_verdict_accuracy** — verdict pass-rate on the human transcript. Pure lint-tool quality.
- **noisy_verdict_accuracy** — verdict pass-rate on the whisper seed. End-to-end audio mode quality.
- **mean_wer** — token-level word error rate, whisper vs human.
- **verdict_shift_rate** — fraction of cases where the clean pass was right but STT noise broke the verdict. The "STT cost" line.
- **by_event_tag** — same metrics partitioned by event class (departure / emergency / handoff / …).

`--skip-noisy` is for inner-loop iteration on the lint pipeline — halves model load by stubbing the noisy pass to always-OOS. Don't ship a baseline with `--skip-noisy`; the gate runs both passes.

## Stage 5: pin the baseline

Once the accuracy is where you want it:

```
HARNESS_CHARACTER_NAME=airton_c1 harness eval atc-audio --save-baseline
```

Writes `character/airton_c1/atc_audio_baseline.json`. Every future `--compare-baseline` (and the pre-push hook) diffs against this file.

The pre-push hook (`scripts/atc_audio_gate.sh`, wired in `.pre-commit-config.yaml`) fires when a push touches `evals/atc_audio.py`, the `utt/` corpus, the baseline file, or any of the `atc_audio_*` scripts. It blocks the push on:

- Aggregate accuracy regression (clean or noisy verdict / citation / combined).
- WER rise (inverted polarity — higher = worse).
- Per-case verdict / citation pass flips beyond `--regression-budget` (default 0).

The hook skips cleanly when no baseline file is present *or* no `utt/*.jsonl` rows exist yet — both are valid initial states, not regressions.

To intentionally update the baseline after a real improvement:

```
HARNESS_CHARACTER_NAME=airton_c1 harness eval atc-audio --save-baseline
git add character/airton_c1/atc_audio_baseline.json
git commit -m "atc-audio: snapshot new baseline (clean verdict +X%)"
```

## End-to-end smoke run

```
export HARNESS_CHARACTER_NAME=airton_c1

uv run python scripts/atc_audio_ingest.py
uv run python scripts/atc_audio_transcribe.py --model mlx-community/whisper-tiny --limit 1
uv run python scripts/atc_audio_label.py --limit 1 --auto-play
harness eval atc-audio
```

Tiny-model + single-clip is enough to verify plumbing in ~30 seconds. Switch back to `whisper-large-v3-mlx` (default) before producing real labels.

## Common questions

**Where do I delete a bad ingest?** Remove `meta/<clip_id>.json`, `norm/<clip_id>.flac`, and `raw/<original-filename>.mp3`. Re-run ingest if you want to re-process the source.

**The label tool segmented one utterance into three (or three into one).** That's a whisper segmentation artifact. v0 treats whisper segments as provisional utterances; merging or splitting after the fact isn't supported. Workarounds: (a) label each fragment and live with the noise, (b) skip the clip and label a different one, (c) hand-edit `utt/<clip_id>.jsonl` after the fact (rows are independent — merging is two rows worth of human work). A real VAD-aware segmenter is a deferred follow-up bead.

**Whisper transcript is unusable on a clip.** Two paths: (a) hit `o` through every utterance — they all become OOS, contributing nothing useful but also nothing harmful; (b) hit `s` to skip the clip entirely (no rows written, the clip can be re-labeled later from a better whisper run with `--force`). The audio-mode eval drops unverified rows, so a half-labeled clip doesn't taint the score.

**The eval ran for hours.** Each labeled row hits the lint pipeline twice (clean + noisy passes) at temperature 0. Big corpus = big runtime. Use `--only` with the small-N pre-push subset for fast feedback; run the full eval ahead of `--save-baseline`.

**The pre-push gate is failing on a clean push.** Check:
- Did you hand-edit a `utt/*.jsonl` row? That counts as a corpus change.
- Did you change the lint tool, the eval scorer, or the lint fixture? Look at `git diff --name-only` against `.pre-commit-config.yaml`'s `files:` regex for the `atc-audio-gate` hook.
- Run `harness eval atc-audio --compare-baseline` directly to see which dimension regressed. If the regression is intentional, re-snapshot with `--save-baseline` and commit the new baseline.

## Pointers

- Script docstrings (`scripts/atc_audio_*.py`) carry the canonical flag reference.
- `src/harness/evals/atc_audio.py` defines the row schema, scoring, and baseline diff.
- Bead history: `harness-d8o2` (ingest) · `harness-xko1` (transcribe) · `harness-hp8k` (label) · `harness-gy5z` (eval) — closed under `harness-o92o`.
