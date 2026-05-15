"""Suppress third-party noise at process startup.

Import this module *before* any `huggingface_hub`, `mlx_lm`,
`transformers`, or `sentence_transformers` import — env vars must be in
place when those libraries read them during their own import. The CLI
entrypoint does this at the very top of `cli.py`; other consumers
should do the same.

Also warms a few process-global resources whose lazy first-use would
otherwise blow up inside the chat loop:

  - `tqdm`'s multiprocessing RLock: sentence-transformers' `encode()`
    calls `tqdm.trange(...)` even when progress bars are disabled.
    `trange()` lazily creates a `multiprocessing.RLock` on first use,
    which triggers `multiprocessing.resource_tracker.ensure_running()`,
    which calls `_posixsubprocess.fork_exec()` with a list of fds to
    pass to the spawned tracker. By the time the embedder runs inside
    a chat session, MLX has opened Metal-related fds whose validation
    fails (`ValueError: bad value(s) in fds_to_keep`). Pre-creating
    the lock at import time — BEFORE MLX/torch touch anything —
    makes the tracker spawn while fds are still clean, then reuse
    forever. Observed 2026-05-15 with airton_c_tfr; see harness-ygvg
    follow-up.

Uses `setdefault` so a user who wants the full firehose can override
any of these via their shell env.
"""

from __future__ import annotations

import os
import warnings

# HF Hub: silence progress bars + telemetry. The "unauthenticated
# requests" warning is left intact deliberately — it's a reminder to set
# HF_TOKEN when rate limits start biting. `HF_HUB_DISABLE_IMPLICIT_TOKEN`
# is intentionally NOT set: that flag prevents HF from picking up a
# cached token, which would defeat the point once a token is added.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Transformers: advisory warnings + verbose logging
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Tokenizers: parallelism warning triggered after fork
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# sentence-transformers emits FutureWarnings about renamed APIs; keep
# those out of the CLI. Scoped to the package so unrelated FutureWarnings
# from user code still surface.
warnings.filterwarnings("ignore", category=FutureWarning, module="sentence_transformers.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")


def _warm_multiprocessing_resource_tracker() -> None:
    """Pre-spawn `multiprocessing.resource_tracker` while fds are clean.

    See the module docstring for the failure mode. The first call to
    `multiprocessing.RLock()` triggers `resource_tracker.ensure_running()`
    which fork-execs the tracker process; doing this here, before any
    MLX/torch/HF code runs, means the tracker is alive and the lock is
    cached before the chat loop's first embedder query.

    Catches any exception so a hostile env (no /tmp, sandboxed runtime,
    forbidden fork) doesn't break import — the failure surfaces later
    at the embedder boundary just as it did before this fix landed."""
    try:
        import multiprocessing

        # Just constructing a lock is enough; resource_tracker.register
        # is what triggers the fork_exec. We don't need to hold the
        # lock — let it drop and be garbage-collected, the tracker
        # process stays alive for the rest of the harness lifetime.
        multiprocessing.RLock()
    except (OSError, ValueError, ImportError):
        # Failing here just defers to the pre-fix lazy behaviour —
        # the embedder will still try the multiprocessing lock on
        # first use, and the chat session will surface the error
        # via the retrieval-error-log path instead of silently
        # warming. Strictly better than crashing import.
        pass


_warm_multiprocessing_resource_tracker()
