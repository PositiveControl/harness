"""Suppress third-party noise at process startup.

Import this module *before* any `huggingface_hub`, `mlx_lm`,
`transformers`, or `sentence_transformers` import — env vars must be in
place when those libraries read them during their own import. The CLI
entrypoint does this at the very top of `cli.py`; other consumers
should do the same.

Also short-circuits one process-global lazy initializer whose first
use would otherwise blow up inside the chat loop:

  - `tqdm`'s multiprocessing RLock: sentence-transformers' `encode()`
    calls `tqdm.trange(...)` even when progress bars are disabled.
    `trange()` lazily creates a `multiprocessing.RLock` on first use,
    which triggers `multiprocessing.resource_tracker.ensure_running()`,
    which calls `_posixsubprocess.fork_exec()` with a list of fds to
    pass to the spawned tracker. By the time the embedder runs inside
    a chat session, MLX has opened Metal-related fds whose validation
    fails (`ValueError: bad value(s) in fds_to_keep`).
    Replacing tqdm's class-level lock with a `threading.RLock` at
    import time bypasses the multiprocessing path entirely — correct
    for our single-process harness. Observed 2026-05-15 with
    airton_c_tfr; see harness-ygvg follow-up. An earlier attempt
    pre-warmed the multiprocessing tracker at import time but didn't
    survive into chat-session embedder calls (tracker invalidated
    somewhere between startup and first encode()).

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


def _install_tqdm_threading_lock() -> None:
    """Pre-install a `threading.RLock` on tqdm so it never tries to
    construct a `multiprocessing.RLock` lazily.

    See the module docstring for the failure mode. The original chain
    fires because `tqdm.tqdm.get_lock()` lazily constructs a
    multiprocessing lock the first time `trange(...)` runs inside the
    embedder, which forces `resource_tracker.ensure_running()` to
    fork-exec while MLX-tainted fds are open.

    A pre-warming approach (eagerly call `multiprocessing.RLock()` at
    import time) was tried first but didn't survive: by the time the
    embedder is reached in a live chat session, something between
    process startup and first encode() invalidates the tracker, and
    the next lock construction re-spawns into the bad-fds state.

    Setting a threading.RLock as tqdm's class-level `_lock` short-
    circuits the lazy path entirely — `get_lock()` returns the
    threading lock, never touches multiprocessing. This is correct
    for our use: the harness is single-process; tqdm's mp-lock would
    only matter if multiple Python processes shared one terminal.

    Catches the import error so installing tqdm into a hostile env
    can't break harness startup."""
    try:
        import threading

        import tqdm  # type: ignore[import-untyped]  # tqdm has no stubs; treated as Any here is fine

        tqdm.tqdm.set_lock(threading.RLock())
    except (ImportError, AttributeError):
        # Failing here just defers to tqdm's default behaviour. If
        # tqdm later trips multiprocessing the retrieval error log
        # path captures the trace — same as before this fix.
        pass


_install_tqdm_threading_lock()
