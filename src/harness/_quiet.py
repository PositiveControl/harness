"""Suppress third-party noise at process startup.

Import this module *before* any `huggingface_hub`, `mlx_lm`,
`transformers`, or `sentence_transformers` import — env vars must be in
place when those libraries read them during their own import. The CLI
entrypoint does this at the very top of `cli.py`; other consumers
should do the same.

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
