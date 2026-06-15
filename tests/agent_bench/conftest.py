"""Put the agent-bench tree on sys.path for its tests.

agent-bench is run as scripts from its own directory (`python runner.py`), so
its modules (`adapters`, `scorers`, `store`) are top-level — not under the
`harness` package. These tests import them directly, so we prepend the
agent-bench dir here. Localized to this conftest so the rest of the suite is
unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path

_AGENT_BENCH = Path(__file__).resolve().parents[2] / "agent-bench"
if str(_AGENT_BENCH) not in sys.path:
    sys.path.insert(0, str(_AGENT_BENCH))
