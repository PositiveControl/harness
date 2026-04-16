"""Model adapters. The rest of the harness must not import model SDKs
directly — go through an adapter so models stay swappable.

`MLXAdapter` is intentionally not re-exported here; importing it would
drag `mlx_lm` into every consumer. Import it from `harness.model.mlx`
when you actually need the type."""

from harness.model.adapter import ChatMessage, ModelAdapter
from harness.model.echo import EchoAdapter
from harness.model.factory import AdapterName, make_adapter

__all__ = [
    "AdapterName",
    "ChatMessage",
    "EchoAdapter",
    "ModelAdapter",
    "make_adapter",
]
