"""Shared lightweight types. Import-light (no isaaclab/torch at module load)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Verdict:
    """Result of a `BaseVerifier` — privileged, never shown to the agent.

    success: did the delivery meet the goal.
    score:   scalar quality, partial score for not ful, verifier-defined.
    info:    free-form diagnostics.
    """

    success: bool
    score: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)
