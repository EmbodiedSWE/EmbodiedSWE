"""BaseVerifier — privileged, hidden scorer of the final delivery.

Standalone and optional: the env can run with none (pure sandbox). A verifier reads the env's
scene/robot state and returns a score/verdict — **free-form (`Any`) for now**; a structured result
type will be added later. Never exposed to the agent — attach it (`env.verify()`) or run it from
the harness in a separate clean process.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .env import BaseEnv


class BaseVerifier(ABC):
    @abstractmethod
    def verify(self, env: BaseEnv) -> Any:
        """Score the current delivery from the env's (privileged) state. Return type is free-form
        for now (a structured verdict will be defined later)."""
