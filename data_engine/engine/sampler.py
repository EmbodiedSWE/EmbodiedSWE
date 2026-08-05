"""sampler — parameters per episode index. PLACEHOLDER.

The real design arrives with the tunable rebuild (README, 05 TODO): draw the
scene cfg's tunables (L4 domain randomization, e.g. friction, mass) and the
solve's Params cfg (L2 solver parameters) as a pure function of the episode
index — low-discrepancy over bands, index 0 = nominal — and record the drawn
parameters in the episode meta. Reset/initialization randomization is NOT
sampled here; it belongs to the scene's own reset and the phase reset conditions.

Until then: every episode is nominal.
"""

from __future__ import annotations


def sample_parameters(episode_index: int) -> dict:
    """Parameters for one episode. Placeholder: nominal (empty) for every index."""
    return {}
