"""Grader for the pc_motherboard_gpu_ram scene: the complete build — 7 bolts, 2 sticks, 1 card.

Ten parts, three families, and every part is worth the same tenth of the score. Each family
climbs the ladder its own mate allows:
    key_grasped                the allen key ever taken in the weld-on-closure grip
                               (`scene.grasp_held` column 0, site order [key, card, ram0, ram1]);
                               a milestone, latched, and implied by a finished build. Under an
                               embodiment with no gripper (robot="null") the grasp state does not
                               exist and the rung reads from that implication alone
    driven                     the mean over the 7 bolts of a linear ramp of tip depth from the
                               start register (`stage_depth`, the hand-started spawn) to
                               `bolt_seat_depth`, counted only while the bolt passes the scene's
                               xy + tilt gates at its nearest hole. A bolt has no free-flight
                               phase of its own — it is threaded in place — so the bolts carry
                               one rung, not three
    ram_grasped / ram_aligned / ram_seated    and
    gpu_grasped / gpu_aligned / gpu_seated    the loose parts, which ARE carried: grasped ->
                               aligned (origin within `*_align_xy` of the seated point, up axis
                               within `*_align_axis_deg` of the slot axis, length axis within
                               `*_align_yaw_deg` of the slot heading — the scene's own non-depth
                               seat gates) -> seated (depth below the slot mouth as a fraction of
                               `*_seat_depth`, counted only while aligned, so the value is 1.0
                               exactly when the scene calls the part seated)
The rungs nest (seated => aligned => grasped), so progress reaches 1.0 exactly when all ten parts
are seated, for a grader built after the picks as well as one that watched them. Success is the
scene's own `success()` (every bolt, both sticks and the card seated), read when the delivery
finishes.
"""

from __future__ import annotations

import math

from robobench.core import BaseGrader
from robobench.suites.assembly.scenes.pc_motherboard_gpu_ram_assembly import PcMotherboardGpuRamAssemblyScene


class PcMotherboardGpuRamAssemblyGrader(BaseGrader):
    """The whole PC built: 7 motherboard bolts driven, both DIMM sticks seated, the card in the x16 slot.

    Ladder (total weight 30, every part worth 3/30): the key's milestone 1/30; the 7 bolts'
    `driven` fraction 7/30 at full depth; each stick's three rungs 2/30 apiece (6/30 seated);
    the card's three rungs 1/30 apiece (3/30 seated). Motherboard fastened = 0.267, memory
    installed = 0.667, card home = 1.00 — the assembly order the scene stages.
    """

    SCENE = PcMotherboardGpuRamAssemblyScene
    # One unit per part. `driven` is a k/7 fraction over seven identical bolts, so it weighs 7;
    # the stick rungs are k/2 fractions over two identical sticks, so they weigh 2 each; the
    # card's rungs weigh 1. The key's single milestone weighs 1, as in pc_motherboard.
    RUBRIC = (
        ("key_grasped", 1, "once"), ("driven", 7),
        ("ram_grasped", 2, "once"), ("ram_aligned", 2), ("ram_seated", 2),
        ("gpu_grasped", 1, "once"), ("gpu_aligned", 1), ("gpu_seated", 1),
    )
    scene: PcMotherboardGpuRamAssemblyScene

    def setup(self) -> None:
        import torch

        c = self.scene.cfg
        self._depth0 = c.stage_depth if c.screw_mechanic else 0.0  # bolt tip depth at the start
        # `grasp_held` exists only while the weld contract is live (a gripper on the stage).
        self._has_grasp = hasattr(self.scene, "grasp_held")
        dev = self.scene.engaged().device
        self._key_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)  # latch
        # per-part latch for the carried parts, in grasp-site order after the key: [card, ram0, ram1]
        self._held = torch.zeros((self.num_envs, 1 + c.num_slots), dtype=torch.bool, device=dev)

    def check_success(self):
        return self.scene.success()  # (num_envs,) — all ten parts of that env seated

    # ---- shared reads ---------------------------------------------------------------------------
    def _latch_held(self):
        """(num_envs, 1 + num_slots) bool, [card, ram0, ram1]: ever held OR already seated (a
        seated part was necessarily carried there, so the rungs nest under the seated rungs)."""
        import torch

        if self._has_grasp:
            self._held |= self.scene.grasp_held[:, 1 : 2 + self.scene.cfg.num_slots]
        seated = self.scene.seated()  # (n, num_holes + num_slots + 1): bolts, sticks, card
        H = self.scene.cfg.num_holes
        return self._held | torch.cat([seated[:, -1:], seated[:, H:-1]], dim=1)

    def _gpu_aligned(self):
        """(num_envs,) bool: the card passes the scene's xy + tilt + heading gates, without depth."""
        c = self.scene.cfg
        rel = self.scene._card_offset_in_case()  # (n, 3), zero at the seated pose — privileged
        xy_ok = rel[:, 0:2].norm(dim=-1) <= c.gpu_align_xy
        up_ok = self.scene._part_axis_cos(self.scene.card, 2) >= math.cos(math.radians(c.gpu_align_axis_deg))
        yaw_ok = self.scene._part_axis_cos(self.scene.card, 0) >= math.cos(math.radians(c.gpu_align_yaw_deg))
        return xy_ok & up_ok & yaw_ok

    def _ram_aligned(self):
        """(num_envs, num_slots) bool: each stick passes the scene's xy + tilt + heading gates."""
        import torch

        c = self.scene.cfg
        rel = self.scene._ram_offsets_in_case()  # (n, S, 3), zero at the seated poses — privileged
        xy_ok = rel[..., 0:2].norm(dim=-1) <= c.ram_align_xy
        up_ok = torch.stack([self.scene._part_axis_cos(r, 2) for r in self.scene.rams], dim=1) >= math.cos(
            math.radians(c.ram_align_axis_deg))
        yaw_ok = torch.stack([self.scene._part_axis_cos(r, 1) for r in self.scene.rams], dim=1) >= math.cos(
            math.radians(c.ram_align_yaw_deg))
        return xy_ok & up_ok & yaw_ok

    # ---- bolt stages ---------------------------------------------------------------------------
    def key_grasped(self):
        if self._has_grasp:
            self._key_grasped |= self.scene.grasp_held[:, 0]
        return (self._key_grasped | self.scene.success()).float()

    def driven(self):
        """(num_envs,) fraction over that env's bolts of tip depth from the staged register to
        the seat, counted only while the bolt passes the scene's gates at its nearest hole."""
        c = self.scene.cfg
        off = self.scene._bolt_offsets_in_case()  # (n, B, H, 3) — privileged read
        near_dist = off[..., :2].norm(dim=-1).min(dim=-1).values
        aligned = (near_dist <= c.bolt_align_xy) & (
            self.scene._bolt_axis_cos() >= math.cos(math.radians(c.bolt_align_axis_deg)))
        depth = self.scene.bolts_engaged()  # (n, B) tip depth below the board face (m)
        drive = ((depth - self._depth0) / (c.bolt_seat_depth - self._depth0)).clamp(0.0, 1.0)
        return (aligned * drive).mean(dim=1)

    # ---- stick stages — each returns a (num_envs,) fraction over that env's sticks -------------
    def ram_grasped(self):
        return self._latch_held()[:, 1:].float().mean(dim=1)

    def ram_aligned(self):
        return self._ram_aligned().float().mean(dim=1)

    def ram_seated(self):
        depth = self.scene.ram_engaged()  # (n, S) blade depth below each DIMM slot mouth (m)
        return (self._ram_aligned() * (depth / self.scene.cfg.ram_seat_depth).clamp(0.0, 1.0)).mean(dim=1)

    # ---- card stages ---------------------------------------------------------------------------
    def gpu_grasped(self):
        return self._latch_held()[:, 0].float()

    def gpu_aligned(self):
        return self._gpu_aligned().float()

    def gpu_seated(self):
        depth = self.scene.gpu_engaged()  # (n,) tab depth below the PCIe slot mouth (m)
        return self._gpu_aligned() * (depth / self.scene.cfg.gpu_seat_depth).clamp(0.0, 1.0)
