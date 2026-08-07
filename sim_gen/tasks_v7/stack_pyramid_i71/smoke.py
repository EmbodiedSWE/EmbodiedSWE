"""Smoke / rubric-REJECTION battery for FalseworkTentScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the latched
credit is monotone along a real trajectory). This battery proves the rubric REJECTS
wrong outcomes, and that the physical claim the task rests on — a lone panel cannot
hold an in-band lean, so the counted tent is genuinely mutually supporting — is
load-bearing physics, not fiat. Every probe is CONSTRUCTED as a settled state
(teleport, real physics steps, judge) — instrumentation, never a solution: no probe
here reaches success().

Checks:
   1. settle/baseline    — seeded reset settles finite; panels flat (tilt ~90), pillar
                           standing at the build centre; score 0, no success;
   2. randomization      — two seeded resets: READBACK build-pad xy + yaw, park-pad
                           xy, panel_a xy all differ;
   3. slot swap          — over 8 resets the panels occupy BOTH slot assignments;
   4. null-policy        — 240 idle steps: score ~0, no success;
   5. SEED STRATEGY      — the seed's plan ("stack the pieces"): panels CONSTRUCTED
                           stacked FLAT on the build pad, pillar parked -> settled,
                           NO success, score ~0 (flat is 90 deg, out of band);
   6. falsework-in-place — both panels leaning ON the pillar (the mid-solve state):
                           pair credit latches but `pillar_clear` refuses the tent —
                           removing the falsework is not optional;
   7. LONE LEAN FALLS    — a single panel released at an in-band lean with NO pillar
                           collapses flat, and the debounced lean latch never fires
                           while it sweeps the band — mutual support is real, and the
                           latch cannot be tricked by a transient;
   8. half-collapse      — one panel flat, the other resting on it at ~85 deg: the
                           tilt band rejects (a fallen "tent" is not a tent);
   9. out-of-zone        — a real settled tent built 0.35 m from the build centre,
                           pillar parked -> the zone clause refuses;
  10. not-parked         — a real settled tent at the build centre but the pillar
                           left standing on open ground -> no success, score capped
                           at 0.85 (the park clause is the only miss);
  11. pillar-adjacent    — tent geometry OK but the pillar standing within `clear_r`
                           of a panel -> `pillar_clear` refuses (a propped tent does
                           not count even if nothing touches);
  12. parallel-lean      — both panels leaning the SAME way (one on the pillar, one
                           on its back), both in band -> the opposed clause refuses;
  13. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.stack_pyramid_i71.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=12)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version on some GPUs and silently rejects RTX
# -> the annotator returns EMPTY frames. Disable the check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()

try:
    from . import scene as task_scene
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene

_qmul, _qz, _qx = task_scene._qmul, task_scene._qz, task_scene._qx

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if rec and _REC["i"] % args.record_every == 0:
            for _f in range(3):  # flush accumulated history (ghosting fix)
                env.sim.render()
            arr = np.asarray(_REC["annot"].get_data())
            if arr.size:
                _REC["frames"].append(arr[..., :3].astype(np.uint8).copy())
        _REC["i"] += 1


def _refresh() -> None:
    _ENV.iscene.update(0.0)


def _all_ids():
    return torch.arange(_ENV.num_envs, device=_ENV.device)


def _write_state(body, st: torch.Tensor) -> None:
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    tilt = scene.panel_tilt_deg()[0]
    pz = scene.pillar.data.root_pos_w[0] - scene.env_origins[0]
    print(f"[smoke] {tag:18s} | tilt=({float(tilt[0]):.1f},{float(tilt[1]):.1f})deg "
          f"pillar=({float(pz[0]):+.3f},{float(pz[1]):+.3f},{float(pz[2]):+.3f}) "
          f"lean={bool(scene._lean[0])} pair={bool(scene._pair[0])} "
          f"free={bool(scene._free[0])} tent={bool(scene.tent_standing()[0])} "
          f"parked={bool(scene.pillar_parked()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.falsework_tent")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.35, -0.95, 0.85)) + o),
                                tuple(np.array((0.32, -0.02, 0.08)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        _REC["annot"] = annot if warm.size else None
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    # ----- pose constructors (pad frame -> world) -------------------------------------------------
    def frame_pose(ctr_xy_w: torch.Tensor, psi: torch.Tensor, y_loc: float,
                   z_loc: float, alpha: float) -> torch.Tensor:
        """Root state at (0, y_loc, z_loc) in a frame at `ctr_xy_w` with yaw psi,
        tipped by qx(alpha) in that frame."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = ctr_xy_w[:, 0] - torch.sin(psi) * y_loc
        st[:, 1] = ctr_xy_w[:, 1] + torch.cos(psi) * y_loc
        st[:, 2] = scene.env_origins[:, 2] + z_loc
        st[:, 3:7] = _qmul(_qz(psi), _qx(torch.full((n,), alpha, device=device)))
        return st

    base_z = c.pad_t + 0.003

    def lean_state(i: int, side: float, theta_deg: float,
                   ctr_xy_w: torch.Tensor, psi: torch.Tensor,
                   base_y: float | None = None) -> torch.Tensor:
        """Panel i at a lean of theta against a plane on side=+/-1 (solve geometry);
        if base_y is given, anchor the base there instead of the pillar face."""
        th = math.radians(theta_deg)
        length = c.panel_l[i]
        if base_y is None:
            face = c.col_y / 2 + 0.002
            top_y = side * (face + (c.panel_t / 2) * math.cos(th))
            base_y = top_y + side * length * math.sin(th)
        ctr_y = base_y - side * (length / 2) * math.sin(th)
        ctr_z = base_z + (length / 2) * math.cos(th)
        # qx(+a) maps local z -> (0, -sin a, cos a): side=+1 needs +th to lean inward.
        return frame_pose(ctr_xy_w, psi, ctr_y, ctr_z, side * th)

    def park_pillar() -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = scene.park_xy
        st[:, 2] = scene.env_origins[:, 2] + c.pad_t + 0.01
        st[:, 3] = 1.0
        _write_state(scene.pillar, st)

    def flat_state(i: int, ctr_xy_w: torch.Tensor, psi: torch.Tensor,
                   z_loc: float) -> torch.Tensor:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = ctr_xy_w
        st[:, 2] = scene.env_origins[:, 2] + z_loc
        st[:, 3:7] = _qmul(_qz(psi), _qx(torch.full((n,), math.pi / 2, device=device)))
        return st

    def construct_tent(ctr_xy_w: torch.Tensor, psi: torch.Tensor) -> None:
        """CONSTRUCT a near-equilibrium tent (panels released slightly apart, tip
        inward, catch each other) and settle it — a real self-supporting state."""
        _write_state(scene.panels["panel_a"],
                     lean_state(0, +1.0, 23.0, ctr_xy_w, psi, base_y=+0.070))
        _write_state(scene.panels["panel_b"],
                     lean_state(1, -1.0, 24.0, ctr_xy_w, psi, base_y=-0.061))
        _step(150)

    # ================= 1. settle / baseline =======================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(120)
    _report("settle")
    _REC["on"] = False
    tilt = scene.panel_tilt_deg()[0]
    pz = float(scene.pillar.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    pd = float((scene.pillar.data.root_pos_w[0, 0:2] - scene.build_xy[0]).norm())
    check("settle/baseline: layout settles finite; panels flat (tilt > 80 deg), "
          "pillar standing at the build centre; score 0, no success",
          bool(scene._finite()[0]) and float(tilt.min()) > 80.0
          and pz < 0.05 and pd < 0.03
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (scene.build_pad.data.root_pos_w[0, 0:2].clone(),
                yaw_of(scene.build_pad.data.root_quat_w[0]),
                scene.park_pad.data.root_pos_w[0, 0:2].clone(),
                scene.panels["panel_a"].data.root_pos_w[0, 0:2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_bp, a_yaw, a_pp, a_pa = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_bp, b_yaw, b_pp, b_pa = readback()
    d_bp = float((a_bp - b_bp).norm())
    d_yawv = dyaw(a_yaw, b_yaw)
    d_pp = float((a_pp - b_pp).norm())
    d_pa = float((a_pa - b_pa).norm())
    print(f"[smoke] randomization deltas: build_xy={d_bp * 1000:.1f}mm "
          f"build_yaw={d_yawv:.1f}deg park_xy={d_pp * 1000:.1f}mm "
          f"panel_a_xy={d_pa * 1000:.1f}mm", flush=True)
    check("randomization-is-real: build-pad xy + yaw, park-pad xy, panel_a xy "
          "readback differ between seeds",
          d_bp > 0.003 and d_yawv > 2.0 and d_pp > 0.003 and d_pa > 0.003)

    # ================= 3. slot-swap coverage ======================================================
    swaps = set()
    for s in range(8):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        swaps.add(bool(scene.slot_swapped[0]))
    print(f"[smoke] over 8 resets: slot_swapped values {sorted(swaps)}", flush=True)
    check("slot swap: both panel-to-slot assignments occur over 8 resets — which "
          "panel lies where must be perceived, not memorized",
          swaps == {False, True})

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED'S OWN STRATEGY =======================================
    # "Stack the pieces" — the seed's only skill. CONSTRUCT the panels stacked FLAT
    # on the build pad (one on the other), pillar parked: flat is 90 deg, the tilt
    # band refuses, score stays ~0.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    park_pillar()
    psi = scene.build_yaw
    _write_state(scene.panels["panel_b"],
                 flat_state(1, scene.build_xy, psi, c.pad_t + c.panel_t / 2 + 0.002))
    _write_state(scene.panels["panel_a"],
                 flat_state(0, scene.build_xy, psi,
                            c.pad_t + c.panel_t * 1.5 + 0.006))
    _step(150)
    _report("seed-strategy")
    tilt = scene.panel_tilt_deg()[0]
    check("negative (SEED strategy): panels stacked flat on the build pad, pillar "
          "parked — the tilt band refuses, no success, score <= 0.01",
          float(tilt.min()) > 80.0 and bool(scene.pillar_parked()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.01)
    _REC["on"] = False

    # ================= 6. falsework still in place ================================================
    # The mid-solve state: both panels leaning ON the pillar at the build centre.
    # Pair credit is earned, but `pillar_clear` refuses the tent — the falsework
    # MUST come out.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    psi = scene.build_yaw
    _write_state(scene.panels["panel_a"], lean_state(0, +1.0, 14.0, scene.build_xy, psi))
    _step(60)
    _write_state(scene.panels["panel_b"], lean_state(1, -1.0, 14.0, scene.build_xy, psi))
    _step(90)
    _report("falsework-in")
    tilt = scene.panel_tilt_deg()[0]
    in_band = c.tilt_lo_deg < float(tilt[0]) < c.tilt_hi_deg \
        and c.tilt_lo_deg < float(tilt[1]) < c.tilt_hi_deg
    check("falsework-in-place: both panels hold settled leans ON the pillar (pair "
          "latches) but pillar_clear refuses the tent — no success, score <= 0.55",
          in_band and bool(scene._pair[0]) and not bool(scene.pillar_clear()[0])
          and not bool(scene.tent_standing()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.55 + 1e-6)
    _REC["on"] = False

    # ================= 7. LONE LEAN FALLS (the mutual-support certificate) ========================
    # Park the pillar, then release ONE panel at a 20 deg in-band lean with nothing
    # to lean on: it must collapse flat, and the debounced lean latch must NOT fire
    # while the panel sweeps through the band with velocity.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    park_pillar()
    _step(60)
    psi = scene.build_yaw
    _write_state(scene.panels["panel_a"],
                 lean_state(0, +1.0, 20.0, scene.build_xy, psi, base_y=+0.030))
    _step(150)
    _report("lone-lean")
    ta = float(scene.panel_tilt_deg()[0, 0])
    check("LONE LEAN FALLS: a single panel released at a 20 deg in-band lean with "
          "no support collapses flat (tilt > 55 deg) and the debounced lean latch "
          "never fires — the tent predicate cannot be met without mutual support",
          ta > c.tilt_hi_deg and not bool(scene._lean[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 8. half-collapse ===========================================================
    # One panel flat, the other resting on it at ~85 deg — a collapsed "tent".
    torch.manual_seed(100)
    env.reset()
    _step(30)
    park_pillar()
    psi = scene.build_yaw
    _write_state(scene.panels["panel_a"],
                 flat_state(0, scene.build_xy, psi, c.pad_t + c.panel_t / 2 + 0.002))
    th = math.radians(84.0)
    lb = c.panel_l[1]
    st = frame_pose(scene.build_xy, psi,
                    -0.15 + (lb / 2) * math.sin(th),
                    base_z + (lb / 2) * math.cos(th), -th)  # u_y > 0: top toward +y
    _write_state(scene.panels["panel_b"], st)
    _step(120)
    _report("half-collapse")
    tb = float(scene.panel_tilt_deg()[0, 1])
    check("half-collapse: one panel flat, the other resting on it at ~85 deg — the "
          "tilt band refuses, no success, score <= 0.01",
          tb > c.tilt_hi_deg and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.01)

    # ================= 9. out-of-zone tent ========================================================
    # A REAL settled tent, built 0.35 m from the build centre (off the pad), pillar
    # parked: everything but the zone clause holds.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    park_pillar()
    psi = scene.build_yaw
    off = torch.stack([-torch.sin(psi), torch.cos(psi)], dim=-1) * 0.35
    construct_tent(scene.build_xy + off, psi)
    _report("out-of-zone")
    top, bot = scene._ends()
    zone_d = float((bot[0, :, 0:2] - scene.build_xy[0]).norm(dim=-1).min())
    apex = float((top[0, 0] - top[0, 1]).norm())
    print(f"[smoke] out-of-zone: nearest base {zone_d:.3f} m from the build centre "
          f"(zone {c.zone_r:.2f} m), apex gap {apex * 1000:.0f}mm", flush=True)
    check("out-of-zone: a real settled tent 0.35 m from the build centre, pillar "
          "parked — the zone clause refuses, no success",
          zone_d > c.zone_r and not bool(scene.tent_standing()[0])
          and not bool(scene.success()[0]))

    # ================= 10. tent OK, pillar not parked =============================================
    # A real tent at the build centre; the pillar left standing on open ground,
    # clear of the tent but off the magenta pad: score reaches the 0.85 cap and
    # stops — success needs the park.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    psi = scene.build_yaw
    st = frame_pose(scene.build_xy, psi, 0.30, 0.004, 0.0)  # off-pad: ground height
    _write_state(scene.pillar, st)
    _step(30)
    construct_tent(scene.build_xy, psi)
    _report("not-parked")
    s10 = float(scene.score()[0])
    check("not-parked: a real free-standing tent (free latch fires) but the pillar "
          "rests on open ground off the magenta pad — no success, score capped at "
          "0.85",
          bool(scene.tent_standing()[0]) and bool(scene._free[0])
          and not bool(scene.pillar_parked()[0])
          and not bool(scene.success()[0]) and 0.80 <= s10 <= 0.85 + 1e-6)
    _REC["on"] = False

    # ================= 11. pillar-adjacent ========================================================
    # Tent geometry OK but the pillar standing within `clear_r` of a panel (not
    # touching): pillar_clear refuses — a tent with the falsework hovering next to
    # it is not demonstrably free-standing.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    psi = scene.build_yaw
    ex = torch.stack([torch.cos(psi), torch.sin(psi)], dim=-1)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:2] = scene.build_xy + ex * 0.13
    st[:, 2] = scene.env_origins[:, 2] + c.pad_t + 0.002
    st[:, 3:7] = _qz(psi)
    _write_state(scene.pillar, st)
    _step(30)
    construct_tent(scene.build_xy, psi)
    _report("pillar-adjacent")
    pos, _u, _v = scene._panel_tensors()
    dmin = float((pos[0, :, 0:2]
                  - scene.pillar.data.root_pos_w[0, None, 0:2]).norm(dim=-1).min())
    tilt = scene.panel_tilt_deg()[0]
    in_band = c.tilt_lo_deg < float(tilt[0]) < c.tilt_hi_deg \
        and c.tilt_lo_deg < float(tilt[1]) < c.tilt_hi_deg
    print(f"[smoke] pillar-adjacent: min pillar-to-panel distance {dmin:.3f} m "
          f"(clear_r {c.clear_r:.2f} m)", flush=True)
    check("pillar-adjacent: tent geometry in band but the pillar stands within "
          "clear_r of a panel — pillar_clear refuses, no success",
          in_band and dmin < c.clear_r and not bool(scene.pillar_clear()[0])
          and not bool(scene.tent_standing()[0]) and not bool(scene.success()[0]))

    # ================= 12. parallel lean ==========================================================
    # Both panels leaning the SAME direction: panel_a on the pillar face, panel_b
    # on panel_a's back — both in band, both settled, but not opposed.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    psi = scene.build_yaw
    _write_state(scene.panels["panel_a"], lean_state(0, +1.0, 14.0, scene.build_xy, psi))
    _step(60)
    _write_state(scene.panels["panel_b"],
                 lean_state(1, +1.0, 25.0, scene.build_xy, psi, base_y=+0.118))
    _step(120)
    _report("parallel-lean")
    tilt = scene.panel_tilt_deg()[0]
    in_band = c.tilt_lo_deg < float(tilt[0]) < c.tilt_hi_deg \
        and c.tilt_lo_deg < float(tilt[1]) < c.tilt_hi_deg
    check("parallel-lean: both panels hold in-band leans facing the SAME way — the "
          "opposed clause refuses the pair and the tent, no success, score <= 0.25",
          in_band and not bool(scene.opposed()[0]) and not bool(scene._pair[0])
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.25 + 1e-6)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.falsework_tent")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    check("video: frames captured and saved to frames.npz", len(_REC["frames"]) > 10)

    n_pass = sum(ok for _n, ok in checks)
    n_tot = len(checks)
    if n_pass == n_tot:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{n_tot}", flush=True)
        code = 0
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{n_tot}", flush=True)
        code = 1

    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 — fail fast, don't hang until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({exc})", flush=True)
        os._exit(1)
