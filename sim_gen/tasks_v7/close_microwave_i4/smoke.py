"""Smoke / rubric-REJECTION battery for DropGateOvenScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the latched
credit is monotone along a real trajectory). This battery proves the rubric REJECTS
wrong outcomes, and that the geometric claims the task rests on — the prop post holds
the gate, the seated gate seals the chamber — are physically load-bearing. Every probe
is CONSTRUCTED as a settled state (teleport, real physics steps, judge) —
instrumentation, never a solution: no probe here reaches success().

Checks:
   1. settle/no-NaN      — seeded reset settles finite; gate RAISED on the post, both
                           cans inside the chamber; score 0, no success;
   2. randomization      — two seeded resets: READBACK oven yaw, oven xy, pad xy and
                           post lateral slot all differ;
   3. arrangement swap   — over 10 resets the blue can occupies BOTH interior slots;
   4. null-policy        — 240 idle steps: the prop post keeps holding the gate up
                           (raised height unchanged), score ~0, no success;
   5. GATE INTERLOCK     — gate seated; the blue can inside is shoved toward the
                           doorway at 3x its own weight for 1.5 s: it never leaves the
                           chamber and the gate stays seated ("once down, the chamber
                           cannot be reached" is physics, not fiat);
   6. SEED STRATEGY      — the seed's plan ("close the door" and nothing else): gate
                           CONSTRUCTED seated with BOTH cans still inside -> no
                           success, score ~0 (the `sealed` latch requires the blue can
                           out, so a bare door-close earns nothing);
   7. wrong object       — RED can on the pad, blue can inside, gate seated -> no
                           success;
   8. near-miss place    — blue can upright on the GROUND just beside the pad (just
                           outside `pad_xy_tol`), gate seated, red in -> no success;
   9. near-miss gate     — gate resting on the TOPPLED post lying in the doorway
                           (bottom ~25 mm above its seat), blue on pad, red in -> no
                           success;
  10. tipped can         — blue can lying on its SIDE on the pad, gate seated, red in
                           -> no success (upright clause);
  11. gate discarded     — gate lying FLAT on the ground outside the doorway, blue on
                           pad, red in -> no success (a removed gate is not a closed
                           gate);
  12. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.close_microwave_i4.smoke --headless
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

_qmul, _qy = task_scene._qmul, task_scene._qy

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


def _write_body(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    gl = scene._oven_local(scene.gate.data.root_pos_w)[0]
    print(f"[smoke] {tag:18s} | gate_z={float(gl[2]):+.3f} "
          f"blue_out={bool(scene.blue_outside()[0])} "
          f"on_pad={bool(scene.blue_on_pad()[0])} red_in={bool(scene.red_inside()[0])} "
          f"seated={bool(scene.gate_seated()[0])} "
          f"out={bool(scene._out[0])} placed={bool(scene._placed[0])} "
          f"sealed={bool(scene._sealed[0])} score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


def _push(body, force_w: torch.Tensor, steps: int) -> None:
    """Apply a world-frame force at the body's CoM for `steps` steps, then clear."""
    n = _ENV.num_envs
    zero = torch.zeros(n, 1, 3, device=_ENV.device)
    f = force_w.view(1, 1, 3).expand(n, 1, 3).contiguous()
    for _ in range(steps):
        body.set_external_force_and_torque(f, zero, env_ids=_all_ids(), is_global=True)
        _step(1)
    body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.drop_gate_oven")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.55, -0.55, 0.60)) + o),
                                tuple(np.array((0.35, 0.08, 0.10)) + o),
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

    def oven_world(loc_xyz) -> torch.Tensor:
        """Oven-local point -> world (per-env)."""
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.oven.data.root_pos_w + quat_apply(scene.oven.data.root_quat_w, loc)

    def oven_quat(q_extra: torch.Tensor | None = None) -> torch.Tensor:
        _refresh()
        q = scene.oven.data.root_quat_w
        return q if q_extra is None else _qmul(q, q_extra)

    def park_post() -> None:
        """Move the post to a rest spot on the ground away from the oven."""
        p = oven_world((0.30, -0.25, 0.0))
        p[:, 2] = env.iscene.env_origins[:, 2] + c.post_xy / 2
        _write_body(scene.post, p,
                    oven_quat(_qy(torch.full((n,), math.pi / 2, device=device))))

    def seat_gate() -> None:
        """CONSTRUCT the gate seated in its slots (post parked first), then settle."""
        park_post()
        _write_body(scene.gate, oven_world((c.x_gate, 0.0, c.z_seat + 0.002)),
                    oven_quat())
        _step(90)

    def blue_to_pad() -> None:
        p = scene.pad.data.root_pos_w.clone()
        p[:, 2] += c.pad_t / 2 + c.can_h / 2 + 0.01
        _write_body(scene.blue, p, None)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    gz = float(scene._oven_local(scene.gate.data.root_pos_w)[0, 2])
    check("settle/no-NaN: layout settles finite; gate RAISED on the post, both cans "
          "inside the chamber; score 0, no success",
          bool(scene._finite()[0]) and gz > c.z_seat + 0.08
          and not bool(scene.blue_outside()[0]) and bool(scene.red_inside()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene.oven.data.root_quat_w[0]),
                scene.oven.data.root_pos_w[0, :2].clone(),
                scene.pad.data.root_pos_w[0, :2].clone(),
                float(scene._oven_local(scene.post.data.root_pos_w)[0, 1]))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_op, a_pp, a_py = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_op, b_pp, b_py = readback()
    d_yawv = dyaw(a_yaw, b_yaw)
    d_op, d_pp = float((a_op - b_op).norm()), float((a_pp - b_pp).norm())
    d_py = abs(a_py - b_py)
    print(f"[smoke] randomization deltas: oven_yaw={d_yawv:.1f}deg "
          f"oven_xy={d_op * 1000:.1f}mm pad_xy={d_pp * 1000:.1f}mm "
          f"post_y={d_py * 1000:.1f}mm", flush=True)
    check("randomization-is-real: oven yaw, oven xy, pad xy, post slot readback differ",
          d_yawv > 2.0 and d_op > 0.003 and d_pp > 0.003 and d_py > 0.003)

    # ================= 3. can arrangement swap ====================================================
    sides = set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        sides.add("+" if float(scene.blue_slot[0]) > 0 else "-")
    print(f"[smoke] over 10 resets: blue can slots {sorted(sides)}", flush=True)
    check("arrangement swap: the blue can occupies BOTH interior slots over 10 resets",
          sides == {"+", "-"})

    # ================= 4. null policy fails (and the prop post is load-bearing) ===================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    gz = float(scene._oven_local(scene.gate.data.root_pos_w)[0, 2])
    check("null-policy-fails: 240 idle steps, the post keeps the gate raised, "
          "score ~0, no success",
          gz > c.z_seat + 0.08 and float(scene.score()[0]) <= 0.05
          and not bool(scene.success()[0]))

    # ================= 5. GATE INTERLOCK: a sealed chamber cannot be raided =======================
    # "once the gate is down you cannot reach the cans": with the gate seated, shove
    # the blue can toward the doorway at 3x its weight for 1.5 s — the can must stay
    # in and the gate must stay seated. This is the physics behind the ordering claim.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    seat_gate()
    _refresh()
    assert bool(scene.gate_seated()[0]), "probe setup: gate must seat in its slots"
    door_w = quat_apply(scene.oven.data.root_quat_w,
                        torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]
    f = 3.0 * c.can_mass * 9.81 * door_w
    for _ in range(9):  # 9 x 20 = 180 steps = 1.5 s of shoving
        _push(scene.blue, f, 20)
    _step(60)
    _report("gate-interlock")
    bx = float(scene._oven_local(scene.blue.data.root_pos_w)[0, 0])
    print(f"[smoke] interlock shove: blue oven-frame x={bx * 1000:.0f}mm "
          f"(gate plane {c.x_gate * 1000:.0f}mm)", flush=True)
    check("GATE INTERLOCK: 3x-weight shove toward the doorway for 1.5 s never gets "
          "the can out of the sealed chamber; the gate stays seated",
          not bool(scene.blue_outside()[0]) and bool(scene.gate_seated()[0])
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6. negative: the SEED'S OWN STRATEGY =======================================
    # "Close the microwave door" — the seed's ONLY act. CONSTRUCT its end state here:
    # the gate seated, NOTHING else done (both cans still inside). Must not be
    # success, and must score ~0: `sealed` only latches with the blue can out.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    seat_gate()
    _report("seed-strategy")
    check("negative (SEED strategy): gate seated with BOTH cans still inside — the "
          "bare door-close earns NO success and score ~0",
          bool(scene.gate_seated()[0]) and bool(scene.red_inside()[0])
          and not bool(scene.blue_outside()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.05)
    _REC["on"] = False

    # ================= 7. negative: wrong object ==================================================
    # The RED can extracted and seated on the pad, the BLUE can left inside, gate
    # seated: a complete, tidy episode — of the wrong can. Must not be success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    p = scene.pad.data.root_pos_w.clone()
    p[:, 2] += c.pad_t / 2 + c.can_h / 2 + 0.01
    _write_body(scene.red, p, None)
    seat_gate()
    _step(60)
    _report("wrong-object")
    check("negative (wrong object): RED can on the pad, blue can sealed inside — "
          "no success",
          not bool(scene.success()[0]) and not bool(scene.blue_on_pad()[0])
          and not bool(scene.red_inside()[0]))
    _REC["on"] = False

    # ================= 8. near-miss: blue can beside the pad ======================================
    # Blue can upright on the GROUND, its centre ~65 mm from the pad axis — just
    # outside `pad_xy_tol` (45 mm) and off the pad top. Everything else perfect.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    p = scene.pad.data.root_pos_w.clone()
    p[:, 0] += c.pad_r + 0.005  # 65 mm from the axis: standing on the ground beside it
    p[:, 2] = env.iscene.env_origins[:, 2] + c.can_h / 2 + 0.005
    _write_body(scene.blue, p, None)
    seat_gate()
    _step(60)
    _report("near-miss-pad")
    d_xy = float((scene.blue.data.root_pos_w[0, :2] - scene.pad.data.root_pos_w[0, :2]).norm())
    print(f"[smoke] near-miss distance from pad axis: {d_xy * 1000:.0f}mm "
          f"(tol {c.pad_xy_tol * 1000:.0f}mm)", flush=True)
    check("near-miss (placement): blue can upright on the ground just beside the pad "
          "— no success, score <= 0.75",
          d_xy > c.pad_xy_tol and not bool(scene.blue_on_pad()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.75)

    # ================= 9. near-miss: gate resting on the toppled post =============================
    # The post knocked over but left LYING in the doorway; the gate falls onto it and
    # hangs ~25 mm above its seat. Blue on pad, red in — only the gate clause fails.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    blue_to_pad()
    # post lying on its side across the doorway floor, under the gate plane
    q_side = oven_quat(_qy(torch.full((n,), math.pi / 2, device=device)))
    _write_body(scene.post, oven_world((c.post_x, 0.0, c.zf + c.post_xy / 2 + 0.002)),
                q_side)
    _write_body(scene.gate, oven_world((c.x_gate, 0.0, c.z_seat + 0.06)), oven_quat())
    _step(120)
    _report("near-miss-gate")
    gz = float(scene._oven_local(scene.gate.data.root_pos_w)[0, 2])
    print(f"[smoke] gate hangs at z={gz * 1000:.1f}mm "
          f"(seat {c.z_seat * 1000:.1f}mm, tol {c.gate_z_tol * 1000:.0f}mm)", flush=True)
    check("near-miss (gate): gate resting on the toppled post, ~2 cm above its seat "
          "— not seated, no success, score <= 0.75",
          gz > c.z_seat + c.gate_z_tol and not bool(scene.gate_seated()[0])
          and bool(scene.blue_on_pad()[0]) and bool(scene.red_inside()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.75)
    _REC["on"] = False

    # ================= 10. negative: blue can tipped on the pad ===================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    p = scene.pad.data.root_pos_w.clone()
    p[:, 2] += c.pad_t / 2 + c.can_r + 0.005
    _write_body(scene.blue, p,
                _qy(torch.full((n,), math.pi / 2, device=device)))
    seat_gate()
    _step(90)
    _report("tipped-can")
    check("negative (tipped): blue can lying on its SIDE on the pad — upright clause "
          "refuses, no success",
          not bool(scene.blue_on_pad()[0]) and not bool(scene.success()[0]))

    # ================= 11. negative: gate discarded ===============================================
    # The gate removed from its slots entirely and laid FLAT on the ground in front
    # of the oven (low z!) — a removed door is not a closed door.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    blue_to_pad()
    park_post()
    g = oven_world((0.30, 0.10, 0.0))
    g[:, 2] = env.iscene.env_origins[:, 2] + c.gate_t / 2 + 0.003
    _write_body(scene.gate, g,
                oven_quat(_qy(torch.full((n,), math.pi / 2, device=device))))
    _step(90)
    _report("gate-discarded")
    check("negative (gate discarded): gate lying flat on the ground outside the "
          "doorway — not seated, no success",
          not bool(scene.gate_seated()[0]) and bool(scene.blue_on_pad()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.75)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.drop_gate_oven")
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
    main()
