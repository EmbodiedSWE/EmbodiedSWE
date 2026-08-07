"""Smoke / rubric-REJECTION battery for HookDenRetrievalScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the latched
credit is monotone along a real rake-out trajectory, two+ seeds). This battery proves
the rubric REJECTS wrong outcomes, and that the mechanism claim the task rests on —
the roof makes a direct lift physically impossible, so the cube must be DRAGGED out —
is physics, not fiat. Every probe is CONSTRUCTED as a settled state (teleport, real
physics steps, judge) — instrumentation, never a solution.

Checks:
   1. settle/no-NaN      — seeded reset settles finite; cargo cube inside the den,
                           hook in the open and NOT engaged; score ~0, no success;
   2. randomization      — two seeded resets: READBACK den yaw, den xy, the cargo
                           cube's den-frame spot and the hook's pose all differ;
   3. side/roll swap     — over 10 resets the cargo cube occupies BOTH sides of the
                           den centreline AND the hook spawns with BOTH toe rolls;
   4. null-policy        — 240 idle steps: score ~0, no success;
   5. ROOF LIFT-BLOCK    — a 2.5x-weight UPWARD pull on the denned cube for 1.5 s:
                           the roof holds it in — the cube never leaves the den (the
                           seed's own "grasp and lift" is physically impossible
                           here; extraction must be a drag through the mouth);
   6. engage geometry    — hook toe constructed IN FRONT of the cube (mouth side,
                           within lateral reach): the engaged latch must NOT set —
                           scraping at the cube's face earns nothing;
   7. engage anchor      — hook toe constructed BEHIND the cube (the raking pocket):
                           engaged latches (0.15) but nothing more — no success;
   8. SEED STRATEGY      — the seed's end state ("lift it up to the target"): cargo
                           cube settled up on the DEN ROOF — elevated, but not on
                           the pedestal: no success, score ~0;
   9. near-miss offset   — cargo cube on the pedestal TOP but 45 mm off-axis
                           (outside on_ped_xy_tol): no success;
  10. near-miss beside   — cargo cube settled on the ground touching the pedestal's
                           base: no success;
  11. wrong object       — RED decoy on the pedestal, cargo still denned: no
                           success, score ~0;
  12. both on top        — cargo centred on the pedestal AND the decoy up there too:
                           the decoy clause alone refuses — no success;
  13. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.pick_and_lift_small_i55.smoke --headless
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

_qmul, _qx, _qz = task_scene._qmul, task_scene._qx, task_scene._qz

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
    gd = scene._den_local(scene.cargo.data.root_pos_w)[0]
    td = scene._den_local(scene.toe_mid_w())[0]
    print(f"[smoke] {tag:16s} | cargo_den=({float(gd[0]):+.3f},{float(gd[1]):+.3f},"
          f"{float(gd[2]):+.3f}) toe_den=({float(td[0]):+.3f},{float(td[1]):+.3f}) "
          f"in_den={bool(scene.in_den(scene.cargo.data.root_pos_w)[0])} "
          f"eng={bool(scene._engaged[0])} ext={bool(scene._extracted[0])} "
          f"on_ped={bool(scene.on_pedestal(scene.cargo)[0])} "
          f"decoy_on={bool(scene.on_pedestal(scene.decoy)[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


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
    env = ENVS.get("simgen.hook_den_retrieval")().build(num_envs=args.num_envs,
                                                        device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.55, -0.90, 0.60)) + o),
                                tuple(np.array((0.28, 0.0, 0.06)) + o),
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

    def den_world(loc_xyz) -> torch.Tensor:
        """Den-local point -> world (z passed through as absolute height)."""
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        out = scene.den.data.root_pos_w + quat_apply(scene.den.data.root_quat_w, loc)
        out = out.clone()
        out[:, 2] = loc[0, 2]
        return out

    def ped_world(loc_xyz) -> torch.Tensor:
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.pedestal.data.root_pos_w + quat_apply(
            scene.pedestal.data.root_quat_w, loc)

    def den_quat(q_extra: torch.Tensor | None = None) -> torch.Tensor:
        _refresh()
        q = scene.den.data.root_quat_w
        return q if q_extra is None else _qmul(q, q_extra)

    def cargo_side() -> float:
        _refresh()
        return 1.0 if float(scene._den_local(scene.cargo.data.root_pos_w)[0, 1]) > 0 \
            else -1.0

    def hook_rolled() -> bool:
        """True when the hook lies toe-to--y (rolled pi about the handle)."""
        _refresh()
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
        return float(quat_apply(scene.hook.data.root_quat_w, ez)[0, 2]) < 0.0

    def write_hook_at(corner_den_xy, s: float) -> None:
        """Construct the hook flat on the den floor/ground, toe pointing at side s."""
        roll = math.pi if s < 0 else 0.0
        z = 0.018 if roll else 0.002
        pos = den_world((corner_den_xy[0], corner_den_xy[1], z))
        write_q = den_quat(_qx(torch.full((n,), roll, device=device)))
        _write_body(scene.hook, pos, write_q)

    def z_ped_top() -> float:
        return c.ped_size[2] / 2 + c.cube_size / 2 + 0.001

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    check("settle/no-NaN: layout settles finite; cargo denned, hook in the open and "
          "not engaged; score ~0, no success",
          bool(scene._finite()[0])
          and bool(scene.in_den(scene.cargo.data.root_pos_w)[0])
          and not bool(scene.hook_engaged()[0])
          and not bool(scene.in_den(scene.hook.data.root_pos_w)[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene.den.data.root_quat_w[0]),
                scene.den.data.root_pos_w[0, :2].clone(),
                scene._den_local(scene.cargo.data.root_pos_w)[0, :2].clone(),
                scene.hook.data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_dp, a_cp, a_hp = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_dp, b_cp, b_hp = readback()
    d_yawv, d_dp = dyaw(a_yaw, b_yaw), float((a_dp - b_dp).norm())
    d_cp, d_hp = float((a_cp - b_cp).norm()), float((a_hp - b_hp).norm())
    print(f"[smoke] randomization deltas: den_yaw={d_yawv:.1f}deg "
          f"den_xy={d_dp * 1000:.1f}mm cargo_den_xy={d_cp * 1000:.1f}mm "
          f"hook_xy={d_hp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: den yaw, den xy, the cargo's den-frame spot and "
          "the hook pose readback differ across seeds",
          d_yawv > 2.0 and d_dp > 0.003 and d_cp > 0.004 and d_hp > 0.010)

    # ================= 3. cargo side + hook roll swap =============================================
    sides, rolls = set(), set()
    for s_i in range(10):
        torch.manual_seed(300 + s_i)
        env.reset()
        sides.add("+" if cargo_side() > 0 else "-")
        rolls.add("rolled" if hook_rolled() else "flat")
    print(f"[smoke] over 10 resets: cargo sides {sorted(sides)}, "
          f"hook rolls {sorted(rolls)}", flush=True)
    check("side/roll swap: the cargo occupies BOTH den sides and the hook spawns "
          "with BOTH toe rolls over 10 resets",
          sides == {"+", "-"} and rolls == {"rolled", "flat"})

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. ROOF LIFT-BLOCK: the seed's grasp-and-lift is impossible ================
    # A 2.5x-weight upward pull on the denned cube for 1.5 s: the roof (60 mm) holds
    # it in. The cube must never leave the den — extraction has to be a DRAG through
    # the mouth. This is the physics behind the strategic claim.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    f_up = torch.tensor([0.0, 0.0, 2.5 * c.cube_mass * 9.81], device=device)
    stayed = True
    zmax = 0.0
    for _ in range(9):  # 9 x 20 = 180 steps = 1.5 s of pulling
        _push(scene.cargo, f_up, 20)
        _refresh()
        stayed &= bool(scene.in_den(scene.cargo.data.root_pos_w)[0])
        zmax = max(zmax, float(scene._den_local(scene.cargo.data.root_pos_w)[0, 2]))
    _step(90)
    _report("roof-block")
    print(f"[smoke] lift-block: cargo peak den-frame z={zmax * 1000:.0f}mm "
          f"(roof underside {c.int_h * 1000:.0f}mm)", flush=True)
    check("ROOF LIFT-BLOCK: a 2.5x-weight upward pull for 1.5 s never gets the "
          "denned cube out (the roof is the physical stop)",
          stayed and bool(scene.in_den(scene.cargo.data.root_pos_w)[0])
          and not bool(scene._extracted[0]) and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6. engage geometry: toe IN FRONT of the cube earns nothing =================
    # The toe placed on the MOUTH side of the cube, within lateral reach: pressing at
    # the cube's face is not a rake — the engaged latch must NOT set.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    s = cargo_side()
    _write_body(scene.cargo, den_world((-0.055, s * 0.030, c.cube_size / 2 + 0.001)),
                den_quat())
    _step(30)
    _refresh()
    gd = scene._den_local(scene.cargo.data.root_pos_w)[0]
    write_hook_at((float(gd[0]) + 0.030, float(gd[1]) - s * 0.033), s)
    _step(60)
    _report("toe-in-front")
    td = scene._den_local(scene.toe_mid_w())[0]
    check("engage geometry: toe IN FRONT of the cube (within lateral reach) does "
          "NOT latch engaged",
          bool(scene.in_den(scene.cargo.data.root_pos_w)[0])
          and float(td[0]) > float(scene._den_local(scene.cargo.data.root_pos_w)[0, 0])
          and abs(float(td[1]) - float(gd[1])) < c.engage_y_reach
          and not bool(scene._engaged[0]) and float(scene.score()[0]) <= 0.01)

    # ================= 7. engage anchor: toe BEHIND the cube = 0.15, nothing more =================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    s = cargo_side()
    _write_body(scene.cargo, den_world((-0.055, s * 0.030, c.cube_size / 2 + 0.001)),
                den_quat())
    _step(30)
    _refresh()
    gd = scene._den_local(scene.cargo.data.root_pos_w)[0]
    write_hook_at((float(gd[0]) - 0.040, float(gd[1]) - s * 0.033), s)
    _step(60)
    _report("toe-behind")
    check("engage anchor: toe BEHIND the cube latches engaged (score 0.15) and "
          "nothing more — no success",
          bool(scene._engaged[0])
          and abs(float(scene.score()[0]) - c.w_engaged) < 0.01
          and not bool(scene.success()[0]))

    # ================= 8. negative: the SEED'S OWN END STATE ======================================
    # The seed's goal is "pick up the cube and lift it up to the target" — an elevated
    # cube. Construct the nearest analogue: the cargo cube settled up on the DEN ROOF.
    # Elevated, in the open air — but not on the pedestal: no credit.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_body(scene.cargo,
                den_world((0.0, 0.0, c.int_h + 0.012 + c.cube_size / 2 + 0.003)),
                den_quat())
    _step(120)
    _report("seed-strategy")
    check("negative (SEED strategy): cargo lifted-up onto the den roof — elevated "
          "but not on the pedestal: no success, score ~0",
          not bool(scene.in_den(scene.cargo.data.root_pos_w)[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.05)
    _REC["on"] = False

    # ================= 9. near-miss: on the pedestal top but off-axis =============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_body(scene.cargo, ped_world((0.045, 0.0, z_ped_top())),
                scene.pedestal.data.root_quat_w)
    _step(120)
    _report("off-axis")
    pl = scene._ped_local(scene.cargo.data.root_pos_w)[0]
    check("near-miss (off-axis): cargo ON the pedestal top but 45 mm off-axis "
          "(outside on_ped_xy_tol) — no success",
          float(pl[2]) > c.ped_size[2] / 2  # it really is up on the top
          and float(pl[:2].norm()) > c.on_ped_xy_tol
          and not bool(scene.on_pedestal(scene.cargo)[0])
          and not bool(scene.success()[0]))

    # ================= 10. near-miss: on the ground beside the pedestal ===========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_body(scene.cargo,
                ped_world((c.ped_size[0] / 2 + c.cube_size / 2 + 0.002, 0.0,
                           -c.ped_size[2] / 2 + c.cube_size / 2 + 0.002)),
                scene.pedestal.data.root_quat_w)
    _step(120)
    _report("beside-ped")
    check("near-miss (beside): cargo settled on the ground touching the pedestal "
          "base — no success",
          not bool(scene.on_pedestal(scene.cargo)[0])
          and not bool(scene.success()[0]))

    # ================= 11. negative: wrong object =================================================
    # The RED decoy placed neatly on the pedestal, the cargo still denned: a complete,
    # tidy pick-and-place — of the wrong cube. No success, and score ~0 (no latches).
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_body(scene.decoy, ped_world((0.0, 0.0, z_ped_top())),
                scene.pedestal.data.root_quat_w)
    _step(120)
    _report("wrong-object")
    check("negative (wrong object): RED decoy on the pedestal, cargo still denned — "
          "no success, score ~0",
          bool(scene.on_pedestal(scene.decoy)[0])
          and bool(scene.in_den(scene.cargo.data.root_pos_w)[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.05)
    _REC["on"] = False

    # ================= 12. negative: both cubes on the pedestal ===================================
    # Cargo centred on the top AND the decoy up there beside it: every clause but
    # ~on_pedestal(decoy) holds — the decoy clause alone must refuse.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_body(scene.cargo, ped_world((-0.025, 0.0, z_ped_top())),
                scene.pedestal.data.root_quat_w)
    _write_body(scene.decoy, ped_world((0.025, 0.0, z_ped_top())),
                scene.pedestal.data.root_quat_w)
    _step(120)
    _report("both-on-top")
    check("negative (both on top): cargo on the pedestal AND the decoy too — the "
          "decoy clause refuses, no success",
          bool(scene.on_pedestal(scene.cargo)[0])
          and bool(scene.on_pedestal(scene.decoy)[0])
          and not bool(scene.success()[0]))

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.hook_den_retrieval")
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
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(2)
