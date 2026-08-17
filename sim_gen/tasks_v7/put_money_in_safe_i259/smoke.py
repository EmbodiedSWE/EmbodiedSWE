"""Smoke / rubric-REJECTION battery for SeesawVaultScene — NullRobot, teleported probes.

solve.py is the acceptance proof (park the counterweight on the pedal pan so its
standing weight swings the lid open, drop the brick through the uncovered mouth,
unpark the weight so gravity reseals). This battery proves the rubric REJECTS
every wrong outcome and that the SEE-SAW TOLL — the task's strategic
differentiator from the seed — is physically load-bearing. Every probe here is
teleport + gravity: the mechanism is judged by what standing weight alone does.

Checks:
   1. settle/no-NaN      — seeded reset settles finite; lid rests on its closed
                           stop, brick flat on its stand; score ~0, no success;
   2. randomization      — two seeded resets: READBACK vault yaw, vault xy, stand
                           position and brick yaw all differ;
   3. side-flip          — across six seeded resets the brick/weight PEDESTAL
                           assignment flips (both sides observed) and the brick
                           readback sits on the flagged stand every time;
   4. null-policy        — 240 idle steps: lid stays on its stop, score ~0,
                           no success;
   5. SEED STRATEGY      — the seed's whole plan ("put the money in the safe"):
                           the brick set down on the vault — it lands ON the
                           closed lid over the mouth and stays there; refused
                           entry, no credit, no success;
   6. unpaid-toll drop   — the solve's OWN deposit drop (same hover, same
                           orientation) with the toll unpaid: the closed lid
                           refuses the brick — it never enters, the lid barely
                           moves; order is forced physically;
   7. brick-cannot-pay   — the brick itself laid on the pedal pan: its torque is
                           under the lid's gravity bias — the lid stays closed
                           (the dedicated counterweight is the only tool);
   8. banked-unsealed    — full deposit but the weight LEFT in the pan: brick
                           inside AND lid open — success false, score <= 0.60
                           (the reseal clause has teeth);
   9. wrong-object       — the WEIGHT constructed inside the chamber, brick still
                           on its pedestal: no credit, no success;
  10. stop-reality       — the parked weight drives the lid to its 60 deg stop
                           and HOLDS it there through a 1 s window; unparking
                           closes it back onto the 0 deg stop by gravity alone;
  11. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.put_money_in_safe_i259.smoke --headless
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

_qmul, _qz = task_scene._qmul, task_scene._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

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
    lid = float(scene.lid_angle_deg()[0])
    p = scene._vault_local(scene.brick.data.root_pos_w)[0]
    w = scene._vault_local(scene.weight.data.root_pos_w)[0]
    print(f"[smoke] {tag:16s} | lid={lid:+7.2f}deg "
          f"brick=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}) "
          f"weight=({float(w[0]):+.3f},{float(w[1]):+.3f},{float(w[2]):+.3f}) "
          f"inside={bool(scene.brick_inside()[0])} closed={bool(scene.lid_closed()[0])} "
          f"open={bool(scene.lid_open()[0])} settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.seesaw_vault")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.30, -1.05, 0.95)) + o),
                                tuple(np.array((0.10, 0.00, 0.22)) + o),
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

    def ang() -> float:
        _refresh()
        return float(scene.lid_angle_deg()[0])

    def brick_local() -> torch.Tensor:
        _refresh()
        return scene._vault_local(scene.brick.data.root_pos_w)[0]

    def vault_world(loc_xyz) -> torch.Tensor:
        _refresh()
        t = torch.zeros(n, 3, device=device)
        t[:] = torch.tensor(loc_xyz, device=device, dtype=torch.float)
        return scene.vault.data.root_pos_w + quat_apply(scene.vault.data.root_quat_w, t)

    def put_brick_vault_local(loc_xyz, extra_yaw_deg: float = 0.0) -> None:
        """Teleport the brick to a vault-local point (zero velocity), long axis on
        the vault x heading + optional extra yaw."""
        q = _qmul(scene.vault.data.root_quat_w,
                  _qz(torch.full((n,), math.radians(extra_yaw_deg), device=device)))
        _write_body(scene.brick, vault_world(loc_xyz), q)

    # the solve's own transport writes, verbatim geometry
    pan_cx = (c.pan_x0 + c.pan_x1) / 2
    pan_floor_top = c.axis_z - 0.020 + 0.004
    fence_top = c.axis_z - 0.001 + c.fence_h / 2

    def park_weight_in_pan() -> None:
        _write_body(scene.weight, vault_world((pan_cx, 0.0, pan_floor_top + 0.006)),
                    scene.vault.data.root_quat_w)

    def unpark_weight() -> None:
        _refresh()
        home = scene.stand_b.data.root_pos_w if bool(scene.brick_on_a[0]) \
            else scene.stand_a.data.root_pos_w
        pos = home.clone()
        pos[:, 2] = pos[:, 2] + c.stand_h + 0.004
        _write_body(scene.weight, pos, None)

    def wait_until(pred, max_steps: int, streak_need: int = 30) -> int:
        streak = 0
        for i in range(max_steps):
            _step(1)
            if pred():
                streak += 1
                if streak >= streak_need:
                    return i + 1
            else:
                streak = 0
        return max_steps

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(180)
    _report("settle")
    _REC["on"] = False
    up_z = float(quat_apply(scene.brick.data.root_quat_w,
                            torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3))[0, 2])
    brick_z = float(scene.brick.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    check("settle/no-NaN: seeded reset settles finite; lid rests on its closed "
          "stop, brick flat on its stand; score ~0, no success",
          bool(scene._finite()[0]) and abs(ang()) < 3.0
          and abs(brick_z - (c.stand_h + c.brick_h / 2)) < 0.02 and abs(up_z) > 0.95
          and float(scene.score()[0]) <= 0.03 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        vyaw = yaw_of(scene.vault.data.root_quat_w[0])
        vp = scene.vault.data.root_pos_w[0, :2].clone()
        sp = scene.stand_a.data.root_pos_w[0, :2].clone()
        byaw = yaw_of(scene.brick.data.root_quat_w[0])
        return vyaw, vp, sp, byaw

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_vp, a_sp, a_byaw = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_vp, b_sp, b_byaw = readback()
    d_vyaw = dyaw(a_yaw, b_yaw)
    d_vp = float((a_vp - b_vp).norm())
    d_sp = float((a_sp - b_sp).norm())
    d_byaw = dyaw(a_byaw, b_byaw)
    print(f"[smoke] randomization deltas: vault_yaw={d_vyaw:.1f}deg "
          f"vault_xy={d_vp * 1000:.1f}mm standA_xy={d_sp * 1000:.1f}mm "
          f"brick_yaw={d_byaw:.1f}deg", flush=True)
    check("randomization-is-real: vault yaw, vault xy, stand position and brick "
          "yaw readback all differ across seeds",
          d_vyaw > 2.0 and d_vp > 0.003 and d_sp > 0.02 and d_byaw > 5.0)

    # ================= 3. side assignment flips (readback matches the flag) =======================
    sides = []
    match_ok = True
    for s in (11, 22, 33, 44, 55, 66):
        torch.manual_seed(s)
        env.reset()
        _step(4)
        _refresh()
        side = bool(scene.brick_on_a[0])
        sides.append(side)
        stand = scene.stand_a if side else scene.stand_b
        d = float((scene.brick.data.root_pos_w[0, :2] - stand.data.root_pos_w[0, :2]).norm())
        match_ok = match_ok and d < 0.10
    print(f"[smoke] side flags over 6 seeds: {sides}", flush=True)
    check("side-flip: the brick/weight pedestal assignment flips across seeds "
          "(both sides observed) and the brick readback sits on the flagged stand",
          any(sides) and not all(sides) and match_ok)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, the lid keeps its closed stop, the "
          "brick stays on its stand, score ~0, no success",
          abs(ang()) < 3.0 and float(scene.score()[0]) <= 0.03
          and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED'S OWN STRATEGY =======================================
    # rlbench/put_money_in_safe: grasp the money, carry it to the safe, SET IT
    # DOWN inside — one place onto a shelf behind an already-open door. Here no
    # open face exists: the exact analogue is setting the brick down on the vault
    # over the deposit mouth. It lands ON the closed lid and stays there —
    # refused entry, no partial credit, never success.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    lid_top = c.axis_z - 0.003 + c.lid_t / 2
    mouth_cx = (c.mouth_x0 + c.mouth_x1) / 2
    put_brick_vault_local((mouth_cx, 0.0, lid_top + c.brick_h / 2 + 0.004),
                          extra_yaw_deg=90.0)
    _step(300)
    _report("seed-on-lid")
    _REC["on"] = False
    p = brick_local()
    check("negative (SEED strategy): the brick set down on the vault lands ON the "
          "closed lid over the mouth and rests there — refused entry, no credit, "
          "no success",
          float(p[2]) > c.roof_z1 and abs(float(p[0])) < c.half_x
          and abs(float(p[1])) < c.half_y and not bool(scene.brick_inside()[0])
          and abs(ang()) < 3.0 and float(scene.score()[0]) <= 0.03
          and not bool(scene.success()[0]))

    # ================= 6. negative: the solve's own drop with the toll UNPAID =====================
    # Same hover, same orientation as the solve's deposit — but the lid is closed.
    # The 10 mm plate over the mouth must refuse the 30 mm brick outright.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    put_brick_vault_local((c.drop_x, 0.0, c.drop_hover_z), extra_yaw_deg=90.0)
    _step(300)
    _report("unpaid-drop")
    _REC["on"] = False
    p = brick_local()
    check("negative (unpaid toll): the solve's own deposit drop against the "
          "CLOSED lid is refused — the brick lands on the lid and never enters, "
          "the lid barely moves; order is forced physically",
          float(p[2]) > c.roof_z1 - 0.005 and float(p[2]) < c.drop_hover_z - 0.05
          and not bool(scene.brick_inside()[0]) and abs(ang()) < 3.0
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.03)

    # ================= 7. negative: the brick cannot pay the toll =================================
    # Lay the brick on the pedal pan (across the fence tops, long axis on y —
    # full pan-centre radius). Its torque is under the lid's closed bias: the
    # lid must NOT open. The counterweight is the only tool that works.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    put_brick_vault_local((pan_cx, 0.0, fence_top + c.brick_h / 2 + 0.004),
                          extra_yaw_deg=90.0)
    _step(360)
    _report("brick-toll")
    _REC["on"] = False
    p = brick_local()
    on_pan = (c.pan_x0 - 0.03 < float(p[0]) < c.pan_x1 + 0.06
              and 0.24 < float(p[2]) < 0.40)
    check("negative (brick cannot pay): the brick laid on the pedal pan loads it "
          "at full radius yet the lid stays closed (torque under the gravity "
          "bias) — no open credit, no success",
          on_pan and ang() < 8.0 and not bool(scene.lid_open()[0])
          and float(scene.score()[0]) < 0.10 and not bool(scene.success()[0]))

    # ================= 8. near-miss: banked but UNSEALED ==========================================
    # Run the real first two phases (weight parked -> lid open -> brick dropped
    # in) and then STOP: brick genuinely inside, lid genuinely open — the reseal
    # clause must refuse it and cap the score at 0.25 + 0.35.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    park_weight_in_pan()
    wait_until(lambda: ang() > c.open_min_deg
               and float(scene.lever.data.root_ang_vel_w[0].norm()) < 0.25, 720)
    assert ang() > c.open_min_deg, f"probe setup: lid must open, lid={ang():.2f}"
    put_brick_vault_local((c.drop_x, 0.0, c.drop_hover_z), extra_yaw_deg=90.0)
    wait_until(lambda: bool(scene.brick_inside()[0])
               and float(scene.brick.data.root_lin_vel_w[0].norm()) < c.settle_speed, 600)
    _step(120)
    _report("banked-unsealed")
    _REC["on"] = False
    check("near-miss (unsealed): brick genuinely inside but the counterweight "
          "left in the pan holds the lid open — success false, score <= 0.60",
          bool(scene.brick_inside()[0]) and bool(scene._in[0])
          and bool(scene.lid_open()[0]) and not bool(scene.lid_closed()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.6000005)

    # ================= 9. near-miss: the WRONG object inside ======================================
    # The counterweight constructed inside the chamber, brick untouched on its
    # pedestal: the rubric judges the BRICK — no credit, no success.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _write_body(scene.weight, vault_world((mouth_cx, 0.0, 0.020)),
                scene.vault.data.root_quat_w)
    _step(240)
    _report("wrong-object")
    _refresh()
    w = scene._vault_local(scene.weight.data.root_pos_w)[0]
    check("near-miss (wrong object): the counterweight inside the chamber and "
          "the brick still on its pedestal earn nothing — no success, score ~0",
          float(w[2]) < c.in_top_z and not bool(scene.brick_inside()[0])
          and float(scene.score()[0]) <= 0.03 and not bool(scene.success()[0]))

    # ================= 10. stop-reality: open stop holds, gravity reseals =========================
    # The parked weight must drive the lid to the 60 deg stop and HOLD it there
    # through a 1 s window (the third hand is real); unparking must close it
    # back onto the 0 deg stop with no applied wrench anywhere.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    park_weight_in_pan()
    wait_until(lambda: ang() > c.open_min_deg
               and float(scene.lever.data.root_ang_vel_w[0].norm()) < 0.25, 720)
    _step(60)
    lo, hi = 1e9, -1e9
    for _ in range(120):  # 1 s hold window
        _step(1)
        a = ang()
        lo, hi = min(lo, a), max(hi, a)
    print(f"[smoke] open-stop hold window: lid in [{lo:+.2f}, {hi:+.2f}] deg "
          f"(stop at {c.open_deg:.0f})", flush=True)
    held = lo > c.open_min_deg and hi < c.open_deg + 2.5
    unpark_weight()
    used = wait_until(lambda: bool(scene.lid_closed()[0])
                      and float(scene.lever.data.root_ang_vel_w[0].norm()) < 0.2, 720)
    _step(60)
    _report("stop-reality")
    _REC["on"] = False
    check("stop-reality: the parked weight holds the lid through a 1 s window on "
          "the 60 deg stop, and gravity alone reseals it onto the closed stop "
          "after the unpark",
          held and ang() < c.closed_max_deg and bool(scene.lid_closed()[0]))

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.seesaw_vault")
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
    except Exception as exc:  # noqa: BLE001 - Kit teardown hangs; die loudly NOW
        print(f"SIM_GEN_SMOKE: FAIL (exception: {exc!r})", flush=True)
        import traceback

        traceback.print_exc()
        os._exit(1)
