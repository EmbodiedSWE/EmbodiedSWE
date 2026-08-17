"""Smoke / rubric-REJECTION battery for DepositVaultScene — NullRobot, teleported probes.

solve.py is the acceptance proof (crank the turret open, force-push the brick
through the window into the pocket, crank back so the sweep drops it into the bin
and reseals). This battery proves the rubric REJECTS every wrong outcome and that
the AIRLOCK — the task's strategic differentiator from the seed — is physically
load-bearing: a sealed hatch refuses a real force-push, the deposit stops are real,
and no constructed partial state reaches success() unless the brick is genuinely
banked AND the hatch resealed.

Checks:
   1. settle/no-NaN      — seeded reset settles finite; turret holds its random
                           seal angle on the damped pivot, brick flat on its stand;
                           score ~0, no success;
   2. randomization      — two seeded resets: READBACK vault yaw, vault xy, stand
                           position, brick yaw and initial turret angle all differ;
   3. null-policy        — 240 idle steps: nothing moves, score ~0, no success;
   4. SEED STRATEGY      — the seed's whole plan ("put the money on the shelf
                           inside"): brick CONSTRUCTED resting on the transfer
                           shelf inside the OPEN pocket (turret at receive) — the
                           exact analogue of money-on-safe-shelf — earns partial
                           credit at most, NO success (nothing is banked, nothing
                           is sealed);
   5. seal-reality       — turret sealed (fresh reset), the solve's own force-push
                           drives the brick at the window for 3 s: the blue slab
                           REFUSES it — the brick stays outside the interior and
                           the hatch barely moves; no success;
   6. roof-drop          — brick dropped onto the vault roof: the axle hole is
                           covered by the turret disc; the brick never reaches the
                           bin, no success;
   7. in-bin-UNSEALED    — brick constructed in the bin but the turret left at
                           90 deg: `in_bin` reads true yet success is false and
                           score stays <= 0.70 — the reseal clause has teeth;
   8. stranded-on-shelf  — hatch sealed, brick constructed on the shelf beside the
                           slab (inside, at shelf level): not loaded, not banked,
                           no success;
   9. window-jam         — brick left half-through the window resting on the sill
                           (pocket open): a stable state, but its radius is outside
                           the pocket — not loaded, not banked, no success;
  10. stop-reality       — 0.4 N.m presses past BOTH ends of travel: the seal stop
                           and the receive stop both arrest the turret within a few
                           degrees — the crank range is real;
  11. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.put_money_in_safe_i174.smoke --headless
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
    th = float(scene.turret_angle_deg()[0])
    p = scene._vault_local(scene.brick.data.root_pos_w)[0]
    print(f"[smoke] {tag:16s} | theta={th:+7.2f}deg "
          f"brick=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}) "
          f"in_bin={bool(scene.brick_in_bin()[0])} loaded={bool(scene.brick_loaded()[0])} "
          f"sealed={bool(scene.sealed()[0])} settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.deposit_vault")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.50, -0.95, 0.95)) + o),
                                tuple(np.array((0.35, 0.00, 0.25)) + o),
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
        return float(scene.turret_angle_deg()[0])

    def brick_local() -> torch.Tensor:
        _refresh()
        return scene._vault_local(scene.brick.data.root_pos_w)[0]

    zero = torch.zeros(n, 1, 3, device=device)

    def turret_torque(tau: float) -> None:
        """Constant torque about the vertical pivot (+ toward receive)."""
        _refresh()
        t_w = torch.zeros(n, 3, device=device)
        t_w[:, 2] = tau
        t_b = quat_apply_inverse(scene.turret.data.root_quat_w, t_w)
        scene.turret.set_external_force_and_torque(zero, t_b.reshape(n, 1, 3))

    def set_turret_angle(deg: float) -> None:
        """Teleport the turret to pivot angle `deg` — both joint origins sit ON the
        vertical axis, so this is a pivot-consistent pure pose write (zero vel).
        The vault is untouched (the rest of the linkage keeps its pose)."""
        _refresh()
        q = _qmul(scene.vault.data.root_quat_w,
                  _qz(torch.full((n,), math.radians(deg), device=device)))
        _write_body(scene.turret, scene.vault.data.root_pos_w, q)

    def put_brick_vault_local(loc_xyz, extra_yaw_deg: float = 0.0) -> None:
        """Teleport the brick to a vault-local point, long axis on the vault x
        heading (+ optional extra yaw), zero velocity."""
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        pos = scene.vault.data.root_pos_w + quat_apply(scene.vault.data.root_quat_w, loc)
        q = _qmul(scene.vault.data.root_quat_w,
                  _qz(torch.full((n,), math.radians(extra_yaw_deg), device=device)))
        _write_body(scene.brick, pos, q)

    def force_push(tgt_local, steps: int) -> None:
        """The solve's own insertion push (same gains/clamp): PD force toward a
        vault-local target + gravity feedforward + weak heading torque. Real
        contact physics; wrench zeroed afterwards."""
        no_action = torch.empty(0, device=device)
        m_brick = float(scene.brick.root_physx_view.get_masses()[0].sum())
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
        tgt = torch.zeros(n, 3, device=device)
        tgt[:] = torch.tensor(tgt_local, device=device)
        for _ in range(steps):
            _refresh()
            q = scene.brick.data.root_quat_w
            p = scene.brick.data.root_pos_w
            v = scene.brick.data.root_lin_vel_w
            w = scene.brick.data.root_ang_vel_w
            tgt_w = scene.vault.data.root_pos_w \
                + quat_apply(scene.vault.data.root_quat_w, tgt)
            err = tgt_w - p
            dist = err.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            f_w = m_brick * 9.81 * ez + 30.0 * err - 9.0 * v \
                + 2.0 * (err / dist) * (dist > 0.005)
            f_norm = f_w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            f_w = f_w * (f_norm.clamp(max=5.0) / f_norm)
            f_b = quat_apply_inverse(q, f_w)
            bx = quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
            hx = quat_apply(scene.vault.data.root_quat_w,
                            torch.tensor([-1.0, 0.0, 0.0], device=device).expand(n, 3))
            t_w = 0.03 * torch.cross(bx, hx, dim=-1) - 0.004 * w
            t_b = quat_apply_inverse(q, t_w)
            scene.brick.set_external_force_and_torque(f_b.reshape(n, 1, 3),
                                                      t_b.reshape(n, 1, 3))
            rec = _REC["on"] and _REC["annot"] is not None
            env.step(no_action, render=rec)
            if rec and _REC["i"] % args.record_every == 0:
                for _f in range(3):
                    env.sim.render()
                arr = np.asarray(_REC["annot"].get_data())
                if arr.size:
                    _REC["frames"].append(arr[..., :3].astype(np.uint8).copy())
            _REC["i"] += 1
        scene.brick.set_external_force_and_torque(zero, zero)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    th0 = float(scene.theta0[0])
    _REC["on"] = True
    _step(180)
    _report("settle")
    _REC["on"] = False
    up_z = float(quat_apply(scene.brick.data.root_quat_w,
                            torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3))[0, 2])
    brick_z = float(scene.brick.data.root_pos_w[0, 2] - env.iscene.env_origins[0, 2])
    check("settle/no-NaN: layout settles finite; turret holds its random seal angle "
          "on the damped pivot, brick flat on its stand; score ~0, no success",
          bool(scene._finite()[0]) and abs(ang() - th0) < 4.0
          and abs(brick_z - (c.stand_h + c.brick_h / 2)) < 0.02 and abs(up_z) > 0.95
          and float(scene.score()[0]) <= 0.03 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        vyaw = yaw_of(scene.vault.data.root_quat_w[0])
        vp = scene.vault.data.root_pos_w[0, :2].clone()
        sp = scene.stand.data.root_pos_w[0, :2].clone()
        byaw = yaw_of(scene.brick.data.root_quat_w[0])
        return vyaw, vp, sp, byaw, float(scene.theta0[0])

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_vp, a_sp, a_byaw, a_t0 = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_vp, b_sp, b_byaw, b_t0 = readback()
    d_vyaw = dyaw(a_yaw, b_yaw)
    d_vp = float((a_vp - b_vp).norm())
    d_sp = float((a_sp - b_sp).norm())
    d_byaw = dyaw(a_byaw, b_byaw)
    d_t0 = abs(a_t0 - b_t0)
    print(f"[smoke] randomization deltas: vault_yaw={d_vyaw:.1f}deg "
          f"vault_xy={d_vp * 1000:.1f}mm stand_xy={d_sp * 1000:.1f}mm "
          f"brick_yaw={d_byaw:.1f}deg theta0={d_t0:.2f}deg", flush=True)
    check("randomization-is-real: vault yaw, vault xy, stand position, brick yaw "
          "and initial turret angle readback all differ across seeds",
          d_vyaw > 2.0 and d_vp > 0.003 and d_sp > 0.02 and d_byaw > 5.0
          and d_t0 > 0.02)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, the turret keeps its seal angle, the "
          "brick stays on its stand, score ~0, no success",
          abs(ang() - th0) < 4.0 and float(scene.score()[0]) <= 0.03
          and not bool(scene.success()[0]))

    # ================= 4. negative: the SEED'S OWN STRATEGY =======================================
    # rlbench/put_money_in_safe: pick the money up and SET IT DOWN ON THE SHELF
    # inside the open safe — done. Construct the exact analogue: pocket open at
    # receive, brick resting on the transfer shelf inside the pocket. That is this
    # scene's mid-plan `loaded` state: partial credit at most, never success (the
    # brick is not in the bin, the hatch is not sealed).
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    set_turret_angle(c.receive_deg - 1.0)
    _step(30)
    put_brick_vault_local((c.insert_com_x, 0.0, c.shelf_z1 + c.brick_h / 2 + 0.003))
    _step(240)
    _report("seed-on-shelf")
    _REC["on"] = False
    p = brick_local()
    on_shelf = abs(float(p[2]) - (c.shelf_z1 + c.brick_h / 2)) < 0.015 \
        and float(p[:2].norm()) < c.load_r_max
    check("negative (SEED strategy): brick set down on the shelf inside the open "
          "pocket — the seed's goal state — is only the mid-plan `loaded` state: "
          "partial credit, NO success",
          on_shelf and bool(scene.brick_loaded()[0]) and not bool(scene.brick_in_bin()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.7000005)

    # ================= 5. seal-reality: the sealed hatch REFUSES a real push ======================
    # Fresh reset (turret on its seal stop). Teleport the brick to the solve's own
    # hover outside the window, then run the solve's own force-push toward the
    # inside target for 3 s: the blue slab must hold it OUTSIDE the interior and
    # the hatch must not be pushed open.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    put_brick_vault_local((c.half_out + 0.11, 0.0, c.shelf_z1 + c.brick_h / 2 + 0.004))
    _step(30)
    force_push((c.insert_com_x, 0.0, c.shelf_z1 + c.brick_h / 2 + 0.002), steps=360)
    _step(90)
    _report("seal-refused")
    _REC["on"] = False
    p = brick_local()
    check("seal-reality: with the hatch sealed the solve's own 3 s force-push at "
          "the window cannot get the brick inside (CoM stays outside the interior "
          "wall line) and barely moves the hatch; no success",
          float(p[0]) > c.half_in and abs(ang()) < 25.0
          and not bool(scene.brick_loaded()[0]) and not bool(scene.brick_in_bin()[0])
          and not bool(scene.success()[0]))

    # ================= 6. negative: roof drop =====================================================
    # Drop the brick onto the roof above the axle hole: the turret disc covers the
    # hole from below — no path into the bin from above.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    put_brick_vault_local((0.04, 0.0, c.roof_z1 + 0.12))
    _step(300)
    _report("roof-drop")
    _REC["on"] = False
    check("negative (roof drop): brick dropped onto the roof over the axle hole "
          "never reaches the bin (the turret disc seals the hole), no success",
          not bool(scene.brick_in_bin()[0]) and not bool(scene.success()[0])
          and bool(scene._finite()[0]))

    # ================= 7. near-miss: in the bin but UNSEALED ======================================
    # The reseal clause must have teeth: brick genuinely in the bin, turret left at
    # 90 deg — `in_bin` true, success false, score capped.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    set_turret_angle(90.0)
    _step(30)
    put_brick_vault_local((0.02, 0.0, 0.015 + c.brick_h / 2 + 0.003))
    _step(180)
    _report("bin-unsealed")
    check("near-miss (unsealed): brick genuinely in the bin but the turret left at "
          "90 deg — in_bin reads true yet success is false and score <= 0.70",
          bool(scene.brick_in_bin()[0]) and not bool(scene.sealed()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.7000005)

    # ================= 8. near-miss: stranded on the shelf, hatch sealed ==========================
    # Brick inside at shelf level beside the slab (between axle and slab, clear of
    # both), hatch sealed: not loaded (pocket faces the rear), not banked.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    put_brick_vault_local((0.055, 0.0, c.shelf_z1 + c.brick_h / 2 + 0.003),
                          extra_yaw_deg=90.0)
    _step(240)
    _report("shelf-stranded")
    p = brick_local()
    check("near-miss (stranded): brick on the shelf inside a SEALED vault rests "
          "stably at shelf level — not loaded, not banked, no success",
          float(p[2]) > c.shelf_z1 and not bool(scene.brick_loaded()[0])
          and not bool(scene.brick_in_bin()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.7000005)

    # ================= 9. near-miss: jammed half-through the window ===============================
    # Pocket open, brick resting half-through the window on the sill: a stable
    # state whose radius is outside the pocket — not loaded, not banked.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    set_turret_angle(c.receive_deg - 1.0)
    _step(30)
    put_brick_vault_local((0.130, 0.0, c.shelf_z1 + c.brick_h / 2 + 0.003))
    _step(240)
    _report("window-jam")
    p = brick_local()
    check("near-miss (window jam): brick left half-through the window on the sill "
          "rests stably outside the pocket radius — not loaded, not banked, no "
          "success",
          float(p[0]) > 0.10 and float(p[2]) > c.shelf_z1
          and not bool(scene.brick_loaded()[0]) and not bool(scene.brick_in_bin()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.7000005)

    # ================= 10. stop-reality: both ends of travel arrest ===============================
    # Press each stop with a steady 0.4 N.m: the turret must stay within a few
    # degrees of the limit — the crank range the solve relies on is real.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    turret_torque(-0.4)
    _step(240)
    turret_torque(0.0)
    _step(30)
    lo_ang = ang()
    set_turret_angle(c.receive_deg - 3.0)
    _step(30)
    turret_torque(0.4)
    _step(240)
    turret_torque(0.0)
    _step(30)
    hi_ang = ang()
    print(f"[smoke] stop press readback: low={lo_ang:+.2f}deg (limit 0) "
          f"high={hi_ang:+.2f}deg (limit {c.receive_deg:.0f})", flush=True)
    check("stop-reality: 0.4 N.m pressed past both ends of travel — the seal stop "
          "and the receive stop both arrest the turret within a few degrees",
          -4.0 < lo_ang < 4.0 and c.receive_deg - 4.0 < hi_ang < c.receive_deg + 4.0)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.deposit_vault")
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
