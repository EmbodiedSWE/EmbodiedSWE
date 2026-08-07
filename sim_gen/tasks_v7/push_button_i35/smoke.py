"""Smoke battery for CamPressScene (sim_gen task `push_button_i35`) — REJECTION-
ONLY: every check either verifies basic health/randomization or executes/constructs
a wrong strategy and asserts the rubric refuses it. success() must never fire
anywhere in the battery.

 1. settle/no-NaN     — wheel seated on the pedestal, follower in the sleeve on
                        step 0, depth ~0, score ~0, no success.
 2. randomization A   — tower yaw spans > 90 deg and xy jitters (readback).
 3. randomization B   — BOTH chiralities drawn and the park angle jitters
                        (readback across resets).
 4. null policy       — 240 idle steps -> score ~0, no success.
 5. SEED strategy     — the seed's whole skill (PRESS the button) executed for
                        real: 30 N straight down on the red cap for 2 s — the
                        follower rests on the staircase, load path to ground:
                        depth stays ~0, score ~0, no success.
 6. reverse rotation  — full-strength wheel torque in the ASCENDING direction:
                        headwall + pin/post jam it within a few degrees; depth
                        ~0, no credit.
 7. pry the button    — 25 N straight up on the follower: the ball foot (wider
                        than the bore) jams under the sleeve, never extracted;
                        released, it re-seats on step 0 with no credit.
 8. pry the wheel     — 25 N straight up on the wheel: the centre-post cap stops
                        the hub inside the seat tolerance; disc_seated never
                        breaks, nothing gained.
 9. off-mount cheat   — the active wheel written to the depot: the follower
                        free-falls until its head catches on the sleeve top at a
                        depth DEEPER than d3 — the disc_seated guard refuses
                        every latch; score ~0, no success.
10. sideways dismount — the wheel written 30 mm off the pedestal axis: the hub
                        ring collides with the centre post and is RECAPTURED
                        (back inside the seat tolerance, still at the park
                        angle); the wheel cannot be dismounted sideways and
                        nothing is gained.
11. near miss         — wheel + follower CONSTRUCTED at step 5 (50 mm, past d2):
                        the honest partial credit is exactly w1+w2; l3 and
                        success refuse.
12. latched credit    — from check 11, the machine constructed BACK to the start
                        park: the w1+w2 latch survives, success does not.
13. rejection audit   — success() observed False at every step of the battery.
14. final no-NaN      — and frames.npz written to the CWD.

Run (forge): python -u -m simgen_tasks.push_button_i35.smoke --headless
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
    from . import scene as task_scene  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": True, "annot": None, "frames": [], "i": 0}
_AUDIT = {"saw_success": False}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        _AUDIT["saw_success"] |= bool(env.scene.success()[0])
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
    print(f"[smoke] {tag:14s} | chir={int(scene.chir[0]):+d} "
          f"psi={float(scene.psi_deg()[0]):+7.2f} "
          f"depth={float(scene.depth()[0]) * 1000:6.2f}mm "
          f"seated={bool(scene.disc_seated()[0])} "
          f"sleeve={bool(scene.in_sleeve()[0])} "
          f"at_low={bool(scene.at_low()[0])} "
          f"L={int(scene._l1[0])}{int(scene._l2[0])}{int(scene._l3[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cam_press")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.80, -0.65, 0.62)) + o),
                                tuple(np.array((0.00, 0.00, 0.15)) + o),
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
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def active_disc():
        return scene.discs[int(scene.chir[0])]

    def tower_pq():
        return (scene.tower.data.root_pos_w.clone(),
                scene.tower.data.root_quat_w.clone())

    def qz_world(extra_deg: float) -> torch.Tensor:
        """Tower yaw + extra (about z), as a world quaternion batch."""
        half = (scene.tower_yaw + math.radians(extra_deg)) / 2
        q = torch.zeros(n, 4, device=device)
        q[:, 0] = torch.cos(half)
        q[:, 3] = torch.sin(half)
        return q

    def mount_disc_at(psi_deg_val: float) -> None:
        """CONSTRUCT: write the ACTIVE wheel on the pedestal at wheel angle psi."""
        tp, _ = tower_pq()
        pos = tp.clone()
        pos[:, 2] += 0.001
        _write_body(active_disc(), pos, qz_world(psi_deg_val))

    def follower_at(depth: float) -> None:
        """CONSTRUCT: write the follower on the sleeve axis at `depth` of travel."""
        tp, _ = tower_pq()
        cy = torch.cos(scene.tower_yaw)
        sy = torch.sin(scene.tower_yaw)
        pos = tp.clone()
        pos[:, 0] += c.sleeve_x * cy
        pos[:, 1] += c.sleeve_x * sy
        pos[:, 2] += c.step0_top - depth + 0.002
        _write_body(scene.follower, pos, qz_world(0.0))

    zero = torch.zeros(n, 1, 3, device=device)

    def push_z(body, fz: float, steps: int) -> float:
        """REAL actuation: pure world/body z force (frame-invariant for yaw-only
        bodies). Returns the extreme follower depth seen while pushing."""
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, 2] = fz
        ext = float(scene.depth()[0])
        for _ in range(steps):
            body.set_external_force_and_torque(f, zero)
            _step(1)
            d = float(scene.depth()[0])
            ext = max(ext, d) if fz < 0 else min(ext, d)
        body.set_external_force_and_torque(zero, zero)
        _step(60)
        return ext

    def torque_disc(tau_z: float, steps: int) -> None:
        tq = torch.zeros(n, 1, 3, device=device)
        tq[:, 0, 2] = tau_z
        disc = active_disc()
        for _ in range(steps):
            disc.set_external_force_and_torque(zero, tq)
            _step(1)
        disc.set_external_force_and_torque(zero, zero)
        _step(60)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(120)
    _report("reset")
    check("settle/no-NaN: wheel seated, follower in the sleeve on step 0, depth ~0, "
          "score ~0, no success",
          bool(scene._finite()[0]) and bool(scene.disc_seated()[0])
          and bool(scene.in_sleeve()[0]) and abs(float(scene.depth()[0])) < 0.005
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 2+3. randomization readback ================================================
    yaws, xys, chirs, parks = [], [], [], []
    for k in range(8):
        torch.manual_seed(20 + k)
        env.reset()
        _step(6)
        _refresh()
        yaws.append(yaw_of(scene.tower.data.root_quat_w[0]))
        xys.append((scene.tower.data.root_pos_w[0]
                    - scene.env_origins[0])[:2].tolist())
        chirs.append(int(scene.chir[0]))
        parks.append(abs(float(scene.psi_start[0])))
    yspan = max(yaws) - min(yaws)
    xystd = float(np.std(np.asarray(xys), axis=0).mean())
    pspan = max(parks) - min(parks)
    print(f"[smoke] readback: yaws={[f'{y:+.0f}' for y in yaws]} span={yspan:.0f} "
          f"xystd={xystd:.3f} chirs={chirs} parks={[f'{p:.1f}' for p in parks]} "
          f"park_span={pspan:.1f}", flush=True)
    check("randomization A: tower yaw spans > 90 deg and xy jitters across resets",
          yspan > 90.0 and xystd > 0.008)
    check("randomization B: BOTH staircase chiralities drawn and the park angle "
          "jitters across resets",
          len(set(chirs)) == 2 and pspan > 1.0
          and all(c.psi_start_deg - c.park_jitter_deg - 0.5 <= p
                  <= c.psi_start_deg + c.park_jitter_deg + 0.5 for p in parks))

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> depth ~0, score ~0, no success",
          abs(float(scene.depth()[0])) < 0.005 and float(scene.score()[0]) <= 0.02
          and not bool(scene.success()[0]))

    # ================= 5. SEED strategy (press the button, for real) ==============================
    torch.manual_seed(41)
    env.reset()
    _step(120)
    dmax = push_z(scene.follower, -30.0, 240)  # 30 N >> any fingertip press
    _report("seed-press")
    check("SEED strategy: 30 N pressed straight down on the red cap for 2 s — the "
          "staircase carries the load, depth stays ~0, score ~0, no success",
          dmax < 0.006 and abs(float(scene.depth()[0])) < 0.006
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 6. reverse rotation blocked ================================================
    s6 = int(scene.chir[0])
    psi_a = float(scene.psi_deg()[0])
    torque_disc(s6 * 1.2, 600)  # ASCENDING direction, full solver torque, 5 s
    psi_b = float(scene.psi_deg()[0])
    dpsi = abs((psi_b - psi_a + 180.0) % 360.0 - 180.0)
    _report("reverse")
    check("reverse rotation: full-strength torque the WRONG way jams on the "
          "headwall + pin/post within a few degrees — depth ~0, no credit",
          dpsi < 12.0 and abs(float(scene.depth()[0])) < 0.006
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 7. pry the button ==========================================================
    _refresh()
    z0 = float(scene._tower_local(scene.follower.data.root_pos_w)[0, 2])
    push_z(scene.follower, +25.0, 240)
    _refresh()
    z_after = float(scene._tower_local(scene.follower.data.root_pos_w)[0, 2])
    _report("pry-button")
    check("pry the button: 25 N straight up jams the ball foot under the sleeve — "
          "never extracted, re-seats on step 0, no credit",
          z_after < c.sleeve_z0 - 2 * c.foot_r + 0.004  # back near rest, not above
          and abs(z_after - z0) < 0.006 and bool(scene.in_sleeve()[0])
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 8. pry the wheel ===========================================================
    seated_through = True
    f8 = torch.zeros(n, 1, 3, device=device)
    f8[:, 0, 2] = 25.0
    d8 = active_disc()
    for _ in range(240):
        d8.set_external_force_and_torque(f8, zero)
        _step(1)
        seated_through = seated_through and bool(scene.disc_seated()[0])
    d8.set_external_force_and_torque(zero, zero)
    _step(60)
    _report("pry-wheel")
    check("pry the wheel: 25 N straight up — the centre-post cap holds the hub "
          "within the seat tolerance; disc_seated never breaks, no credit",
          seated_through and bool(scene.disc_seated()[0])
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 9. off-mount cheat =========================================================
    torch.manual_seed(51)
    env.reset()
    _step(90)
    depot = scene.env_origins.clone()
    depot[:, 0] += c.depot[0]
    depot[:, 1] += float(int(scene.chir[0])) * c.depot[1]
    depot[:, 2] += 0.2
    _write_body(active_disc(), depot)   # wheel gone: follower free-falls
    _step(240)
    _report("off-mount")
    d9 = float(scene.depth()[0])
    check("off-mount cheat: wheel removed, follower head catches on the sleeve top "
          "DEEPER than d3 — the disc_seated guard refuses every latch",
          d9 > c.d3 and not bool(scene.disc_seated()[0])
          and not bool(scene._l1[0]) and not bool(scene._l2[0])
          and not bool(scene._l3[0]) and float(scene.score()[0]) <= 0.02
          and not bool(scene.success()[0]))

    # ================= 10. sideways dismount ======================================================
    torch.manual_seed(61)
    env.reset()
    _step(90)
    s10 = int(scene.chir[0])
    psi_park = float(scene.psi_start[0])
    tp, _ = tower_pq()
    pos = tp.clone()
    pos[:, 0] += 0.030 * torch.cos(scene.tower_yaw)   # 30 mm off the pedestal axis
    pos[:, 1] += 0.030 * torch.sin(scene.tower_yaw)
    pos[:, 2] += 0.001
    _write_body(active_disc(), pos, qz_world(psi_park))
    _step(120)
    _report("dismount")
    dpsi10 = abs((float(scene.psi_deg()[0]) - psi_park + 180.0) % 360.0 - 180.0)
    # depth may read a few mm NEGATIVE here (the depenetration jolt of resolving a
    # 30 mm overlap can perch the ball on a step/headwall edge) — that is upward,
    # i.e. zero progress; only positive descent would ever be creditable.
    check("sideways dismount: the wheel written 30 mm off the pedestal axis is "
          "RECAPTURED by the centre post — re-seated near the park angle, no "
          "descent, nothing gained",
          bool(scene.disc_seated()[0]) and dpsi10 < 25.0
          and float(scene.depth()[0]) < 0.006
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 11. near miss ==============================================================
    torch.manual_seed(71)
    env.reset()
    _step(90)
    s11 = int(scene.chir[0])
    mount_disc_at(-s11 * (c.step0_deg + 5 * c.step_pitch_deg))
    follower_at(5 * c.riser)
    _step(150)
    _report("near-miss")
    s_nm = float(scene.score()[0])
    check("near miss: machine constructed at step 5 (50 mm, past d2) — partial "
          "credit is exactly w1+w2; l3 and success refuse",
          bool(scene._l1[0]) and bool(scene._l2[0]) and not bool(scene._l3[0])
          and abs(s_nm - (c.w1 + c.w2)) < 0.01 and not bool(scene.success()[0]))

    # ================= 12. latched credit =========================================================
    mount_disc_at(-s11 * c.psi_start_deg)
    follower_at(0.0)
    _step(150)
    _report("latch-keep")
    s_lk = float(scene.score()[0])
    check("latched credit: machine constructed BACK to the start park — the w1+w2 "
          "latch survives, success does not",
          bool(scene._l1[0]) and bool(scene._l2[0])
          and abs(float(scene.depth()[0])) < 0.006
          and abs(s_lk - (c.w1 + c.w2)) < 0.01 and not bool(scene.success()[0]))

    # ================= 13 + 14. audit, video, verdict =============================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.cam_press")
        print(f"[smoke] wrote {args.out}: {arr.shape}", flush=True)
    check("video: >10 frames recorded", len(_REC["frames"]) > 10)

    n_pass = sum(1 for _, ok in checks if ok)
    if n_pass == len(checks):
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
        code = 0
    else:
        for nm, ok in checks:
            if not ok:
                print(f"[smoke] FAILED: {nm}", flush=True)
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        code = 1
    # Hard exit: Kit teardown hangs — watchdog then die.
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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
