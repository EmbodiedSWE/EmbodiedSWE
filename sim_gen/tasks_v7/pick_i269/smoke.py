"""Smoke / rubric-REJECTION battery for RelayTunnelScene — NullRobot, teleported probes.

solve.py is the acceptance proof (fingertip push + blue-cube relay seats the red cube,
1.0). This battery proves the rubric REJECTS wrong outcomes and that the physical
claims the task rests on are load-bearing: the roof really forbids top-down delivery
(a cube pressed onto it at 2x weight stays out), the 16 mm step really discriminates
(on-slab rest and the analytic lip-straddle pose both sit ABOVE the seated z-window),
a blue-first end state really fails (the wrong cube seated in the well pays nothing),
and success() refuses a moving cube while the non-success score caps at exactly 0.75.
Every probe is CONSTRUCTED (teleport, real physics steps, judge) — instrumentation,
never a solution: success() is monitored at EVERY step and must never turn True
anywhere in this battery (the audit is itself a check).

Checks:
   1. settle/no-NaN     — seeded reset settles finite; cubes on the ground OUTSIDE the
                          garage (readback); authored masses; score ~0, no success;
   2. randomization     — seeded resets: garage xy + yaw, red and blue spawns all
                          differ pairwise across seeds (READBACK); the red/blue slot
                          swap takes BOTH values across seeds 0..7;
   3. null policy       — 240 idle steps: nothing latches, score ~0, no success;
   4. ROOF SEAL         — the seed's own strategy (deliver from above) aimed at the
                          well: the red cube set on the roof directly over the well and
                          PRESSED down at 2x weight for a full second. The roof takes
                          the load (probe real: z readback), nothing latches, score ~0;
   5. near-miss entered — red constructed ON the slab just inside the mouth: `entered`
                          latches (0.25) but not deep / seated — depth matters;
   6. near-miss deep    — red constructed on the slab just past half-depth, 8 mm short
                          of the lip: `deep` latches (0.50) but the ON-SLAB rest z
                          (readback ~0.041) sits above the seated window — no seat
                          credit without the 16 mm drop;
   7. lip-straddle      — the BLUE cube HELD at the ANALYTIC straddle pose (nose down
                          over the lip, pitch asin(step/s)): centre x is in the well
                          span yet z (~0.032) is rejected by the seated z-window;
                          released, it topples and lands flat in the well (readback —
                          the pose is physically adjacent to seating, not arbitrary);
                          red meanwhile dismantled to the floor — the latched 0.50
                          survives (latch semantics);
   8. BLUE-FIRST FAILS  — fresh episode, constructed out-of-order end state: the BLUE
                          cube seated + settled in the well, red stuck behind on the
                          slab. in_well(blue) holds by geometry, yet no success and
                          score stays at red's `entered` 0.25 — object identity and
                          order are load-bearing;
   9. settle gate + cap — red constructed IN the well but sliding at 0.4 m/s: all
                          three latches fire, the non-success score is capped at
                          exactly 0.75, success() refuses while it moves; the probe is
                          dismantled before it can settle, and the 0.75 survives;
  10. rejection audit   — success() was never True at any step of this battery;
  11. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.pick_i269.smoke --headless
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

_qapply, _qz, _qy = task_scene._qapply, task_scene._qz, task_scene._qy


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of wxyz quaternions (..., 4)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() -----------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
_AUD = {"on": False, "hits": 0}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if _AUD["on"] and bool(env.scene.success()[0]):
            _AUD["hits"] += 1
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


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    rl = scene._garage_local(scene.red.data.root_pos_w)[0]
    bl = scene._garage_local(scene.blue.data.root_pos_w)[0]
    print(f"[smoke] {tag:16s} | red_l=({float(rl[0]):+.3f},{float(rl[1]):+.3f},"
          f"{float(rl[2]):+.3f}) blue_l=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
          f"{float(bl[2]):+.3f}) ent={bool(scene.entered(scene.red)[0])} "
          f"deep={bool(scene.deep(scene.red)[0])} well={bool(scene.in_well(scene.red)[0])} "
          f"l=({int(scene._l_entered[0])},{int(scene._l_deep[0])},{int(scene._l_seated[0])}) "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main --------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.relay_tunnel")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    zero = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.30, -0.80, 0.65)) + o),
                                tuple(np.array((0.42, 0.0, 0.06)) + o),
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

    def gq() -> torch.Tensor:
        _refresh()
        return scene.garage.data.root_quat_w

    def g_yaw() -> float:
        q = gq()[0]
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    def red_local() -> torch.Tensor:
        _refresh()
        return scene._garage_local(scene.red.data.root_pos_w)[0]

    def blue_local() -> torch.Tensor:
        _refresh()
        return scene._garage_local(scene.blue.data.root_pos_w)[0]

    def write_local(body, local_xyz, *, local_quat=None, lin_vel_w=None) -> None:
        """Teleport a body to a garage-frame pose (orientation composed with the
        garage's). Pure state construction — the judge and physics do the rest."""
        _refresh()
        st = torch.zeros(n, 13, device=device)
        loc = torch.tensor([list(local_xyz)], device=device, dtype=torch.float)
        st[:, 0:3] = scene.garage.data.root_pos_w + _qapply(gq(), loc.expand(n, 3))
        q = gq()
        if local_quat is not None:
            q = _qmul(q, local_quat.to(device).expand(n, 4))
        st[:, 3:7] = q
        if lin_vel_w is not None:
            st[:, 7:10] = lin_vel_w
        body.write_root_state_to_sim(st, _all_ids())
        _refresh()

    # ================= 1. settle / no-NaN / baseline zero =======================================
    env.reset(seed=0)
    _AUD["on"] = True
    _step(240)
    _report("reset")
    rl, bl = red_local(), blue_local()
    m_red = float(scene.red.root_physx_view.get_masses().sum())
    m_blue = float(scene.blue.root_physx_view.get_masses().sum())
    print(f"[smoke] mass readback: red={m_red:.3f} blue={m_blue:.3f}", flush=True)
    check("settle/no-NaN: seeded reset settles finite; cubes rest on the ground in "
          "front of the apron (readback); masses authored; score ~0, no success",
          bool(scene._finite()[0])
          and float(rl[0]) < -c.apron_len and float(bl[0]) < -c.apron_len
          and abs(float(rl[2]) - c.cube_s / 2) < 0.006
          and abs(float(bl[2]) - c.cube_s / 2) < 0.006
          and abs(m_red - c.cube_mass) < 0.02 and abs(m_blue - c.cube_mass) < 0.02
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization readback ================================================
    garages, yaws, reds, blues, swaps = [], [], [], [], []
    for s in (0, 1, 2):
        env.reset(seed=s)
        _step(30)
        _refresh()
        gp = (scene.garage.data.root_pos_w - scene.env_origins)[0]
        garages.append((float(gp[0]), float(gp[1])))
        yaws.append(g_yaw())
        r, b = red_local(), blue_local()
        reds.append((float(r[0]), float(r[1])))
        blues.append((float(b[0]), float(b[1])))
    for s in range(8):
        env.reset(seed=s)
        _refresh()
        swaps.append(float(scene.swap[0]))
    print(f"[smoke] rand readback: garages={garages} yaws={[f'{y:+.1f}' for y in yaws]} "
          f"reds={reds} blues={blues} swaps={swaps}", flush=True)

    def pairwise(vals, tol) -> bool:
        return all(abs(a - b) > tol for i, a in enumerate(vals) for b in vals[i + 1:])

    check("randomization: garage xy+yaw and red/blue spawn xy differ pairwise across "
          "seeds 0/1/2; the red/blue slot swap takes both values across seeds 0..7",
          pairwise([g[0] for g in garages], 3e-4) and pairwise([g[1] for g in garages], 3e-4)
          and pairwise(yaws, 0.1)
          and pairwise([r[0] for r in reds], 3e-4) and pairwise([r[1] for r in reds], 3e-4)
          and pairwise([b[0] for b in blues], 3e-4) and pairwise([b[1] for b in blues], 3e-4)
          and (1.0 in swaps) and (-1.0 in swaps))

    # ================= 3. null policy ===========================================================
    env.reset(seed=0)
    _step(240)
    _report("null-policy")
    check("null policy: 240 idle steps latch nothing — score ~0, no success",
          float(scene.score()[0]) <= 0.01 and not bool(scene._l_entered[0])
          and not bool(scene._l_deep[0]) and not bool(scene._l_seated[0])
          and not bool(scene.success()[0]))

    # ================= 4. ROOF SEAL: pressed onto the roof over the well ========================
    # The seed's own strategy — carry the cube to the target and place it from above —
    # aimed straight at the well: set the red cube on the roof directly OVER the well
    # and press DOWN at 2x its weight for a full second.
    _REC["on"] = True
    z_on_roof = c.z_roof_top + c.cube_s / 2
    write_local(scene.red, (0.125, 0.0, z_on_roof + 0.004))
    _step(60)
    press = torch.zeros(n, 1, 3, device=device)
    press[0, 0, 2] = -2.0 * c.cube_mass * 9.81
    on_roof = True
    for _ in range(120):
        scene.red.set_external_force_and_torque(press, zero, env_ids=_all_ids(),
                                                is_global=True)
        _step(1)
        z = float(red_local()[2])
        on_roof = on_roof and (z_on_roof - 0.008 < z < z_on_roof + 0.04)
    scene.red.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    _step(60)
    _report("roof-press")
    check("roof seal: red cube pressed onto the roof directly over the well at 2x "
          "weight stays ON the roof (probe real, z readback) — nothing latches",
          on_roof and not bool(scene.entered(scene.red)[0])
          and not bool(scene._l_entered[0]) and not bool(scene._l_seated[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 5. near-miss: entered but shallow ========================================
    write_local(scene.red, (0.045, 0.0, c.z_on_slab + 0.002))
    _step(60)
    _report("entered-shallow")
    check("near-miss entered: red on the slab just inside the mouth — `entered` "
          "latches (0.25) but deep/seated do not: depth matters",
          bool(scene._l_entered[0]) and not bool(scene._l_deep[0])
          and not bool(scene._l_seated[0])
          and abs(float(scene.score()[0]) - c.w_entered) < 1e-4
          and not bool(scene.success()[0]))

    # ================= 6. near-miss: deep but ON the slab =======================================
    # 2 mm past deep_x, CoM still 8 mm inside the slab edge: rests flat ON the slab.
    write_local(scene.red, (c.deep_x + 0.002, 0.0, c.z_on_slab + 0.002))
    _step(90)
    rl = red_local()
    _report("deep-on-slab")
    check("near-miss deep: red on the slab past half-depth — `deep` latches (0.50) "
          "but the on-slab rest z (readback) sits ABOVE the seated window: no seat "
          "credit without the 16 mm drop",
          bool(scene._l_deep[0]) and not bool(scene._l_seated[0])
          and abs(float(rl[2]) - c.z_on_slab) < 0.005
          and float(rl[2]) > c.seat_z_hi + 0.008
          and not bool(scene.in_well(scene.red)[0])
          and abs(float(scene.score()[0]) - (c.w_entered + c.w_deep)) < 1e-4
          and not bool(scene.success()[0]))

    # ================= 7. lip-straddle rest is rejected =========================================
    # Dismantle red to the floor first (transport); its latched 0.50 must survive.
    write_local(scene.red, (-0.30, -0.20, c.cube_s / 2 + 0.002))
    _step(30)
    # Analytic straddle rest: bottom face on the lip corner, nose down with the front
    # bottom edge on the well floor — pitch asin(step/s), CoM at z ~ 0.0317 (> window).
    lean = math.asin(c.step / c.cube_s)
    x_com = c.well_x0 + (c.cube_s / 2) * math.cos(lean) + (c.cube_s / 2) * math.sin(lean)
    z_com = c.step / 2 + (c.cube_s / 2) * math.cos(lean)
    write_local(scene.blue, (x_com, 0.0, z_com + 0.002),
                local_quat=_qy(torch.tensor([lean])))
    _refresh()
    # At the held straddle pose the centre x IS in the well span but z sits above the
    # seated window — the 16 mm step is what makes z discriminative.
    bl_held = blue_local()
    straddle_pose_ok = (c.seat_x_lo < float(bl_held[0]) < c.seat_x_hi
                        and float(bl_held[2]) > c.seat_z_hi + 0.002
                        and not bool(scene.in_well(scene.blue)[0]))
    # Release: on this geometry the straddle is not a stable rest — the cube topples
    # forward off the lip and lands flat INSIDE the well (a real dynamics readback,
    # asserted, so the pose above was physically adjacent to seating, not arbitrary).
    _step(180)
    bl = blue_local()
    _report("lip-straddle")
    check("lip-straddle rejected: blue HELD at the analytic straddle pose has centre "
          "x in the well span but z above the seated window (in_well False — the "
          "16 mm step discriminates); released, it topples and lands flat in the "
          "well (readback); red's latched 0.50 survives its dismantle to the floor",
          straddle_pose_ok
          and c.seat_x_lo < float(bl[0]) < c.seat_x_hi and abs(float(bl[1])) < 0.02
          and float(bl[2]) < c.seat_z_hi
          and abs(float(scene.score()[0]) - (c.w_entered + c.w_deep)) < 1e-4
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 8. BLUE-FIRST end state fails ============================================
    env.reset(seed=3)
    _step(60)
    _REC["on"] = True
    # Constructed out-of-order outcome: BLUE seated + settled in the well, RED stuck
    # behind it on the slab (the single lane means this is unrecoverable).
    write_local(scene.blue, (0.120, 0.0, c.z_seated + 0.002))
    _step(90)
    write_local(scene.red, (0.055, 0.0, c.z_on_slab + 0.002))
    _step(90)
    bl = blue_local()
    _report("blue-first")
    check("blue-first fails: the BLUE cube seated+settled in the well (in_well(blue) "
          "True by geometry) with red stuck behind — no success, no seat credit, "
          "score stays at red's `entered` 0.25: identity and order are load-bearing",
          bool(scene.in_well(scene.blue)[0]) and abs(float(bl[2]) - c.z_seated) < 0.004
          and bool(scene._l_entered[0]) and not bool(scene._l_seated[0])
          and abs(float(scene.score()[0]) - c.w_entered) < 1e-4
          and not bool(scene.success()[0]))

    # ================= 9. settle gate + exact score cap =========================================
    env.reset(seed=4)
    _step(60)
    # Red IN the well but sliding at 0.4 m/s along the channel: every latch fires,
    # the non-success score caps at exactly 0.75, success refuses while moving.
    vel = _qapply(gq(), torch.tensor([[0.4, 0.0, 0.0]], device=device))
    write_local(scene.red, (0.112, 0.0, c.z_seated + 0.002), lin_vel_w=vel)
    moving_ok = True
    for _ in range(4):
        _step(1)
        v = float(scene.red.data.root_lin_vel_w[0].norm())
        moving_ok = moving_ok and v > 0.1 and bool(scene.in_well(scene.red)[0]) \
            and not bool(scene.settled_red()[0]) and not bool(scene.success()[0])
    s_cap = float(scene.score()[0])
    _report("settle-gate")
    # dismantle before it can settle into a real success; the latched 0.75 survives
    write_local(scene.red, (-0.30, -0.20, c.cube_s / 2 + 0.002))
    _step(60)
    s_after = float(scene.score()[0])
    _report("cap-latched")
    _REC["on"] = False
    check("settle gate + cap: a sliding in-well red cube latches everything, the "
          "non-success score is capped at exactly 0.75, success refuses while it "
          "moves, and the latched 0.75 survives its dismantle",
          moving_ok and bool(scene._l_seated[0]) and bool(scene._l_deep[0])
          and s_cap >= 0.75 - 1e-4 and s_cap <= 0.75 + 1e-5
          and abs(s_after - 0.75) < 1e-4
          and not bool(scene.success()[0]))

    # ================= 10. rejection audit ======================================================
    check("rejection audit: success() was never True at any step of this battery",
          _AUD["hits"] == 0)

    # ================= save + verdict ===========================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.relay_tunnel")
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
