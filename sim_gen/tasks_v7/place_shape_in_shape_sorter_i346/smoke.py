"""Smoke / rubric-REJECTION battery for CounterpoiseRackScene — NullRobot probes.

solve.py is the acceptance proof (both seeds: hover-drop the torque-cancelling
assignment, the beam settles level, score 1.0). This battery proves the rubric
REJECTS wrong outcomes and that the physics it rests on — the beam as its own
arithmetic verifier — is load-bearing. Every probe is CONSTRUCTED (teleport /
hover-drop, real physics steps, judge the SETTLED state); success() is audited
at EVERY step and must never turn True anywhere in this battery.

Wrong-pairing probes drop cubes exactly the way the solve does (hover above the
walls, gravity seats them), so "geometric insertion succeeds mechanically" is
demonstrated, and then the beam — not a geometry test — delivers the verdict.

Checks:
   1. settle/no-NaN   — seeded reset settles finite: both plugs seated, plug
                        mass readback on the ledger (m*r = 0.0192 both arms),
                        beam LEVEL, cubes on the ground unseated, score 0;
   2. randomization   — 4 seeds (GPU RNG pairs can collide), max-pairwise
                        readback: rig xy + yaw, cube spawns, plug pocket pair
                        (occupancy config) all vary;
   3. null-policy     — 240 idle steps: score ~0, no success;
   4. FLAGSHIP (seed strategy) — the cubes SWAPPED: orange at 2r, blue at r.
                        Pure insertion succeeds mechanically (both seat
                        latches fire) but 0.32*2r != 0.16*r: the beam settles
                        tilted >= 8 deg — score capped 0.60, no success;
   5. same-radius     — both cubes at the one radius free on BOTH arms:
                        equal r, unequal mass -> tilted >= 8 deg, no success;
   6. same-arm        — both cubes in the two free pockets of ONE arm: the
                        beam slams to its stop, no success;
   7. half-load latch — orange alone in a free pocket: beam pinned, latched
                        0.30 credit; cube removed to the ground: the beam
                        re-levels on its own (plug ledger) and the 0.30
                        SURVIVES — but success never fires;
   8. plug dislodged  — the left plug parked on the ground: the "level lock"
                        was the plugs' cancelling torques, so the beam falls
                        to a stop — plugs_seated False, no success (the
                        describe() warning is physical, not decorative);
   9. balanced-but-unseated — a LEVEL cheat: blue perched ON TOP of a plug at
                        radius 2r', orange seated at r' opposite (torques
                        cancel exactly, the beam holds level and still) — yet
                        the cube z-band refuses the perch: no success, score
                        caps at the orange's 0.30 (config-dependent probe:
                        the battery seed-searches for a qualifying episode);
  10. bare-bar rest   — blue resting on the un-walled bar between pockets
                        (near a torque-balancing abscissa), orange seated:
                        the slot-radius clause refuses (no pocket there), no
                        success;
  11. rejection audit — success() was never True at any step of this battery;
  12. frames.npz      — video captured and saved to the CWD (> 10 frames).

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.place_shape_in_shape_sorter_i346.smoke --headless
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

_qapply = task_scene._qapply

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

# ----- module state wired up in main() ------------------------------------------------------------
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


def _place(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None) -> None:
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
    tl = math.degrees(float(scene.tilt()[0]))
    print(f"[smoke] {tag:18s} | tilt={tl:+.2f}deg "
          f"h_seat={bool(scene.cube_seated(scene.cube_h)[0])} "
          f"l_seat={bool(scene.cube_seated(scene.cube_l)[0])} "
          f"plugs={bool(scene.plugs_seated()[0])} level={bool(scene.level()[0])} "
          f"still={int(scene._still_count[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.counterpoise_rack")().build(num_envs=args.num_envs,
                                                       device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.35, -0.80, 0.55)) + o),
                                tuple(np.array((0.40, 0.00, 0.12)) + o),
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

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def pt(x: float, y: float, z: float) -> torch.Tensor:
        p = torch.tensor([x, y, z], device=device).expand(n, 3).clone()
        return p + env.iscene.env_origins

    def plug_config() -> tuple[float, float]:
        """(r_l, r_r) live plug seat radii for env 0."""
        seat = scene._plug_seat[0]
        return float(-seat[0]), float(seat[1])

    def free_slots(r_l: float, r_r: float) -> dict[float, list[float]]:
        slots = tuple(c.slots)
        return {-1.0: [s for s in slots if abs(s - r_l) > 1e-6],
                +1.0: [s for s in slots if abs(s - r_r) > 1e-6]}

    def correct_plan(r_l: float, r_r: float) -> tuple[float, float]:
        """(side, r_h): orange at side*r_h, blue at -side*2*r_h (the solve's law)."""
        free = free_slots(r_l, r_r)
        for side in (-1.0, +1.0):
            for r_h in free[side]:
                if any(abs(2 * r_h - s) < 1e-6 for s in free[-side]):
                    return side, r_h
        raise AssertionError(f"no balancing assignment for config ({r_l}, {r_r})")

    def beam_hover(x_local: float, z_local: float):
        qb = scene.beam.data.root_quat_w.clone()
        local = torch.tensor([x_local, 0.0, z_local], device=device).expand(n, 3)
        return scene.beam.data.root_pos_w + _qapply(qb, local), qb

    def drop_pair(x_h: float | None, x_l: float | None,
                  z_h: float = 0.050, z_l: float = 0.050) -> None:
        """Hover-drop the orange (at beam-frame x_h) and blue (x_l) cubes one
        step apart onto the LIVE beam, then let everything swing and settle
        (4+ s: the damped pendulum reaches its verdict). None = leave cube."""
        _refresh()
        if x_h is not None:
            pos, qb = beam_hover(x_h, z_h)
            _place(scene.cube_h, pos, qb)
        _step(1)
        if x_l is not None:
            pos, qb = beam_hover(x_l, z_l)
            _place(scene.cube_l, pos, qb)
        _step(510)

    def park_cubes() -> None:
        _place(scene.cube_h, pt(0.85, -0.45, c.cube / 2 + 0.002))
        _place(scene.cube_l, pt(-0.05, -0.45, c.cube / 2 + 0.002))

    def tilt_deg() -> float:
        _refresh()
        return math.degrees(float(scene.tilt()[0]))

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _AUD["on"] = True
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    r_l, r_r = plug_config()
    m_l = float(scene.plug_masses()[0])
    m_r = float(scene.plug_masses()[1])
    print(f"[smoke] plug ledger: left r={r_l:.3f} m={m_l:.3f} (m*r={m_l * r_l:.5f}) "
          f"right r={r_r:.3f} m={m_r:.3f} (m*r={m_r * r_r:.5f})", flush=True)
    check("settle/no-NaN: seeded reset settles finite — both plugs seated, plug "
          "masses on the ledger (m*r=0.0192 both arms), beam level, cubes on "
          "the ground unseated, score 0, no success",
          bool(scene._finite()[0]) and bool(scene.plugs_seated()[0])
          and abs(m_l * r_l - c.plug_c) < 1e-4 and abs(m_r * r_r - c.plug_c) < 1e-4
          and abs(r_l - r_r) > 1e-6 and bool(scene.level()[0]) and abs(tilt_deg()) < 2.0
          and not bool(scene.cube_seated(scene.cube_h)[0])
          and not bool(scene.cube_seated(scene.cube_l)[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (3-seed max-pairwise readback) ===================
    def readback():
        _refresh()
        return (scene.gantry.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.gantry.data.root_quat_w[0]),
                scene.cube_h.data.root_pos_w[0, :2].clone(),
                scene.cube_l.data.root_pos_w[0, :2].clone(),
                plug_config())

    obs = []
    for sd in (101, 202, 303, 404):
        torch.manual_seed(sd)
        env.reset()
        _step(10)
        obs.append(readback())
    prs = [(i, j) for i in range(len(obs)) for j in range(i + 1, len(obs))]
    d_rig = max(float((obs[i][0] - obs[j][0]).norm()) for i, j in prs)
    d_yaw = max(dyaw(obs[i][1], obs[j][1]) for i, j in prs)
    d_ch = max(float((obs[i][2] - obs[j][2]).norm()) for i, j in prs)
    d_cl = max(float((obs[i][3] - obs[j][3]).norm()) for i, j in prs)
    cfg_vary = any(obs[i][4] != obs[j][4] for i, j in prs)
    print(f"[smoke] randomization deltas (max pairwise over 3 seeds): "
          f"rig_xy={d_rig * 1000:.1f}mm rig_yaw={d_yaw:.1f}deg "
          f"cube_h={d_ch * 1000:.1f}mm cube_l={d_cl * 1000:.1f}mm "
          f"plug_cfgs={[o[4] for o in obs]}", flush=True)
    check("randomization-is-real: rig xy+yaw, both cube spawns and the plug "
          "pocket pair all vary across 3 seeds (readback)",
          d_rig > 0.005 and d_yaw > 1.5 and d_ch > 0.005 and d_cl > 0.005
          and cfg_vary)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 4. FLAGSHIP: the seed's own strategy (pure insertion) ======================
    # The cubes SWAPPED: orange dropped at the 2r pocket, blue at the r pocket —
    # every piece "fits its hole" and both SEAT LATCHES fire (insertion succeeds
    # mechanically, exactly like the shape-sorter seed) — but 0.32*2r != 0.16*r
    # leaves >= 0.28 N*m and the beam settles hard-tilted: no success, score
    # capped at the 0.60 latch ledger.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    r_l, r_r = plug_config()
    side, r_h = correct_plan(r_l, r_r)
    _REC["on"] = True
    drop_pair(x_h=-side * 2 * r_h, x_l=side * r_h)   # swapped!
    _report("flagship-swap")
    _REC["on"] = False
    check("FLAGSHIP (seed strategy rejected): swapped pairing — both cubes "
          "physically seat in pockets (latches fire, score 0.60 cap) but the "
          "beam settles tilted >= 8 deg: not level, no success",
          float(scene.score()[0]) >= 0.59 and float(scene.score()[0]) <= 0.601
          and abs(tilt_deg()) >= 8.0 and not bool(scene.level()[0])
          and not bool(scene.success()[0]))

    # ================= 5. same radius, both arms ==================================================
    # Exactly one radius is free on BOTH arms (the one no plug occupies). Equal
    # radii + 2:1 masses -> net 0.16*r_c kg*m, the beam tips.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    r_l, r_r = plug_config()
    r_c = next(s for s in c.slots if abs(s - r_l) > 1e-6 and abs(s - r_r) > 1e-6)
    drop_pair(x_h=+r_c, x_l=-r_c)
    _report("same-radius")
    check("same-radius rejected: cubes at the shared free radius on opposite "
          "arms — equal arms, unequal masses, tilted >= 8 deg, no success",
          abs(tilt_deg()) >= 8.0 and not bool(scene.level()[0])
          and not bool(scene.success()[0]))

    # ================= 6. same arm ================================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    r_l, r_r = plug_config()
    f_left = free_slots(r_l, r_r)[-1.0]
    drop_pair(x_h=-f_left[0], x_l=-f_left[1])
    _report("same-arm")
    check("same-arm rejected: both cubes on one arm — the beam slams to its "
          "stop (tilt >= 10 deg), no success",
          abs(tilt_deg()) >= 10.0 and not bool(scene.success()[0]))

    # ================= 7. half load: latch fires, survives removal, never success =================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    r_l, r_r = plug_config()
    side, r_h = correct_plan(r_l, r_r)
    _REC["on"] = True
    drop_pair(x_h=side * r_h, x_l=None)          # orange alone
    _report("half-load")
    half_tilt = abs(tilt_deg())
    half_latch = float(scene.score()[0])
    park_cubes()                                  # remove it again
    _step(420)                                    # the plug ledger re-levels the beam
    _report("half-removed")
    _REC["on"] = False
    check("half-load latch: orange alone pins the beam (tilt >= 8 deg) and "
          "latches 0.30; after removal the beam re-levels on its own and the "
          "0.30 survives — success never fires",
          half_tilt >= 8.0 and abs(half_latch - c.w_seat) < 1e-5
          and bool(scene.level()[0]) and abs(float(scene.score()[0]) - c.w_seat) < 1e-5
          and not bool(scene.success()[0]))

    # ================= 8. plug dislodged: the level lock is physical ==============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _refresh()
    _place(scene.plug_l, pt(0.85, 0.35, c.plug_h / 2 + 0.002))
    _step(420)
    _report("plug-dislodged")
    check("plug-dislodged rejected: removing one plug breaks the cancelling "
          "ledger — the beam falls to a stop (tilt >= 8 deg), plugs_seated "
          "False, no success",
          abs(tilt_deg()) >= 8.0 and not bool(scene.plugs_seated()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.01)

    # ================= 9. balanced-but-unseated: the z-band is load-bearing =======================
    # The one LEVEL cheat: blue perched ON TOP of a plug at radius r_p (adding
    # 0.16*r_p) while orange sits in the free pocket at r_p/2 opposite (adding
    # 0.32*r_p/2) — the torques cancel EXACTLY, the beam holds level and still,
    # every clause but one is satisfied... and the cube z-band refuses the perch
    # (centre ~60 mm >> 42 mm ceiling). Config-dependent: seed-search for an
    # episode whose plug layout admits it (4 of the 6 configs do).
    found = None
    for sd in range(100, 160, 7):
        torch.manual_seed(sd)
        env.reset()
        _step(30)
        r_l, r_r = plug_config()
        free = free_slots(r_l, r_r)
        for p_side, r_p in ((-1.0, r_l), (+1.0, r_r)):
            r_half = r_p / 2
            if any(abs(r_half - s) < 1e-6 for s in free[-p_side]):
                found = (sd, p_side, r_p, r_half)
                break
        if found:
            break
    assert found, "no qualifying plug config found in the seed search"
    sd, p_side, r_p, r_half = found
    print(f"[smoke] perch probe: seed {sd}, plug at {p_side:+.0f}*{r_p:.3f}, "
          f"orange at {-p_side:+.0f}*{r_half:.3f}", flush=True)
    plug_top = c.bar_t / 2 + c.plug_h  # beam-frame z of the plug's top face
    _REC["on"] = True
    drop_pair(x_h=-p_side * r_half, x_l=p_side * r_p,
              z_l=plug_top + c.cube / 2 + 0.004)   # blue released just above the plug top
    _report("perch-balanced")
    _REC["on"] = False
    blue_z = float(scene._beam_local(scene.cube_l.data.root_pos_w)[0, 2])
    check("balanced-but-unseated rejected: blue perched ON a plug balances the "
          "beam exactly (level, still, orange seated) yet the cube z-band "
          "refuses the perch — no success, score capped at 0.30",
          bool(scene.level()[0]) and bool(scene.cube_seated(scene.cube_h)[0])
          and blue_z > c.cube_z_hi
          and not bool(scene.cube_seated(scene.cube_l)[0])
          and float(scene.score()[0]) <= c.w_seat + 1e-5
          and not bool(scene.success()[0]))

    # ================= 10. bare-bar rest: the slot-radius clause is load-bearing ==================
    # Blue set down on the un-walled bar between the 0.12 and 0.24 cells
    # (x=0.18: >= 55 mm from either slot radius even if it slides to a wall),
    # orange seated at its planned pocket. No pocket there -> never seated.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    r_l, r_r = plug_config()
    side, r_h = correct_plan(r_l, r_r)
    drop_pair(x_h=side * r_h, x_l=-side * 0.180, z_l=0.050)
    _report("bare-bar")
    bar_x = abs(float(scene._beam_local(scene.cube_l.data.root_pos_w)[0, 0]))
    check("bare-bar rejected: blue resting on the un-walled bar (no pocket at "
          "its abscissa) is never cube_seated — no success",
          not bool(scene.cube_seated(scene.cube_l)[0])
          and min(abs(bar_x - s) for s in c.slots) > c.seat_x_tol
          and not bool(scene.success()[0]))

    # ================= 11. rejection audit ========================================================
    check("rejection audit: success() was never True at any step of this battery",
          _AUD["hits"] == 0)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.counterpoise_rack")
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
