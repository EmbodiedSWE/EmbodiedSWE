"""Smoke / rubric-REJECTION battery for BayonetLidScene — NullRobot, teleported probes.

solve.py is the acceptance proof (keyed insertion by gravity + clockwise twist by
torque reaches success, monotone latched credit). This battery proves the rubric
REJECTS wrong outcomes — above all the SEED task's own end state (a lid simply SET
DOWN on the pot) — and that the physical claims the task rests on are load-bearing:
the tab ring keeps an unkeyed lid 26 mm too high, the stops make the coupling
directional (counter-clockwise jams), the overhanging tabs physically RETAIN an
engaged lid against a straight pull while a disengaged lid pulls straight out, and
the decoy's oversized lugs cannot enter the collar at all. Every probe is
CONSTRUCTED (teleport, real physics steps, judge) — instrumentation, never a
solution: no probe here reaches success().

Checks:
   1. settle/no-NaN      — seeded reset settles finite; both lids flat on the
                           ground; score ~0, no success;
   2. randomization      — seeded resets: READBACK pot xy AND pot yaw differ, and
                           the red lid's start SIDE swaps across seeds;
   3. null-policy        — 240 idle steps: score ~0, no success (nothing moves);
   4. SEED STRATEGY      — the seed's whole plan ("put the bowl ON the plate"):
                           the lid dropped ON TOP of the pot with lugs over the
                           TABS rests on the brass ring ~86 mm up — out of the
                           50..70 mm seat band -> NO success;
   5. seated-not-locked  — the lid dropped through the entry gaps (delta ~ 90°)
                           seats on the ledge but is NOT twisted -> no success,
                           score <= 0.75;
   6. near-miss twist    — the seated lid at delta ~ 48°, a partial clockwise
                           twist that stopped short of the stops -> outside the
                           locked window, no success;
   7. CCW jam            — clockwise is the ONLY locking direction: torque the
                           seated lid COUNTER-clockwise; the lugs jam against the
                           stops (~106°) without ever entering the locked window
                           -> no success;
   8. decoy keyed out    — the blue-knobbed decoy dropped aligned over the pot
                           CANNOT descend into the collar (lugs 78 mm > wall
                           72 mm): rests near rim height, no success;
   9. RETENTION contrast — tabs retain an ENGAGED lid: at delta ~ 48° (lugs under
                           the tabs) a 3x-weight straight pull for 0.75 s cannot
                           extract it; at delta ~ 90° (lugs in the gaps) a mere
                           1.5x-weight pull lifts it straight out;
  10. flipped lid        — the lid upside-down in the collar rests out of the seat
                           band with its +z inverted -> not seated, no success;
  11. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_kitchen_scene7_put_the_white_bowl_on_the_plate_i3.smoke --headless
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

_qy = task_scene._qy
_qz = task_scene._qz
_yaw_of = task_scene._yaw_of

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
    xy, z = scene._rel(scene.lid)
    e = float(torch.rad2deg(scene.engagement())[0])
    print(f"[smoke] {tag:18s} | lid xy={float(xy[0]) * 1000:6.1f}mm "
          f"z={float(z[0]) * 1000:6.1f}mm delta={e:6.1f}deg "
          f"seated={bool(scene.seated()[0])} locked={bool(scene.locked()[0])} "
          f"lift={bool(scene._lift[0])} over={bool(scene._over[0])} "
          f"seat={bool(scene._seat[0])} score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


def _pull(body, force_w: torch.Tensor, steps: int, z_stop: float | None = None) -> float:
    """Apply a world-frame force at the CoM for up to `steps` steps (early-out once
    the pot-frame z of the RED LID passes `z_stop`), then clear. Returns the max
    pot-frame z of the red lid seen during the pull."""
    scene = _ENV.scene
    n = _ENV.num_envs
    zero = torch.zeros(n, 1, 3, device=_ENV.device)
    f = force_w.view(1, 1, 3).expand(n, 1, 3).contiguous()
    z_max = -1.0
    for _ in range(steps):
        body.set_external_force_and_torque(f, zero, env_ids=_all_ids(), is_global=True)
        _step(1)
        _refresh()
        z_max = max(z_max, float(scene._rel(scene.lid)[1][0]))
        if z_stop is not None and z_max > z_stop:
            break
    body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    return z_max


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.lid_bayonet")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.15, -0.85, 0.75)) + o),
                                tuple(np.array((0.30, 0.00, 0.05)) + o),
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

    def pot_c() -> torch.Tensor:
        _refresh()
        return scene.pot.data.root_pos_w.clone()

    def pot_yaw() -> float:
        _refresh()
        return float(_yaw_of(scene.pot.data.root_quat_w)[0])

    def lid_yaw_for(delta_deg: float) -> torch.Tensor:
        """(N,) lid yaw whose engagement (mod 120°) equals `delta_deg`."""
        y = pot_yaw() + math.radians(delta_deg)
        return torch.full((n,), y, device=device)

    def drop_lid(body, delta_deg: float, hover_dz: float = 0.012) -> None:
        """CONSTRUCT: hold `body` just above the rim at engagement `delta_deg`,
        release, let gravity act, settle. Mirrors solve's insertion primitive."""
        p = pot_c()
        p[:, 2] += c.pot_collar_top + c.lid_disc_t + hover_dz
        _write_body(body, p, _qz(lid_yaw_for(delta_deg)))
        _step(150)

    def seat_lid(delta_deg: float) -> None:
        """CONSTRUCT: the red lid resting on the internal ledge at engagement
        `delta_deg` (teleport 3 mm above the seat, settle onto it)."""
        p = pot_c()
        p[:, 2] += c.pot_ledge_top + 0.003
        _write_body(scene.lid, p, _qz(lid_yaw_for(delta_deg)))
        _step(90)

    def ccw_twist(tau: float, steps: int, w_max: float = 1.5) -> None:
        """Speed-governed COUNTER-clockwise torque (+z) for `steps` steps, then clear."""
        zero = torch.zeros(n, 1, 3, device=device)
        for _ in range(steps):
            wz = scene.lid.data.root_ang_vel_w[:, 2]
            t = torch.zeros(n, 1, 3, device=device)
            t[:, 0, 2] = torch.where(wz < w_max,
                                     torch.full_like(wz, tau),
                                     torch.zeros_like(wz))
            scene.lid.set_external_force_and_torque(zero, t, env_ids=_all_ids(),
                                                    is_global=True)
            _step(1)
        scene.lid.set_external_force_and_torque(zero, zero, env_ids=_all_ids())

    def delta_deg() -> float:
        _refresh()
        return float(torch.rad2deg(scene.engagement())[0])

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(150)
    _report("settle")
    _REC["on"] = False
    _refresh()
    lid_z = float((scene.lid.data.root_pos_w - scene.env_origins)[0, 2])
    check("settle/no-NaN: layout settles finite, both lids flat on the ground, "
          "score ~0, no success",
          bool(torch.isfinite(scene.lid.data.root_pos_w).all())
          and bool(torch.isfinite(scene.decoy.data.root_pos_w).all())
          and lid_z < 0.05 and float(scene.score()[0]) <= 0.03
          and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (scene.pot.data.root_pos_w[0, :2].clone(), pot_yaw(),
                int(scene.lid_side[0]))

    sides = set()
    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_p, a_y, a_s = readback()
    sides.add(a_s)
    d_p = d_y = 0.0  # max deltas vs the reference seed, over 8 probe seeds
    for s in range(8):
        torch.manual_seed(202 + s)
        env.reset()
        _step(10)
        b_p, b_y, b_s = readback()
        sides.add(b_s)
        d_p = max(d_p, float((a_p - b_p).norm()))
        dy = abs(math.degrees(a_y - b_y))
        d_y = max(d_y, min(dy, 360.0 - dy))
    print(f"[smoke] randomization deltas: pot_xy={d_p * 1000:.1f}mm "
          f"pot_yaw={d_y:.1f}deg sides_seen={sorted(sides)}", flush=True)
    check("randomization-is-real: pot xy AND pot yaw readback differ, and the red "
          "lid's start side swaps across seeds",
          d_p > 0.003 and d_y > 5.0 and sides == {-1, 1})

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 4. negative: the SEED'S OWN STRATEGY =======================================
    # "Put the white bowl ON the plate" — one set-down on the target, no keying, no
    # twist. CONSTRUCT it: the lid released over the pot with its lugs over the
    # TABS (delta ~ 30°). It comes to rest ON the brass ring, ~86 mm up — 16+ mm
    # above the seat band. The seed's whole plan physically cannot reach the seat.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    drop_lid(scene.lid, 30.0)
    _report("seed-strategy")
    _refresh()
    z_rest = float(scene._rel(scene.lid)[1][0])
    check("negative (SEED strategy): lid set down ON the pot, lugs over the tabs — "
          "rests on the brass ring out of the seat band, NO success",
          z_rest > 0.075 and not bool(scene.seated()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.30)
    _REC["on"] = False

    # ================= 5. negative: seated but NOT twisted ========================================
    # The keyed insertion alone (the drop through the gaps) is worth at most 0.75:
    # seated at delta ~ 90° is out of the locked window.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    drop_lid(scene.lid, float(c.entry_deg))
    _report("seated-unlocked")
    check("negative (no twist): lid dropped through the gaps seats on the ledge at "
          "delta ~ 90° — seated but NOT locked, no success, score <= 0.75",
          bool(scene.seated()[0]) and not bool(scene.locked()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.75)
    _REC["on"] = False

    # ================= 6. near-miss: partial twist ================================================
    # A clockwise twist that stops short of the stops: seated at delta ~ 48°, just
    # 8° outside the locked window — still not success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    seat_lid(48.0)
    _report("partial-twist")
    check("near-miss (partial twist): lid seated at delta ~ 48°, short of the "
          "stops — outside the locked window, no success",
          bool(scene.seated()[0]) and not bool(scene.locked()[0])
          and not bool(scene.success()[0]) and 40.0 < delta_deg() < 60.0)

    # ================= 7. negative: COUNTER-clockwise jams ========================================
    # Directionality is physical: from the entry angle, CCW torque drives the lugs
    # INTO the stops almost immediately (~106°); the locked window is never entered.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    drop_lid(scene.lid, float(c.entry_deg))
    assert bool(scene.seated()[0]), "probe setup: lid must seat before the CCW twist"
    never_locked = True
    for _ in range(10):
        ccw_twist(0.19, 25)
        _step(15)
        never_locked = never_locked and not bool(scene.locked()[0])
    _step(60)
    _report("ccw-jam")
    e_end = delta_deg()
    check("negative (CCW jam): counter-clockwise torque jams the lugs against the "
          "stops (~106°) — never enters the locked window, no success",
          bool(scene.seated()[0]) and never_locked and not bool(scene.locked()[0])
          and 95.0 < e_end < 120.0 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 8. negative: the DECOY is keyed out ========================================
    # The blue-knobbed decoy's lugs (78 mm) overshoot the collar wall (72 mm): even
    # dropped perfectly aligned with the gaps it cannot descend into the collar.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    p = pot_c()
    p[:, 2] += c.pot_collar_top + c.lid_disc_t + 0.012
    _write_body(scene.decoy, p, _qz(lid_yaw_for(float(c.entry_deg))))
    _step(150)
    _refresh()
    dz = float((scene.decoy.data.root_pos_w - scene.pot.data.root_pos_w)[0, 2])
    _report("decoy-drop")
    print(f"[smoke] decoy rest height above pot base: {dz * 1000:.1f}mm "
          f"(seat band tops out at {c.seat_z_hi * 1000:.0f}mm)", flush=True)
    check("negative (decoy keyed out): the blue lid dropped aligned over the pot "
          "cannot enter the collar — rests near rim height, no success",
          dz > 0.075 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.03)
    _REC["on"] = False

    # ================= 9. RETENTION contrast: the tabs are load-bearing ===========================
    # Engaged (lugs under the tabs, delta ~ 48°): a 3x-weight straight pull for
    # 0.75 s cannot extract the lid — the overhanging tabs catch the lugs.
    # Disengaged (lugs in the gaps, delta ~ 90°): a 1.5x-weight pull lifts it
    # straight out. Retention is contact geometry, not bookkeeping.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    seat_lid(48.0)
    assert bool(scene.seated()[0]), "probe setup: lid must seat at delta ~ 48"
    f_hold = torch.tensor([0.0, 0.0, 3.0 * c.lid_mass * 9.81], device=device)
    z_eng = _pull(scene.lid, f_hold, 90)
    _step(60)
    _refresh()
    held = z_eng < 0.080 and bool(scene.seated()[0])
    _report("retention-pull")
    torch.manual_seed(100)
    env.reset()
    _step(30)
    seat_lid(float(c.entry_deg))
    assert bool(scene.seated()[0]), "probe setup: lid must seat at delta ~ 90"
    f_lift = torch.tensor([0.0, 0.0, 1.5 * c.lid_mass * 9.81], device=device)
    z_dis = _pull(scene.lid, f_lift, 90, z_stop=0.20)
    _report("extraction-pull")
    print(f"[smoke] straight pull: engaged max z={z_eng * 1000:.1f}mm (3x weight, "
          f"held under the tabs) vs disengaged max z={z_dis * 1000:.1f}mm "
          f"(1.5x weight, extracted)", flush=True)
    check("RETENTION interlock: engaged lid (delta ~ 48°) survives a 3x-weight "
          "straight pull under the tabs; disengaged lid (delta ~ 90°) pulls "
          "straight out at 1.5x weight",
          held and z_dis > 0.15 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 10. negative: flipped lid ==================================================
    # The lid upside-down in the collar: the knob hangs down through the ledge
    # opening, the lugs rest ON the ledge — origin above the seat band and the
    # lid's +z inverted. Not seated, no success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    p = pot_c()
    p[:, 2] += 0.075
    _write_body(scene.lid, p, _qy(torch.full((n,), math.pi, device=device)))
    _step(120)
    _report("flipped-lid")
    up_z = float(scene.lid_up()[0, 2])
    check("negative (flipped): lid upside-down in the collar — +z inverted, out of "
          "the seat band, not seated, no success",
          up_z < 0.0 and not bool(scene.seated()[0])
          and not bool(scene.success()[0])
          and bool(torch.isfinite(scene.lid.data.root_pos_w).all()))

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.lid_bayonet")
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
