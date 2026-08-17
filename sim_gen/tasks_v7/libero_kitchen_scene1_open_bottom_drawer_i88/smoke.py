"""Smoke battery for BayonetDrawerScene (sim_gen task
`libero_kitchen_scene1_open_bottom_drawer_i88`) — REJECTION-ONLY: every check
either verifies basic health/randomization or CONSTRUCTS (or physically drives)
a settled wrong outcome and asserts the rubric refuses it. success() must never
fire anywhere in the battery.

 1. settle/no-NaN     — key resting near its sampled floor pose, drawer sealed
                        at 0, score ~0, no success.
 2. randomization A   — the key's sampled floor pose (x, y, yaw) varies across
                        resets and the settled key tracks it every time (readback).
 3. randomization B   — bowl and plate xy poses vary across resets (readback).
 4. null policy       — 240 idle steps -> nothing moves, score ~0, no success.
 5. front push probe  — REAL force on the judged drawer (the only purchase-free
                        move available): first constructed 2 cm open, a 15 N
                        inward push drives it back to its inner stop (asserted:
                        it actually moved — the force pathway is live), then
                        keeps pressing: the drawer cannot be pushed open and
                        stays at 0; no success. (Pulling the drawer BODY with an
                        external wrench would bypass the very no-purchase
                        geometry being claimed, so it is not a valid probe; the
                        impossibility is geometric — smooth flush slab, 4 mm
                        gap, inner stop — and asserted in the cfg contract.)
 6. untwisted pull    — the SEED's reflex (pull without the twist): the key is
                        constructed fully inserted but FLAT, then a REAL servo
                        pull retreats +x — the crossbar passes back out through
                        the slot and the key simply extracts (asserted: it moved
                        way out); the drawer stays sealed; score <= keyed credit
                        (0.15), no success.
 7. pre-twisted key   — the key constructed OUTSIDE the cabinet already rolled
                        90 deg, then a REAL servo push toward the slot: the
                        vertical 46 mm crossbar cannot pass the 16 mm slot — the
                        key jams on the slab face (asserted: it moved inward and
                        stopped short), keyed never fires, drawer sealed,
                        score ~0, no success.
 8. decoy misuse      — the full REAL strategy executed at the DECOY slot in
                        the fixed panel: insert (decoy_keyed fires — asserted),
                        twist, pull. The key locks to the CABINET: the pull
                        cannot extract it (asserted: crossbar still captive) and
                        moves no drawer; keyed stays False, score ~0, no success.
 9. near miss         — drawer constructed settled at ~7 cm (past the crack
                        latch, short of the 8 cm goal): capped partial credit
                        only, no success.
10. open but moving   — drawer written at 10 cm WITH outward velocity: the
                        settle gate refuses success on the moving state
                        (checked without stepping; then retracted).
11. wrong object      — bowl and plate shoved off the cabinet to the ground:
                        score ~0, no success.
12. latched credit    — the REAL coupling driven partway (genuine servo insert
                        + twist: keyed and locked latches fire physically),
                        then the key STOLEN away to the floor: latches survive,
                        the drawer never opened, success does not fire.
13. rejection audit   — success() observed False at every step of the battery.
14. final no-NaN.
15. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_open_bottom_drawer_i88.smoke --headless
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

# ----- key wrench-servo gains (same limited hand as solve.py) ----------------------------------
KP_P = 150.0
KD_P = 6.0
F_CLAMP = 20.0
KR = 0.02
T_CLAMP = 0.06

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


def _write_body(body, pos_w: torch.Tensor, vel_x: float = 0.0) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    st[:, 3] = 1.0
    st[:, 7] = vel_x
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    p = scene.cross_local()[0]
    print(f"[smoke] {tag:16s} | open={float(scene.drawer_open()[0]):+.4f} "
          f"cross=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}) "
          f"tilt={float(scene.cross_tilt()[0]):.2f} "
          f"latch(k/l/c)=({int(scene._keyed_l[0])},{int(scene._locked_l[0])},"
          f"{int(scene._crack_l[0])}) settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bayonet_drawer")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import axis_angle_from_quat, quat_apply_inverse, quat_conjugate, quat_mul

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.30, -1.05, 0.85)) + o),
                                tuple(np.array((-0.10, 0.00, 0.30)) + o),
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

    def succ() -> bool:
        s = bool(scene.success()[0])
        _AUDIT["saw_success"] |= s
        return s

    def origin(dx: float, dy: float, dz: float) -> torch.Tensor:
        return (scene.env_origins[0:1]
                + torch.tensor([dx, dy, dz], device=device)).expand(n, 3)

    zero = torch.zeros(n, 1, 3, device=device)

    def key_quat(theta: float) -> torch.Tensor:
        """Yaw pi (key +x faces the cabinet) composed with roll theta about world x."""
        q = torch.zeros(n, 4, device=device)
        q[:, 2] = -math.sin(theta / 2.0)
        q[:, 3] = math.cos(theta / 2.0)
        return q

    def write_key(px: float, py: float, pz: float, theta: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = origin(px, py, pz)
        st[:, 3:7] = key_quat(theta)
        scene.key.write_root_state_to_sim(st, _all_ids())
        _refresh()

    def key_ramp(steps: int, tail: int, p_from, p_to, th_from: float, th_to: float,
                 done=None) -> None:
        """REAL actuation: the same limited 6-DOF wrench servo as solve.py,
        ramping a pose target (what a hand on the knob does)."""
        p0 = torch.tensor(p_from, device=device).expand(n, 3)
        p1 = torch.tensor(p_to, device=device).expand(n, 3)
        for i in range(steps + tail):
            a = min(1.0, i / max(steps, 1))
            p_t = p0 + (p1 - p0) * a
            q_t = key_quat(th_from + (th_to - th_from) * a)
            p = scene.key.data.root_pos_w - scene.env_origins
            v = scene.key.data.root_lin_vel_w
            q = scene.key.data.root_quat_w
            f_w = KP_P * (p_t - p) - KD_P * v
            f_w[:, 2] += c.key_mass * 9.81
            f_w = f_w.clamp(min=-F_CLAMP, max=F_CLAMP)
            q_err = quat_mul(q_t, quat_conjugate(q))
            aa = axis_angle_from_quat(q_err)
            t_w = (KR * aa).clamp(min=-T_CLAMP, max=T_CLAMP)
            f_b = quat_apply_inverse(q, f_w)
            t_b = quat_apply_inverse(q, t_w)
            scene.key.set_external_force_and_torque(f_b.reshape(n, 1, 3), t_b.reshape(n, 1, 3))
            _step(1)
            if done is not None and a >= 1.0 and done():
                break
        scene.key.set_external_force_and_torque(zero, zero)
        _step(30)

    def push_drawer(tgt: float, steps: int, *, kp: float, kd: float, clamp: float) -> None:
        """REAL actuation: a PD force along the drawer's prismatic axis toward
        opening `tgt`, force-limited to what a hand would apply."""
        for _ in range(steps):
            d = scene.drawer_open()
            v = scene.drawer.data.root_lin_vel_w[:, 0]
            fx = (kp * (tgt - d) - kd * v).clamp(min=-clamp, max=clamp)
            f_w = torch.zeros(n, 3, device=device)
            f_w[:, 0] = fx
            f_b = quat_apply_inverse(scene.drawer.data.root_quat_w, f_w)
            scene.drawer.set_external_force_and_torque(f_b.reshape(n, 1, 3), zero)
            _step(1)
        scene.drawer.set_external_force_and_torque(zero, zero)
        _step(60)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(150)
    _report("reset")
    kp0 = (scene.key.data.root_pos_w - scene.env_origins)[0]
    check("settle/no-NaN: key resting near its sampled floor pose, drawer "
          "sealed at 0, score ~0, no success",
          bool(scene._finite()[0])
          and float((kp0[:2] - scene.k0[0, :2]).norm()) < 0.03
          and float(scene.drawer_open()[0]) < 0.005
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2+3. randomization readback ================================================
    ks, kset, yaws, xys = [], [], [], []
    for k in range(8):
        torch.manual_seed(20 + k)
        env.reset()
        _step(60)
        _refresh()
        ks.append(scene.k0[0, :2].tolist())
        yaws.append(float(scene.k0[0, 2]))
        kset.append((scene.key.data.root_pos_w[0] - scene.env_origins[0])[:2].tolist())
        bw = (scene.bowl.data.root_pos_w[0] - scene.env_origins[0])[:2].tolist()
        pl = (scene.plate.data.root_pos_w[0] - scene.env_origins[0])[:2].tolist()
        xys.append(bw + pl)
    kstd = float(np.std(np.asarray(ks), axis=0).mean())
    yaw_span = max(yaws) - min(yaws)
    tracks = all(math.hypot(a[0] - b[0], a[1] - b[1]) < 0.03 for a, b in zip(ks, kset))
    xystd = float(np.std(np.asarray(xys), axis=0).mean())
    print(f"[smoke] readback: key xystd={kstd:.3f} yaw_span={yaw_span:.2f} "
          f"tracks={tracks} topper_xystd={xystd:.3f}", flush=True)
    check("randomization A: the key's sampled floor pose (x, y, yaw) varies "
          "across resets and the settled key tracks it every time",
          kstd > 0.01 and yaw_span > 0.5 and tracks)
    check("randomization B: bowl and plate xy poses vary across resets",
          xystd > 0.008)

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> drawer stays sealed, score ~0, no success",
          float(scene.drawer_open()[0]) < 0.005
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 5. front push probe (real force on the judged drawer) =====================
    torch.manual_seed(41)
    env.reset()
    _step(120)
    # construct the drawer 2 cm open (below every latch), settled
    _write_body(scene.drawer, origin(0.020, 0.0, 0.0))
    _step(60)
    d_before = float(scene.drawer_open()[0])
    # real inward push: it slides back to its inner stop — the force pathway is live
    push_drawer(-0.05, 240, kp=300.0, kd=30.0, clamp=15.0)
    d_mid = float(scene.drawer_open()[0])
    # keep pressing against the stop: the drawer cannot be pushed open
    push_drawer(-0.05, 120, kp=300.0, kd=30.0, clamp=15.0)
    _report("front-push")
    check("front push probe: a real 15 N push on the judged drawer drives it "
          "back to its inner stop (it moved — the force is live) and pressing "
          "harder cannot open it; score ~0, no success",
          d_before >= 0.014 and d_before - d_mid >= 0.012
          and float(scene.drawer_open()[0]) < 0.004
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 6. untwisted pull (the seed's reflex without the twist) ====================
    torch.manual_seed(51)
    env.reset()
    _step(120)
    # construct the key fully inserted but FLAT (keyed latch fires by design)
    write_key(c.key_ins_x, 0.0, c.slot_z, 0.0)
    _step(30)
    assert bool(scene.keyed()[0]), "probe setup: flat-inserted key must be keyed"
    # real pull straight back out — no twist, no interlock
    key_ramp(360, 180, (c.key_ins_x, 0.0, c.slot_z), (0.30, 0.0, c.slot_z), 0.0, 0.0,
             done=lambda: float((scene.key.data.root_pos_w - scene.env_origins)[0, 0]) > 0.25)
    _report("untwisted-pull")
    kx6 = float((scene.key.data.root_pos_w - scene.env_origins)[0, 0])
    check("untwisted pull: a real servo pull on the flat-inserted key simply "
          "extracts it back through the slot (it moved way out); the drawer "
          "stays sealed; score <= keyed credit, no success",
          kx6 >= 0.15
          and float(scene.drawer_open()[0]) < 0.005
          and float(scene.score()[0]) <= c.w_keyed + 1e-4 and not succ())

    # ================= 7. pre-twisted key cannot enter ============================================
    torch.manual_seed(61)
    env.reset()
    _step(120)
    # construct the key OUTSIDE, already rolled 90 deg
    write_key(0.090, 0.0, c.slot_z, math.pi / 2)
    _step(30)
    kx7a = float((scene.key.data.root_pos_w - scene.env_origins)[0, 0])
    # real push toward the slot: the vertical crossbar jams on the slab face
    key_ramp(300, 180, (0.090, 0.0, c.slot_z), (c.key_ins_x, 0.0, c.slot_z),
             math.pi / 2, math.pi / 2)
    _report("pre-twisted")
    kx7b = float((scene.key.data.root_pos_w - scene.env_origins)[0, 0])
    check("pre-twisted key: a real servo push cannot pass the vertical 46 mm "
          "crossbar through the 16 mm slot — the key moved inward but jams "
          "outside (keyed never fires), drawer sealed, score ~0, no success",
          kx7a - kx7b >= 0.010 and kx7b >= 0.050
          and not bool(scene._keyed_l[0])
          and float(scene.drawer_open()[0]) < 0.005
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 8. decoy misuse (full strategy at the wrong slot) ==========================
    torch.manual_seed(71)
    env.reset()
    _step(120)
    write_key(0.090, 0.0, c.decoy_z, 0.0)  # staging hover at the DECOY slot
    _step(30)
    # real insert + twist at the decoy
    key_ramp(300, 180, (0.090, 0.0, c.decoy_z), (c.key_ins_x, 0.0, c.decoy_z), 0.0, 0.0,
             done=lambda: bool(scene.decoy_keyed()[0]))
    decoy_hit = bool(scene.decoy_keyed()[0])
    key_ramp(300, 180, (c.key_ins_x, 0.0, c.decoy_z), (c.key_ins_x, 0.0, c.decoy_z),
             0.0, math.pi / 2,
             done=lambda: float(scene.cross_tilt()[0]) > 0.95)
    # real pull: the key is now locked to the FIXED panel — it cannot extract
    key_ramp(360, 180, (c.key_ins_x, 0.0, c.decoy_z), (0.30, 0.0, c.decoy_z),
             math.pi / 2, math.pi / 2)
    _report("decoy-misuse")
    cross_x8 = float(scene.cross_local()[0, 0])
    check("decoy misuse: the full real strategy at the DECOY slot keys the "
          "fixed panel (asserted hit), locks the key to the CABINET (the pull "
          "cannot extract the crossbar), moves no drawer; keyed stays False, "
          "score ~0, no success",
          decoy_hit and cross_x8 <= -(c.panel_t - 0.005)
          and not bool(scene._keyed_l[0])
          and float(scene.drawer_open()[0]) < 0.005
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 9. near miss (7 cm settled, goal is 8 cm) ==================================
    torch.manual_seed(81)
    env.reset()
    _step(120)
    _write_body(scene.drawer, origin(0.070, 0.0, 0.0))
    _step(120)
    _report("near-miss")
    s9 = float(scene.score()[0])
    check("near miss: drawer constructed settled at ~7 cm — past the crack "
          "latch but short of the 8 cm goal: capped partial credit, no success",
          0.060 <= float(scene.drawer_open()[0]) < c.open_goal
          and c.w_crack - 1e-4 <= s9 <= 0.60 + 1e-4 and not succ())

    # ================= 10. open but moving (settle gate) ==========================================
    torch.manual_seed(91)
    env.reset()
    _step(120)
    _write_body(scene.drawer, origin(0.100, 0.0, 0.0), vel_x=0.30)
    moving_rejected = not succ()  # judged WITHOUT stepping: open enough but moving
    still_open = float(scene.drawer_open()[0]) >= c.open_goal
    _write_body(scene.drawer, origin(0.020, 0.0, 0.0))  # retract before any stepping
    _step(90)
    _report("open-moving")
    check("open but moving: drawer written at 10 cm with outward velocity — "
          "the settle gate refuses success on the moving state",
          moving_rejected and still_open and not succ())

    # ================= 11. wrong object (distractors shoved off) ==================================
    torch.manual_seed(101)
    env.reset()
    _step(120)
    _write_body(scene.bowl, origin(0.60, 0.55, c.bowl_h / 2 + 0.002))
    _write_body(scene.plate, origin(0.60, -0.55, c.plate_h / 2 + 0.002))
    _step(180)
    _report("wrong-object")
    check("wrong object: bowl and plate shoved off the cabinet to the ground — "
          "score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 12. latched credit (real insert+twist, then key stolen) ====================
    torch.manual_seed(111)
    env.reset()
    _step(120)
    write_key(0.090, 0.0, c.slot_z, 0.0)  # transport to the staging hover
    _step(30)
    # drive the REAL coupling: genuine servo insert, then genuine twist
    key_ramp(300, 240, (0.090, 0.0, c.slot_z), (c.key_ins_x, 0.0, c.slot_z), 0.0, 0.0,
             done=lambda: bool(scene.keyed()[0]))
    key_ramp(300, 240, (c.key_ins_x, 0.0, c.slot_z), (c.key_ins_x, 0.0, c.slot_z),
             0.0, math.pi / 2,
             done=lambda: bool(scene.locked()[0]) and float(scene.cross_tilt()[0]) > 0.95)
    latched = bool(scene._keyed_l[0]) and bool(scene._locked_l[0])
    # steal the key away to the floor: latches must survive, success must not fire
    write_key(0.45, 0.30, c.key_z0, 0.0)
    _step(90)
    _report("latch-steal")
    s12 = float(scene.score()[0])
    check("latched credit: the real coupling driven through insert+twist "
          "(keyed+locked latches from genuine physics), then the key stolen "
          "away — latches survive, the drawer never opened, no success",
          latched and not bool(scene.keyed()[0])
          and float(scene.drawer_open()[0]) < 0.005
          and c.w_keyed + c.w_locked - 1e-4 <= s12 <= 0.60 + 1e-4 and not succ())

    # ================= 13-15. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.bayonet_drawer")
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
