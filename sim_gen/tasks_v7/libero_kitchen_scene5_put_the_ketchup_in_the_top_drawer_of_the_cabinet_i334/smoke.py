"""Smoke battery for LetterboxCradleScene (sim_gen task
`libero_kitchen_scene5_put_the_ketchup_in_the_top_drawer_of_the_cabinet_i334`) —
REJECTION-ONLY: every check either verifies basic health/randomization or
CONSTRUCTS/EXECUTES a wrong outcome and asserts the rubric refuses it.
success() must never fire anywhere in the battery. (The cradle is JOINTED to
the kinematic vault — the revolute limits ARE the hard stops — so a
"stolen-cradle" fake is structurally impossible; the riding guard additionally
checks `cradle_on_hinge`.) NOTE: an upright bottle standing on the chamber
floor would be GENUINE success (upright inside the vault, however produced),
so no probe constructs or routes through that state.

 1. settle/no-NaN     — bottle standing at its sampled apron pose, cradle
                        resting on its 0-deg stop, score ~0, no success.
 2. randomization     — the bottle's staged (x, y) varies across resets with
                        real span on both axes, and the SETTLED bottle tracks
                        the sample, upright on the apron, every time.
 3. null policy       — 240 idle steps -> score ~0, no success.
 4. SEED strategy     — the seed's whole skill (carry the UPRIGHT bottle to
                        the receptacle and insert it) executed for REAL: a
                        force-capped push drives the standing bottle at the
                        slot. The header refuses it: the bottle is NEVER
                        (inside AND upright) at any step, no success. (If the
                        shove happens to topple it, it can only continue
                        lying — which is the task's own plan, not the seed's.)
 5. wrong order       — the EMPTY cradle is cranked past over-centre for real
                        (it tips onto its 90-stop and STAYS — bistable), and
                        only then is the bottle laid and pushed through the
                        slot: it strands lying low in the chamber, nothing
                        left to erect it. Erect credit stays ~0 (the riding
                        guard), score <= laid+inside, no success.
 6. under-crank       — the full legitimate entry (lay + real push to the
                        seat), then a real crank to ~50 deg — BELOW the loaded
                        over-centre (~68 deg) — and release: the cradle falls
                        BACK to its rest stop and lays the bottle down again.
                        Partial erect credit latches (< cap), never success.
 7. lying-inside fake — bottle CONSTRUCTED lying on the chamber floor (inside,
                        settled, wrong pose): laid+inside credit only, no
                        success.
 8. roof-perch fake   — bottle CONSTRUCTED standing on the vault roof
                        (upright, settled, wrong place): outside the chamber
                        box, score ~0, no success.
 9. rejection audit   — success() observed False at every step of the battery.
10. final no-NaN.
11. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene5_put_the_ketchup_in_the_top_drawer_of_the_cabinet_i334.smoke --headless
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

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
    p = (scene.bottle.data.root_pos_w - scene.env_origins)[0]
    print(f"[smoke] {tag:16s} | b=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}) "
          f"upz={float(scene.bottle_up()[0, 2]):+.2f} "
          f"phi={math.degrees(float(scene.phi()[0])):+.1f}deg "
          f"latch(l/i/e)=({int(scene._flaid[0])},{int(scene._finside[0])},"
          f"{float(scene._ferect[0]):.2f}) riding={bool(scene.bottle_riding()[0])} "
          f"settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    from isaaclab.utils.math import quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.letterbox_cradle_vault")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    zero13 = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.45, -1.15, 1.25)) + o),
                                tuple(np.array((-0.05, 0.00, 0.55)) + o),
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
        return (scene.env_origins
                + torch.tensor([dx, dy, dz], device=device)).expand(n, 3)

    def bpos():
        return (scene.bottle.data.root_pos_w - scene.env_origins)[0]

    def phi_deg() -> float:
        return math.degrees(float(scene.phi()[0]))

    def lying_quat() -> torch.Tensor:
        """wxyz quat rotating +90 deg about Y: body +z (cap) -> world +x."""
        q = torch.zeros(n, 4, device=device)
        h = math.sqrt(0.5)
        q[:, 0] = h
        q[:, 2] = h
        return q

    def clear_wrench() -> None:
        scene.bottle.set_external_force_and_torque(zero13, zero13)
        scene.cradle.set_external_force_and_torque(zero13, zero13)

    def lay_in_lane() -> None:
        """TRANSPORT construct (what a pick-and-lay-down produces): the bottle
        hovering lying over the guide lane; it drops ~15 mm and settles."""
        _write_body(scene.bottle, origin(0.14, 0.0, c.apron_z1 + c.bottle_r + 0.015),
                    lying_quat())
        _step(120)

    def push_bottle(v_des: float, steps: int, x_stop: float | None,
                    watch_upright_inside: bool = False) -> bool:
        """REAL actuation: a velocity-capped horizontal push at the bottle's CoM
        (|F| <= push_f_max), world -x, rotated into the body frame each step.
        Returns True if (inside AND upright) was ever observed while pushing."""
        seen = False
        stall, x_last = 0, float(bpos()[0])
        for i in range(steps):
            x = float(bpos()[0])
            if x_stop is not None and x <= x_stop:
                break
            if i % 30 == 29:
                stall = stall + 1 if abs(x - x_last) < 0.001 else 0
                x_last = x
                if stall >= 3:
                    break
            vx = scene.bottle.data.root_lin_vel_w[:, 0]
            fw = torch.zeros(n, 3, device=device)
            fw[:, 0] = (30.0 * (v_des - vx)).clamp(-c.push_f_max, c.push_f_max)
            fb = quat_apply_inverse(scene.bottle.data.root_quat_w, fw)
            scene.bottle.set_external_force_and_torque(fb.reshape(n, 1, 3), zero13)
            _step(1)
            if watch_upright_inside:
                seen |= bool((scene.bottle_inside() & scene.bottle_upright())[0])
        clear_wrench()
        _step(90)
        return seen

    def crank_servo(phi_target_deg: float, steps: int) -> None:
        """REAL actuation: a ramped, torque-limited servo on the cradle about
        its hinge axis (a hand on the external knob; |tau| <= crank_tau_max).
        A pure Y-torque on a body that only rotates about Y is frame-invariant."""
        tgt_final = math.radians(phi_target_deg)
        phi0 = float(scene.phi()[0])
        for i in range(steps):
            tgt = phi0 + (tgt_final - phi0) * min(1.0, (i + 1) / (0.7 * steps))
            phi = scene.phi()
            w = scene.cradle.data.root_ang_vel_w[:, 1]
            tau = -(2.0 * (tgt - phi) + 0.3 * w).clamp(-c.crank_tau_max, c.crank_tau_max)
            tw = torch.zeros(n, 3, device=device)
            tw[:, 1] = tau
            scene.cradle.set_external_force_and_torque(zero13, tw.reshape(n, 1, 3))
            _step(1)
        clear_wrench()

    seat_x = c.hinge_x + c.foot_x1 + c.bottle_l / 2  # bottle centre x when seated
    stand_z = c.apron_z1 + c.bottle_l / 2 + 0.002

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(150)
    _report("reset")
    p = bpos()
    sx, sy = float(scene.stage_xy[0, 0]), float(scene.stage_xy[0, 1])
    check("settle/no-NaN: bottle standing at its sampled apron pose, cradle "
          "resting on its 0-deg stop, score ~0, no success",
          bool(scene._finite()[0]) and bool(scene.bottle_upright()[0])
          and abs(float(p[0]) - sx) < 0.01 and abs(float(p[1]) - sy) < 0.01
          and abs(float(p[2]) - stand_z) < 0.012
          and abs(phi_deg()) < 2.0 and bool(scene.cradle_on_hinge()[0])
          and float(scene.score()[0]) <= 0.02 and not succ())

    # ================= 2. randomization readback ==================================================
    xs, ys, tracks, upright = [], [], True, True
    for k in range(10):
        torch.manual_seed(20 + k)
        env.reset()
        _step(90)
        _refresh()
        p = bpos()
        sx, sy = float(scene.stage_xy[0, 0]), float(scene.stage_xy[0, 1])
        xs.append(sx)
        ys.append(sy)
        tracks = tracks and abs(float(p[0]) - sx) < 0.01 and abs(float(p[1]) - sy) < 0.01 \
            and abs(float(p[2]) - stand_z) < 0.012
        upright = upright and bool(scene.bottle_upright()[0])
    xspan, yspan = max(xs) - min(xs), max(ys) - min(ys)
    print(f"[smoke] readback: x={[f'{v:.3f}' for v in xs]} span={xspan * 1000:.0f}mm "
          f"y={[f'{v:.3f}' for v in ys]} span={yspan * 1000:.0f}mm "
          f"tracks={tracks} upright={upright}", flush=True)
    check("randomization: the staged (x, y) varies across resets (span x > 40 mm, "
          "y > 60 mm) and the settled bottle tracks the sample, upright, every time",
          xspan > 0.040 and yspan > 0.060 and tracks and upright)

    # ================= 3. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> score ~0, no success",
          float(scene.score()[0]) <= 0.02 and not succ())

    # ================= 4. SEED strategy (upright insertion, executed for real) ====================
    torch.manual_seed(41)
    env.reset()
    _step(120)
    # transport the STANDING bottle into the lane mouth (what the seed's carry does) ...
    _write_body(scene.bottle, origin(0.15, 0.0, stand_z + 0.004))
    _step(90)
    # ... then push it at the slot for real: the seed's insertion attempt.
    seen_ui = push_bottle(-0.08, 300, None, watch_upright_inside=True)
    _report("seed-skill")
    p = bpos()
    check("SEED strategy: the upright bottle pushed at the slot for real — the "
          "header refuses upright passage: never (inside AND upright) at any "
          "step, root never through the wall while upright, no success",
          (not seen_ui) and not succ()
          and not bool((scene.bottle_inside() & scene.bottle_upright())[0])
          and abs(phi_deg()) < 3.0)

    # ================= 5. wrong order (crank first, insert second) ================================
    torch.manual_seed(51)
    env.reset()
    _step(120)
    crank_servo(80.0, 480)          # crank the EMPTY cradle past over-centre, for real
    _step(240)                      # release: bistable — it falls onto the 90-stop and stays
    _report("empty-cranked")
    phi_empty_held = phi_deg()
    fe_empty = float(scene._ferect[0])
    lay_in_lane()                   # only now lay the bottle ...
    push_bottle(-0.15, 700, None)   # ... and push it through the slot for real
    _report("wrong-order")
    p = bpos()
    check("wrong order: the empty cradle cranked past over-centre stays erected "
          "(bistable) and earns NO erect credit (riding guard); the bottle then "
          "pushed through the slot strands lying low in the chamber — nothing "
          "left to erect it: score <= laid+inside, no success",
          phi_empty_held >= 86.0 and fe_empty <= 0.02
          and float(scene._ferect[0]) <= 0.02
          and bool(scene.bottle_lying()[0]) and not bool(scene.bottle_riding()[0])
          and float(p[2]) < 0.50 and float(p[0]) < -0.03
          and float(scene.score()[0]) <= c.w_laid + c.w_inside + 0.02
          and not succ())

    # ================= 6. under-crank (release below over-centre) =================================
    torch.manual_seed(61)
    env.reset()
    _step(120)
    lay_in_lane()
    push_bottle(-0.15, 900, seat_x + 0.004)   # the legitimate entry, executed for real
    _report("seated")
    seated_ok = bool(scene.bottle_riding()[0]) and bool(scene.bottle_inside()[0]) \
        and abs(phi_deg()) < 3.0
    crank_servo(50.0, 420)                    # raise to ~50 deg — BELOW loaded over-centre
    phi_held = phi_deg()
    _report("under-crank-held")
    _step(360)                                # release: gravity drops the cradle back
    _report("under-crank-free")
    s6 = float(scene.score()[0])
    check("under-crank: the seated cradle raised to ~50 deg (below the ~68 deg "
          "loaded over-centre) and released falls BACK to its rest stop, laying "
          "the bottle down again — partial erect credit latches below the cap, "
          "never success",
          seated_ok and 40.0 <= phi_held <= 62.0
          and phi_deg() < 10.0 and bool(scene.bottle_lying()[0])
          and 0.55 <= s6 <= 0.85 and not succ())

    # ================= 7. lying-inside fake =======================================================
    torch.manual_seed(71)
    env.reset()
    _step(120)
    # lying on the chamber floor, off the cradle's lane (clear of bed and axle)
    _write_body(scene.bottle, origin(-0.16, -0.095, c.floor_z1 + c.bottle_r + 0.004),
                lying_quat())
    _step(180)
    _report("lying-inside")
    check("lying-inside fake: the bottle constructed lying on the chamber floor "
          "(inside, settled, wrong pose) — laid+inside credit only, no success",
          bool(scene.bottle_inside()[0]) and bool(scene.bottle_lying()[0])
          and not bool(scene.bottle_upright()[0])
          and float(scene.score()[0]) <= c.w_laid + c.w_inside + 0.02
          and not succ())

    # ================= 8. roof-perch fake =========================================================
    torch.manual_seed(81)
    env.reset()
    _step(120)
    _write_body(scene.bottle, origin(-0.16, 0.0, c.roof_z1 + c.bottle_l / 2 + 0.002))
    _step(180)
    _report("roof-perch")
    check("roof-perch fake: the bottle constructed standing on the vault roof "
          "(upright, settled, wrong place) — outside the chamber box, score ~0, "
          "no success",
          bool(scene.bottle_upright()[0]) and not bool(scene.bottle_inside()[0])
          and float(scene.score()[0]) <= 0.02 and not succ())

    # ================= 9-11. audit, no-NaN, video =================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.letterbox_cradle_vault")
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
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
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
