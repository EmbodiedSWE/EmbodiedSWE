"""Smoke battery for CapsizeRecoveryScene (sim_gen task
`living_room_scene2_pick_up_the_orange_juice_and_put_it_in_the_basket_i178`).

Every wrong outcome is CONSTRUCTED as a settled physical state and asserted REJECTED:

- reset sanity: settled, capsized, juice COVERED, caged body-frame 'containment' earns
  nothing (score ~0), success False, finite state
- randomization by READBACK across 8 seeds: crate xy jitter + free yaw; caged carton
  offset + yaw vary (and the carton is covered at every seed); milk carton SIDE flips
  (Bernoulli) + xy jitter
- null policy: 2.5 s hands-off, no credit
- SEED strategy (put the carton 'in the basket' where it is): carton set on the
  capsized crate's upturned base -> NOT contained (below the floor slab in the crate
  frame), not success
- side-lying cavity trap: carton placed INSIDE the sideways cavity -> body-frame
  containment True but crate not upright -> no loaded latch, not success
- upright empty crate: unveil+right latches only (~0.50), not success
- WRONG carton: white milk dropped into the upright crate -> success excluded, no
  containment credit
- BOTH cartons inside: exclusion clause holds -> capped 0.65, never success
- near-miss: juice standing AGAINST the outside wall of the upright crate -> not
  contained, not success
- success reproduction (constructed upright crate + the solve's drop recipe):
  success True, score 1.0
- latch regression x2: yank the juice out, then knock the crate back over -> success
  collapses immediately but the latched 0.65 base credit persists (never 1.0)
- audit: success() was never True at any rejection-battery judged point
- final: no NaN anywhere

Records viewport rgb frames to frames.npz (cwd). Prints `SIM_GEN_SMOKE: ALL PASS n/n`.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os
import threading

import numpy as np
import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

C45 = math.cos(math.pi / 4)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.capsize_recovery")().build(num_envs=args.num_envs,
                                                      device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    origin = scene.env_origins

    from isaaclab.utils.math import quat_apply, quat_mul

    # --- recording (viewport rgb annotator) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.55, -1.10, 0.85)) + o),
                                tuple(np.array((0.55, 0.00, 0.10)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action)
            if (annot is not None and step_i % args.record_every == 0
                    and len(frames) < args.max_frames):
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    checks: list[tuple[str, bool]] = []
    reject_violated = [False]

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def judge(tag: str, *, expect_reject: bool = True) -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        if expect_reject and ok:
            reject_violated[0] = True
        print(f"[smoke] {tag:24s} | up_z={float(scene.crate_up()[0, 2]):+.3f} "
              f"covered={bool(scene.covered()[0])} "
              f"upright={bool(scene.upright()[0])} "
              f"grounded={bool(scene.grounded()[0])} "
              f"j_in={bool(scene.contained(scene.juice)[0])} "
              f"m_in={bool(scene.contained(scene.milk)[0])} "
              f"unv={bool(scene._unveiled_ever[0])} "
              f"rgt={bool(scene._righted_ever[0])} "
              f"ld={bool(scene._loaded_ever[0])} "
              f"still={bool(scene._still()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)
        return s, ok

    def finite_all() -> bool:
        ok = True
        for b in (scene.crate, scene.juice, scene.milk):
            ok = ok and bool(torch.isfinite(b.data.root_state_w).all())
        return ok

    def place(body, pos_env, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor(pos_env, device=device).expand(n, 3) + origin
        st[:, 3:7] = torch.tensor(quat, device=device).expand(n, 4)
        body.write_root_state_to_sim(st, all_ids)

    def drop_in(body, y_loc: float, max_steps: int = 900) -> None:
        """The solve's deposit recipe: stage the carton LYING (long axis along the
        crate-local x), bottom 30 mm above the rim, in the crate's CURRENT frame."""
        b_pos, b_quat = scene.crate.data.root_pos_w, scene.crate.data.root_quat_w
        rim = c.floor_t + c.wall_h
        loc = torch.tensor([0.0, y_loc, rim + 0.030 + c.carton_w / 2],
                           device=device).expand(n, 3)
        p_w = b_pos + quat_apply(b_quat, loc)
        qy90 = torch.tensor([C45, 0.0, C45, 0.0], device=device).expand(n, 4)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = p_w
        st[:, 3:7] = quat_mul(b_quat, qy90)
        body.write_root_state_to_sim(st, all_ids)
        quiet = 0
        for _ in range(max_steps):
            step(1)
            quiet = quiet + 1 if bool(scene._still()[0]) else 0
            if quiet >= 30:
                break

    def settle(k: int) -> None:
        step(k)

    def crate_xy() -> tuple[float, float]:
        p = (scene.crate.data.root_pos_w - origin)[0]
        return float(p[0]), float(p[1])

    # =========================== 1. reset sanity ============================================
    env.reset(seed=101)
    settle(60)
    s, ok = judge("reset+settle")
    check("reset: settled+capsized, juice covered, caged body-frame containment earns "
          "nothing (score <= 0.02), success False, no NaN",
          finite_all() and not ok and s <= 0.02 and bool(scene._still()[0])
          and bool(scene.covered()[0]) and bool(scene.contained(scene.juice)[0])
          and float(scene.crate_up()[0, 2]) < -0.95)

    # =========================== 2-4. randomization by readback =============================
    cxs, cys, psis, joffx, joffy, jyaws, msides, mxs = [], [], [], [], [], [], [], []
    covered_all = True
    for sd in range(8):
        env.reset(seed=sd)
        settle(5)
        cx, cy = crate_xy()
        cq = scene.crate.data.root_quat_w[0]
        psis.append(math.degrees(2.0 * math.atan2(float(cq[2]), float(cq[1]))))
        jp = (scene.juice.data.root_pos_w - origin)[0]
        jq = scene.juice.data.root_quat_w[0]
        jyaws.append(math.degrees(2.0 * math.atan2(-float(jq[1]), float(jq[0]))))
        mp = (scene.milk.data.root_pos_w - origin)[0]
        cxs.append(cx)
        cys.append(cy)
        joffx.append(float(jp[0]) - cx)
        joffy.append(float(jp[1]) - cy)
        msides.append(float(mp[1]) > 0)
        mxs.append(float(mp[0]))
        covered_all = covered_all and bool(scene.covered()[0])
    print(f"[smoke] readback 8 seeds: crate_x={[f'{v:+.3f}' for v in cxs]} "
          f"crate_y={[f'{v:+.3f}' for v in cys]} psi={[f'{v:+.0f}' for v in psis]}",
          flush=True)
    print(f"[smoke] joff_x={[f'{v:+.3f}' for v in joffx]} "
          f"joff_y={[f'{v:+.3f}' for v in joffy]} jyaw={[f'{v:+.0f}' for v in jyaws]} "
          f"milk+y={msides} milk_x={[f'{v:+.3f}' for v in mxs]}", flush=True)
    check("randomization: crate xy jitter + free yaw (readback spread)",
          max(cxs) - min(cxs) > 0.01 and max(cys) - min(cys) > 0.01
          and max(psis) - min(psis) > 30.0)
    check("randomization: caged carton offset + yaw vary; covered at every seed",
          (max(joffx) - min(joffx) > 0.004 or max(joffy) - min(joffy) > 0.004)
          and max(jyaws) - min(jyaws) > 10.0 and covered_all)
    check("randomization: milk SIDE flips across seeds + xy jitter",
          any(msides) and not all(msides) and max(mxs) - min(mxs) > 0.01)

    # =========================== 5. null policy =============================================
    env.reset(seed=102)
    settle(300)
    s, ok = judge("null policy 2.5s")
    check("null policy: 2.5 s hands-off leaves score <= 0.02, no success",
          s <= 0.02 and not ok)

    # =========================== 6. SEED strategy rejected ==================================
    # 'Put the carton in the basket where it is': the only reachable surface of the
    # capsized crate is its upturned base — set the carton there, settled.
    env.reset(seed=103)
    settle(30)
    cx, cy = crate_xy()
    place(scene.juice, (cx, cy, c.floor_t + c.wall_h + c.carton_w / 2 + 0.004),
          quat=(C45, 0.0, C45, 0.0))
    settle(240)
    jz = float((scene.juice.data.root_pos_w - origin)[0, 2])
    s, ok = judge("seed: on upturned base")
    check("seed strategy: carton set ON the capsized crate's upturned base -> resting "
          "there but NOT contained (below the floor slab in the crate frame), "
          "not success",
          jz > 0.125 and not bool(scene.contained(scene.juice)[0])
          and s <= 0.21 and not ok)

    # =========================== 7. side-lying cavity trap ==================================
    # Crate ON ITS SIDE (mouth sideways), carton resting INSIDE the sideways cavity:
    # body-frame containment is True — and correctly earns nothing (not upright).
    env.reset(seed=104)
    settle(30)
    place(scene.crate, (0.70, 0.0, 0.081), quat=(C45, 0.0, C45, 0.0))  # R_y(90): +x down
    place(scene.juice, (0.755, 0.0, 0.034), quat=(C45, -C45, 0.0, 0.0))
    settle(240)
    s, ok = judge("side-lying cavity")
    check("side-lying trap: carton inside the SIDEWAYS cavity -> body-frame "
          "containment True but crate not upright -> no loaded latch, score <= 0.21, "
          "not success",
          bool(scene.contained(scene.juice)[0]) and not bool(scene.upright()[0])
          and not bool(scene._loaded_ever[0]) and s <= 0.21 and not ok)

    # =========================== 8. upright empty crate =====================================
    env.reset(seed=105)
    settle(30)
    place(scene.crate, (0.70, 0.0, 0.002))
    settle(240)
    s, ok = judge("upright empty")
    check("upright empty crate: unveil+right latches only (0.49 <= score <= 0.502), "
          "not success",
          bool(scene.upright()[0]) and bool(scene.grounded()[0])
          and 0.49 <= s <= 0.502 and not ok)

    # =========================== 9. WRONG carton rejected ===================================
    drop_in(scene.milk, 0.033)
    s, ok = judge("milk in crate")
    check("wrong carton: WHITE milk inside the upright crate -> success excluded, no "
          "containment credit (score <= 0.502)",
          bool(scene.contained(scene.milk)[0]) and s <= 0.502 and not ok)

    # =========================== 10. BOTH cartons: exclusion ================================
    drop_in(scene.juice, -0.033)
    s, ok = judge("both cartons inside")
    check("exclusion: juice AND milk inside the upright crate -> capped 0.651, "
          "never success",
          bool(scene.contained(scene.juice)[0]) and bool(scene.contained(scene.milk)[0])
          and s <= 0.651 and not ok)

    # =========================== 11. outside-wall near-miss =================================
    env.reset(seed=106)
    settle(30)
    place(scene.crate, (0.70, 0.0, 0.002))
    settle(120)
    place(scene.juice, (0.70 - c.out / 2 - c.carton_w / 2 - 0.002, 0.0,
                        c.carton_h / 2 + 0.002))
    settle(240)
    s, ok = judge("leaning outside wall")
    check("near-miss: juice standing AGAINST the outside wall of the upright crate -> "
          "not contained, score <= 0.502, not success",
          not bool(scene.contained(scene.juice)[0]) and s <= 0.502 and not ok)

    # =========================== 12. success reproduction ===================================
    env.reset(seed=107)
    settle(30)
    place(scene.crate, (0.70, 0.0, 0.002))
    settle(120)
    drop_in(scene.juice, 0.0, max_steps=1100)
    quiet = 0
    for _ in range(400):
        step(1)
        quiet = quiet + 1 if bool(scene.success()[0]) else 0
        if quiet >= 30:
            break
    s, ok = judge("goal state", expect_reject=False)
    check("success reproduction: upright grounded crate + juice dropped in, settled -> "
          "success True, score 1.0",
          ok and s >= 0.999)

    # =========================== 13-14. latch regression ====================================
    place(scene.juice, (0.20, -0.45, c.carton_w / 2 + 0.002), quat=(C45, 0.0, C45, 0.0))
    settle(240)
    s, ok = judge("juice yanked out")
    check("latch regression: juice yanked out -> success collapses, latched credit "
          "stays (0.64 <= score <= 0.651)",
          not ok and 0.64 <= s <= 0.651
          and not bool(scene.contained(scene.juice)[0]))

    place(scene.crate, (0.70, 0.0, c.floor_t + c.wall_h + 0.002), quat=(0.0, 1.0, 0.0, 0.0))
    settle(240)
    s, ok = judge("crate knocked over")
    check("latch regression: crate knocked back capsized -> not upright, not success, "
          "latched credit stays (0.64 <= score <= 0.651)",
          not ok and not bool(scene.upright()[0]) and 0.64 <= s <= 0.651)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() never True at any rejection-battery judged point",
          not reject_violated[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.capsize_recovery")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        for nm, okc in checks:
            if not okc:
                print(f"[smoke] FAILED CHECK: {nm}", flush=True)
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)

    code = 0 if all_ok else 1
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
    except Exception as e:  # noqa: BLE001 — fail fast, don't idle until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
