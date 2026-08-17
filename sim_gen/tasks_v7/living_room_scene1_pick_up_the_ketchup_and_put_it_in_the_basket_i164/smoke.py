"""Smoke battery for HookHangBasketScene (sim_gen task
`living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket_i164`).

Every wrong outcome is CONSTRUCTED as a settled physical state and asserted REJECTED:

- reset sanity: settled, score ~0, finite state
- randomization by READBACK across 8 seeds: high/low peg SIDE flips, bottle SLOT swap
  (bottles always on opposite slots), continuous jitter (peg z, stand yaw, basket xy)
- null policy: 2.5 s hands-off, no credit
- SEED strategy (ketchup into the basket where it lies on the floor): contained but not
  hung -> score capped at the 0.10 containment latch, not success
- LOW-PEG decoy: bar hooked on the low peg -> the 186 mm hang depth grounds the basket
  into a leaning rest; hang band + upright + clearance reject it, ~0 credit
- hang alone (high peg, empty basket): 0.25, not success
- WRONG bottle (brown bbq) dropped into the hung basket: rejected, no containment credit
- BOTH bottles inside: exclusion clause holds -> capped 0.65, not success
- success reproduction (the solve recipe): success True, score 1.0
- latch regression x2: yank the ketchup out, then knock the basket off the peg ->
  success collapses immediately but the latched 0.65 base credit persists (never 1.0)
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


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.hook_hang_basket")().build(num_envs=args.num_envs,
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
        env.sim.set_camera_view(tuple(np.array((1.35, -0.95, 0.80)) + o),
                                tuple(np.array((0.40, 0.00, 0.22)) + o),
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
        print(f"[smoke] {tag:22s} | eng={bool(scene.engaged_hung()[0])} "
              f"k_in={bool(scene.contained(scene.ketchup)[0])} "
              f"bbq_in={bool(scene.contained(scene.bbq)[0])} "
              f"hung_ever={bool(scene._hung_ever[0])} "
              f"in_ever={bool(scene._in_ever[0])} "
              f"loaded_ever={bool(scene._loaded_ever[0])} "
              f"still={bool(scene._still()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)
        return s, ok

    def finite_all() -> bool:
        ok = True
        for b in (scene.stand, scene.peg_a, scene.peg_b, scene.basket,
                  scene.ketchup, scene.bbq):
            ok = ok and bool(torch.isfinite(b.data.root_state_w).all())
        return ok

    def place(body, pos_env, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor(pos_env, device=device).expand(n, 3) + origin
        st[:, 3:7] = torch.tensor(quat, device=device).expand(n, 4)
        body.write_root_state_to_sim(st, all_ids)

    def peg_z(peg) -> float:
        return float((peg.data.root_pos_w - origin)[0, 2])

    def hang_on_high(max_steps: int = 720) -> bool:
        """The solve recipe: stage the bar 25 mm above the HIGH peg, hands off."""
        high = scene.peg_a if peg_z(scene.peg_a) > peg_z(scene.peg_b) else scene.peg_b
        p_pos, p_quat = high.data.root_pos_w, high.data.root_quat_w
        bar_loc = torch.tensor([0.105, 0.0, c.peg_r + c.bar_t / 2 + 0.025],
                               device=device).expand(n, 3)
        bar_w = p_pos + quat_apply(p_quat, bar_loc)
        off = torch.tensor([0.0, 0.0, c.bar_z], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = bar_w - quat_apply(p_quat, off)
        st[:, 3:7] = p_quat
        scene.basket.write_root_state_to_sim(st, all_ids)
        quiet = 0
        for _ in range(max_steps):
            step(1)
            eng = bool(scene.engaged_hung()[0])
            still = float(scene.basket.data.root_lin_vel_w[0].norm()) < c.settle_speed
            quiet = quiet + 1 if (eng and still) else 0
            if quiet >= 30:
                return True
        return False

    def drop_into_basket(body, x_loc: float, max_steps: int = 900) -> None:
        """Stage a bottle upright just above the basket's CURRENT opening; drop."""
        b_pos, b_quat = scene.basket.data.root_pos_w, scene.basket.data.root_quat_w
        rim = c.floor_t + c.wall_h
        drop_loc = torch.tensor([x_loc, 0.0, rim], device=device).expand(n, 3)
        p_w = b_pos + quat_apply(b_quat, drop_loc)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = p_w[:, 0:2]
        st[:, 2] = p_w[:, 2] + 0.008 + c.body_h / 2
        st[:, 3] = 1.0
        body.write_root_state_to_sim(st, all_ids)
        quiet = 0
        for _ in range(max_steps):
            step(1)
            quiet = quiet + 1 if bool(scene._still()[0]) else 0
            if quiet >= 30:
                break

    def settle(k: int) -> None:
        step(k)

    # =========================== 1. reset sanity ============================================
    env.reset(seed=101)
    settle(60)
    s, ok = judge("reset+settle")
    check("reset: settled, no NaN, success False, score <= 0.02",
          finite_all() and not ok and s <= 0.02 and bool(scene._still()[0]))

    # =========================== 2-4. randomization by readback =============================
    highs, yaws, hz, k_ys, q_ys, b_xy = [], [], [], [], [], []
    opposite = True
    for sd in range(8):
        env.reset(seed=sd)
        settle(5)
        za, zb = peg_z(scene.peg_a), peg_z(scene.peg_b)
        highs.append(za > zb)
        hz.append(max(za, zb))
        sq = scene.stand.data.root_quat_w[0]
        yaws.append(math.degrees(2.0 * math.atan2(float(sq[3]), float(sq[0]))))
        ky = float((scene.ketchup.data.root_pos_w - origin)[0, 1])
        qy = float((scene.bbq.data.root_pos_w - origin)[0, 1])
        k_ys.append(ky)
        q_ys.append(qy)
        opposite = opposite and (ky * qy < 0)
        b = (scene.basket.data.root_pos_w - origin)[0]
        b_xy.append((float(b[0]), float(b[1])))
    print(f"[smoke] readback 8 seeds: high_is_a={highs} peg_z={[f'{z:.3f}' for z in hz]}",
          flush=True)
    print(f"[smoke] yaw={[f'{y:+.1f}' for y in yaws]} k_y={[f'{y:+.2f}' for y in k_ys]} "
          f"basket={[(f'{x:.2f}', f'{y:+.2f}') for x, y in b_xy]}", flush=True)
    check("randomization: HIGH peg side flips across seeds (readback)",
          any(highs) and not all(highs))
    check("randomization: bottle slots swap; bottles always on opposite slots",
          any(y > 0 for y in k_ys) and any(y < 0 for y in k_ys) and opposite)
    spread = (max(hz) - min(hz) > 0.004 and max(yaws) - min(yaws) > 3.0
              and max(x for x, _ in b_xy) - min(x for x, _ in b_xy) > 0.01)
    check("randomization: continuous jitter present (peg z, stand yaw, basket xy)", spread)

    # =========================== 5. null policy =============================================
    env.reset(seed=102)
    settle(300)
    s, ok = judge("null policy 2.5s")
    check("null policy: hands-off leaves score <= 0.02, no success", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy rejected ==================================
    # Ketchup into the basket where it lies ON THE FLOOR (the seed task's goal state).
    drop_into_basket(scene.ketchup, -0.030, max_steps=360)
    s, ok = judge("seed: in grounded basket")
    check("seed strategy: ketchup contained in the GROUNDED basket -> containment latch "
          "only (score <= 0.105), not success, not hung",
          bool(scene.contained(scene.ketchup)[0]) and not bool(scene.engaged_hung()[0])
          and s <= 0.105 and not ok)

    # =========================== 7. LOW-PEG decoy rejected ==================================
    # Construct the only physically reachable low-peg 'hang': bar hooked ON the low peg
    # with the basket tilted, bottom resting on the floor (hang depth 186 mm > peg
    # height). Staged just above that rest, tilted 75 deg about the bar axis.
    env.reset(seed=103)
    settle(30)
    low = scene.peg_a if peg_z(scene.peg_a) < peg_z(scene.peg_b) else scene.peg_b
    p_pos, p_quat = low.data.root_pos_w, low.data.root_quat_w
    bar_loc = torch.tensor([0.100, 0.0, c.peg_r + c.bar_t / 2 + 0.004],
                           device=device).expand(n, 3)
    bar_w = p_pos + quat_apply(p_quat, bar_loc)
    th = math.radians(75.0)
    q_tilt = torch.tensor([math.cos(th / 2), 0.0, math.sin(th / 2), 0.0],
                          device=device).expand(n, 4)
    b_quat = quat_mul(p_quat, q_tilt)
    off = torch.tensor([0.0, 0.0, c.bar_z], device=device).expand(n, 3)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = bar_w - quat_apply(b_quat, off)
    st[:, 3:7] = b_quat
    scene.basket.write_root_state_to_sim(st, all_ids)
    bar_local = scene._peg_local(low, scene._bar_center_w())[0]
    print(f"[smoke] low-peg construct: bar peg-local=({float(bar_local[0]):+.3f},"
          f"{float(bar_local[1]):+.3f},{float(bar_local[2]):+.3f})", flush=True)
    assert 0.015 <= float(bar_local[0]) <= c.peg_len - 0.006 \
        and abs(float(bar_local[2])) <= 0.04, "low-peg construct did not start on the peg"
    settle(360)
    s, ok = judge("low-peg decoy")
    check("low-peg decoy: bar on the LOW peg leaves the basket grounded/leaning -> "
          "not engaged-hung, score <= 0.02, not success",
          not bool(scene.engaged_hung()[0]) and not bool(scene._hung_ever[0])
          and s <= 0.02 and not ok)

    # =========================== 8. hang alone is 0.25, not success =========================
    env.reset(seed=104)
    settle(30)
    hung = hang_on_high()
    s, ok = judge("hang alone")
    check("hang alone (high peg, empty basket): engaged-hung, score ~0.25, not success",
          hung and bool(scene.engaged_hung()[0]) and 0.24 <= s <= 0.26 and not ok)

    # =========================== 9. WRONG bottle rejected ===================================
    drop_into_basket(scene.bbq, -0.040)
    s, ok = judge("bbq in hung basket")
    check("wrong bottle: BROWN bbq inside the hung basket -> no containment credit "
          "(score <= 0.26), not success",
          bool(scene.contained(scene.bbq)[0]) and bool(scene.engaged_hung()[0])
          and s <= 0.26 and not ok)

    # =========================== 10. BOTH bottles: exclusion ================================
    drop_into_basket(scene.ketchup, 0.032)
    s, ok = judge("both bottles inside")
    check("exclusion: ketchup AND bbq inside the hung basket -> capped 0.651, "
          "never success",
          bool(scene.contained(scene.ketchup)[0]) and bool(scene.contained(scene.bbq)[0])
          and bool(scene.engaged_hung()[0]) and s <= 0.651 and not ok)

    # =========================== 11. success reproduction ===================================
    env.reset(seed=105)
    settle(30)
    hung = hang_on_high()
    drop_into_basket(scene.ketchup, -0.040, max_steps=1100)
    quiet = 0
    for _ in range(400):
        step(1)
        quiet = quiet + 1 if bool(scene.success()[0]) else 0
        if quiet >= 30:
            break
    s, ok = judge("solve recipe", expect_reject=False)
    check("success reproduction: hang + gentle drop -> success True, score 1.0",
          hung and ok and s >= 0.999)

    # =========================== 12-13. latch regression ====================================
    place(scene.ketchup, (0.08, -0.38, c.body_h / 2 + 0.002))
    settle(240)
    s, ok = judge("ketchup removed")
    check("latch regression: ketchup yanked out -> success collapses, latched credit "
          "stays (0.64 <= score <= 0.651)",
          not ok and 0.64 <= s <= 0.651 and not bool(scene.contained(scene.ketchup)[0]))

    place(scene.basket, (0.30, -0.12, 0.002))
    settle(240)
    s, ok = judge("basket knocked down")
    check("latch regression: basket knocked off the peg -> not engaged, not success, "
          "latched credit stays (0.64 <= score <= 0.651)",
          not ok and not bool(scene.engaged_hung()[0]) and 0.64 <= s <= 0.651)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() never True at any rejection-battery judged point",
          not reject_violated[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.hook_hang_basket")
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
