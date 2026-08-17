"""Smoke battery for SluiceHopperCatchScene (sim_gen task
`living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket_i214`).

Every wrong outcome is CONSTRUCTED as a settled physical state (or driven by real
forces) and asserted REJECTED:

- reset sanity: bottle rolled down against the seated gate, settled, score ~0, finite
- randomization by READBACK across 8 seeds: hopper SIDE flips; continuous jitter in
  hopper x / yaw, bottle ramp position, basket + decoy xy; basket and decoy always on
  the side OPPOSITE the hopper
- null policy: 2.5 s hands-off, no credit
- SEED strategy (put the visible floor bottle into the basket): that bottle is the
  brown DECOY -> contained but score ~0, not success
- INTERLOCK: a velocity-capped horizontal force shoves the ketchup against the seated
  gate for 2.5 s — the probe must actually PRESS (readback), the bottle must never
  cross the lip plane, the gate must not unseat, no release credit
- discharge WITHOUT staging the basket: the real force-extraction recipe with the
  basket left at spawn -> bottle lands on the open floor; extract + release latch
  (0.45) but not success
- near-miss: ketchup settled leaning against the basket's OUTER wall: not contained
- wrong place: ketchup settled on the hopper ROOF: no release, ~0 credit
- success reproduction (the solve recipe): success True, score 1.0
- exclusion: brown decoy added INTO the loaded basket -> success collapses, cap 0.60
- latch regression: ketchup yanked out to the floor -> success collapses, the latched
  0.60 base persists (never 1.0 again)
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
    env = ENVS.get("simgen.sluice_hopper_catch")().build(num_envs=args.num_envs,
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
        env.sim.set_camera_view(tuple(np.array((1.40, -1.00, 0.85)) + o),
                                tuple(np.array((0.38, 0.00, 0.20)) + o),
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
        g_loc = scene._hopper_local(scene.gate.data.root_pos_w)[0]
        k_loc = scene._hopper_local(scene.ketchup.data.root_pos_w)[0]
        print(f"[smoke] {tag:22s} | gate_z={float(g_loc[2]):+.3f} "
              f"k_loc_x={float(k_loc[0]):+.3f} "
              f"staged_ever={bool(scene._staged_ever[0])} "
              f"extr_ever={bool(scene._extracted_ever[0])} "
              f"rel_ever={bool(scene._released_ever[0])} "
              f"k_in={bool(scene.contained(scene.ketchup)[0])} "
              f"bbq_in={bool(scene.contained(scene.bbq)[0])} "
              f"still={bool(scene._still()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)
        return s, ok

    def finite_all() -> bool:
        ok = True
        for b in (scene.hopper, scene.gate, scene.basket, scene.ketchup, scene.bbq):
            ok = ok and bool(torch.isfinite(b.data.root_state_w).all())
        return ok

    def place(body, pos_env, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor(pos_env, device=device).expand(n, 3) + origin
        st[:, 3:7] = torch.tensor(quat, device=device).expand(n, 4)
        body.write_root_state_to_sim(st, all_ids)

    zero_wrench = torch.zeros(n, 1, 3, device=device)

    def stage_basket() -> None:
        """The solve recipe P1: basket under the lip, hopper frame, one pose write."""
        h_pos, h_quat = scene.hopper.data.root_pos_w, scene.hopper.data.root_quat_w
        tgt = torch.tensor([c.lip_x + c.land_dx, 0.0, 0.0], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = h_pos + quat_apply(h_quat, tgt)
        st[:, 2] = 0.004
        st[:, 3:7] = h_quat
        scene.basket.write_root_state_to_sim(st, all_ids)
        step(90)

    def extract_gate(max_steps: int = 900) -> bool:
        """The solve recipe P2: velocity-capped lift force until the slab is clear of
        the rails (readback), then force off and park the free gate on open floor."""
        lift_f, v_cap, clear_z = 2.2, 0.30, 0.46
        sustain_f = 0.6 * c.gate_mass * 9.81
        best_z = float(scene._hopper_local(scene.gate.data.root_pos_w)[0, 2])
        stall = 0
        freed = False
        for _ in range(max_steps):
            vz = float(scene.gate.data.root_lin_vel_w[0, 2])
            f = torch.zeros(n, 1, 3, device=device)
            f[:, 0, 2] = lift_f if vz < v_cap else sustain_f
            scene.gate.set_external_force_and_torque(f, zero_wrench, env_ids=all_ids,
                                                     is_global=True)
            step(1)
            z = float(scene._hopper_local(scene.gate.data.root_pos_w)[0, 2])
            if z > best_z + 0.005:
                best_z, stall = z, 0
            else:
                stall += 1
                if stall >= 240:
                    lift_f = min(lift_f + 0.6, 8.0)
                    stall = 0
            if z >= clear_z:
                freed = True
                break
        scene.gate.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                 env_ids=all_ids, is_global=True)
        if freed:
            side = float(scene._side[0])
            place(scene.gate, (0.10, -side * 0.45, 0.010),
                  (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0))
        return freed

    def wait_success(max_steps: int) -> bool:
        quiet = 0
        for _ in range(max_steps):
            step(1)
            quiet = quiet + 1 if bool(scene.success()[0]) else 0
            if quiet >= 30:
                return True
        return False

    def drop_lying_into_basket(body, max_steps: int = 600) -> None:
        """Stage a bottle LYING (axis along the basket's long side) just above the
        basket cavity, off-axis from whatever is already inside; drop, settle."""
        b_pos, b_quat = scene.basket.data.root_pos_w, scene.basket.data.root_quat_w
        k_loc_x = float(scene._basket_local(scene.ketchup.data.root_pos_w)[0, 0])
        x_off = -0.033 if k_loc_x > 0.0 else 0.033
        drop = torch.tensor([x_off, 0.0, c.floor_t + c.wall_h + c.body_r + 0.010],
                            device=device).expand(n, 3)
        p_w = b_pos + quat_apply(b_quat, drop)
        qx = torch.tensor([math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0.0, 0.0],
                          device=device).expand(n, 4)  # bottle +z -> +y (lying)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = p_w
        st[:, 3:7] = quat_mul(b_quat, qx)
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
    env.reset(seed=201)
    settle(300)  # the ketchup rolls down the ramp and rests against the gate
    g_loc = scene._hopper_local(scene.gate.data.root_pos_w)[0]
    k_loc = scene._hopper_local(scene.ketchup.data.root_pos_w)[0]
    s, ok = judge("reset+settle")
    check("reset: settled, finite, gate seated, ketchup enclosed behind it, "
          "score <= 0.02, no success",
          finite_all() and not ok and s <= 0.02
          and abs(float(g_loc[2]) - c.gate_seat_z) < 0.02
          and float(k_loc[0]) < c.lip_x
          and bool(scene._still()[0]))

    # =========================== 2-3. randomization by readback =============================
    sides, yaws, hxs, bxs, b_ys, q_ys = [], [], [], [], [], []
    opposite = True
    for sd in range(8):
        env.reset(seed=sd)
        settle(5)
        h = (scene.hopper.data.root_pos_w - origin)[0]
        hq = scene.hopper.data.root_quat_w[0]
        sides.append(float(h[1]) > 0)
        yaws.append(math.degrees(2.0 * math.atan2(float(hq[3]), float(hq[0]))))
        hxs.append(float(h[0]))
        bxs.append(float(scene._hopper_local(scene.ketchup.data.root_pos_w)[0, 0]))
        by = float((scene.basket.data.root_pos_w - origin)[0, 1])
        qy = float((scene.bbq.data.root_pos_w - origin)[0, 1])
        b_ys.append(by)
        q_ys.append(qy)
        opposite = opposite and (by * float(h[1]) < 0) and (qy * float(h[1]) < 0)
    print(f"[smoke] readback 8 seeds: hopper_left={sides} "
          f"yaw={[f'{y:+.1f}' for y in yaws]} hx={[f'{x:.3f}' for x in hxs]}", flush=True)
    print(f"[smoke] bottle_ramp_x={[f'{x:+.3f}' for x in bxs]} "
          f"basket_y={[f'{y:+.2f}' for y in b_ys]} bbq_y={[f'{y:+.2f}' for y in q_ys]}",
          flush=True)
    check("randomization: hopper SIDE flips across seeds; basket and decoy always "
          "on the opposite side (readback)",
          any(sides) and not all(sides) and opposite)
    spread = (max(yaws) - min(yaws) > 3.0 and max(hxs) - min(hxs) > 0.008
              and max(bxs) - min(bxs) > 0.01
              and max(abs(a - b) for a, b in zip(b_ys, b_ys[1:])) > 0.005)
    check("randomization: continuous jitter present (hopper x/yaw, bottle ramp "
          "position, basket xy)", spread)

    # =========================== 4. null policy =============================================
    env.reset(seed=202)
    settle(300)
    s, ok = judge("null policy 2.5s")
    check("null policy: hands-off leaves score <= 0.02, no success", s <= 0.02 and not ok)

    # =========================== 5. SEED strategy rejected ==================================
    # The seed's plan: grasp the visible standing bottle, put it in the basket. The
    # only graspable floor bottle here is the brown DECOY.
    b_pos, b_quat = scene.basket.data.root_pos_w, scene.basket.data.root_quat_w
    drop = torch.tensor([0.0, 0.0, c.floor_t + c.wall_h + c.body_r + 0.010],
                        device=device).expand(n, 3)
    qx = torch.tensor([math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0.0, 0.0],
                      device=device).expand(n, 4)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = b_pos + quat_apply(b_quat, drop)
    st[:, 3:7] = quat_mul(b_quat, qx)
    scene.bbq.write_root_state_to_sim(st, all_ids)
    settle(300)
    s, ok = judge("seed: decoy in basket")
    check("seed strategy: the visible floor bottle (brown decoy) placed in the basket "
          "-> contained but score <= 0.02, not success",
          bool(scene.contained(scene.bbq)[0]) and s <= 0.02 and not ok)

    # =========================== 6. interlock: seated gate seals the hopper =================
    # Velocity-capped horizontal shove on the ketchup toward the discharge for 2.5 s.
    # The probe must actually PRESS the bottle against the gate (readback), and the
    # sealed hopper must hold: no lip crossing, no release credit, gate still seated.
    env.reset(seed=203)
    settle(300)
    h_quat = scene.hopper.data.root_quat_w
    ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
    push_dir = quat_apply(h_quat, ex)
    max_x = -1.0
    crossed = False
    for _ in range(300):
        v_along = float((scene.ketchup.data.root_lin_vel_w * push_dir).sum(dim=-1)[0])
        f = (push_dir * (3.0 if v_along < 0.25 else 0.0)).view(n, 1, 3).contiguous()
        scene.ketchup.set_external_force_and_torque(f, zero_wrench, env_ids=all_ids,
                                                    is_global=True)
        step(1)
        kx = float(scene._hopper_local(scene.ketchup.data.root_pos_w)[0, 0])
        max_x = max(max_x, kx)
        crossed = crossed or kx >= c.lip_x + c.release_dx
    scene.ketchup.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                env_ids=all_ids, is_global=True)
    settle(120)
    g_loc = scene._hopper_local(scene.gate.data.root_pos_w)[0]
    print(f"[smoke] interlock probe: max ketchup local x={max_x:+.3f} "
          f"(lip at {c.lip_x:.3f}), gate_z={float(g_loc[2]):+.3f}", flush=True)
    s, ok = judge("interlock shove")
    check("interlock: 2.5 s forced shove PRESSES the bottle against the seated gate "
          "(probe moved, readback) but it never crosses the lip; gate stays seated; "
          "no release credit",
          max_x >= 0.085 and not crossed and not bool(scene._released_ever[0])
          and abs(float(g_loc[2]) - c.gate_seat_z) < 0.02 and s <= 0.02 and not ok)

    # =========================== 7. discharge without the basket ============================
    # The REAL extraction recipe with the basket left at spawn: the bottle discharges
    # onto the open floor. Extract + release latch; no containment, no success.
    env.reset(seed=204)
    settle(300)
    freed = extract_gate()
    settle(600)
    s, ok = judge("discharge, no basket")
    check("discharge without staging: gate force-extracted, bottle lands on the open "
          "floor -> extract+release latched (0.44 <= score <= 0.46), not success",
          freed and bool(scene._extracted_ever[0]) and bool(scene._released_ever[0])
          and not bool(scene.contained(scene.ketchup)[0])
          and 0.44 <= s <= 0.46 and not ok)

    # =========================== 8. near-miss: against the OUTER wall =======================
    # Ketchup settled upright leaning on the basket's outer wall: close, not inside.
    b_pos, b_quat = scene.basket.data.root_pos_w, scene.basket.data.root_quat_w
    out = torch.tensor([c.out_x / 2 + c.body_r + 0.004, 0.0, c.body_h / 2 + 0.004],
                       device=device).expand(n, 3)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = b_pos + quat_apply(b_quat, out)
    st[:, 2] = c.body_h / 2 + 0.004  # standing on the floor beside the wall
    st[:, 3:7] = b_quat
    scene.ketchup.write_root_state_to_sim(st, all_ids)
    settle(240)
    s, ok = judge("outer-wall near-miss")
    check("near-miss: ketchup settled against the basket's OUTER wall -> not "
          "contained, not success, no new credit (score <= 0.46)",
          not bool(scene.contained(scene.ketchup)[0]) and s <= 0.46 and not ok)

    # =========================== 9. wrong place: on the hopper ROOF =========================
    env.reset(seed=205)
    settle(120)
    h_pos, h_quat = scene.hopper.data.root_pos_w, scene.hopper.data.root_quat_w
    roof = torch.tensor([-0.0125, 0.0, c.roof_z + 0.012 + c.body_r + 0.004],
                        device=device).expand(n, 3)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = h_pos + quat_apply(h_quat, roof)
    st[:, 3:7] = quat_mul(h_quat, qx)
    scene.ketchup.write_root_state_to_sim(st, all_ids)
    settle(300)
    s, ok = judge("on the roof")
    check("wrong place: ketchup resting ON the hopper roof -> no release credit, "
          "score <= 0.02, not success",
          not bool(scene._released_ever[0]) and s <= 0.02 and not ok)

    # =========================== 10. success reproduction ===================================
    env.reset(seed=206)
    settle(300)
    stage_basket()
    freed = extract_gate()
    done = wait_success(1500)
    s, ok = judge("solve recipe", expect_reject=False)
    check("success reproduction: stage basket + force-extract gate -> gravity "
          "discharge into the basket, success True, score 1.0",
          freed and done and ok and s >= 0.999)

    # =========================== 11. exclusion: decoy joins the loaded basket ===============
    drop_lying_into_basket(scene.bbq)
    s, ok = judge("decoy added to load")
    check("exclusion: brown decoy added INTO the loaded basket -> success collapses, "
          "latched credit capped (0.599 <= score <= 0.601)",
          bool(scene.contained(scene.bbq)[0]) and not ok and 0.599 <= s <= 0.601)

    # =========================== 12. latch regression =======================================
    side = float(scene._side[0])
    place(scene.ketchup, (0.12, -side * 0.40, c.body_h / 2 + 0.002))
    place(scene.bbq, (0.30, -side * 0.22, c.body_h / 2 + 0.002))
    settle(240)
    s, ok = judge("ketchup removed")
    check("latch regression: ketchup yanked out to the floor -> success collapses, "
          "latched base persists (0.599 <= score <= 0.601), never 1.0",
          not ok and not bool(scene.contained(scene.ketchup)[0])
          and 0.599 <= s <= 0.601)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() never True at any rejection-battery judged point",
          not reject_violated[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.sluice_hopper_catch")
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
