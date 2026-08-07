"""Smoke / rubric-rejection battery for CartFerryScene (sim_gen task
`libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i61`) — NullRobot,
teleported/forced probe states (instrumentation, NOT a solution), RECORDED.

Battery:
  1-2. settle/no-NaN    — reset settles finite: cart at the home end of the lane, bowl
                          on the ledge under the roof; score ~0, no latches, no success;
  3. randomization      — READBACK across seeds: the FIXTURE root xy+yaw move (all
                          predicates are fixture-frame), cart lane position and bowl
                          ledge position/yaw move;
  4. null-policy-fails  — 240 idle steps -> score ~0, no success, bowl still on ledge;
  5-7. oracle x3 seeds  — cart docked, bowl dropped onto the docked counter top
                          (transfer latch), cart force-pulled home with the bowl riding
                          -> success() and score 1.0; persists 240 further steps;
  8-9. monotonicity     — 0 (idle) < 0.20 (docked) < 0.50 (+transfer) < 0.65 (+ride,
                          cart paused mid-lane: near-miss home, NO success) < 1.0
                          (home = success); partials < 1.0, latched credit never drops;
  10. negative A (seed-analog) — bowl slid off the ledge with the cart NOT docked (the
                          only analog of "just take the bowl and put it there" this
                          scene admits) -> falls through the SLOT, void latch, score
                          capped, no success;
  11. negative A2 (void permanence) — after the loss, bowl set aboard + cart home ->
                          STILL no success, score stays capped (the latch is permanent);
  12. negative B (undocked transfer is physically impossible) — cart just outside the
                          forward band, bowl released over the gap -> falls into the
                          slot (nothing bridges it);
  13. negative C (near-miss aboard-x) — bowl resting on the plate's front-edge overhang,
                          outside the fully-supported band -> not aboard, no success;
  14. negative D (inverted) — bowl UPSIDE-DOWN on the docked counter top -> not aboard,
                          no success;
  15. negative E (not home) — proper dock + transfer but the cart left at the dock ->
                          no success, score exactly 0.50;
  16. calibration       — bowl drop x-offset sweep across the counter top (published
                          in/out table for the aboard band).

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i61.smoke --headless
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
# RTX recipe: kit may mis-decode the driver version and silently reject RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i61 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cart_ferry")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(1, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-1.25, -1.00, 0.85)) + o),
                                tuple(np.array((-0.10, 0.0, 0.25)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

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

    def settle_until(pred, max_steps: int = 400, poll: int = 10,
                     min_steps: int = 30) -> bool:
        """Poll `pred` while stepping. ALWAYS steps at least `min_steps` first: right
        after a teleport all velocities are zero, so settled()-style predicates are
        vacuously true before physics has run (and post_step latches never fire)."""
        step(min_steps)
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        cxy = scene.cart_fix_xy()[0]
        bf = scene._to_fix(scene.bowl.data.root_pos_w)[0]
        bc = scene._to_cart(scene.bowl.data.root_pos_w)[0]
        print(f"[smoke] {tag:16s} cart_fix=({cxy[0]:+.3f},{cxy[1]:+.3f}) "
              f"bowl_fix=({bf[0]:+.3f},{bf[1]:+.3f},{bf[2]:+.3f}) "
              f"bowl_cart=({bc[0]:+.3f},{bc[1]:+.3f},{bc[2]:+.3f}) "
              f"docked={bool(scene.cart_docked()[0])} aboard={bool(scene.bowl_on_cart()[0])} "
              f"home={bool(scene.cart_home()[0])} void={bool(scene._voided_ever[0])} "
              f"settled={bool(scene.settled()[0])} score={sc():.2f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402

    def fix_to_world(loc) -> torch.Tensor:
        p = quat_apply(scene.fixture.data.root_quat_w[0:1],
                       torch.tensor([loc], device=device))[0]
        return p + scene.fixture.data.root_pos_w[0]

    def fixq() -> tuple:
        q = scene.fixture.data.root_quat_w[0]
        return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    def tp(body, pos_w, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor([float(v) for v in pos_w], device=device)
        st[:, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        body.write_root_state_to_sim(st, all_ids)

    def v_along(body, axis: torch.Tensor) -> float:
        return float((body.data.root_lin_vel_w[0] * axis).sum())

    def clear_forces() -> None:
        scene.cart.set_external_force_and_torque(zero3, zero3)
        scene.bowl.set_external_force_and_torque(zero3, zero3)

    def drive(body, axis: torch.Tensor, fmag: float, vmax: float, done,
              max_steps: int) -> bool:
        """Pulsed push along world `axis` with the pod-dependent frame probe
        (toggle raw-world <-> body-frame encoding if progress stalls)."""
        mode = 0
        last_probe = float((body.data.root_pos_w[0] * axis).sum())
        for i in range(max_steps):
            if done():
                clear_forces()
                return True
            f = fmag if v_along(body, axis) < vmax else 0.0
            fw = f * axis
            if mode == 1:
                fw = quat_apply_inverse(body.data.root_quat_w[0:1], fw.view(1, 3))[0]
            scene.cart.set_external_force_and_torque(zero3, zero3)
            scene.bowl.set_external_force_and_torque(zero3, zero3)
            body.set_external_force_and_torque(fw.view(1, 1, 3), zero3)
            step(1)
            if (i + 1) % 40 == 0:
                cur = float((body.data.root_pos_w[0] * axis).sum())
                if cur - last_probe < 0.001:
                    mode ^= 1
                last_probe = cur
        clear_forces()
        return done()

    def lane_axis() -> torch.Tensor:
        return quat_apply(scene.fixture.data.root_quat_w[0:1],
                          torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]

    def dock_cart() -> None:
        """Teleport the cart into the dock seat (probe state, 3 mm off the stop)."""
        tp(scene.cart, fix_to_world([-0.238, 0.0, c.cart_root_z + 0.002]), fixq())
        settle_until(lambda: bool(scene.cart_docked()[0]) and bool(scene.settled()[0]),
                     max_steps=200)

    def drop_bowl_fix(x_fix: float, y_fix: float = 0.0, z_fix: float = 0.417,
                      inverted: bool = False) -> None:
        """Release the bowl just above the counter-top height at a fixture-frame xy."""
        q = (0.0, 1.0, 0.0, 0.0) if inverted else (1.0, 0.0, 0.0, 0.0)
        tp(scene.bowl, fix_to_world([x_fix, y_fix, z_fix]), q)
        settle_until(lambda: bool(scene.settled()[0]), max_steps=300)

    def pull_home(stop_x: float) -> bool:
        return drive(scene.cart, -lane_axis(), 20.0, 0.08,
                     lambda: float(scene.cart_fix_xy()[0, 0]) < stop_x, 2400)

    def bowl_fix():
        return scene._to_fix(scene.bowl.data.root_pos_w)[0]

    # =========================== 1-2. settle / no-NaN ============================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("settled")
    st0 = torch.cat([scene.fixture.data.root_state_w, scene.cart.data.root_state_w,
                     scene.bowl.data.root_state_w], dim=-1)
    bf = bowl_fix()
    cxy = scene.cart_fix_xy()[0]
    check("settle: states finite, cart at the home end of the lane, bowl on the ledge "
          "(fixture frame), everything settled",
          bool(torch.isfinite(st0).all()) and float(cxy[0]) < c.home_x
          and 0.15 < float(bf[0]) < 0.25 and abs(float(bf[1])) < 0.05
          and 0.40 < float(bf[2]) < 0.43 and bool(scene.settled()[0]))
    check("settle: score ~0 at reset, no success, no latches",
          sc() <= 0.005 and not bool(scene.success()[0])
          and not bool(scene._docked_ever[0]) and not bool(scene._transferred_ever[0])
          and not bool(scene._ride_ever[0]) and not bool(scene._home_ever[0])
          and not bool(scene._voided_ever[0]))

    # =========================== 3. randomization is real ========================================
    reads = []
    for s in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(s)
        env.reset()
        step(4)
        fp = (scene.fixture.data.root_pos_w[0] - scene.env_origins[0])
        fq = scene.fixture.data.root_quat_w[0]
        fyaw = 2.0 * math.atan2(float(fq[3]), float(fq[0]))
        bl = bowl_fix()
        ex = quat_apply(scene.bowl.data.root_quat_w[0:1],
                        torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        byaw = math.atan2(float(ex[1]), float(ex[0]))
        reads.append((float(fp[0]), float(fp[1]), fyaw,
                      float(scene.cart_fix_xy()[0, 0]), float(bl[0]), byaw))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (fix_x, fix_y, fix_yaw, cart_x, bowl_x, "
          f"bowl_yaw):\n{np.round(arr, 3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: fixture root xy+yaw, cart lane position, bowl ledge position "
          "and yaw all move across seeds (READBACK from sim)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.05
          and spread[3] > 0.015 and spread[4] > 0.015 and spread[5] > 0.5)

    # =========================== 4. null policy fails ============================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: score ~0, no success, bowl still on the ledge after 240 idle steps",
          sc() <= 0.02 and not bool(scene.success()[0])
          and 0.40 < float(bowl_fix()[2]) < 0.43)

    # =========================== 5-7. oracle on 3 seeds ==========================================
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        step(30)
        dock_cart()
        drop_bowl_fix(-0.10)
        report(f"oracle{s}-transfer")
        okp = pull_home(c.home_x - 0.02)
        ok = okp and settle_until(lambda: bool(scene.success()[0]), max_steps=300)
        step(240)  # persistence: success must not flicker off
        report(f"oracle{s}-final")
        check(f"oracle seed {s}: dock -> transfer -> ferry home = success, score 1.0, "
              f"persists 240 steps",
              ok and bool(scene.success()[0]) and sc() == 1.0)

    # =========================== 8-9. rubric monotonicity ========================================
    torch.manual_seed(41)
    env.reset()
    step(30)
    s0 = sc()
    dock_cart()
    s_dock = sc()
    report("mono-docked")
    drop_bowl_fix(-0.10)
    s_tr = sc()
    report("mono-transfer")
    pull_home(-0.35)  # pause mid-lane: past the ride latch, short of home
    settle_until(lambda: bool(scene.settled()[0]), max_steps=200)
    s_ride = sc()
    near_home_no_success = not bool(scene.success()[0]) and not bool(scene.cart_home()[0])
    report("mono-ride")
    pull_home(c.home_x - 0.02)
    settle_until(lambda: bool(scene.success()[0]), max_steps=300)
    s_home = sc()
    report("mono-final")
    print(f"[smoke] monotonicity ladder: idle={s0:.3f} docked={s_dock:.3f} "
          f"transfer={s_tr:.3f} ride={s_ride:.3f} home={s_home:.3f}", flush=True)
    check("monotonicity: idle < docked (0.20) < +transfer (0.50) < +ride (0.65, "
          "mid-lane = near-miss home, NO success) < home (1.0); latches never drop",
          s0 <= 0.02 and 0.18 <= s_dock <= 0.22 and 0.48 <= s_tr <= 0.52
          and 0.63 <= s_ride <= 0.67 and near_home_no_success and s_home == 1.0
          and s0 < s_dock < s_tr < s_ride < s_home)
    check("monotonicity: partial states score < 1.0", max(s0, s_dock, s_tr, s_ride) < 1.0)

    # =========================== 10. negative A: undocked slide-off ==============================
    # The only analog of the seed's "just move the bowl there" this scene admits: slide
    # the bowl out of the alcove with the cart still at home. Nothing bridges the slot.
    torch.manual_seed(51)
    env.reset()
    step(30)
    tp(scene.bowl, fix_to_world([0.0, 0.0, 0.413]))  # over the slot, cart at home
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    report("undocked-loss")
    check("negative A (seed-analog): bowl slid off the ledge with the cart undocked "
          "falls through the SLOT -> void latch, score capped <= 0.20, no success",
          bool(scene.bowl_in_slot()[0]) and bool(scene._voided_ever[0])
          and not bool(scene.success()[0]) and sc() <= 0.20)

    # =========================== 11. negative A2: void is permanent ==============================
    dock_cart()
    tp(scene.cart, fix_to_world([-0.45, 0.0, c.cart_root_z + 0.002]), fixq())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=200)
    drop_bowl_fix(-0.30)  # aboard the (home) cart
    step(120)
    report("void-permanent")
    check("negative A2 (void permanence): after the loss, bowl set aboard + cart home "
          "-> STILL no success, score stays capped <= 0.20",
          not bool(scene.success()[0]) and sc() <= 0.20 + 1e-4  # float32 0.2 epsilon
          and bool(scene._voided_ever[0]))

    # =========================== 12. negative B: undocked transfer is impossible =================
    torch.manual_seed(61)
    env.reset()
    step(30)
    # Cart mid-lane, well outside the forward band: plate front edge at fixture x
    # ~ -0.115, ledge front at +0.055 -> a 17 cm open gap over the slot. The bowl
    # (10.4 cm) released at x=-0.02 clears both edges and free-falls into the slot.
    tp(scene.cart, fix_to_world([-0.38, 0.0, c.cart_root_z + 0.002]), fixq())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=200)
    fwd = bool(scene.cart_forward()[0])
    tp(scene.bowl, fix_to_world([-0.02, 0.0, 0.413]))  # released over the gap
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    report("gap-drop")
    check("negative B (undocked transfer impossible): cart outside the forward "
          "band leaves the gap open -> bowl falls into the slot, no aboard/transfer",
          not fwd and bool(scene.bowl_in_slot()[0]) and bool(scene._voided_ever[0])
          and not bool(scene.bowl_on_cart()[0]) and not bool(scene._transferred_ever[0])
          and not bool(scene.success()[0]))

    # =========================== 13. negative C: front-edge overhang =============================
    torch.manual_seed(71)
    env.reset()
    step(30)
    dock_cart()
    drop_bowl_fix(0.005)  # rests on the plate's front-edge overhang (cart x ~ +0.24)
    step(60)
    bc = scene._to_cart(scene.bowl.data.root_pos_w)[0]
    report("edge-overhang")
    check("negative C (near-miss aboard-x): bowl resting on the plate's front-edge "
          "overhang, outside the fully-supported band -> not aboard, no success",
          float(bc[0]) > c.aboard_x_hi and not bool(scene.bowl_on_cart()[0])
          and not bool(scene._transferred_ever[0]) and not bool(scene.success()[0]))

    # =========================== 14. negative D: inverted ========================================
    torch.manual_seed(81)
    env.reset()
    step(30)
    dock_cart()
    drop_bowl_fix(-0.10, inverted=True)
    step(60)
    report("inverted")
    check("negative D (inverted): bowl UPSIDE-DOWN on the docked counter top -> not "
          "aboard, no transfer credit, no success",
          not bool(scene.bowl_upright()[0]) and not bool(scene.bowl_on_cart()[0])
          and not bool(scene._transferred_ever[0]) and not bool(scene.success()[0]))

    # =========================== 15. negative E: transfer done, cart not home ====================
    torch.manual_seed(91)
    env.reset()
    step(30)
    dock_cart()
    drop_bowl_fix(-0.10)
    step(120)
    report("not-home")
    check("negative E (not home): proper dock + transfer but the cart left at the dock "
          "-> no success, score exactly 0.50",
          bool(scene.bowl_on_cart()[0]) and bool(scene._transferred_ever[0])
          and not bool(scene.cart_home()[0]) and not bool(scene.success()[0])
          and 0.48 <= sc() <= 0.52)

    # =========================== 16. calibration =================================================
    print("[smoke] CALIBRATION: bowl drop x (fixture frame, cart docked) -> aboard "
          "(rest x in the CART frame)", flush=True)
    cal = []
    for x_off in (-0.16, -0.10, 0.005, -0.30):
        torch.manual_seed(101)
        env.reset()
        step(20)
        dock_cart()
        drop_bowl_fix(x_off)
        step(120)
        bc = scene._to_cart(scene.bowl.data.root_pos_w)[0]
        cal.append((x_off, float(bc[0]), bool(scene.bowl_on_cart()[0])))
        print(f"[smoke]   drop_x={x_off * 1000:+5.0f}mm rest_cart_x="
              f"{float(bc[0]) * 1000:+5.0f}mm aboard={cal[-1][2]}", flush=True)
    check("calibration: bowl resting inside the fully-supported counter-top band counts "
          "aboard; front-edge overhang and rear handle-side margin do not",
          cal[0][2] and cal[1][2] and not cal[2][2] and not cal[3][2])

    # =========================== save + verdict ==================================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.cart_ferry")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, ok in checks:
            if not ok:
                print(f"[smoke]   FAILED: {nm}", flush=True)
    code = 0 if all_ok else 1
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()
