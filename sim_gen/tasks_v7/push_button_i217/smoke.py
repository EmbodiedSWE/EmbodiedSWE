"""smoke — REJECTION battery for the RecoilButtonScene rubric (NullRobot, RECORDED).

This module is NOT a solution (solve.py already proves the rubric ACCEPTS the correct
outcome on two seeds). Every check here CONSTRUCTS a wrong outcome as a settled state
(teleports are instrumentation; forces go through the scene's frame-encoded probe
buffers) and asserts the rubric REJECTS it:

  1. settle/no-NaN     — reset settles finite: plunger at rest (~0 depth), pawl up on the
                         stem, not seated, score ~0;
  2. randomization     — READBACK: anvil xy/yaw and cartridge distance/bearing/yaw differ
                         across seeded resets; the trio stays assembled at every spawn;
  3. null policy       — 360 idle steps -> score ~0, no success;
  4. seed strategy     — the seed's plan (poke the red cap where the button stands): a
                         ramped 4.5 N fingertip push on the UNDOCKED cap shoves the whole
                         cartridge >= 4 cm across the floor (probe is real), compresses the
                         plunger < the 26 mm line, never drops the pawl, never succeeds;
  5. spring-back       — releasing that poke: the plunger returns out (~0 depth), nothing
                         latched;
  6. seated, no press  — the cartridge teleport-constructed SEATED in the dock: 0.45-band
                         credit only, no success;
  7. partial press     — docked press to ~15 mm then release: springs back out, pawl up,
                         score < 0.9 band, no success;
  8. latch retention   — carrying the partially-credited cartridge back OUT of the dock:
                         latched score keeps, success stays False;
  9. latched unseated  — the LATCHED state (plunger at depth, pawl in the groove)
                         constructed on open ground far from the dock: it persists
                         hands-off (the mechanism is real) but scores ~0 — seat required;
 10. wrong-way dock    — cartridge REVERSED in the pocket, cap braced on the backwall, the
                         housing pushed 12 N: this really LATCHES the button (probe is
                         real: depth + pawl verified) yet seated=False -> no success,
                         score stays in the approach band;
 11. fake latch       — teleporting ONLY the pawl down with the plunger out: depenetration
                         pops it back onto the stem; no latch, no success;
 12. no accidental success — success() never fired during the whole battery;
 13. final no-NaN     — every body finite at the end.

Records video frames -> frames.npz in the CWD.
Run (forge): python -u -m simgen_tasks.push_button_i217.smoke --headless
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--max_sec", type=float, default=1200.0)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version and silently rejects RTX -> the
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
from simgen_tasks.push_button_i217 import scene as scene_mod  # noqa: E402

G = scene_mod.G

threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                       os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.recoil_button")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids = torch.zeros(1, dtype=torch.long, device=device)

    # --- recording (viewport rgb annotator, the proven forge mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.35, -1.45, 0.95)) + o),
                                tuple(np.array((0.02, 0.0, 0.10)) + o),
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
    saw_success = False

    def step(k: int) -> None:
        nonlocal step_i, saw_success
        for _ in range(k):
            env.step(no_action)
            saw_success = saw_success or bool(scene.success()[0])
            if (annot is not None and step_i % args.record_every == 0
                    and len(frames) < args.max_frames):
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def report(tag: str) -> None:
        p = scene.cart_in_dock()[0]
        print(f"[smoke] {tag:16s} d={float(scene.depth()[0]) * 1000:6.1f}mm "
              f"pawl_dz={float(scene.pawl_dz()[0]) * 1000:6.1f}mm "
              f"dock=({p[0]:+.3f},{p[1]:+.3f}) seated={bool(scene.seated_now()[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def push_off() -> None:
        for k in scene.push_w:
            scene.push_w[k].zero_()

    def rigid_move(pos_w: torch.Tensor, quat_w: torch.Tensor, settle: int = 60) -> None:
        """Transport the TRIO by one rigid transform (cartridge root -> given pose),
        preserving the internal plunger/pawl state. Zero velocity."""
        p0 = scene.cartridge.data.root_pos_w[0].clone()
        q0 = scene.cartridge.data.root_quat_w[0].clone()
        q_t = quat_mul(quat_w.view(1, 4), quat_inv(q0.view(1, 4)))[0]
        p_t = pos_w - quat_apply(q_t.view(1, 4), p0.view(1, 3))[0]
        for body in (scene.cartridge, scene.plunger, scene.pawl):
            st = torch.zeros(1, 13, device=device)
            st[0, 0:3] = quat_apply(q_t.view(1, 4), body.data.root_pos_w[0].view(1, 3))[0] + p_t
            st[0, 3:7] = quat_mul(q_t.view(1, 4), body.data.root_quat_w[0].view(1, 4))[0]
            body.write_root_state_to_sim(st, ids)
        step(settle)

    def seat_pose() -> tuple[torch.Tensor, torch.Tensor]:
        pos = scene.seat_pos_w()[0].clone()
        pos[2] = scene.cartridge.data.root_pos_w[0, 2]
        return pos, scene.anvil.data.root_quat_w[0].clone()

    # =========================== 1. settle / no-NaN =========================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all())
              for b in (scene.anvil, scene.cartridge, scene.plunger, scene.pawl))
    check("settle: finite, plunger at rest, pawl up on the stem, not seated, score ~0",
          fin and abs(float(scene.depth()[0])) < 0.004
          and float(scene.pawl_dz()[0]) > -0.006
          and not bool(scene.seated_now()[0]) and float(scene.score()[0]) <= 0.005)

    # =========================== 2. randomization is real ===================================
    reads = []
    for s in (21, 22, 23, 24, 25, 26):
        env.reset(seed=s)
        step(30)
        ap = (scene.anvil.data.root_pos_w - scene.env_origins)[0]
        ayaw = float(scene._yaw(scene.anvil.data.root_quat_w)[0])
        cyaw = float(scene._yaw(scene.cartridge.data.root_quat_w)[0])
        dist = float((scene.cartridge.data.root_pos_w[0, :2]
                      - scene.seat_pos_w()[0, :2]).norm())
        intact = (abs(float(scene.depth()[0])) < 0.004
                  and -0.006 < float(scene.pawl_dz()[0]) < 0.001)
        reads.append((float(ap[0]), float(ap[1]), ayaw, dist, cyaw, intact))
    arr = np.array([r[:5] for r in reads])
    print(f"[smoke] randomization readback (anvil_x, anvil_y, anvil_yaw, spawn_dist, "
          f"cart_yaw):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: anvil pose, spawn distance and cartridge yaw vary (readback), "
          "spawn distance always in band, trio assembled at every spawn",
          spread[0] > 0.02 and spread[1] > 0.05 and spread[2] > 0.15
          and spread[3] > 0.03 and spread[4] > 1.0
          and all(0.36 <= r[3] <= 0.52 for r in reads) and all(r[5] for r in reads))

    # =========================== 3. null policy =============================================
    env.reset(seed=31)
    step(360)
    report("null-policy")
    check("null policy: score ~0 and no success after 360 idle steps",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 4./5. seed strategy: poke the free-standing cap ============
    env.reset(seed=41)
    step(60)
    start = scene.cartridge.data.root_pos_w[0, :2].clone()
    max_d, min_pawl = -1.0, 1.0
    for i in range(480):
        f = 4.5 * min(1.0, (i + 1) / 240.0)  # ramp to 4.5 N, then hold
        scene.push_w["plunger"][0] = -scene.cart_axis_w()[0] * f
        step(1)
        max_d = max(max_d, float(scene.depth()[0]))
        min_pawl = min(min_pawl, float(scene.pawl_dz()[0]))
    moved = float((scene.cartridge.data.root_pos_w[0, :2] - start).norm())
    report("poke-held")
    print(f"[smoke]   poke outcome: moved={moved * 1000:.0f}mm peak_d={max_d * 1000:.1f}mm "
          f"min_pawl_dz={min_pawl * 1000:.1f}mm", flush=True)
    check("seed strategy: a 4.5 N cap poke on the FREE cartridge shoves it >= 40 mm away "
          "(probe is real), peak depth < the 26 mm line, pawl never drops, no success",
          moved >= 0.040 and max_d < c.press_depth_ok and min_pawl > -c.pawl_drop_ok
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.16)
    push_off()
    step(120)
    report("poke-released")
    check("spring-back: plunger returns out after the poke, nothing latched",
          float(scene.depth()[0]) < 0.005 and float(scene.pawl_dz()[0]) > -0.006
          and not bool(scene.success()[0]))

    # =========================== 6./7./8. seated constructs =================================
    env.reset(seed=61)
    step(60)
    sp, sq = seat_pose()
    rigid_move(sp, sq, settle=240)
    report("seated-no-press")
    check("seated, no press: seat credit band only, no success",
          bool(scene.seated_now()[0]) and not bool(scene.success()[0])
          and 0.44 <= float(scene.score()[0]) <= 0.55)
    # partial press to ~15 mm, then release
    peak = 0.0
    for _ in range(400):
        axis = scene.cart_axis_w()[0]
        d = float(scene.depth()[0])
        peak = max(peak, d)
        if d >= 0.015:
            break
        vp = float(((scene.plunger.data.root_lin_vel_w[0]
                     - scene.cartridge.data.root_lin_vel_w[0]) * (-axis)).sum())
        f = max(0.0, min(16.0, c.spring_f0 + c.spring_k * d + 10.0 * (0.05 - vp)))
        scene.push_w["plunger"][0] = -axis * f
        step(1)
    push_off()
    step(120)
    report("partial-press")
    check("partial press: reached ~15 mm then sprang back out, pawl up, sub-click "
          "credit only, no success",
          0.014 <= peak < c.press_depth_ok and float(scene.depth()[0]) < 0.005
          and float(scene.pawl_dz()[0]) > -0.006 and not bool(scene.success()[0])
          and 0.44 <= float(scene.score()[0]) < 0.90)
    s_before = float(scene.score()[0])
    # carry it back OUT of the dock: latched credit keeps, success stays False
    away = scene.seat_pos_w()[0] + scene.dock_out_w()[0] * 0.40
    away[2] = scene.cartridge.data.root_pos_w[0, 2]
    rigid_move(away, sq, settle=180)
    report("carried-out")
    check("latch retention: carrying the cartridge back out keeps the latched score, "
          "success stays False",
          float(scene.score()[0]) >= s_before - 1e-4 and not bool(scene.seated_now()[0])
          and not bool(scene.success()[0]))

    # =========================== 9. latched but UNSEATED ====================================
    env.reset(seed=81)
    step(60)
    pc = scene.cartridge.data.root_pos_w[0].clone()
    qc = scene.cartridge.data.root_quat_w[0].clone()
    st = torch.zeros(1, 13, device=device)
    st[0, 0:3] = pc + quat_apply(qc.view(1, 4),
                                 torch.tensor([[-0.033, 0.0, 0.0]], device=device))[0]
    st[0, 3:7] = qc
    scene.plunger.write_root_state_to_sim(st, ids)
    st = torch.zeros(1, 13, device=device)
    st[0, 0:3] = pc + quat_apply(qc.view(1, 4),
                                 torch.tensor([[0.0, 0.0, -0.010]], device=device))[0]
    st[0, 3:7] = qc
    scene.pawl.write_root_state_to_sim(st, ids)
    step(240)
    report("latched-unseated")
    check("latched but unseated: the latch itself persists hands-off on open ground "
          "(depth >= 26 mm, pawl in the groove) yet scores ~0 — the seat is required",
          float(scene.depth()[0]) >= c.press_depth_ok
          and float(scene.pawl_dz()[0]) <= -c.pawl_drop_ok
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.02)

    # =========================== 10. wrong-way dock =========================================
    env.reset(seed=91)
    step(60)
    rq = quat_mul(scene.anvil.data.root_quat_w[0].view(1, 4),
                  torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device))[0]  # yaw + 180
    rp = scene.anvil.data.root_pos_w[0] + scene.dock_out_w()[0] * (G.CAP_X1 + 0.004)
    rp = rp.clone()
    rp[2] = scene.cartridge.data.root_pos_w[0, 2]
    rigid_move(rp, rq, settle=60)
    at_stop = 0
    for i in range(500):
        f = 12.0 * min(1.0, (i + 1) / 120.0)
        scene.push_w["cartridge"][0] = -scene.dock_out_w()[0] * f
        step(1)
        if float(scene.depth()[0]) >= G.STROKE - 0.0015:
            at_stop += 1
            if at_stop >= 10:
                break
    for _ in range(96):  # hold braced while the pawl falls
        scene.push_w["cartridge"][0] = -scene.dock_out_w()[0] * 12.0
        step(1)
    push_off()
    step(150)
    report("wrong-way")
    check("wrong-way dock: cap braced on the backwall really latches the button (probe "
          "is real: depth + dropped pawl) but the cartridge is not seated -> no "
          "success, approach-band score only",
          float(scene.depth()[0]) >= c.press_depth_ok
          and float(scene.pawl_dz()[0]) <= -c.pawl_drop_ok
          and not bool(scene.seated_now()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.16)

    # =========================== 11. fake latch (pawl teleported down) ======================
    env.reset(seed=101)
    step(60)
    pc = scene.cartridge.data.root_pos_w[0].clone()
    qc = scene.cartridge.data.root_quat_w[0].clone()
    st = torch.zeros(1, 13, device=device)
    st[0, 0:3] = pc + quat_apply(qc.view(1, 4),
                                 torch.tensor([[0.0, 0.0, -0.010]], device=device))[0]
    st[0, 3:7] = qc
    scene.pawl.write_root_state_to_sim(st, ids)
    # Probe-is-real guard: verify the teleport APPLIED by reading the physx view
    # directly, BEFORE any physics step — depenetration ejects the pawl from the
    # 9 mm interpenetration within the very first substep (that ejection is the
    # phenomenon under test, so sampling pawl_dz() after step(1) would race it).
    from isaaclab.utils.math import quat_apply_inverse
    p_read = scene.pawl.root_physx_view.get_transforms()[0, :3].to(device)
    planted_dz = float(quat_apply_inverse(qc.view(1, 4),
                                          (p_read - pc).view(1, 3))[0, 2])
    pawl_planted = planted_dz <= -c.pawl_drop_ok
    print(f"[smoke]   fake-latch planted rel z = {planted_dz * 1000:.1f}mm", flush=True)
    step(151)
    report("fake-latch")
    check("fake latch: teleporting only the pawl down (plunger out) is depenetrated "
          "back onto the stem — no latch, no success",
          pawl_planted and float(scene.pawl_dz()[0]) > -0.0065
          and abs(float(scene.depth()[0])) < 0.005 and not bool(scene.success()[0]))

    # =========================== 12./13. global =============================================
    check("no accidental success: success() never fired during the whole battery",
          not saw_success)
    fin = all(bool(torch.isfinite(b.data.root_state_w).all())
              for b in (scene.anvil, scene.cartridge, scene.plunger, scene.pawl))
    check("final: every body state finite", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.recoil_button")
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
