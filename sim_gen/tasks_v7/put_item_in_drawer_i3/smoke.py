"""Smoke / rubric-REJECTION battery for StopperVaultScene (sim_gen task
`put_item_in_drawer_i3`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — force-pushed doorway transit + rib-arrested
stopper seating, both through contact dynamics — is the acceptance evidence that the
rubric ACCEPTS a correct outcome). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome as a settled state and asserts the rubric
REJECTS it; no probe in this battery ever reaches success(), and a final audit check
asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: cube + both stoppers
                            standing outside the vault, everything still, score ~0;
  3-4. randomization      — READBACK over 8 seeded resets: cube/stopper side swap
                            flips, blue/red slot order flips; vault yaw + xy, cube
                            xy + yaw all vary;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed strategy       — the end state the seed's plan (open the receptacle, drop
                            the item in from above) produces on a roofed vault: the
                            cube settled ON THE ROOF -> rejected;
  7.  near-miss deposit   — cube settled in the doorway, mostly through but NOT
                            fully past the front wall -> rejected;
  8.  out-of-order / empty seal — BLUE stopper seated in the pocket (constructed,
                            verified seated) with the cube still outside -> no
                            success AND the order-gated seal term earns ~nothing;
  9.  order forced        — with the stopper seated, the cube force-pushed at the
                            doorway for 2.5 s stays OUTSIDE (the seated stopper
                            physically blocks the only entry) -> rejected;
  10. wrong stopper       — the narrow RED decoy standing in the doorway (cube and
                            blue outside) -> rejected (it can never seal);
  11. decoy-inside clause — cube inside + BLUE seated + RED inside the interior ->
                            everything else satisfied, the red-clear clause alone
                            rejects (score capped 0.85);
  12. near-miss seat      — cube inside, BLUE in the tunnel but 20 mm shy of the
                            seat depth -> rejected;
  13. misaligned seat     — cube inside, BLUE at seat depth but yawed 90 deg (slab
                            edge-on, doorway NOT covered) -> the axis clause
                            rejects;
  14. latched credit      — deposit credit earned, then the cube yanked back out ->
                            the latched score holds (credit does not evaporate),
                            still no success;
  15. rejection audit     — success() was never True at ANY judged point;
  16. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.put_item_in_drawer_i3.smoke --headless
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
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX -> the
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

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.stopper_vault")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.70, -1.05, 0.85)) + o),
                                tuple(np.array((0.42, 0.00, 0.10)) + o),
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

    ever_success = [False]

    def judge() -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def vault_pose() -> tuple[torch.Tensor, float]:
        vp = (scene.vault.data.root_pos_w - scene.env_origins)[0]
        q = scene.vault.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return vp, yaw

    def report(tag: str) -> None:
        cl = scene._local(scene.cube)[0]
        bl = scene._local(scene.blue)[0]
        rl = scene._local(scene.red)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | cube_v=({float(cl[0]):+.3f},{float(cl[1]):+.3f},"
              f"{float(cl[2]):.3f}) blue_v=({float(bl[0]):+.3f},{float(bl[1]):+.3f}) "
              f"red_v=({float(rl[0]):+.3f},{float(rl[1]):+.3f}) "
              f"inside={bool(scene._cube_inside_now()[0])} "
              f"seated={bool(scene._blue_seated_now()[0])} "
              f"red_clear={bool(scene._red_clear()[0])} "
              f"dep={bool(scene._deposited[0])} seal={float(scene._seal_max[0]):.3f} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_world(body, x: float, y: float, z: float,
                    quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def place_local(body, x_l: float, y_l: float, z: float,
                    extra_yaw_deg: float = 0.0) -> None:
        """Place `body` at a pose given in the VAULT's frame (xy) + world z, oriented
        to the vault's yaw (+ extra) — probe constructor robust to vault jitter."""
        vp, vyaw = vault_pose()
        wx = float(vp[0]) + math.cos(vyaw) * x_l - math.sin(vyaw) * y_l
        wy = float(vp[1]) + math.sin(vyaw) * x_l + math.cos(vyaw) * y_l
        half = (vyaw + math.radians(extra_yaw_deg)) / 2
        place_world(body, wx, wy, z, (math.cos(half), 0.0, 0.0, math.sin(half)))

    def seat_blue() -> None:
        """Construct the BLUE stopper seated in the pocket (2 mm outside the exact
        rib-arrest depth — inside every seat tolerance, no authored overlap)."""
        place_local(scene.blue, c.seat_x + 0.002, 0.0, c.slab_h / 2 + 0.002)

    def push_cube_at_doorway(steps: int, force: float = 4.0) -> None:
        """Drive the cube toward the doorway with a horizontal world-frame force
        along the vault's inward axis (the solve's transit push, reused as a probe)."""
        _vp, vyaw = vault_pose()
        in_dir = torch.tensor([-math.cos(vyaw), -math.sin(vyaw), 0.0], device=device)
        f_w = (in_dir * force).view(1, 1, 3).expand(n, 1, 3).contiguous()
        for _ in range(steps):
            scene.cube.set_external_force_and_torque(f_w, zero_wrench,
                                                     env_ids=all_ids, is_global=True)
            step(1)
        scene.cube.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                 env_ids=all_ids)
        step(30)

    def states_finite() -> bool:
        return bool(torch.isfinite(scene.cube.data.root_state_w).all()
                    and torch.isfinite(scene.blue.data.root_state_w).all()
                    and torch.isfinite(scene.red.data.root_state_w).all()
                    and torch.isfinite(scene.vault.data.root_state_w).all())

    def all_still() -> bool:
        return (float(scene.cube.data.root_lin_vel_w[0].norm()) < c.settle_speed
                and float(scene.blue.data.root_lin_vel_w[0].norm()) < c.settle_speed
                and float(scene.red.data.root_lin_vel_w[0].norm()) < c.settle_speed)

    cube_rest_z = c.cube_s / 2 + 0.002
    slab_rest_z = c.slab_h / 2 + 0.002

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    outside = (float(scene._local(scene.cube)[0, 0]) > c.mouth_x + 0.02
               and float(scene._local(scene.blue)[0, 0]) > c.mouth_x + 0.02
               and bool(scene._red_clear()[0]))
    check("settle: states finite, cube + both stoppers standing outside the vault, "
          "everything still", states_finite() and outside and all_still())
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(2)
        vp, vyaw = vault_pose()
        yaw_dev = math.degrees(math.atan2(math.sin(vyaw - math.pi),
                                          math.cos(vyaw - math.pi)))
        cp = (scene.cube.data.root_pos_w - scene.env_origins)[0]
        bp = (scene.blue.data.root_pos_w - scene.env_origins)[0]
        rp_ = (scene.red.data.root_pos_w - scene.env_origins)[0]
        q = scene.cube.data.root_quat_w[0]
        cyaw = math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))
        cube_side = 1.0 if float(cp[1]) > 0 else 0.0
        blue_near = 1.0 if abs(float(bp[1])) < abs(float(rp_[1])) else 0.0
        reads.append((float(vp[0]), float(vp[1]), yaw_dev, float(cp[0]), float(cp[1]),
                      cyaw, cube_side, blue_near))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (vault_x, vault_y, vault_yaw_dev_deg, "
          f"cube_x, cube_y, cube_yaw_deg, cube_side_pos_y, blue_in_near_slot):\n{arr}",
          flush=True)
    check("randomization: cube/stopper side swap flips AND blue/red slot order flips "
          "across seeded resets (readback)",
          0.0 < arr[:, 6].mean() < 1.0 and 0.0 < arr[:, 7].mean() < 1.0)
    vault_var = (float(arr[:, 2].max() - arr[:, 2].min()) > 2.0
                 and float((arr[:, 0:2].max(axis=0) - arr[:, 0:2].min(axis=0)).max()) > 0.008)
    cube_var = (float(arr[:, 3].max() - arr[:, 3].min()) > 0.01
                and float(np.abs(arr[:, 4]).max() - np.abs(arr[:, 4]).min()) > 0.01
                and float(arr[:, 5].max() - arr[:, 5].min()) > 20.0)
    check("randomization: vault yaw + xy and cube xy + yaw all vary (readback)",
          vault_var and cube_var)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: drop from above ==========================
    # The seed's plan — open the receptacle, then lower the item in from above — has
    # no purchase on a fully roofed vault: the drop lands the cube ON THE ROOF.
    torch.manual_seed(41)
    env.reset()
    step(10)
    place_local(scene.cube, 0.0, 0.0, c.roof_z + c.roof_t + c.cube_s / 2 + 0.003)
    step(80)
    report("seed-strategy")
    s, ok = judge()
    cz = float((scene.cube.data.root_pos_w - scene.env_origins)[0, 2])
    check("seed strategy: cube dropped from above settles ON THE ROOF — not inside, "
          "no success, score < 0.3",
          cz > 0.15 and not bool(scene._cube_inside_now()[0]) and not ok and s < 0.3)

    # =========================== 7. near-miss: cube not fully past the wall =================
    torch.manual_seed(51)
    env.reset()
    step(10)
    place_local(scene.cube, c.inside_x_max + 0.020, 0.0, cube_rest_z)
    step(60)
    report("near-miss-dep")
    s, ok = judge()
    check("near-miss deposit: cube settled in the doorway, mostly through but NOT "
          "fully past the front wall — not inside, no success, score <= 0.25",
          not bool(scene._cube_inside_now()[0]) and not ok and s <= 0.25)

    # =========================== 8. out-of-order / sealing an empty vault ===================
    torch.manual_seed(61)
    env.reset()
    step(10)
    seat_blue()
    step(40)
    report("empty-seal")
    s, ok = judge()
    seated = bool(scene._blue_seated_now()[0])
    check("out-of-order: BLUE stopper seated (verified) with the cube still outside — "
          "no success, order-gated seal term earns ~nothing (score <= 0.05)",
          seated and not bool(scene._cube_inside_now()[0]) and not ok and s <= 0.05)

    # =========================== 9. order physically forced =================================
    # Same state: with the stopper seated, drive the cube at the doorway for 2.5 s —
    # the seated stopper blocks the only entry; the cube must stay outside.
    place_local(scene.cube, 0.30, 0.0, cube_rest_z)
    step(20)
    push_cube_at_doorway(300)
    report("blocked-entry")
    s, ok = judge()
    cx_v = float(scene._local(scene.cube)[0, 0])
    check("order forced: cube force-pushed at the doorway for 2.5 s with the stopper "
          "seated stays OUTSIDE (blocked at the stopper face) — not inside, no success",
          cx_v > 0.150 and not bool(scene._cube_inside_now()[0])
          and bool(scene._blue_seated_now()[0]) and not ok)

    # =========================== 10. wrong stopper: the RED decoy ===========================
    torch.manual_seed(71)
    env.reset()
    step(10)
    place_local(scene.red, c.seat_x + 0.002, 0.0, slab_rest_z)
    step(40)
    report("red-in-doorway")
    s, ok = judge()
    check("wrong stopper: narrow RED decoy standing in the doorway (cube and blue "
          "outside) — no success, red-clear clause violated, score <= 0.2",
          not bool(scene._red_clear()[0]) and not bool(scene._blue_seated_now()[0])
          and not ok and s <= 0.2)

    # =========================== 11. decoy-inside clause ====================================
    # Cube inside + BLUE seated + RED inside the interior: everything else satisfied;
    # the red-clear clause ALONE rejects (this probe never reaches success).
    torch.manual_seed(81)
    env.reset()
    step(10)
    place_local(scene.cube, -0.010, -0.045, cube_rest_z)
    step(20)
    place_local(scene.red, -0.045, 0.045, slab_rest_z)
    step(20)
    seat_blue()
    step(40)
    report("decoy-inside")
    s, ok = judge()
    check("decoy-inside: cube inside + BLUE seated + RED inside the interior — the "
          "red-clear clause alone rejects, no success, score <= 0.85",
          bool(scene._cube_inside_now()[0]) and bool(scene._blue_seated_now()[0])
          and not bool(scene._red_clear()[0]) and not ok and s <= 0.85)

    # =========================== 12. near-miss seat: 20 mm shy ==============================
    torch.manual_seed(91)
    env.reset()
    step(10)
    place_local(scene.cube, -0.010, 0.0, cube_rest_z)
    step(20)
    place_local(scene.blue, c.seat_x + 0.020, 0.0, slab_rest_z)
    step(40)
    report("shy-seat")
    s, ok = judge()
    check("near-miss seat: cube inside, BLUE in the tunnel but 20 mm shy of the seat "
          "depth — seat clause rejects, no success, score <= 0.85",
          bool(scene._cube_inside_now()[0]) and not bool(scene._blue_seated_now()[0])
          and not ok and s <= 0.85)

    # =========================== 13. misaligned seat: yawed 90 deg ==========================
    torch.manual_seed(101)
    env.reset()
    step(10)
    place_local(scene.cube, -0.010, 0.0, cube_rest_z)
    step(20)
    place_local(scene.blue, c.seat_x + 0.002, 0.0, slab_rest_z, extra_yaw_deg=90.0)
    step(40)
    report("misaligned")
    s, ok = judge()
    check("misaligned seat: cube inside, BLUE at seat depth but yawed 90 deg (slab "
          "edge-on, doorway NOT covered) — axis clause rejects, no success",
          bool(scene._cube_inside_now()[0]) and not bool(scene._blue_seated_now()[0])
          and not ok)

    # =========================== 14. latched credit survives regression =====================
    torch.manual_seed(111)
    env.reset()
    step(10)
    place_local(scene.cube, -0.010, 0.0, cube_rest_z)
    step(30)
    s_a, ok_a = judge()
    place_world(scene.cube, 0.20, 0.30, cube_rest_z)  # yank it back out
    step(60)
    report("regressed")
    s_b, ok_b = judge()
    check("latched credit: deposit credit earned then the cube yanked back outside — "
          f"score holds ({s_a:.3f} -> {s_b:.3f}), >= 0.30, still no success",
          bool(scene._deposited[0]) and abs(s_a - s_b) < 1e-3 and s_a >= 0.30
          and not ok_a and not ok_b)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", states_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.stopper_vault")
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
    # Hard exit: Kit teardown hangs — watchdog then die.
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()
