"""Smoke / rubric-REJECTION battery for TagHangersScene (sim_gen task
`pick_single_egad_i4`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — force-thread each tag onto its matching peg and let
it drop into real suspension — is the acceptance evidence that the rubric ACCEPTS a
correct outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong (or
partial) outcome as a settled state and asserts the rubric REJECTS it — plus applied-
force probes that prove the mechanism is physically real (a hung tag really is carried
by the peg; a lifted tag really cannot score). No probe in this battery ever reaches
success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite and still, score ~0 at rest;
  3-4. randomization      — READBACK over 6 seeded resets: the red stand's slot and
                            yaw vary (stand permutation is real); the red tag's side
                            and pose vary (side swap is real);
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's plan (lift the object into the air, hold it
                            raised): 2 N up on the red tag for 1.5 s, then release ->
                            never success, latched score stays small;
  7.  wrong color         — red tag physically HUNG on the BLUE peg (verified by
                            geometry readback) -> NOT success, no thread credit,
                            score < 0.9;
  8.  decoy stand         — blue tag physically hung on the GRAY peg -> NOT success,
                            score < 0.9;
  9.  suspension honesty  — the wrong-color hang from a fresh construction survives a
                            0.8 N lateral shove (its own weight, 0.5 s): a draped fake
                            would be swept off, a THREADED tag is retained laterally by
                            the peg through the hole — real contact, not scripting;
  10. draped on top       — red tag resting FLAT on top of its matching peg (aperture
                            axis vertical, hole NOT threaded) -> never success over
                            240 steps (the ring-below-axis clause rejects
                            resting-from-above);
  11. near-miss below     — red tag settled on the ground at the foot of its matching
                            stand -> NOT success, score < 0.9;
  12. partial             — ONE correct hang (red on red), blue on the ground -> NOT
                            success, score < 0.9;
  13. latched credit      — teleporting the hung red tag back to the ground leaves the
                            latched score unchanged (credit does not evaporate);
  14. rejection audit     — success() was never True at ANY judged point;
  15. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.pick_single_egad_i4.smoke --headless
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
    env = ENVS.get("simgen.tag_hangers")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.05, -0.95, 0.75)) + o),
                                tuple(np.array((0.18, 0.00, 0.18)) + o),
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

    def stand_pose(nm: str) -> tuple[torch.Tensor, float]:
        sp = (scene.stands[nm].data.root_pos_w - scene.env_origins)[0]
        q = scene.stands[nm].data.root_quat_w[0]
        return sp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def report(tag: str) -> None:
        s, ok = judge()
        bits = []
        for nm in scene.names:
            loc = scene._stand_local(nm, nm)[0]
            bits.append(f"{nm}=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
                        f"{float(loc[2]):.3f})h{int(bool(scene.hanging(nm)[0]))}")
        print(f"[smoke] {tag:16s} | " + " ".join(bits)
              + f" score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_tag(nm: str, x: float, y: float, z: float, yaw: float = 0.0,
                  flat: bool = False, settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        if flat:  # qz(yaw) * qy(90 deg)
            c45 = math.cos(math.pi / 4)
            st[:, 3] = math.cos(yaw / 2) * c45
            st[:, 4] = -math.sin(yaw / 2) * c45
            st[:, 5] = math.cos(yaw / 2) * c45
            st[:, 6] = math.sin(yaw / 2) * c45
        else:
            st[:, 3] = math.cos(yaw / 2)
            st[:, 6] = math.sin(yaw / 2)
        st[:, 0:3] += env.iscene.env_origins
        scene.tags[nm].write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def hang_probe(tag_nm: str, stand_nm: str, settle_steps: int = 120) -> None:
        """Construct a physically hung state: ring mid-peg, 7 mm below the axis (1 mm
        above resting contact), upright, aligned — then let contact carry it."""
        sp, yaw = stand_pose(stand_nm)
        mid_x = (c.hang_x_lo + c.hang_x_hi) / 2
        az = float(scene._peg_axis_z(torch.tensor([mid_x]))[0])
        wx = float(sp[0]) + math.cos(yaw) * mid_x
        wy = float(sp[1]) + math.sin(yaw) * mid_x
        place_tag(tag_nm, wx, wy, az - c.hang_drop + 0.001, yaw=yaw,
                  settle_steps=settle_steps)

    def apply_force_steps(nm: str, fx: float, fy: float, fz: float, k: int) -> None:
        """Apply a constant world-frame force to a tag for k steps, then clear it."""
        f_w = torch.tensor([fx, fy, fz], device=device).view(1, 1, 3).expand(n, 1, 3)
        tag = scene.tags[nm]
        for _ in range(k):
            tag.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                              env_ids=all_ids, is_global=True)
            step(1)
        tag.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all())
              for b in (*scene.tags.values(), *scene.stands.values()))
    ring_z = [float((scene.tags[nm].data.root_pos_w - scene.env_origins)[0, 2])
              for nm in scene.names]
    check("settle: all states finite, tags at rest low on the ground",
          fin and all(z < 0.05 for z in ring_z)
          and all(bool(scene.settled(nm)[0]) for nm in scene.names))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        rp, ryaw = stand_pose("red")
        tp = (scene.tags["red"].data.root_pos_w - scene.env_origins)[0]
        reads.append((float(rp[0]), float(rp[1]), ryaw, float(tp[0]), float(tp[1])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (red_stand_x, red_stand_y, red_stand_yaw, "
          f"red_tag_x, red_tag_y):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: red stand slot + yaw vary across seeded resets (readback)",
          spread[1] > 0.10 and spread[2] > 0.05)
    check("randomization: red tag side/pose varies across seeded resets (readback)",
          spread[4] > 0.10)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy: lift the object ==========================
    # The seed's whole plan is "grasp the object and raise it" (checker: z-shift
    # 7.5 cm). Do exactly that with a real force: 2 N up on the red tag for 1.5 s
    # (raises it well past 7.5 cm), then release. It must never count.
    torch.manual_seed(41)
    env.reset()
    step(10)
    apply_force_steps("red", 0.0, 0.0, 2.0, 180)
    z_peak = float((scene.tags["red"].data.root_pos_w - scene.env_origins)[0, 2])
    _s, _ok = judge()
    step(120)  # release, fall back, settle
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy (lift the tag into the air): it rose well past the seed's "
          f"7.5 cm (peak z={z_peak:.2f} m) yet latched score stays <= 0.15, never success",
          z_peak > 0.15 and s <= 0.15 and not ever_success[0])

    # =========================== 7. wrong color: red tag on the blue peg ====================
    torch.manual_seed(51)
    env.reset()
    step(10)
    hang_probe("red", "blue")
    report("wrong-color")
    geo = bool(scene._hang_geom("red", "blue", y_tol=c.hang_y_tol,
                                dz_lo=c.hang_dz_lo, dz_hi=c.hang_dz_hi)[0])
    s, ok = judge()
    check("wrong color: red tag PHYSICALLY hangs on the blue peg (geometry readback "
          "confirms suspension) yet NOT success, no thread credit, score < 0.9",
          geo and not ok and float(scene.thread_latch[0, 0]) < 0.5 and s < 0.9)

    # =========================== 8. decoy stand: blue tag on the gray peg ===================
    torch.manual_seed(61)
    env.reset()
    step(10)
    hang_probe("blue", "gray")
    report("decoy-stand")
    geo = bool(scene._hang_geom("blue", "gray", y_tol=c.hang_y_tol,
                                dz_lo=c.hang_dz_lo, dz_hi=c.hang_dz_hi)[0])
    s, ok = judge()
    check("decoy stand: blue tag physically hangs on the GRAY peg yet NOT success, "
          "score < 0.9", geo and not ok and s < 0.9)

    # =========================== 9. suspension honesty: threaded retention ==================
    # Still the wrong-color state (no success possible): shove the hung tag SIDEWAYS
    # with its own weight (0.8 N lateral, 0.5 s). A tag merely draped/leaning would be
    # swept off; a THREADED tag is retained laterally because the washer's side bar
    # catches on the peg through the hole. After release it must re-hang on the peg.
    # (A hard downward press is NOT used: pressing a light washer onto the kinematic
    # peg pumps a solver ratchet that walks it off the open tip — a sim artifact, and
    # not the invariant under test. Through-threading IS the invariant.)
    torch.manual_seed(71)
    env.reset()
    step(10)
    hang_probe("red", "blue")
    _, yaw_b = stand_pose("blue")
    loc0 = scene._stand_local("red", "blue")[0].clone()
    fx_l, fy_l = -math.sin(yaw_b) * 0.8, math.cos(yaw_b) * 0.8  # +y in blue's frame
    for _k in range(3):
        apply_force_steps("red", fx_l, fy_l, 0.0, 20)
        _l = scene._stand_local("red", "blue")[0]
        _v = scene.tags["red"].data.root_lin_vel_w[0]
        print(f"[smoke] shove diag t={(_k + 1) * 20}: loc=({float(_l[0]):+.4f},"
              f"{float(_l[1]):+.4f},{float(_l[2]):.4f}) |v|={float(_v.norm()):.3f}",
              flush=True)
    step(120)
    loc1 = scene._stand_local("red", "blue")[0]
    still_on = bool(scene._hang_geom("red", "blue", y_tol=0.013, dz_lo=-0.016,
                                     dz_hi=0.005)[0])
    report("lateral-shove")
    s, ok = judge()
    check("suspension honesty: a 0.8 N lateral shove (1x tag weight, 0.5 s) cannot "
          "detach the threaded tag — the peg through the aperture retains it and it "
          f"re-hangs (y {float(loc0[1]):+.3f}->{float(loc1[1]):+.3f})",
          still_on and not ok)

    # =========================== 10. draped on top: hole not threaded =======================
    # Rest the red tag FLAT on top of its matching peg: washer plane horizontal
    # (aperture axis vertical), handle plate cantilevered sideways into free air —
    # the hole is NOT threaded and, tumbling off sideways, the peg's side can never
    # enter the aperture. The ring center reads ABOVE the axis; whether it balances
    # or falls, it must never count. (An upright washer dropped from above would
    # horseshoe-thread the peg — a legitimate hang, not a fake — so the fake must be
    # constructed flat.)
    torch.manual_seed(81)
    env.reset()
    step(10)
    sp, yaw = stand_pose("red")
    mid_x = (c.hang_x_lo + c.hang_x_hi) / 2
    # clear the up-tilted far end of the peg under the flat washer
    az_far = float(scene._peg_axis_z(torch.tensor([mid_x + c.outer_half]))[0])
    wx = float(sp[0]) + math.cos(yaw) * mid_x
    wy = float(sp[1]) + math.sin(yaw) * mid_x
    place_tag("red", wx, wy, az_far + c.peg_w / 2 + c.washer_t / 2 + 0.002,
              yaw=yaw + math.pi / 2, flat=True, settle_steps=0)
    never = True
    for _ in range(8):
        step(30)
        _s, _ok = judge()
        never = never and not _ok and not bool(scene.hanging("red")[0])
    report("draped-on-top")
    check("draped on top (hole not threaded): ring-above-axis is rejected — never "
          "hanging, never success, wherever it ends up", never)

    # =========================== 11. near-miss: at the foot of the stand ====================
    torch.manual_seed(91)
    env.reset()
    step(10)
    sp, yaw = stand_pose("red")
    fx = float(sp[0]) + math.cos(yaw) * 0.16
    fy = float(sp[1]) + math.sin(yaw) * 0.16
    place_tag("red", fx, fy, c.plate_t / 2 + 0.004, yaw=yaw, flat=True, settle_steps=60)
    report("near-miss-foot")
    s, ok = judge()
    check("near-miss: red tag settled on the ground at the foot of its own stand — "
          "NOT success, not hanging, score < 0.9",
          not bool(scene.hanging("red")[0]) and not ok and s < 0.9)

    # =========================== 12. partial: one correct hang ==============================
    torch.manual_seed(101)
    env.reset()
    step(10)
    hang_probe("red", "red")
    report("partial-hang")
    s, ok = judge()
    one_ok = bool((scene.hanging("red") & scene.settled("red"))[0])
    check("partial: red tag correctly hung but blue still on the ground — NOT success, "
          "score < 0.9", one_ok and not ok and s < 0.9)

    # =========================== 13. latched credit survives moving back ====================
    s_hung, _ = judge()
    place_tag("red", c.tag_x, 0.0, c.plate_t / 2 + 0.004, flat=True, settle_steps=40)
    report("moved-back")
    s_back, ok = judge()
    check("latched credit: returning the hung red tag to the ground leaves the latched "
          "score unchanged", abs(s_back - s_hung) < 1e-3 and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all())
              for b in (*scene.tags.values(), *scene.stands.values()))
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.tag_hangers")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(okc for _nm, okc in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, okc in checks:
            if not okc:
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
