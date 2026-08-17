"""Smoke / rubric-REJECTION battery for BananaKilnScene (sim_gen task
`track_banana_i240`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — lay the banana into the open channel, ram it
through the tunnel with the pusher, withdraw — is the acceptance evidence that the
rubric ACCEPTS a correct outcome). Every probe here CONSTRUCTS a wrong (or partial)
settled outcome and asserts the rubric REJECTS it; no probe in this battery ever
reaches success(), and a final audit check asserts exactly that.

  1.  settle/no-NaN       — banana loose at its staging spot, rammer withdrawn in
                            the dock, kiln settled; states finite; authored masses
                            read back (custom spawners apply no cfg schemas);
  2.  reset worthless     — score ~0, no success — and the rammer STARTS retracted,
                            so the retraction conjunct alone can never earn credit;
  3-4. randomization      — READBACK over 8 seeded resets: kiln xy+yaw vary within
                            the declared band; banana staging xy and rammer start
                            depth jitter are real;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  carry exclusion     — the SEED strategy (transport a held banana to the goal)
                            is impossible: a banana pressed DOWN onto the chamber
                            with 6 N rides the ROOF (measured support height — the
                            probe is not vacuous) and never reads inside;
  7.  flick dies on grit  — a 2 m/s shuffleboard flick down the channel (ballistic
                            delivery, the no-tool shortcut) is stopped by the gritty
                            floor well short of the sill; the chamber latch never
                            sets;
  8.  tool drivable       — positive control of the mechanism: 10 N on the rammer
                            advances it >= 15 cm up the channel (measured), yet
                            rammer motion alone earns NOTHING (score stays ~0);
  9.  tunnel near-miss    — a banana settled in the tunnel 6 cm short of the sill:
                            partial latch only (0.25), no success;
  10. perch / z-gate      — with the rammer parked in the tunnel: a banana held at
                            head-perch height over the chamber floor never reads
                            inside (z bound), while the same xy ON the floor does
                            (positive control — success stays blocked);
  11. no-retraction       — banana genuinely inside + rammer still in the tunnel:
                            every other conjunct True, success False;
  12. latched credit      — removing that banana to the open floor does not
                            evaporate the latched chamber credit;
  13. settle gate         — the success configuration with the banana still sliding
                            at ~0.45 m/s is refused at that instant (deconstructed
                            before it can settle into a genuine success);
  14. rejection audit     — success() never True at ANY judged point; final no-NaN;
                            frames.npz saved.

Run (forge): python -u -m simgen_tasks.track_banana_i240.smoke --headless
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
import traceback

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
    env = ENVS.get("simgen.banana_kiln")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.45, -0.95, 0.85)) + o),
                                tuple(np.array((0.25, 0.12, 0.06)) + o),
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

    def report(tag: str) -> None:
        b = scene.banana_local()[0]
        r = scene.rammer_local()[0]
        s, ok = judge()
        print(f"[smoke] {tag:14s} | ban_local=({float(b[0]):+.3f},{float(b[1]):+.3f},"
              f"{float(b[2]):.3f}) ram_lx={float(r[0]):+.3f} "
              f"fed={bool(scene._fed_ever[0])} tun={bool(scene._tun_ever[0])} "
              f"cham={bool(scene._cham_ever[0])} in_cham={bool(scene.inside_chamber()[0])} "
              f"retr={bool(scene.retracted()[0])} still={bool(scene._still()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def kq() -> torch.Tensor:
        return scene.kiln.data.root_quat_w

    def k2w(lx: float, ly: float, lz: float) -> torch.Tensor:
        """Kiln-local -> world (N, 3), live kiln pose (env origins included)."""
        lp = torch.tensor([lx, ly, lz], device=device).expand(n, 3)
        return scene.kiln.data.root_pos_w + quat_apply(kq(), lp)

    def place_local(body, lx: float, ly: float, lz: float, vel_along: float = 0.0,
                    yaw90: bool = False) -> None:
        """Teleport a body to a kiln-local pose (kiln-aligned, optional +90 deg yaw),
        with an optional initial velocity along the kiln's local +x."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = k2w(lx, ly, lz)
        if yaw90:
            q90 = torch.zeros(n, 4, device=device)
            q90[:, 0] = q90[:, 3] = 0.7071068
            qb = q90
            w1, x1, y1, z1 = kq().unbind(-1)
            w2, x2, y2, z2 = qb.unbind(-1)
            st[:, 3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            st[:, 4] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            st[:, 5] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            st[:, 6] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        else:
            st[:, 3:7] = kq()
        if vel_along != 0.0:
            ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
            st[:, 7:10] = quat_apply(kq(), ex) * vel_along
        body.write_root_state_to_sim(st, all_ids)

    def place_world(body, x: float, y: float, z: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 0:3] += env.iscene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    # =========================== 1-2. settle / no-NaN / reset worthless =====================
    torch.manual_seed(11)
    env.reset()
    step(60)
    report("reset")
    fin0 = bool(torch.isfinite(scene.kiln.data.root_state_w).all()
                and torch.isfinite(scene.rammer.data.root_state_w).all()
                and torch.isfinite(scene.banana.data.root_state_w).all())
    m_kiln = float(env.iscene["kiln"].root_physx_view.get_masses()[0].sum())
    m_ram = float(env.iscene["rammer"].root_physx_view.get_masses()[0].sum())
    m_ban = float(env.iscene["banana"].root_physx_view.get_masses()[0].sum())
    print(f"[smoke] mass readback: kiln={m_kiln:.2f} rammer={m_ram:.3f} "
          f"banana={m_ban:.3f}", flush=True)
    b = scene.banana_local()[0]
    check("settle: states finite, authored masses read back (kiln ~30 kg, rammer "
          "~0.4 kg, banana ~0.12 kg), banana loose on the open floor, everything "
          "still",
          fin0 and abs(m_kiln - c.kiln_mass) < 0.5 and abs(m_ram - c.ram_mass) < 0.05
          and abs(m_ban - c.ban_mass) < 0.02 and float(b[2]) < 0.03
          and bool(scene._still()[0]))
    s, ok = judge()
    check("reset worthless: score ~0, no success — and the rammer STARTS retracted, "
          "so the retraction conjunct alone can never earn credit",
          s <= 0.02 and not ok and bool(scene.retracted()[0]))

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(2)
        kx, ky = float(scene._kiln_xy[0, 0]), float(scene._kiln_xy[0, 1])
        kyaw = math.degrees(float(scene._kiln_yaw[0]))
        bl = scene.banana_local()[0]
        rl = scene.rammer_local()[0]
        reads.append((kx, ky, kyaw, float(bl[0]), float(bl[1]), float(rl[0])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (kiln_x, kiln_y, kiln_yaw_deg, "
          f"ban_lx, ban_ly, ram_lx):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: kiln xy and yaw vary across seeded resets (readback: yaw "
          "spread > 8 deg within the declared band, xy spread > 1 cm)",
          spread[2] > 8.0 and (spread[0] > 0.01 or spread[1] > 0.01)
          and np.abs(arr[:, 2]).max() <= c.kiln_yaw_deg + 1.0)
    check("randomization: banana staging xy jitter (> 2 cm spread, kiln-local "
          "readback) and rammer start-depth jitter (> 8 mm spread) are real",
          (spread[3] > 0.02 or spread[4] > 0.02) and spread[5] > 0.008)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. carry exclusion (the seed strategy) =====================
    # The seed transports a HELD banana along free-space waypoints to the goal. Here
    # the goal region is roofed: press a banana down onto the chamber with 6 N (a
    # firm carried-object approach from above) — it rides the ROOF, never inside.
    torch.manual_seed(41)
    env.reset()
    step(20)
    roof_top = c.floor_t + c.cham_h + c.wall_t  # kiln-local roof top
    place_local(scene.banana, (c.tun_x1 + c.cham_x1) / 2, 0.0, roof_top + 0.06)
    never_in = True
    zs = []
    f = torch.zeros(n, 1, 3, device=device)
    f[:, 0, 2] = -6.0
    scene.banana.set_external_force_and_torque(f, zero3, env_ids=all_ids, is_global=True)
    for _ in range(90):
        env.step(no_action)
        never_in = never_in and not bool(scene.inside_chamber()[0])
        zs.append(float(scene.banana_local()[0, 2]))
    scene.banana.set_external_force_and_torque(zero3, zero3, env_ids=all_ids)
    zi = float(scene.banana_local()[0, 2])
    print(f"[smoke] roof press: settled kiln-local z={zi:.3f} "
          f"(roof top {roof_top:.3f}), min z={min(zs):.3f}", flush=True)
    report("roof-press")
    s, ok = judge()
    check("carry exclusion: a banana pressed DOWN onto the chamber with 6 N rides "
          f"the ROOF (supported at z~{zi:.3f} >= roof top — measured, not vacuous) "
          "and never reads inside; score stays ~0",
          never_in and min(zs) > roof_top - 0.005 and abs(zi - roof_top - c.seg_h / 2) < 0.02
          and s <= 0.02 and not ok)

    # =========================== 7. ballistic flick dies on the grit ========================
    torch.manual_seed(51)
    env.reset()
    step(20)
    place_local(scene.banana, c.dock_x0 + 0.03, 0.0, c.ban_rest_z + 0.002,
                vel_along=2.0, yaw90=True)
    far_reach, v_launch = -1.0, 0.0
    for i in range(150):
        env.step(no_action)
        far_reach = max(far_reach, float(scene.banana_local()[0, 0]))
        if i == 0:  # non-vacuity: the launch really happened (readback, not intent)
            v_launch = float(scene.banana.data.root_lin_vel_w[0].norm())
    report("flick")
    s, ok = judge()
    check("flick dies on grit: a 2 m/s shuffleboard launch from the channel start "
          f"(launch speed readback {v_launch:.2f} m/s after 1 step — the probe is real) "
          f"tumbles to a stop at kiln-local x={far_reach:+.3f} (sill at {c.sill_x:+.3f}) — "
          "the chamber latch never sets, no success",
          v_launch > 0.8 and far_reach < c.sill_x - 0.05
          and not bool(scene._cham_ever[0]) and not ok)

    # =========================== 8. tool drivable, motion alone worthless ===================
    torch.manual_seed(61)
    env.reset()
    step(20)
    r0 = float(scene.rammer_local()[0, 0])
    ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
    for _ in range(300):
        u = quat_apply(kq(), ex)[0]
        v = float(scene.rammer.data.root_lin_vel_w[0, 0:2].norm())
        f = torch.zeros(n, 1, 3, device=device)
        if v < 0.25:  # speed-capped: no coast slam into the far end
            f[:, 0, 0:2] = 10.0 * u[0:2]
        scene.rammer.set_external_force_and_torque(f, zero3, env_ids=all_ids,
                                                   is_global=True)
        env.step(no_action)
        if float(scene.rammer_local()[0, 0]) > -0.05:
            break
    scene.rammer.set_external_force_and_torque(zero3, zero3, env_ids=all_ids)
    step(60)
    r1 = float(scene.rammer_local()[0, 0])
    report("tool-drive")
    s, ok = judge()
    check("tool drivable (positive control): 10 N on the rammer advances it "
          f"{(r1 - r0) * 1000:.0f} mm up the channel (>= 150), yet rammer motion "
          "alone earns NOTHING (banana untouched, score ~0)",
          r1 - r0 >= 0.15 and not bool(scene.retracted()[0]) and s <= 0.02 and not ok)

    # =========================== 9. tunnel near-miss ========================================
    torch.manual_seed(71)
    env.reset()
    step(10)
    place_local(scene.banana, c.sill_x - 0.06, 0.0, c.ban_rest_z + 0.002, yaw90=True)
    step(60)
    report("near-miss")
    s9, ok = judge()
    check("tunnel near-miss: a banana settled in the tunnel 6 cm short of the sill "
          "latches only the tunnel stage (score 0.25), no success",
          bool(scene.in_tunnel()[0]) and not bool(scene.inside_chamber()[0])
          and not ok and 0.24 <= s9 <= 0.26)

    # =========================== 10-11. perch z-gate + retraction conjunct ==================
    torch.manual_seed(81)
    env.reset()
    step(10)
    # park the rammer IN the tunnel first: success stays structurally blocked while
    # this episode constructs in-chamber states
    place_local(scene.rammer, 0.0, 0.0, c.head_z + 0.002)
    step(20)
    assert not bool(scene.retracted()[0]), "rammer parking failed"
    cx = (c.sill_x + c.cham_x1) / 2
    perch_never = True
    for _ in range(25):
        place_local(scene.banana, cx, 0.0, c.floor_t + c.head_h + c.seg_h / 2, yaw90=True)
        env.step(no_action)
        perch_never = perch_never and not bool(scene.inside_chamber()[0])
    # positive control: the same xy ON the chamber floor reads inside
    place_local(scene.banana, cx, 0.0, c.ban_rest_z + 0.002, yaw90=True)
    step(60)
    report("in-cham")
    inside_reads = bool(scene.inside_chamber()[0])
    s10, ok10 = judge()
    check("perch / z-gate: a banana at rammer-head-perch height over the chamber "
          "floor is NEVER inside (z bound), while the same xy ON the floor IS "
          "(positive control)", perch_never and inside_reads)
    conj = {"inside": inside_reads, "still": bool(scene._still()[0]),
            "retracted": bool(scene.retracted()[0])}
    print(f"[smoke] no-retraction conjuncts: {conj}", flush=True)
    check("no-retraction: banana genuinely inside + everything still, but the "
          "rammer head still in the tunnel — success False, chamber latch only "
          "(score 0.25)",
          conj["inside"] and conj["still"] and not conj["retracted"] and not ok10
          and 0.24 <= s10 <= 0.26)

    # =========================== 12. latched credit survives regression =====================
    place_world(scene.banana, -0.10, -0.25, c.seg_h / 2 + 0.003)
    step(50)
    report("regressed")
    s12, ok = judge()
    check("latched credit: removing the banana from the chamber to the open floor "
          f"does not evaporate the latched credit ({s10:.3f} -> {s12:.3f}), never "
          "success", s12 >= s10 - 1e-3 and not ok)

    # =========================== 13. settle gate ============================================
    torch.manual_seed(91)
    env.reset()
    step(10)
    # rammer stays retracted (reset pose): the ONLY blocked conjunct is stillness
    place_local(scene.banana, c.sill_x + 0.02, 0.0, c.ban_rest_z + 0.002,
                vel_along=0.45, yaw90=True)
    env.step(no_action)
    sp = float(scene.banana.data.root_lin_vel_w[0].norm())
    in_ch = bool(scene.inside_chamber()[0])
    s, ok = judge()
    gate = in_ch and sp > 0.15 and not ok
    # deconstruct BEFORE it can settle into a genuine success
    place_world(scene.banana, -0.10, -0.25, c.seg_h / 2 + 0.003)
    step(30)
    check("settle gate: the success configuration with the banana still sliding at "
          f"{sp:.2f} m/s inside the chamber is refused at that instant", gate)

    # =========================== 14. audit + no-NaN =========================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = bool(torch.isfinite(scene.kiln.data.root_state_w).all()
               and torch.isfinite(scene.rammer.data.root_state_w).all()
               and torch.isfinite(scene.banana.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.banana_kiln")
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
    try:
        main()
    except BaseException:  # noqa: BLE001 — Kit teardown hangs on exception; die loudly
        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        threading.Timer(10.0, lambda: os._exit(2)).start()
        os._exit(2)
