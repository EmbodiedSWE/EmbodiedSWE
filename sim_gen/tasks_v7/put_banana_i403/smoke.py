"""Smoke / rubric-REJECTION battery for FragilePackScene (sim_gen task `put_banana_i403`)
— NullRobot, teleported/dropped probe states, RECORDED.

This is NOT a solution (solve.py — cradle install + compliant tray catch — is the
acceptance evidence that the rubric ACCEPTS a correct outcome). Every probe here
CONSTRUCTS a wrong (or partial) outcome — as a settled state or as an honest physical
drop — and asserts the rubric REJECTS it; no probe reaches success(), and a final audit
asserts exactly that.

IMPORTANT ORDERING: a snapped core weld is PERMANENT for the process (PhysX joint breaks
do not heal on env.reset()), so every probe that needs an intact orb (1-9) runs BEFORE
the two break probes (10-12).

  1-2. settle/no-NaN      — reset layout settles finite: orb intact (core concentric),
                            cradle + orb loose on open ground, nothing seated/packed;
                            score ~0 at rest, no success;
  3-4. randomization      — READBACK over 8 seeded resets: crate xy + yaw really move,
                            cradle xy + yaw and orb xy really vary, and the orb is
                            INTACT in every draw;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  suspension control  — the cradle still on OPEN GROUND: the orb dropped 4 cm onto
                            its sprung tray is CAUGHT INTACT and the suspension visibly
                            compresses (asserted — the shock absorber is real, and the
                            break in #10 is the floor's fault, not the drop's) — but the
                            cradle was never installed: NOT success, score ~0;
  7.  no-cushion delivery — the orb lowered GENTLY onto the bare crate floor (zero
                            release height — it survives) with the cradle left outside:
                            intact, inside the crate, but not on the tray and nothing
                            seated: NOT success, score ~0 — care without the cradle
                            earns nothing;
  8.  beside the tray     — cradle seated in the crate, intact orb settled on the crate
                            floor BESIDE it: seated credit only, NOT success, score<0.9;
  9.  latched credit      — yanking the cradle back out of the crate afterwards leaves
                            the latched score unchanged (credit does not evaporate),
                            still no success;
  10. seed strategy       — the seed's aerial plan executed literally: carry the orb
                            over the crate wall and RELEASE it — it free-falls onto the
                            bare crate floor and the core weld SNAPS (separation
                            asserted from ~0 to >= tolerance): permanently broken,
                            score ~0, no success;
  11. end-state-identical — (flagship) the broken orb placed on the tray of a freshly
                            seated cradle: the GEOMETRIC terminal conjunction holds
                            (seated_now AND on_tray_now both True) yet NOT success —
                            only the break history distinguishes it;
  12. foul freeze         — in that fully-assembled state nothing was ever latched
                            post-break: score stays ~0;
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.put_banana_i403.smoke --headless
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

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.fragile_pack")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.35, -1.00, 0.90)) + o),
                                tuple(np.array((0.45, 0.00, 0.05)) + o),
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
        oo = (scene.orb.data.root_pos_w - scene.env_origins)[0]
        k = (scene.cradle.data.root_pos_w - scene.env_origins)[0]
        kl = scene._local(scene.crate, scene.cradle.data.root_pos_w)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | orb=({float(oo[0]):+.3f},{float(oo[1]):+.3f},"
              f"{float(oo[2]):.3f}) cradle=({float(k[0]):+.3f},{float(k[1]):+.3f},"
              f"{float(k[2]):.3f}) cradle_loc=({float(kl[0]):+.3f},{float(kl[1]):+.3f}) "
              f"core_off={float(scene.core_offset()[0]) * 1000:.1f}mm "
              f"intact={bool(scene.intact_now()[0])} "
              f"seated={bool(scene.cradle_seated_now()[0])} "
              f"on_tray={bool(scene.orb_on_tray_now()[0])} "
              f"L(seat={bool(scene._seated[0])},app={float(scene._app_max[0]):.3f},"
              f"pack={bool(scene._packed[0])},brk={bool(scene._broken[0])}) "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def crate_local_to_world(loc) -> torch.Tensor:
        p = torch.tensor(loc, device=device).expand(n, 3)
        return scene.crate.data.root_pos_w + quat_apply(scene.crate.data.root_quat_w, p)

    def place_pair(x_w: torch.Tensor, y_w: torch.Tensor, z: float) -> None:
        """Teleport the INTACT orb+core pair together (the weld sees zero relative
        motion), zero velocity (probe constructor)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = x_w
        st[:, 1] = y_w
        st[:, 2] = z
        st[:, 3] = 1.0
        scene.orb.write_root_state_to_sim(st.clone(), all_ids)
        scene.core.write_root_state_to_sim(st, all_ids)

    def place_shell(x_w: torch.Tensor, y_w: torch.Tensor, z: float) -> None:
        """Teleport the shell ALONE (post-break probes; the loose core stays behind)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = x_w
        st[:, 1] = y_w
        st[:, 2] = z
        st[:, 3] = 1.0
        scene.orb.write_root_state_to_sim(st, all_ids)

    def seat_cradle(off_ly: float = 0.0) -> None:
        """Teleport the cradle+tray (jointed pair, written together, joint at rest) to
        the crate floor, crate-aligned, offset off_ly along crate-local y."""
        cq = scene.crate.data.root_quat_w.clone()
        p = crate_local_to_world((0.0, off_ly, 0.0))
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = p[:, :2]
        st[:, 2] = 0.002
        st[:, 3:7] = cq
        scene.cradle.write_root_state_to_sim(st, all_ids)
        loc = torch.tensor([c.tray_rest_lx, 0.0, c.tray_rest_lz], device=device)
        off = quat_apply(cq, loc.expand(n, 3))
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = p[:, :2] + off[:, :2]
        st[:, 2] = 0.002 + c.tray_rest_lz
        st[:, 3:7] = cq
        scene.tray.write_root_state_to_sim(st, all_ids)

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = all(torch.isfinite(b.data.root_state_w).all()
               for b in (scene.crate, scene.cradle, scene.tray, scene.orb, scene.core))
    still = (float(scene.orb.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.cradle.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, orb INTACT (core concentric), cradle + orb loose on "
          "open ground (nothing seated, nothing on the tray), everything still",
          bool(fin0) and bool(scene.intact_now()[0])
          and not bool(scene.cradle_seated_now()[0])
          and not bool(scene.orb_on_tray_now()[0]) and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    intact_all = True
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(30)
        cr = (scene.crate.data.root_pos_w - scene.env_origins)[0]
        cq = scene.crate.data.root_quat_w[0]
        cyaw = math.degrees(2.0 * math.atan2(float(cq[3]), float(cq[0])))
        k = (scene.cradle.data.root_pos_w - scene.env_origins)[0]
        kq = scene.cradle.data.root_quat_w[0]
        kyaw = math.degrees(2.0 * math.atan2(float(kq[3]), float(kq[0])))
        oo = (scene.orb.data.root_pos_w - scene.env_origins)[0]
        intact_all = intact_all and bool(scene.intact_now()[0])
        reads.append((float(cr[0]), float(cr[1]), cyaw, float(k[0]), float(k[1]), kyaw,
                      float(oo[0]), float(oo[1])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (crate_x, crate_y, crate_yaw, cradle_x, "
          f"cradle_y, cradle_yaw, orb_x, orb_y):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: crate xy jitter AND crate yaw really move across seeded "
          "resets (readback)",
          spread[0] > 0.015 and spread[1] > 0.015 and spread[2] > 40.0)
    check("randomization: cradle xy + yaw AND orb xy really vary (readback), and the "
          "orb is INTACT in every draw",
          spread[3] > 0.015 and spread[4] > 0.015 and spread[5] > 40.0
          and spread[6] > 0.02 and spread[7] > 0.02 and intact_all)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. suspension control (cradle NOT installed) ===============
    # The cradle still sits on OPEN GROUND. Drop the intact orb from 4 cm above its
    # riding height onto the sprung tray: the suspension must visibly compress
    # (asserted — the shock absorber is real, not decoration) and the orb must survive.
    # But the cradle was never installed in the crate, so this earns ~nothing: the
    # assembled-outside near-miss and the fragility CONTROL in one probe (the break in
    # #10 is the hard floor's fault, not the drop's).
    torch.manual_seed(41)
    env.reset()
    step(30)
    tray_rest_z = float(scene.tray.data.root_pos_w[0, 2])
    ride_z = tray_rest_z + c.orb_tray_lz
    kp = scene.tray.data.root_pos_w
    place_pair(kp[:, 0], kp[:, 1], ride_z + 0.04)
    min_tray = tray_rest_z
    for _ in range(90):
        step(1)
        min_tray = min(min_tray, float(scene.tray.data.root_pos_w[0, 2]))
    step(60)
    report("cushion-control")
    s, ok = judge()
    compress = tray_rest_z - min_tray
    print(f"[smoke] cushion control: tray compressed {compress * 1000:.1f} mm "
          f"(rest z {tray_rest_z:.3f} -> min {min_tray:.3f})", flush=True)
    check("suspension control: orb dropped 4 cm onto the sprung tray is CAUGHT INTACT "
          "and the suspension visibly compressed (>= 8 mm) — but the cradle was never "
          "installed: NOT success, score ~0 (<= 0.02)",
          bool(scene.intact_now()[0]) and compress >= 0.008
          and not bool(scene.cradle_seated_now()[0]) and s <= 0.02 and not ok)

    # =========================== 7. gentle no-cushion delivery ==============================
    # A careful solver COULD lower the orb intact onto the bare crate floor (zero
    # release height survives — fragility is physics, not a proximity alarm). The
    # rubric must still reject it: no cradle, nothing on the tray, nothing earned.
    torch.manual_seed(51)
    env.reset()
    step(10)
    p = crate_local_to_world((0.0, 0.0, 0.0))
    place_pair(p[:, 0], p[:, 1], c.orb_rest_z + 0.002)
    step(90)
    report("gentle-floor")
    s, ok = judge()
    ol = scene._local(scene.crate, scene.orb.data.root_pos_w)[0]
    in_crate = (abs(float(ol[0])) < c.crate_hx - c.orb_r
                and abs(float(ol[1])) < c.crate_hy - c.orb_r
                and float(ol[2]) < c.crate_wall_h)
    check("no-cushion delivery: orb set down GENTLY on the bare crate floor stays "
          "INTACT (zero-height set-down survives) but the cradle is outside — NOT "
          "success, score ~0 (<= 0.02)",
          bool(scene.intact_now()[0]) and in_crate and s <= 0.02 and not ok)

    # =========================== 8. beside the tray (cradle seated) =========================
    # Cradle seated in the crate (constructed), intact orb settled on the crate floor
    # in the free strip BESIDE the slab: seated credit latches, the pack term must not.
    torch.manual_seed(61)
    env.reset()
    step(10)
    seat_cradle(off_ly=-0.02)
    step(60)  # seated latches
    p = crate_local_to_world((0.0, 0.072, 0.0))
    place_pair(p[:, 0], p[:, 1], c.orb_rest_z + 0.002)
    step(60)
    report("beside-tray")
    s8, ok = judge()
    check("beside the tray: cradle seated + intact orb settled on the crate floor "
          "next to it — seated credit only, NOT success, score < 0.9",
          bool(scene.cradle_seated_now()[0]) and bool(scene.intact_now()[0])
          and not bool(scene.orb_on_tray_now()[0]) and bool(scene._seated[0])
          and not bool(scene._packed[0]) and not ok and s8 < 0.9)

    # =========================== 9. latched credit survives regression ======================
    # (same episode) yank the cradle back OUT of the crate to open ground: the latched
    # seated/approach credit must not evaporate, and the state is of course no success.
    cq = scene.crate.data.root_quat_w.clone()
    away = crate_local_to_world((0.0, -0.35, 0.0))
    st = torch.zeros(n, 13, device=device)
    st[:, 0:2] = away[:, :2]
    st[:, 2] = 0.002
    st[:, 3:7] = cq
    scene.cradle.write_root_state_to_sim(st, all_ids)
    loc = torch.tensor([c.tray_rest_lx, 0.0, c.tray_rest_lz], device=device)
    off = quat_apply(cq, loc.expand(n, 3))
    st = torch.zeros(n, 13, device=device)
    st[:, 0:2] = away[:, :2] + off[:, :2]
    st[:, 2] = 0.002 + c.tray_rest_lz
    st[:, 3:7] = cq
    scene.tray.write_root_state_to_sim(st, all_ids)
    step(60)
    report("regressed")
    s9, ok = judge()
    check("latched credit: yanking the cradle back out of the crate leaves the latched "
          f"score unchanged ({s8:.3f} -> {s9:.3f}), still no success",
          abs(s9 - s8) < 1e-3 and not bool(scene.cradle_seated_now()[0]) and not ok)

    # ======= BREAK PROBES LAST: a snapped weld is permanent for the process =================
    # =========================== 10. seed strategy: over-the-wall release ===================
    # The seed's entire plan executed literally: carry the payload over the container
    # wall and RELEASE it. It free-falls ~13 cm onto the bare crate floor; the shell
    # stops in ~a solver step and the core's inertia SNAPS the brittle weld. Separation
    # is asserted from ~0 to >= tolerance — the sensor demonstrably tripped.
    torch.manual_seed(71)
    env.reset()
    step(10)
    off0 = float(scene.core_offset()[0])
    p = crate_local_to_world((0.0, 0.0, 0.0))
    place_pair(p[:, 0], p[:, 1], 0.15)  # just above the 8 cm wall, then hands off
    step(150)
    report("seed-drop")
    s, ok = judge()
    off1 = float(scene.core_offset()[0])
    print(f"[smoke] seed drop: core offset {off0 * 1000:.1f} mm -> {off1 * 1000:.1f} mm "
          f"(break tol {c.break_tol * 1000:.0f} mm)", flush=True)
    check("seed strategy: the orb released over the crate wall free-falls onto the "
          "bare floor and the core weld SNAPS (separation asserted) — permanently "
          "broken, score ~0 (<= 0.02), no success",
          off0 < 0.004 and off1 >= c.break_tol and bool(scene._broken[0])
          and not bool(scene.intact_now()[0]) and s <= 0.02 and not ok)

    # =========================== 11-12. end-state-identical + foul freeze ===================
    # (same episode — the orb is broken now) Seat the cradle and park the broken shell
    # exactly at its riding pose on the tray: the GEOMETRIC terminal conjunction holds
    # (seated_now AND on_tray_now both True) — only the break history distinguishes
    # this from success. The rubric must reject it AND must have latched nothing.
    seat_cradle()
    step(60)
    tray_z = float(scene.tray.data.root_pos_w[0, 2])
    kp = scene.tray.data.root_pos_w
    place_shell(kp[:, 0], kp[:, 1], tray_z + c.orb_tray_lz + 0.002)
    step(90)
    report("broken-assembled")
    s11, ok = judge()
    geom = bool(scene.cradle_seated_now()[0]) and bool(scene.orb_on_tray_now()[0])
    check("end-state-identical (flagship): broken orb on the tray of the seated cradle "
          "— the geometric terminal conjunction HOLDS (asserted) yet NOT success: only "
          "the break history distinguishes it",
          geom and not bool(scene.intact_now()[0]) and not ok)
    check("foul freeze: nothing latched after the break — score stays ~0 (<= 0.02) in "
          "the fully-assembled state",
          not bool(scene._seated[0]) and not bool(scene._packed[0]) and s11 <= 0.02)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(torch.isfinite(b.data.root_state_w).all()
              for b in (scene.crate, scene.cradle, scene.tray, scene.orb, scene.core))
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.fragile_pack")
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
