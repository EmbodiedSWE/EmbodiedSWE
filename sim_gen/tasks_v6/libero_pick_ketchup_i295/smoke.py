"""Smoke / rubric-REJECTION battery for PostboxDepositScene (sim_gen task
`libero_pick_ketchup_i295`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — tray transport, force push through the flap,
gravity descent — is the acceptance evidence that the rubric ACCEPTS a correct
outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial)
outcome as a settled state and asserts the rubric REJECTS it; no probe in this battery
ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: flap hanging shut, all three
                            bottles standing at their slots, everything still, score ~0;
  3-4. randomization      — READBACK over 8 seeded resets: the bottle->slot PERMUTATION
                            varies (ketchup occupies >= 2 distinct slots; the three
                            bottles always occupy three distinct slots) and per-slot xy
                            jitter is real;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed strategy       — the seed's plan (lower the bottle in from above) dies on
                            the ROOF: ketchup released over the box settles ON the
                            roof -> score ~0, no success;
  7.  near-miss outside   — ketchup lying on the tray, nose at the aperture mouth,
                            settled, flap shut: staged but NOT deposited -> no
                            success, score <= 0.25 (tray credit only);
  8.  jam half-through    — ketchup bridging the sill, nose inside, centre outside;
                            the flap rests propped on it -> no success, score <= 0.6;
  9.  flap propped open   — ketchup fully inside ON THE FLOOR but the mustard decoy
                            jammed in the aperture props the flap open -> the
                            flap-shut clause alone rejects: no success, score <= 0.85;
  10. wrong object        — the MUSTARD decoy inside the shut box, ketchup at its
                            slot -> score ~0, no success (color identification is
                            load-bearing);
  11. decoy exclusion     — ketchup AND mustard both settled inside the shut box:
                            every other gate passes, the decoy clause alone rejects
                            -> no success, score <= 0.85;
  12. latched credit      — teleporting the ketchup back OUT afterwards leaves the
                            latched score unchanged (credit does not evaporate),
                            still no success;
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

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
    env = ENVS.get("simgen.postbox_deposit")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.55, -1.05, 0.85)) + o),
                                tuple(np.array((0.45, 0.00, 0.18)) + o),
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

    def flap_deg() -> float:
        return float(scene.flap_open_deg()[0])

    def report(tag: str) -> None:
        p = scene._local(scene.ketchup)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | flap={flap_deg():6.1f} deg "
              f"ketchup_loc=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):.3f}) "
              f"k_in={bool(scene._inside_box(scene.ketchup)[0])} "
              f"m_in={bool(scene._inside_box(scene.mustard)[0])} "
              f"tray={bool(scene._tray[0])} eng={bool(scene._eng[0])} "
              f"crossed={bool(scene._crossed[0])} desc={float(scene._desc_max[0]):.3f} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += env.iscene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def place_local(body, lx: float, ly: float, lz: float, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        """Teleport `body` to depot-local coordinates (probe constructor)."""
        place(body, c.depot_pos[0] + lx, c.depot_pos[1] + ly, lz, quat)

    LIE_X = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)  # cap toward +x
    LIE_Y = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)  # cap toward -y

    def pose_flap(open_deg: float) -> None:
        """Follower-only re-pose of the flap about its unchanged hinge (probe
        constructor / the same joint-coordinate write reset uses)."""
        h = -math.radians(open_deg) / 2
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = c.depot_pos[0] + c.flap_hinge_x
        st[:, 1] = c.depot_pos[1]
        st[:, 2] = c.flap_hinge_z
        st[:, 3], st[:, 5] = math.cos(h), math.sin(h)
        st[:, 0:3] += env.iscene.env_origins
        scene.flap.write_root_state_to_sim(st, all_ids)

    def obj_xy(body) -> tuple[float, float]:
        p = (body.data.root_pos_w - scene.env_origins)[0]
        return float(p[0]), float(p[1])

    slots = np.array([c.slot_a, c.slot_b, c.slot_c])

    def nearest_slot(body) -> int:
        x, y = obj_xy(body)
        return int(np.argmin(((slots - np.array([x, y])) ** 2).sum(axis=1)))

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("show")
    fin0 = (torch.isfinite(scene.flap.data.root_state_w).all()
            and torch.isfinite(scene.ketchup.data.root_state_w).all()
            and torch.isfinite(scene.mustard.data.root_state_w).all()
            and torch.isfinite(scene.mayo.data.root_state_w).all())
    kz = float((scene.ketchup.data.root_pos_w - scene.env_origins)[0, 2])
    still = (float(scene.ketchup.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.flap.data.root_ang_vel_w[0].norm()) < c.flap_omega)
    check("settle: states finite, flap hanging shut, ketchup standing at its slot, "
          "all still, nothing inside the box",
          bool(fin0) and flap_deg() <= c.closed_tol_deg
          and abs(kz - c.ketchup_body_h / 2) < 0.01 and still
          and not bool(scene._inside_box(scene.ketchup)[0])
          and not bool(scene._inside_box(scene.mustard)[0])
          and not bool(scene._inside_box(scene.mayo)[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        kx, ky = obj_xy(scene.ketchup)
        trip = (nearest_slot(scene.ketchup), nearest_slot(scene.mustard),
                nearest_slot(scene.mayo))
        reads.append((kx, ky, *trip))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (ketchup_x, ketchup_y, k_slot, m_slot, "
          f"y_slot):\n{arr}", flush=True)
    k_slots = arr[:, 2].astype(int)
    check("randomization: the bottle->slot permutation varies across seeded resets "
          "(ketchup lands in >= 2 distinct slots) AND the three bottles always occupy "
          "three distinct slots (readback)",
          len(set(k_slots.tolist())) >= 2
          and all(len({int(r[2]), int(r[3]), int(r[4])}) == 3 for r in arr))
    jit = 0.0
    for slot_id in set(k_slots.tolist()):
        grp = arr[k_slots == slot_id]
        if len(grp) >= 2:
            jit = max(jit, float((grp[:, 0:2].max(axis=0) - grp[:, 0:2].min(axis=0)).max()))
    check("randomization: per-slot spawn jitter is real (readback spread > 4 mm within "
          "a slot group)", jit > 0.004)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy dies on the roof ==========================
    # The seed's plan — carry the bottle above the receptacle and release it — lands on
    # the CLOSED ROOF here. Construct: ketchup released standing just above the roof.
    torch.manual_seed(41)
    env.reset()
    step(30)
    roof_top = c.box_h + c.roof_t
    place_local(scene.ketchup, c.t + c.in_d / 2, 0.0,
                roof_top + c.ketchup_body_h / 2 + 0.005)
    step(150)
    report("seed-strategy")
    s, ok = judge()
    kloc = scene._local(scene.ketchup)[0]
    check("seed strategy: ketchup released from above settles ON the roof, never "
          "enters — score ~0 (<= 0.05), no success",
          float(kloc[2]) > c.box_h - 0.02
          and not bool(scene._inside_box(scene.ketchup)[0]) and not ok and s <= 0.05)

    # =========================== 7. near-miss: staged outside, flap shut ====================
    # Ketchup lying on the tray with its nose at the aperture mouth (the solve's own
    # staging pose), settled: tray credit only — the deposit clauses reject.
    torch.manual_seed(51)
    env.reset()
    step(30)
    place_local(scene.ketchup, -0.115, 0.0, c.sill_z + c.ketchup_body_r + 0.003, LIE_X)
    step(120)
    report("near-miss-out")
    s, ok = judge()
    check("near-miss: ketchup staged on the tray at the aperture mouth, flap shut, "
          "settled — NOT success, score <= 0.25 (tray credit only)",
          bool(scene._tray[0]) and flap_deg() <= c.closed_tol_deg
          and not bool(scene._inside_box(scene.ketchup)[0]) and not ok and s <= 0.25)

    # =========================== 8. jam: half-through, flap propped on the bottle ===========
    # Nose past the wall plane, centre still outside: the flap dropped onto it rests
    # propped open. Even if the flap should nudge it back out over time, the centre
    # never enters the box: NOT success either way.
    torch.manual_seed(61)
    env.reset()
    step(30)
    pose_flap(55.0)
    place_local(scene.ketchup, -0.025, 0.0, c.sill_z + c.ketchup_body_r + 0.003, LIE_X)
    step(180)
    report("jam-halfway")
    s, ok = judge()
    kx = float(scene._local(scene.ketchup)[0, 0])
    check("jam: ketchup bridging the sill (centre outside), flap resting propped on "
          f"it (flap={flap_deg():.1f} deg) — NOT success, score <= 0.6",
          kx < c.box_lo[0] and not bool(scene._inside_box(scene.ketchup)[0])
          and not ok and s <= 0.6)

    # =========================== 9. flap propped open by the decoy ==========================
    # Ketchup fully inside ON the floor; the mustard decoy jammed in the aperture props
    # the flap open. Containment passes, decoy-exclusion passes (mustard's centre is
    # outside) — the flap-shut clause alone must reject.
    torch.manual_seed(71)
    env.reset()
    step(30)
    pose_flap(55.0)
    # drop the ketchup into the interior airspace (clear of the ramp and the swept
    # flap) so it settles on the box floor without teleporting into geometry
    place_local(scene.ketchup, 0.24, 0.0, 0.22, LIE_Y)
    place_local(scene.mustard, -0.030, 0.0, c.sill_z + c.mustard_body_r + 0.003, LIE_X)
    step(240)
    report("flap-propped")
    s, ok = judge()
    check("flap propped: ketchup settled inside on the floor but the decoy jammed in "
          f"the aperture holds the flap open (flap={flap_deg():.1f} deg) — the "
          "flap-shut clause alone rejects: NOT success, score <= 0.85",
          bool(scene._inside_box(scene.ketchup)[0]) and flap_deg() > c.closed_tol_deg
          and not bool(scene._inside_box(scene.mustard)[0]) and not ok and s <= 0.85)

    # =========================== 10. wrong object: decoy deposited ==========================
    torch.manual_seed(81)
    env.reset()
    step(30)
    pose_flap(55.0)
    place_local(scene.mustard, 0.24, 0.0, 0.22, LIE_Y)
    step(240)  # decoy drops to the floor; empty aperture: the flap falls shut
    report("wrong-object")
    s, ok = judge()
    check("wrong object: MUSTARD decoy inside the shut box, ketchup at its slot — "
          "score ~0 (<= 0.05), no success (color identification is load-bearing)",
          bool(scene._inside_box(scene.mustard)[0]) and flap_deg() <= c.closed_tol_deg
          and not ok and s <= 0.05)

    # =========================== 11. decoy-exclusion clause =================================
    # Same episode: ALSO deposit the ketchup. Ketchup inside + flap shut + settled —
    # every other gate passes; the decoy clause alone must reject.
    place_local(scene.ketchup, 0.24, 0.05, 0.25, LIE_Y)
    step(240)
    report("both-inside")
    s11, ok = judge()
    k_still = float(scene.ketchup.data.root_lin_vel_w[0].norm()) < c.settle_speed
    check("decoy exclusion: ketchup AND mustard both settled inside the shut box — "
          "all other gates pass, the decoy clause alone rejects: NOT success, "
          "score <= 0.85",
          bool(scene._inside_box(scene.ketchup)[0])
          and bool(scene._inside_box(scene.mustard)[0])
          and flap_deg() <= c.closed_tol_deg and k_still and not ok and s11 <= 0.85)

    # =========================== 12. latched credit survives regression =====================
    place(scene.ketchup, 0.20, -0.35, c.ketchup_body_h / 2 + 0.002)
    step(60)
    report("regressed")
    s12, ok = judge()
    check("latched credit: teleporting the ketchup back OUT of the box leaves the "
          f"latched score unchanged ({s11:.3f} -> {s12:.3f}), still no success",
          abs(s12 - s11) < 1e-3 and not bool(scene._inside_box(scene.ketchup)[0])
          and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.flap.data.root_state_w).all()
           and torch.isfinite(scene.ketchup.data.root_state_w).all()
           and torch.isfinite(scene.mustard.data.root_state_w).all()
           and torch.isfinite(scene.mayo.data.root_state_w).all()
           and torch.isfinite(scene.depot.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.postbox_deposit")
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
    except Exception as exc:  # noqa: BLE001 — die loudly, never hang in Kit teardown
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({exc})", flush=True)
        os._exit(1)
