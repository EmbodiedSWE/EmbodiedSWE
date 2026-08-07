"""Smoke / rubric-REJECTION battery for KebabSpitScene (sim_gen task
`meat_off_grill_i64`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — the cooked group force-conveyed along the rail and
off the tip into the tray — is the acceptance evidence that the rubric ACCEPTS a
correct outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong (or
partial) outcome as a settled state and asserts the rubric REJECTS it; no probe in
this battery ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN     — reset layout settles finite: every present piece threaded
                           on the rail, still, score ~0, no success;
  3-4. randomization     — READBACK over 8 seeded resets: cooked/raw counts vary,
                           fixture yaw spread and tip-standoff jitter are real;
  5.  null policy        — 240 idle steps -> score ~0, no success;
  6.  captivity probe    — the outermost cooked piece hauled straight UP at ~6x its
                           weight, then sideways, stays THREADED (the rail through
                           its channel is the interlock; there is no lift-off exit);
  7.  seed strategy      — the end state the SEED's plan produces (lift the meat off
                           the grill, set it beside): cooked pieces placed settled on
                           the ground beside the grill bed -> score ~0, no success
                           (only the tray counts, and the real rail never lets a
                           piece off sideways — see check 6);
  8.  overshoot          — the outermost RAW piece in the tray along with every
                           cooked piece (the butt-end push outcome) -> raw-on-rail
                           clause alone rejects: NOT success, score <= 0.75;
  9.  near-miss tray     — one cooked piece settled on the ground BESIDE the tray
                           (missed the drop), the rest inside -> NOT success;
  10. near-miss tip      — every cooked piece conveyed to just short of the tip but
                           still ON the rail -> NOT success, score <= 0.20 (advance
                           credit only; off/tray credit needs real exits);
  11. partial            — one cooked piece in the tray, the boundary cooked piece
                           still threaded -> NOT success, partial credit visible;
  12. latched credit     — teleporting the served piece back OUT afterwards leaves
                           the latched score unchanged, still no success;
  13. wrong object       — an ABSENT (parked) piece constructed into the tray while
                           the present pieces are all correct -> the absent-piece
                           clause alone rejects: NOT success;
  14. rejection audit    — success() was never True at ANY judged point;
  15. final no-NaN       — all task-object states finite at the end.

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
    env = ENVS.get("simgen.kebab_spit")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.35, -0.55, 0.75)) + o),
                                tuple(np.array((0.42, 0.05, 0.13)) + o),
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

    def counts() -> tuple[int, int]:
        return int(scene.n_cooked[0]), int(scene.n_raw[0])

    def report(tag: str) -> None:
        nc, nr = counts()
        cx = ", ".join(f"{float(scene._fix_local(scene.cooked[i])[0, 0]):+.3f}"
                       for i in range(nc))
        tray = [bool(scene._in_tray(scene.cooked[i])[0]) for i in range(nc)]
        rail_r = [bool(scene._on_rail(scene.raw[i])[0]) for i in range(nr)]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | nc={nc} nr={nr} cooked_x=[{cx}] in_tray={tray} "
              f"raw_on_rail={rail_r} score={s:.3f} success={ok} frames={len(frames)}",
              flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def fix_quat() -> torch.Tensor:
        return scene.spit.data.root_quat_w

    def place_local(body, x: float, y: float, z: float, threaded: bool = False) -> None:
        """Teleport `body` to a fixture-local point of the fixture's CURRENT pose
        (probe constructor). `threaded=True` aligns the channel with the rail."""
        loc = torch.tensor([x, y, z], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.spit.data.root_pos_w + quat_apply(fix_quat(), loc)
        if threaded:
            st[:, 3:7] = fix_quat()
        else:
            st[:, 3] = 1.0
        body.write_root_state_to_sim(st, all_ids)

    def place_depot(body, slot: int) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = c.park_pos[0] + 0.12 * slot
        st[:, 1] = c.park_pos[1] + 0.30
        st[:, 2] = c.piece_s / 2 + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def fix_yaw() -> float:
        q = scene.spit.data.root_quat_w[0]
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    def all_bodies():
        return [scene.spit] + scene.cooked + scene.raw

    def finite_all() -> bool:
        return bool(all(torch.isfinite(b.data.root_state_w).all() for b in all_bodies()))

    def drop_into_tray(bodies) -> None:
        """Probe constructor: release pieces one by one over the tray and settle."""
        for j, body in enumerate(bodies):
            place_local(body, 0.27 + 0.05 * (j % 3), -0.04 + 0.045 * (j // 3), 0.20)
            step(90)
        step(120)

    z_rest = c.rail_z + c.chan_half - c.rail_a / 2 + 0.001

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("show")
    nc, nr = counts()
    threaded = all(bool(scene._on_rail(b)[0]) for b in scene.cooked[:nc] + scene.raw[:nr])
    still = all(float(b.data.root_lin_vel_w[0].norm()) < c.settle_speed
                for b in scene.cooked[:nc] + scene.raw[:nr])
    check("settle: states finite, every present piece threaded on the rail, all still",
          finite_all() and threaded and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        nc, nr = counts()
        outer_x = float(scene._fix_local(scene.cooked[0])[0, 0])
        reads.append((nc, nr, fix_yaw(), c.tip_x - outer_x - c.piece_s / 2))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (n_cooked, n_raw, yaw_deg, standoff_m):\n"
          f"{arr}", flush=True)
    combos = {(int(r[0]), int(r[1])) for r in reads}
    check("randomization: cooked/raw counts vary across seeded resets (readback)",
          len(combos) >= 2)
    yaw_spread = float(arr[:, 2].max() - arr[:, 2].min())
    stand_spread = float(arr[:, 3].max() - arr[:, 3].min())
    check("randomization: fixture yaw spread (> 4 deg) and tip-standoff jitter "
          "(> 6 mm) are real (readback)", yaw_spread > 4.0 and stand_spread > 0.006)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. captivity probe =========================================
    # The scene's core claim: a threaded piece has NO exit except past the tip. Haul
    # the outermost cooked piece straight UP at ~6x its weight for 1.5 s, then drag it
    # sideways — it must stay threaded (and thus the seed's lift-off plan is dead).
    torch.manual_seed(31)
    env.reset()
    step(30)
    probe = scene.cooked[0]
    up = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
    side = quat_apply(fix_quat(), torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3))
    for _ in range(180):
        probe.set_external_force_and_torque((up * 3.0).view(n, 1, 3).contiguous(),
                                            zero_wrench, env_ids=all_ids, is_global=True)
        env.step(no_action)
    for _ in range(120):
        probe.set_external_force_and_torque((side * 2.0).view(n, 1, 3).contiguous(),
                                            zero_wrench, env_ids=all_ids, is_global=True)
        env.step(no_action)
    probe.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    step(120)
    report("captivity")
    s, ok = judge()
    check("captivity: outermost cooked piece hauled UP at ~6x weight then sideways "
          "STAYS THREADED on the rail (no lift-off exit exists), no success",
          bool(scene._on_rail(probe)[0]) and not ok)

    # =========================== 7. seed strategy: lift off, set beside =====================
    # The seed's plan — grasp the meat, lift it off the grill, set it down beside —
    # constructed as its end state: cooked pieces settled on the ground beside the
    # grill bed. Only the tray counts; score stays ~0.
    torch.manual_seed(41)
    env.reset()
    step(30)
    nc, nr = counts()
    for i in range(nc):
        place_local(scene.cooked[i], -0.10 + 0.08 * i, 0.22, c.piece_s / 2 + 0.003)
    step(180)
    report("seed-strategy")
    s, ok = judge()
    beside = all(not bool(scene._in_tray(scene.cooked[i])[0])
                 and not bool(scene._on_rail(scene.cooked[i])[0]) for i in range(nc))
    check("seed strategy: cooked pieces 'lifted off and set beside' the grill on the "
          "ground — score ~0 (<= 0.05), no success (only the tray counts)",
          beside and not ok and s <= 0.05)

    # =========================== 8. overshoot: raw piece served too ==========================
    # The butt-end push outcome: every cooked piece AND the outermost raw piece end
    # in the tray. Raw-on-rail clause alone must reject. Raw goes FIRST so success is
    # never True at any judged instant.
    torch.manual_seed(51)
    env.reset()
    step(30)
    nc, nr = counts()
    drop_into_tray([scene.raw[0]] + [scene.cooked[i] for i in range(nc)])
    report("overshoot")
    s8, ok = judge()
    cooked_in = all(bool(scene._in_tray(scene.cooked[i])[0]) for i in range(nc))
    check("overshoot: every cooked piece AND a raw piece settled in the tray (the "
          "butt-end push outcome) — raw-on-rail clause alone rejects: NOT success, "
          "score <= 0.75",
          cooked_in and bool(scene._in_tray(scene.raw[0])[0]) and not ok and s8 <= 0.75)

    # =========================== 9. near-miss: missed the tray ===============================
    torch.manual_seed(61)
    env.reset()
    step(30)
    nc, nr = counts()
    if nc > 1:
        drop_into_tray([scene.cooked[i] for i in range(1, nc)])
    place_local(scene.cooked[0], 0.31, c.tray_hw + 0.08, c.piece_s / 2 + 0.003)
    step(150)
    report("missed-tray")
    s, ok = judge()
    check("near-miss tray: one cooked piece settled on the ground BESIDE the tray "
          "(missed the drop), the rest inside — NOT success",
          not bool(scene._in_tray(scene.cooked[0])[0]) and not ok)

    # =========================== 10. near-miss: conveyed short of the tip ====================
    torch.manual_seed(71)
    env.reset()
    step(30)
    nc, nr = counts()
    for i in range(nc):  # re-thread the cooked group just short of the tip
        place_local(scene.cooked[i], c.tip_x - 0.045 - (c.piece_s + 0.006) * i, 0.0,
                    z_rest, threaded=True)
    step(150)
    report("short-of-tip")
    s, ok = judge()
    on_rail_all = all(bool(scene._on_rail(scene.cooked[i])[0]) for i in range(nc))
    check("near-miss tip: cooked group conveyed to just short of the tip but still ON "
          "the rail — NOT success, score <= 0.20 (advance credit only; off/tray "
          "credit needs real exits)", on_rail_all and not ok and s <= 0.20)

    # =========================== 11-12. partial + latched credit ============================
    # Find a seed with n_cooked >= 2 (and an absent cooked body for check 13).
    sd_two = None
    for sd in range(81, 141):
        torch.manual_seed(sd)
        env.reset()
        if counts()[0] == 2:
            sd_two = sd
            break
    print(f"[smoke] n_cooked=2 seed hunt -> {sd_two}", flush=True)
    step(30)
    nc, nr = counts()
    drop_into_tray([scene.cooked[0]])
    report("partial")
    s11, ok = judge()
    check("partial: one cooked piece served, the boundary cooked piece still "
          "threaded — NOT success, partial credit visible (0.10 <= score <= 0.75)",
          nc == 2 and bool(scene._in_tray(scene.cooked[0])[0])
          and bool(scene._on_rail(scene.cooked[1])[0]) and not ok
          and 0.10 <= s11 <= 0.75)

    place_depot(scene.cooked[0], 6)
    step(60)
    report("regressed")
    s12, ok = judge()
    check("latched credit: teleporting the served piece back OUT of the tray leaves "
          f"the latched score unchanged ({s11:.3f} -> {s12:.3f}), still no success",
          abs(s12 - s11) < 1e-3 and not ok)

    # =========================== 13. wrong object: absent piece in the tray ==================
    # Same seed (n_cooked=2 -> cooked_2 is ABSENT). The absent piece goes into the
    # tray FIRST, then the present pieces are made all-correct: the absent-piece
    # clause alone rejects, and success is never True at any judged instant.
    torch.manual_seed(sd_two)
    env.reset()
    step(30)
    nc, nr = counts()
    drop_into_tray([scene.cooked[2]])  # the ABSENT body
    drop_into_tray([scene.cooked[i] for i in range(nc)])
    report("absent-in-tray")
    s, ok = judge()
    present_ok = (all(bool(scene._in_tray(scene.cooked[i])[0]) for i in range(nc))
                  and all(bool(scene._on_rail(scene.raw[i])[0]) for i in range(nr)))
    check("wrong object: an ABSENT (parked) piece constructed into the tray while "
          "every present piece is correct — the absent-piece clause alone rejects: "
          "NOT success",
          nc == 2 and bool(scene._in_tray(scene.cooked[2])[0]) and present_ok and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.kebab_spit")
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
