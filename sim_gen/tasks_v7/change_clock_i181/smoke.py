"""Smoke / rubric-REJECTION battery for CardClockScene (sim_gen task
`change_clock_i181`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — old card pulled out of the display slot by a
velocity-limited contact force, dropped into the tray, matching card pulled from
the rack and pressed down through the funnel until the slot seats it — is the
acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport
here is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a
settled state and asserts the rubric REJECTS it; no probe in this battery ever
reaches success(), and a final audit check asserts exactly that.

  1.  settle/no-NaN     — reset layout settles finite: displayed card seated in
                          the display slot, the other three upright in the rack,
                          all still, score ~0, no success;
  2.  randomization     — READBACK over 8 seeded resets: whole-console yaw + xy
                          offset are real;
  3.  randomization     — READBACK over the same resets: WHICH card is displayed
                          varies, WHICH hour is the target varies (never the
                          displayed one), the target hour's tile really sits on
                          the green indicator each time, and the rack
                          permutation varies;
  4.  null policy       — 240 idle steps -> score ~0, no success;
  5.  seed strategy     — the seed's plan (ROTATE the clock's state) executed
                          here: the displayed card spun 180 deg in place stays
                          the same card — rotation cannot change the pip count:
                          score ~0, no success (the exchange is unavoidable);
  6.  wrong card        — a DISTRACTOR card (neither displayed nor target)
                          constructed seated in the display slot (old card moved
                          aside): it physically seats but earns NO seating
                          credit and no success — identity is load-bearing;
  7.  near-miss seat    — target card lying FLAT across the slot mouth instead
                          of seated: z + upright gates reject;
  8.  inverted card     — target card seated but FLIPPED (pips-down, hidden in
                          the slot): the pips-up gate rejects;
  9.  discard missing   — target card properly seated but the old card dumped on
                          the base plate instead of the tray: score caps at
                          extract+seat (~0.60), NO success;
  10. latched credit    — from state 9, the seated target teleported back out:
                          latched credit survives (still ~0.60), still no
                          success (non-success cap holds);
  11. wrong discard     — a DISTRACTOR dropped into the tray while the old card
                          is still displayed: no tray credit (identity again);
  12. tray near-miss    — old card resting against the tray's OUTER wall:
                          beside the tray is not inside it — no tray credit;
  13. rejection audit   — success() was never True at ANY judged point;
  14. final no-NaN      — all task-object states finite at the end.

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
# RTX recipe: kit mis-decodes some driver versions and silently rejects RTX ->
# the annotator returns EMPTY frames. Disable the driver check.
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)",
                                       flush=True), os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.card_clock")().build(num_envs=args.num_envs,
                                               device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.95, -0.85, 0.75)) + o),
                                tuple(np.array((0.00, 0.02, 0.08)) + o),
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

    ever_success = [False]

    def judge() -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def d_t() -> tuple[int, int]:
        return int(scene.displayed_idx[0]), int(scene.target_idx[0])

    def report(tag: str) -> None:
        loc, up_z, _still = scene._card_tensors()
        d, t = d_t()
        dl, tl = loc[0, d], loc[0, t]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | old(card_{c.hours[d]})=({float(dl[0]):+.3f},"
              f"{float(dl[1]):+.3f},{float(dl[2]):.3f}) "
              f"target(card_{c.hours[t]})=({float(tl[0]):+.3f},{float(tl[1]):+.3f},"
              f"{float(tl[2]):.3f}) up_z={float(up_z[0, t]):+.2f} "
              f"seated_now={bool(scene._gather(scene._seated_display(), scene.target_idx)[0])} "
              f"in_tray_now={bool(scene._gather(scene._in_tray(), scene.displayed_idx)[0])} "
              f"latches=({bool(scene._extracted_ever[0])},{bool(scene._tray_ever[0])},"
              f"{bool(scene._seated_ever[0])}) score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def loc_of(body) -> torch.Tensor:
        rel = body.data.root_pos_w - scene.console.data.root_pos_w
        return quat_apply_inverse(scene.console.data.root_quat_w, rel)

    def place_local(body, x: float, y: float, z: float, quat=None) -> None:
        """Teleport `body` to a console-local pose of the console's CURRENT pose
        (probe constructor: builds wrong/partial outcomes directly)."""
        s_pos = scene.console.data.root_pos_w
        s_quat = scene.console.data.root_quat_w
        loc = torch.tensor([x, y, z], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = s_pos + quat_apply(s_quat, loc)
        if quat is not None:
            eq = torch.tensor(quat, device=device).expand(n, 4)
            st[:, 3:7] = quat_mul(s_quat, eq)
        else:
            st[:, 3:7] = s_quat
        body.write_root_state_to_sim(st, all_ids)

    def finite_all() -> bool:
        ok = bool(torch.isfinite(scene.console.data.root_state_w).all())
        for b in scene.cards + scene.tiles:
            ok = ok and bool(torch.isfinite(b.data.root_state_w).all())
        return ok

    def console_pose() -> tuple[float, float, float]:
        p = (scene.console.data.root_pos_w - scene.env_origins)[0]
        q = scene.console.data.root_quat_w[0]
        return (float(p[0]), float(p[1]),
                math.degrees(2.0 * math.atan2(float(q[3]), float(q[0]))))

    Q_FLAT = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)  # +90 about x
    Q_FLIP = (0.0, 1.0, 0.0, 0.0)  # 180 about x: pips-down
    Q_SPIN = (0.0, 0.0, 0.0, 1.0)  # 180 about z: spun in place

    def park_old_on_ground(d: int) -> None:
        """Move the displayed card OUT of the slot to the ground beside the
        console (lying flat) — frees the one-card slot for a probe insert."""
        place_local(scene.cards[d], -0.55, -0.30, c.card_t / 2 + 0.002,
                    quat=Q_FLAT)

    # =========================== 1. settle / no-NaN =========================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("show")
    d, t = d_t()
    loc, up_z, still = scene._card_tensors()
    disp_ok = bool(scene._gather(scene._seated_display(), scene.displayed_idx)[0])
    rack_ok = True
    for i in range(4):
        if i == d:
            continue
        li = loc[0, i]
        rack_ok = rack_ok and abs(float(li[1]) - c.rack_y) < 0.02 \
            and abs(float(li[2]) - c.rack_seat_z) < 0.01 \
            and float(up_z[0, i]) > 0.95
    s, ok = judge()
    check("settle: states finite, displayed card seated in the display slot, other "
          "three upright in their rack slots, all still, score ~0 (<= 0.02), no "
          "success",
          finite_all() and disp_ok and rack_ok and bool(still[0].all())
          and s <= 0.02 and not ok)

    # =========================== 2-3. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        cx, cy, cyaw = console_pose()
        d, t = d_t()
        # which card occupies rack slot 0 (local x ~ rack_xs[0], y ~ rack_y)
        loc, _up, _still = scene._card_tensors()
        r0 = -1
        for i in range(4):
            if abs(float(loc[0, i, 0]) - c.rack_xs[0]) < 0.02 \
                    and abs(float(loc[0, i, 1]) - c.rack_y) < 0.02:
                r0 = i
        # the target hour's tile must sit on the green indicator; others parked
        tl = loc_of(scene.tiles[t])[0]
        tile_on_easel = abs(float(tl[0]) - c.easel_x) < 0.02 \
            and abs(float(tl[1]) - c.easel_y) < 0.02
        others_parked = all(float(loc_of(scene.tiles[i])[0][0]) > 0.5
                            for i in range(4) if i != t)
        reads.append((cx, cy, cyaw, d, t, r0,
                      1.0 if (tile_on_easel and others_parked) else 0.0))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (console_x, console_y, yaw_deg, "
          f"displayed, target, rack0_card, tile_ok):\n{arr}", flush=True)
    yaw_spread = float(arr[:, 2].max() - arr[:, 2].min())
    xy_spread = float((arr[:, 0:2].max(axis=0) - arr[:, 0:2].min(axis=0)).max())
    check("randomization: whole-console yaw spread (> 2 deg) and xy offset spread "
          "(> 4 mm) are real (readback)", yaw_spread > 2.0 and xy_spread > 0.004)
    check("randomization: displayed card varies, target hour varies and is never "
          "the displayed one, the target tile really sits on the indicator every "
          "seed (others parked), rack permutation varies (readback)",
          len(set(arr[:, 3])) >= 2 and len(set(arr[:, 4])) >= 2
          and bool((arr[:, 3] != arr[:, 4]).all()) and bool((arr[:, 6] == 1.0).all())
          and len(set(arr[:, 5])) >= 2 and bool((arr[:, 5] >= 0).all()))

    # =========================== 4. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 5. seed strategy: ROTATE the clock =========================
    # The seed's plan is to ROTATE an articulated knob to change the reading.
    # Here rotation is a no-op: the displayed card spun 180 deg in place is the
    # SAME card with the SAME pip count (pips on both faces) — nothing scores.
    torch.manual_seed(41)
    env.reset()
    step(30)
    d, t = d_t()
    lz = float(loc_of(scene.cards[d])[0, 2])
    place_local(scene.cards[d], c.stand_x, c.stand_y, lz, quat=Q_SPIN)
    step(120)
    report("seed-strategy")
    still_seated = bool(scene._gather(scene._seated_display(),
                                      scene.displayed_idx)[0])
    s, ok = judge()
    check("seed strategy: displayed card ROTATED 180 deg in place stays the same "
          "seated card — rotation cannot change the pip count, score ~0 (<= 0.02), "
          "no success (the seed's rotate-the-state plan is a no-op here)",
          still_seated and s <= 0.02 and not ok)

    # =========================== 6. wrong card in the display ===============================
    # A DISTRACTOR (neither displayed nor target) constructed seated in the slot
    # (old card first moved aside to free the one-card slot): it physically
    # seats, but the seat credit gathers at target_idx — identity is load-bearing.
    torch.manual_seed(51)
    env.reset()
    step(30)
    d, t = d_t()
    x = next(i for i in range(4) if i not in (d, t))
    park_old_on_ground(d)
    place_local(scene.cards[x], c.stand_x, c.stand_y, c.disp_seat_z + 0.003)
    step(150)
    report("wrong-card")
    xr = bool(scene._seated_display()[0, x])
    s, ok = judge()
    check("wrong card: DISTRACTOR seated in the display slot physically reads "
          "seated but earns NO seating credit and no success (score = extraction "
          "only, <= 0.17) — pip identity is load-bearing",
          xr and not bool(scene._seated_ever[0]) and s <= c.w_extract + 0.02
          and not ok)

    # =========================== 7. near-miss: flat across the mouth ========================
    torch.manual_seed(61)
    env.reset()
    step(30)
    d, t = d_t()
    park_old_on_ground(d)
    place_local(scene.cards[t], c.stand_x, c.stand_y,
                c.disp_mouth + c.funnel_len + 0.02, quat=Q_FLAT)
    step(180)
    report("near-miss-flat")
    s, ok = judge()
    check("near-miss seat: target card dropped LYING FLAT across the slot mouth "
          "rests high and sideways — z and upright gates reject: no seating "
          "credit, no success",
          not bool(scene._gather(scene._seated_display(), scene.target_idx)[0])
          and not bool(scene._seated_ever[0]) and s <= c.w_extract + 0.02
          and not ok)

    # =========================== 8. inverted card ===========================================
    torch.manual_seed(71)
    env.reset()
    step(30)
    d, t = d_t()
    park_old_on_ground(d)
    place_local(scene.cards[t], c.stand_x, c.stand_y, c.disp_seat_z + 0.003,
                quat=Q_FLIP)
    step(150)
    report("inverted")
    loc, up_z, _still = scene._card_tensors()
    s, ok = judge()
    check("inverted card: target card seated UPSIDE-DOWN (pips hidden inside the "
          "slot, up_z ~ -1) is rejected by the pips-up gate: no seating credit, "
          "no success",
          float(up_z[0, t]) < -0.9
          and abs(float(loc[0, t, 2]) - c.disp_seat_z) < 0.02
          and not bool(scene._seated_ever[0]) and s <= c.w_extract + 0.02
          and not ok)

    # =========================== 9-10. discard missing + latched credit =====================
    torch.manual_seed(81)
    env.reset()
    step(30)
    d, t = d_t()
    # old card dumped flat on the BASE PLATE (well clear of the tray), target
    # constructed seated properly: extract + seat credit, but NO success.
    place_local(scene.cards[d], 0.13, -0.01, c.base_z + c.card_t / 2 + 0.002,
                quat=Q_FLAT)
    place_local(scene.cards[t], c.stand_x, c.stand_y, c.disp_seat_z + 0.003)
    step(150)
    report("discard-missing")
    s9, ok = judge()
    seated_now = bool(scene._gather(scene._seated_display(), scene.target_idx)[0])
    check("discard missing: target card properly seated but old card dumped on "
          f"the base plate (not the tray): score = extract+seat ~0.60 (got "
          f"{s9:.3f}), NO success — both halves of the exchange are required",
          seated_now and bool(scene._seated_ever[0])
          and abs(s9 - (c.w_extract + c.w_seat)) < 0.011 and not ok)
    # regression: the seated target teleported back out to the ground
    place_local(scene.cards[t], -0.55, 0.30, c.card_t / 2 + 0.002, quat=Q_FLAT)
    step(90)
    report("regressed")
    s10, ok = judge()
    check("latched credit: teleporting the seated target back OUT leaves the "
          f"latched score unchanged ({s9:.3f} -> {s10:.3f}), still no success "
          "(non-success cap holds)",
          abs(s10 - s9) < 1e-3
          and not bool(scene._gather(scene._seated_display(),
                                     scene.target_idx)[0]) and not ok)

    # =========================== 11. wrong card in the tray =================================
    torch.manual_seed(91)
    env.reset()
    step(30)
    d, t = d_t()
    x = next(i for i in range(4) if i not in (d, t))
    place_local(scene.cards[x], c.tray_x, c.tray_y, 0.12, quat=Q_FLAT)
    step(180)
    report("wrong-discard")
    x_in = bool(scene._in_tray()[0, x])
    s, ok = judge()
    check("wrong discard: a DISTRACTOR dropped into the tray (it physically "
          "settles inside) earns NO tray credit while the old card is still "
          "displayed: score ~0, no success — the tray credit gathers at the "
          "displayed card",
          x_in and not bool(scene._tray_ever[0]) and s <= 0.02 and not ok)

    # =========================== 12. tray near-miss: outside the wall =======================
    torch.manual_seed(101)
    env.reset()
    step(30)
    d, t = d_t()
    # old card laid flat on the base plate right against the tray's outer -y
    # wall (long axis points -y under Q_FLAT: centre 50 mm + wall clearance out)
    place_local(scene.cards[d], c.tray_x,
                c.tray_y - c.tray_ihy - 0.010 - c.card_h / 2 - 0.002,
                c.base_z + c.card_t / 2 + 0.002, quat=Q_FLAT)
    step(150)
    report("tray-near-miss")
    dl = loc_of(scene.cards[d])[0]
    s, ok = judge()
    check("tray near-miss: old card resting against the tray's OUTER wall — "
          "beside the tray is not inside it: extraction credit only "
          "(<= 0.17), no tray credit, no success",
          not bool(scene._tray_ever[0]) and float(dl[2]) < 0.05
          and s <= c.w_extract + 0.02 and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.card_clock")
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
    except Exception as exc:  # noqa: BLE001 — die fast, Kit teardown hangs
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({exc})", flush=True)
        os._exit(1)
