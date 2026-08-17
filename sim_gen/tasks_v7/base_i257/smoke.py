"""Smoke / rubric-REJECTION battery for CargoTramScene (sim_gen task `base_i257`) —
NullRobot, teleported probe states + force probes, RECORDED.

This is NOT a solution (solve.py — drop the red cube into the cart, push the loaded
cart up the ramp into the roofed pocket, seat it — is the acceptance evidence that the
rubric ACCEPTS a correct outcome; it passes on seeds 0/1). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state and
asserts the rubric REJECTS it — plus force probes that prove the ORDER-FORCING is
physically real: solve's OWN push servo parks the EMPTY cart just fine (parking needs
no cargo — physics doesn't force the order, the canopy does), and then the 40 mm cube
demonstrably cannot be dropped into that parked cart from anywhere above.
No probe in this battery ever reaches success(), and a final audit asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: cart level on the flat, both
                            cubes loose on the open floor, nothing loaded; score ~0;
  3-4. randomization      — READBACK over 6 seeded resets: the cargo/decoy slot pair
                            varies (>= 3 distinct ordered assignments), xy jitter and
                            free yaw vary, the cart's start x varies; every reset sane;
  5.  null policy         — 240 idle steps -> score ~0, no success, cart still on the
                            flat;
  6.  order gate 1        — solve's OWN velocity servo (<= 6 N) drives the EMPTY cart
                            up the ramp and seats it in the pocket (non-vacuous: ~0.5 m
                            of driven displacement, seated readback) — and earns
                            NOTHING: no latch, no success. Parking is physically easy;
                            only the rubric's cargo demand makes it worthless;
  7.  order gate 2        — with the cart parked, the cargo dropped from above lands on
                            the CANOPY, and dropped at the canopy's front lip it is
                            refused by the sub-cube-width slit/strip: the bucket of a
                            parked cart is unreachable — loading MUST precede parking;
  8.  seed-strategy       — the seed family's move (carry the cube straight to the
                            goal) is worthless: the cargo settled on the summit pocket
                            floor (beside where the cart would park) scores nothing —
                            containment is judged in the CART's body frame;
  9.  roof dump           — the cargo settled on the canopy roof scores nothing;
  10. bridge near-miss    — a LOADED cart resting tilted with its tail perched on the
                            crest (z window and upright cone both PASS) is rejected by
                            the park x window alone: almost-in is not parked;
  11. wrong object        — the BLUE decoy settled inside the parked cart's bucket
                            counts for NOTHING (identity, not geometry);
  12. exclusivity         — the cargo stacked into the same bucket ON TOP of the decoy
                            earns every partial latch (~0.60) but success stays False
                            while the decoy rides along;
  13. latched credit      — teleporting the cargo (first) and decoy (second) out of the
                            bucket leaves the latched score unchanged (credit never
                            evaporates) — and still no success;
  14. settle gate         — cart + cargo IN every position window but sliding backward
                            at ~0.25 m/s is NOT success (velocity gates are real);
                            the cargo is yanked out before it can settle;
  15. rejection audit     — success() was never True at ANY judged point;
  16. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.base_i257.smoke --headless
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
    from isaaclab.utils.math import quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cargo_tram")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.55, -1.25, 0.95)) + o),
                                tuple(np.array((0.50, -0.08, 0.10)) + o),
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

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        s, ok = judge()
        cp, gp = rel(scene.cart), rel(scene.cargo)
        print(f"[smoke] {tag:16s} |"
              f" cart=({float(cp[0]):+.3f},{float(cp[1]):+.3f},{float(cp[2]):.3f})"
              f" cargo=({float(gp[0]):+.3f},{float(gp[1]):+.3f},{float(gp[2]):.3f})"
              f" loaded={bool(scene.cargo_in_cart()[0])}"
              f" decoy_in={bool(scene.decoy_in_cart()[0])}"
              f" parked={bool(scene.cart_parked()[0])}"
              f" score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, xyz, quat=None, vel=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor(xyz, device=device)
        if quat is None:
            st[:, 3] = 1.0
        else:
            st[:, 3:7] = torch.tensor(quat, device=device)
        if vel is not None:
            st[:, 7:10] = torch.tensor(vel, device=device)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def wrench(body, f3: torch.Tensor) -> None:
        """World force expressed in the body's current link frame (the house
        convention: is_global=True silently drops torques on this stack)."""
        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f3.view(1, 3).expand(n, 3)).unsqueeze(1),
            torch.zeros(n, 1, 3, device=device),
            env_ids=all_ids)

    def park_cart_probe() -> float:
        """Solve's OWN P2 push controller, verbatim (+x velocity servo toward
        0.10 m/s, clamped to [0, 6] N, then the gentle 1.5 N back-wall seat press):
        the honest order-gate probe, here driving the EMPTY cart. Returns the driven
        displacement (non-vacuity evidence); releases afterwards."""
        f3 = torch.zeros(3, device=device)
        x0 = float(rel(scene.cart)[0])
        for _ in range(2200):
            rx = float(rel(scene.cart)[0])
            if rx > 0.8145:
                break
            vx = float(scene.cart.data.root_lin_vel_w[0, 0])
            f3[0] = max(min(80.0 * (0.10 - vx), 6.0), 0.0)
            wrench(scene.cart, f3)
            step(1)
        f3[0] = 1.5
        for _ in range(90):
            wrench(scene.cart, f3)
            step(1)
        wrench(scene.cart, zero3)
        step(45)
        return float(rel(scene.cart)[0]) - x0

    def layout_sane(tag: str) -> bool:
        """Reset honesty: cart level on the flat inside its start band, both cubes
        loose on the open floor in the scatter apron, nothing loaded."""
        cp = rel(scene.cart)
        ok = (c.cart_start_x[0] - 0.01 <= float(cp[0]) <= c.cart_start_x[1] + 0.01
              and abs(float(cp[1])) < 0.01
              and 0.018 <= float(cp[2]) <= 0.024
              and bool(scene.cart_upright()[0]))
        for b in (scene.cargo, scene.decoy):
            p = rel(b)
            ok = ok and 0.14 < float(p[0]) < 0.39 and -0.39 < float(p[1]) < -0.14 \
                and float(p[2]) < 0.045
        ok = ok and not bool(scene.cargo_in_cart()[0]) \
            and not bool(scene.decoy_in_cart()[0]) and not bool(scene.cart_parked()[0])
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}: cart={cp.tolist()} "
                  f"cargo={rel(scene.cargo).tolist()} decoy={rel(scene.decoy).tolist()}",
                  flush=True)
        return ok

    park_rest_x = c.back_x[0] - c.cart_len / 2  # 0.817, seated against the back wall

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    bodies = (scene.cart, scene.cargo, scene.decoy)
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; cart level on the flat, cubes loose on the open "
          "floor, nothing loaded", fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    slots = np.array(c.cube_slots)
    pairs: set[tuple[int, int]] = set()
    cart_xs, yaws, cg_xy, dc_xy = [], [], [], []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        cart_xs.append(float(rel(scene.cart)[0]))
        assign = []
        for b, acc in ((scene.cargo, cg_xy), (scene.decoy, dc_xy)):
            p = rel(b)
            xy = np.array([float(p[0]), float(p[1])])
            acc.append(xy)
            assign.append(int(np.argmin(((slots - xy) ** 2).sum(axis=1))))
            q = b.data.root_quat_w[0]
            yaws.append(2.0 * math.atan2(float(q[3]), float(q[0])))
        pairs.add(tuple(assign))
    cg, dc = np.array(cg_xy), np.array(dc_xy)
    print(f"[smoke] randomization readback: cart_x={np.round(cart_xs, 4).tolist()}\n"
          f"[smoke] cargo xy:\n{np.round(cg, 4)}\n[smoke] decoy xy:\n{np.round(dc, 4)}\n"
          f"[smoke] (cargo,decoy) slot pairs: {sorted(pairs)}", flush=True)
    cg_sp = cg.max(axis=0) - cg.min(axis=0)
    dc_sp = dc.max(axis=0) - dc.min(axis=0)
    yaw_spread = max(yaws) - min(yaws)
    cart_spread = max(cart_xs) - min(cart_xs)
    check("randomization: the (cargo, decoy) slot pair varies across seeds "
          f"({len(pairs)} distinct ordered assignments) and cube xy jitters "
          f"(spreads cargo=({cg_sp[0]:.3f},{cg_sp[1]:.3f}) "
          f"decoy=({dc_sp[0]:.3f},{dc_sp[1]:.3f}))",
          len(pairs) >= 3 and sane and all(v > 0.015 for v in (*cg_sp, *dc_sp)))
    check(f"randomization: cube yaw is free (readback spread {yaw_spread:.2f} rad) and "
          f"the cart's start x varies (spread {cart_spread:.3f})",
          yaw_spread > 0.5 and cart_spread > 0.02)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0, no success, cart still on the flat after 240 idle "
          "steps", s <= 0.02 and not ok
          and float(rel(scene.cart)[0]) < c.flat_x[1] - c.cart_len / 2)

    # =========================== 6. order gate 1: empty cart parks, earns nothing ==========
    env.reset(seed=41)
    step(60)
    disp = park_cart_probe()
    report("empty-park")
    cp = rel(scene.cart)
    s, ok = judge()
    empty_parked = (c.park_x[0] <= float(cp[0]) <= c.park_x[1]
                    and c.park_z[0] <= float(cp[2]) <= c.park_z[1])
    check("order gate 1: solve's own servo parks the EMPTY cart (driven "
          f"{disp * 100:.1f} cm, seated at x={float(cp[0]):.3f} — non-vacuous: parking "
          "needs no cargo, physics alone does not force the order) yet earns NOTHING: "
          "score ~0, no success", empty_parked and disp > 0.30 and s <= 0.02 and not ok)

    # =========================== 7. order gate 2: canopy blocks loading =====================
    # (a) straight down from above: the cube lands on the ROOF, not in the bucket
    place(scene.cargo, (0.84, 0.0, 0.28), settle_steps=60)
    gp1 = rel(scene.cargo).clone()
    in1 = bool(scene.cargo_in_cart()[0])
    report("drop-on-roof")
    # (b) at the canopy's front lip: refused by the sub-cube slit/strip
    place(scene.cargo, (0.778, 0.0, 0.26), settle_steps=90)
    gp2 = rel(scene.cargo).clone()
    in2 = bool(scene.cargo_in_cart()[0])
    report("drop-at-lip")
    s, ok = judge()
    check("order gate 2: with the cart parked, the cargo dropped from above rests on "
          f"the canopy (z={float(gp1[2]):.3f}) and dropped at the canopy's front lip "
          f"it is refused (rest=({float(gp2[0]):+.3f},{float(gp2[2]):.3f}), never in "
          "the bucket): a parked cart cannot be loaded — no credit, no success",
          not in1 and not in2 and float(gp1[2]) > c.wall_top + 0.01
          and s <= 0.02 and not ok
          and c.park_x[0] <= float(rel(scene.cart)[0]) <= c.park_x[1])

    # =========================== 8. seed-strategy: cube straight to the goal ================
    env.reset(seed=51)
    step(30)
    place(scene.cargo, (0.83, 0.0, 0.14), settle_steps=60)  # free space under the canopy
    report("cube-in-pocket")
    gp = rel(scene.cargo)
    s, ok = judge()
    check("seed-strategy reject: the cargo carried straight to the summit and settled "
          f"on the pocket floor (({float(gp[0]):+.3f},{float(gp[2]):.3f})) counts for "
          "NOTHING — containment is judged in the cart's body frame, and the cart is "
          "still on the flat", 0.78 < float(gp[0]) < 0.87
          and c.pocket_top < float(gp[2]) < c.pocket_top + 0.035
          and not bool(scene.cargo_in_cart()[0]) and s <= 0.02 and not ok)

    # =========================== 9. roof dump ===============================================
    place(scene.cargo, (0.84, 0.0, 0.28), settle_steps=60)
    report("cube-on-roof")
    gp = rel(scene.cargo)
    s, ok = judge()
    check("roof dump reject: the cargo settled on the canopy roof "
          f"(z={float(gp[2]):.3f}) counts for NOTHING",
          c.wall_top + c.canopy_t < float(gp[2]) < c.wall_top + c.canopy_t + 0.04
          and not bool(scene.cargo_in_cart()[0]) and s <= 0.02 and not ok)

    # =========================== 10. bridge near-miss (x window does the work) =============
    env.reset(seed=61)
    step(30)
    # cart resting tilted ~5 deg: tail perched on the crest edge, nose on the pocket
    # floor — center x = 0.8108, BELOW the park x window's low edge (0.8125)
    ang = math.asin((c.crest_z - c.pocket_top) / c.cart_len)  # ~4.99 deg nose-down
    qw, qy = math.cos(ang / 2), math.sin(ang / 2)
    bx = c.crest_x + (c.cart_len / 2) * math.cos(ang)  # bottom-face center x
    bz = (c.crest_z + c.pocket_top) / 2 + 0.0005
    place(scene.cart, (bx, 0.0, bz), quat=(qw, 0.0, qy, 0.0), settle_steps=0)
    # cargo inserted INSIDE the tilted bucket (feasible: free interior space)
    sa, ca_ = math.sin(ang), math.cos(ang)
    lz = c.cart_floor_t + c.cube_edge / 2 + 0.002
    place(scene.cargo, (bx + lz * sa, 0.0, bz + lz * ca_), quat=(qw, 0.0, qy, 0.0),
          settle_steps=20)
    report("bridge-perch")
    cp = rel(scene.cart)
    s, ok = judge()
    z_in = c.park_z[0] <= float(cp[2]) <= c.park_z[1]
    up_in = bool(scene.cart_upright()[0])
    check("bridge near-miss: the LOADED cart perched with its tail on the crest "
          f"(x={float(cp[0]):.3f}, tilt ~{math.degrees(ang):.1f} deg) passes the z "
          f"window ({z_in}) and the upright cone ({up_in}) but the park x window "
          "alone rejects it: almost-in is not parked",
          bool(scene.cargo_in_cart()[0]) and z_in and up_in
          and float(cp[0]) < c.park_x[0] and not bool(scene.cart_parked()[0]) and not ok)
    # clear the perch before the next construct (probe debris discipline)
    place(scene.cargo, (0.25, 0.30, c.cube_edge / 2 + 0.002), settle_steps=10)

    # =========================== 11. wrong object (identity) ================================
    env.reset(seed=71)
    step(30)
    place(scene.cart, (park_rest_x, 0.0, c.pocket_top + 0.0005), settle_steps=30)
    place(scene.decoy, (park_rest_x, 0.0, c.pocket_top + c.cart_floor_t
                        + c.cube_edge / 2 + 0.002), settle_steps=30)
    report("decoy-in-cart")
    s, ok = judge()
    cp = rel(scene.cart)
    check("identity: the BLUE decoy settled inside the PARKED cart's bucket counts "
          "for NOTHING — no credit, no success",
          bool(scene.decoy_in_cart()[0]) and not bool(scene.cargo_in_cart()[0])
          and c.park_x[0] <= float(cp[0]) <= c.park_x[1] and s <= 0.02 and not ok)

    # =========================== 12. exclusivity (decoy blocks the last 0.4) ================
    place(scene.cargo, (park_rest_x, 0.0, c.pocket_top + c.cart_floor_t
                        + 1.5 * c.cube_edge + 0.004), settle_steps=45)
    report("both-in-cart")
    s, ok = judge()
    check("exclusivity: the cargo stacked into the bucket ON TOP of the decoy earns "
          f"every partial latch (score={s:.2f}) but success stays False while the "
          "decoy rides along",
          bool(scene.cargo_in_cart()[0]) and bool(scene.decoy_in_cart()[0])
          and 0.55 <= s <= 0.62 and not ok)

    # =========================== 13. latched credit survives removal ========================
    s_before, _ok = judge()
    # remove the CARGO first (never leave cargo alone in a parked cart mid-probe)
    place(scene.cargo, (0.20, 0.30, c.cube_edge / 2 + 0.002), settle_steps=10)
    place(scene.decoy, (0.33, 0.30, c.cube_edge / 2 + 0.002), settle_steps=30)
    report("cubes-removed")
    s_after, ok = judge()
    check("latched credit: teleporting the cargo (first) then the decoy out of the "
          f"bucket leaves the latched score unchanged ({s_before:.2f} -> "
          f"{s_after:.2f}) while the bucket reads empty",
          abs(s_after - s_before) < 1e-3 and not bool(scene.cargo_in_cart()[0])
          and not bool(scene.decoy_in_cart()[0]) and not ok)

    # =========================== 14. settle gate ============================================
    env.reset(seed=101)
    step(30)
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = park_rest_x
    st[:, 2] = c.pocket_top + 0.0005
    st[:, 3] = 1.0
    st[:, 7] = -0.25
    st[:, 0:3] += scene.env_origins
    scene.cart.write_root_state_to_sim(st, all_ids)
    st2 = st.clone()
    st2[:, 2] += c.cart_floor_t + c.cube_edge / 2 + 0.002
    scene.cargo.write_root_state_to_sim(st2, all_ids)
    step(1)
    v_now = float(scene.cart.data.root_lin_vel_w[0].norm())
    pos_ok = bool(scene.cargo_in_cart()[0]) and bool(scene.cart_parked()[0])
    report("settle-gate")
    _s, ok = judge()
    gate_ok = pos_ok and v_now > c.settle_lin and not ok
    # yank the cargo BEFORE it can settle in place (this battery must never succeed)
    place(scene.cargo, (0.10, 0.45, c.cube_edge / 2 + 0.002), settle_steps=30)
    check("settle gate: cart+cargo inside every position window but sliding at "
          f"{v_now:.2f} m/s is NOT success (velocity gates are real)", gate_ok)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.cargo_tram")
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
    try:
        main()
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)
