"""Smoke battery for ServingShelfScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT a solution (the solution is solve.py — raise the leaf onto its keeper
stop with a hinge-torque servo, swing the brace out, lower the leaf onto it, drop
the platter; the Franka strategy is TASK.md's embodiment argument). Drives here
use the same hand-scale numbers as solve.py (shared numbers = shared honesty);
constructed outcomes are written as CONSISTENT hinge poses (pure quat writes about
the live joints) and then SETTLED through contact before being judged.

One linear run, 14 named checks:
  1. settle    — clean reset: finite everywhere, both leaves physically hanging on
                 their lower stops, braces stowed, platter physically AT its
                 sampled cart-top pose (readback through the cart transform),
                 score ~0, no success;
  2. tiles     — the GREEN tile physically floats on the sampled serve side and
                 the GREY tile on the other (world readback vs the cfg formula);
  3. random    — across 8 seeds: both serve sides drawn, cart xy/yaw vary, platter
                 start varies; cart yaw and platter pose readback-verified;
  4. null      — 4 s of nothing: score < 0.05, no success;
  5. keeper    — the SEED's whole strategy (swing the articulation open, done):
                 servo the leaf up, RELEASE — it rests over-vertical on the keeper
                 stop on its own; up latch only, score ~0.15, NO success;
  6. no-prop   — a LEVEL leaf with its brace stowed is not a restable state: the
                 leaf falls back through level to its hanging stop; seat latch
                 stays 0, score ~0, never success (the brace is load-bearing);
  7. blocked   — ordering is GEOMETRY: with the leaf hanging, the full working
                 brace torque (solve.py's cap) CANNOT deploy the brace — the
                 sweep jams on the hanging leaf far short of deployed; releasing
                 leaves no spurious credit;
  8. clears    — non-vacuousness companion: the SAME constant torque deploys the
                 brace past 75 deg once the leaf is up on the keeper;
  9. platter   — deployed structure but the platter still on the cart top: the
                 leaf-frame window rejects it, NO success;
 10. exactness — full correct structure (brace deployed, leaf seated level on it,
                 platter gravity-dropped onto the shelf) -> success() and
                 score == 1.0, still true 1 s later hands-off;
 11. both-open — deploying the DECOY side too REVOKES success (the grey side must
                 stay folded); the latched credit remains — it never evaporates;
 12. wrong-side— the same structure built on the GREY side with the platter served
                 there scores ~0: wrong place is not partial credit;
 13. near-miss — platter dropped just past the shelf's outer edge tumbles off:
                 not on the shelf, NO success;
 14. frames    — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.open_grill_i260.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=10)
parser.add_argument("--max_frames", type=int, default=280)
parser.add_argument("--out", type=str, default="frames.npz")  # CWD — the pipeline fetches it
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe (proven on this render stack): kit mis-decodes the driver version and
# silently rejects RTX -> the annotator returns EMPTY frames.
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
    from simgen_tasks.open_grill_i260 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
_wd = threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

# Same hand-scale leaf servo as solve.py (shared numbers = shared honesty).
KX_L = 3.0
W_L = 1.5
KV_L = 0.15

PROBE_SEED = 5


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.serving_shelf")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids0 = torch.tensor([0], device=device)

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.95, -0.85, 0.85)) + o),
                                tuple(np.array((0.0, 0.0, 0.28)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (800, 500))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames — check the RTX recipe",
                  flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action, render=annot is not None)
            if annot is not None and step_i % args.record_every == 0 \
                    and len(frames) < args.max_frames:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def report(tag: str) -> None:
        p = scene.platter_local()[0]
        print(f"[smoke] {tag:14s} | leaf={math.degrees(float(scene.target_leaf_angle()[0])):+7.1f}deg "
              f"brace={math.degrees(float(scene.target_brace_angle()[0])):+6.1f}deg "
              f"decoy=({math.degrees(float(scene.decoy_leaf_angle()[0])):+6.1f}, "
              f"{math.degrees(float(scene.decoy_brace_angle()[0])):+5.1f})deg "
              f"plat=({float(p[0]) * 1000:+6.1f}, {float(p[1]) * 1000:+6.1f}, "
              f"{float(p[2]) * 1000:+6.1f})mm on_shelf={bool(scene.platter_on_shelf()[0])} "
              f"latch=(u={float(scene.up_latch[0]):.0f}, b={float(scene.brace_latch[0]):.0f}, "
              f"s={float(scene.seat_latch[0]):.0f}, p={float(scene.plat_latch[0]):.0f}) "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def finite_all() -> bool:
        return all(bool(torch.isfinite(b.data.root_state_w).all())
                   for b in scene._bodies().values()) \
            and bool(torch.isfinite(scene.score()).all())

    def reset(seed: int) -> None:
        torch.manual_seed(seed)
        env.reset()
        step(30)

    # --- construct helpers: CONSISTENT hinge poses (pure rotation about the live joint) ---
    def _q_x(theta: float) -> torch.Tensor:
        return torch.tensor([[math.cos(theta / 2), math.sin(theta / 2), 0.0, 0.0]],
                            device=device)

    def _q_z(phi: float) -> torch.Tensor:
        return torch.tensor([[math.cos(phi / 2), 0.0, 0.0, math.sin(phi / 2)]], device=device)

    def _write_hinged(body, s: float, local_anchor, q_local) -> None:
        q_cart = scene.cart.data.root_quat_w[0:1]
        p_cart = scene.cart.data.root_pos_w[0:1]
        q_mount = scene._q_mount[0 if s > 0 else 1].unsqueeze(0)
        a = torch.tensor([list(local_anchor)], device=device)
        st = torch.zeros(1, 13, device=device)
        st[:, 0:3] = p_cart + quat_apply(q_cart, a)
        st[:, 3:7] = quat_mul(q_cart, quat_mul(q_mount, q_local))
        body.write_root_state_to_sim(st, ids0)

    def set_leaf(s: float, theta: float) -> None:
        _write_hinged(scene.leaves[s], s, (0.0, s * c.hinge_y, c.hinge_z), _q_x(theta))

    def set_brace(s: float, phi: float) -> None:
        _write_hinged(scene.braces[s], s,
                      (-s * c.pivot_x_abs, s * (c.hinge_y - 0.030), c.pivot_z), _q_z(phi))

    def drop_platter(s: float, y_leaf: float) -> None:
        """Transport-only teleport: platter to FREE AIR above the side-`s` leaf at
        leaf-frame y offset `y_leaf`, zero velocity; gravity does the serve."""
        leaf = scene.leaves[s]
        drop = torch.zeros(1, 13, device=device)
        off = torch.tensor([[0.0, y_leaf, 0.05]], device=device)
        drop[0, 0:3] = leaf.data.root_pos_w[0] + quat_apply(leaf.data.root_quat_w, off)[0]
        drop[0, 3:7] = leaf.data.root_quat_w[0]
        scene.platter.write_root_state_to_sim(drop, ids0)
        for _ in range(30):
            step(10)
            if bool(scene.settled()[0]):
                break

    def leaf_servo_up(col: int, budget: int = 2400) -> bool:
        """solve.py's raise servo, verbatim numbers: gravity ff + rate cascade to
        the keeper stop; releases the drive on exit."""
        tgt = c.leaf_hi + math.radians(4.0)
        for _ in range(budget):
            th = float(scene.target_leaf_angle()[0])
            w = float(scene.target_rate()[0])
            if th > math.radians(91.0) and abs(w) < 0.3:
                scene.leaf_tau[0, col] = 0.0
                return True
            w_des = max(-W_L, min(W_L, KX_L * (tgt - th)))
            scene.leaf_tau[0, col] = c.leaf_mgr * math.cos(th) + KV_L * (w_des - w)
            step(1)
        scene.leaf_tau[0, col] = 0.0
        return False

    def brace_push(col: int, tau: float, secs: float) -> tuple[float, float]:
        """Constant pivot torque (the cap — solve's working magnitude) for `secs`;
        returns (max brace angle, max leaf angle) seen. Releases the drive."""
        ph_max, th_max = -9.0, -9.0
        for _ in range(int(secs * 120)):
            scene.brace_tau[0, col] = tau
            step(1)
            ph_max = max(ph_max, float(scene.target_brace_angle()[0]))
            th_max = max(th_max, float(scene.target_leaf_angle()[0]))
        scene.brace_tau[0, col] = 0.0
        return ph_max, th_max

    deg = math.degrees

    # ========================= 1. settle / clean-slate ========================================
    reset(PROBE_SEED)
    step(120)
    report("reset")
    side = float(scene.side[0])
    col = 0 if side > 0 else 1
    th_t, th_d = float(scene.target_leaf_angle()[0]), float(scene.decoy_leaf_angle()[0])
    ph_t, ph_d = float(scene.target_brace_angle()[0]), float(scene.decoy_brace_angle()[0])
    # platter readback through the sampled cart transform
    q_cart = scene.cart.data.root_quat_w[0:1]
    want = scene.cart.data.root_pos_w[0] + quat_apply(q_cart, scene.plat_start[0:1])[0]
    plat_rb = float((scene.platter.data.root_pos_w[0] - want).norm())
    print(f"[smoke] seed={PROBE_SEED} side={side:+.0f} leaves=({deg(th_t):.1f}, {deg(th_d):.1f})deg "
          f"braces=({deg(ph_t):.1f}, {deg(ph_d):.1f})deg plat_rb_err={plat_rb * 1000:.2f}mm",
          flush=True)
    check("settle: clean reset (finite; leaves hanging on their stops; braces stowed; "
          "platter AT its sampled cart-top pose; score ~0)",
          finite_all() and abs(deg(th_t) + 80.0) < 4.0 and abs(deg(th_d) + 80.0) < 4.0
          and abs(deg(ph_t)) < 4.0 and abs(deg(ph_d)) < 4.0 and plat_rb < 0.008
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 2. marker tiles readback =======================================
    tile_err = 0.0
    for name, s in (("green", side), ("grey", -side)):
        want_local = torch.tensor([[0.0, s * (c.cart_hw + 0.003 + c.tile_size[1] / 2),
                                    c.tile_z]], device=device)
        want = scene.cart.data.root_pos_w[0] + quat_apply(q_cart, want_local)[0]
        got = scene.tiles[name].data.root_pos_w[0]
        tile_err = max(tile_err, float((got - want).norm()))
    print(f"[smoke] tile readback err={tile_err * 1000:.2f}mm (green on side {side:+.0f})",
          flush=True)
    check("tiles: GREEN physically floats on the sampled serve side, GREY on the other",
          tile_err < 0.003)

    # ========================= 3. randomization across seeds ==================================
    draws = []
    ok_rb = True
    for sd in (11, 12, 13, 14, 15, 16, 17, 18):
        reset(sd)
        s = float(scene.side[0])
        cx, cy, cyaw = (float(v) for v in scene.cart_pose0[0, :3])
        # cart pose readback: position and yaw of the cart body
        cp = scene.cart.data.root_pos_w[0] - scene.env_origins[0]
        qc = scene.cart.data.root_quat_w[0]
        yaw_rb = 2.0 * math.atan2(float(qc[3]), float(qc[0]))
        ok_rb &= abs(float(cp[0]) - cx) < 0.005 and abs(float(cp[1]) - cy) < 0.005
        ok_rb &= abs(math.atan2(math.sin(yaw_rb - cyaw), math.cos(yaw_rb - cyaw))) < 0.03
        want = scene.cart.data.root_pos_w[0] \
            + quat_apply(scene.cart.data.root_quat_w[0:1], scene.plat_start[0:1])[0]
        ok_rb &= float((scene.platter.data.root_pos_w[0] - want).norm()) < 0.008
        # the green tile physically follows the sampled side (cart-local y sign == side)
        d_tile = (scene.tiles["green"].data.root_pos_w[0:1]
                  - scene.cart.data.root_pos_w[0:1])
        tile_y_local = float(quat_apply_inverse(scene.cart.data.root_quat_w[0:1], d_tile)[0, 1])
        ok_rb &= tile_y_local * s > 0.10
        draws.append((int(s), round(cx * 1000, 1), round(cy * 1000, 1),
                      round(deg(cyaw), 1), round(float(scene.plat_start[0, 0]) * 1000, 1)))
    print(f"[smoke] draws (side, cart_x_mm, cart_y_mm, cart_yaw_deg, plat_x_mm): {draws}",
          flush=True)
    sides = {d[0] for d in draws}
    check("randomization is real (both sides drawn; cart xy/yaw + platter start vary; "
          "cart + platter readback-verified)",
          ok_rb and sides == {-1, 1}
          and len({d[3] for d in draws}) >= 5 and len({d[4] for d in draws}) >= 5)

    # ========================= 4. null policy =================================================
    reset(PROBE_SEED)
    step(480)  # 4 s of nothing
    report("null")
    check("null policy: score < 0.05 and no success",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 5. keeper stop == the seed's end state =========================
    reset(PROBE_SEED)
    side = float(scene.side[0])
    col = 0 if side > 0 else 1
    ok_up = leaf_servo_up(col)
    step(240)  # 2 s hands off — the over-vertical rest must hold on its own
    report("keeper")
    th = float(scene.target_leaf_angle()[0])
    check("keeper (seed strategy): leaf swung open and LEFT — rests over-vertical on the "
          "stop hands-off; up latch only, score ~0.15, no success",
          ok_up and deg(th) > 88.0 and float(scene.up_latch[0]) == 1.0
          and not bool(scene.success()[0])
          and 0.10 < float(scene.score()[0]) < 0.20)

    # ========================= 6. a level leaf without its brace FALLS ========================
    reset(PROBE_SEED)
    side = float(scene.side[0])
    set_leaf(side, 0.0)  # constructed level, brace still stowed
    never_succ = True
    for _ in range(24):  # 2 s
        step(10)
        never_succ &= not bool(scene.success()[0])
    report("no-prop")
    th = float(scene.target_leaf_angle()[0])
    check("no-prop: a LEVEL leaf with a stowed brace falls back to hanging — the brace is "
          "load-bearing; seat latch 0, score ~0, never success",
          deg(th) < -45.0 and float(scene.seat_latch[0]) == 0.0
          and float(scene.score()[0]) < 0.05 and never_succ and finite_all())

    # ========================= 7. blocked sweep: order is geometry ============================
    reset(PROBE_SEED)
    side = float(scene.side[0])
    col = 0 if side > 0 else 1
    ph_max, th_max = brace_push(col, c.brace_tau_max, 3.0)
    step(120)  # release
    report("blocked")
    print(f"[smoke] blocked sweep: brace_max={deg(ph_max):.1f}deg leaf_max={deg(th_max):.1f}deg",
          flush=True)
    check("blocked: with the leaf hanging, the FULL working brace torque jams far short of "
          "deployed (order is geometry); no spurious credit after release",
          deg(ph_max) < 60.0 and deg(float(scene.target_brace_angle()[0])) < 60.0
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.05
          and finite_all())

    # ========================= 8. the same torque deploys once the leaf is up ================
    set_leaf(side, c.leaf_hi - math.radians(1.0))  # construct: leaf on the keeper
    step(120)
    ph_max2, _ = brace_push(col, c.brace_tau_max, 3.0)
    step(60)
    report("clears")
    check("clears: the SAME constant torque deploys the brace past 75 deg once the leaf is "
          "raised (the blocked probe is not vacuous)",
          deg(ph_max2) > 75.0
          and deg(float(scene.target_brace_angle()[0])) > 75.0)

    # ========================= 9. deployed structure, platter still on the cart top ==========
    reset(PROBE_SEED)
    side = float(scene.side[0])
    set_brace(side, math.radians(90.0))
    set_leaf(side, 0.0)
    step(240)  # leaf settles onto the brace bar by contact
    report("no-platter")
    th = float(scene.target_leaf_angle()[0])
    check("platter: correct deployed structure but platter still on the cart top — the "
          "leaf-frame window rejects it, no success",
          abs(th) < c.leaf_level_tol and not bool(scene.platter_on_shelf()[0])
          and not bool(scene.success()[0]))

    # ========================= 10. exactness: the goal state scores 1.0 ======================
    drop_platter(side, c.plat_seat_y)
    report("goal-state")
    good = bool(scene.success()[0]) and abs(float(scene.score()[0]) - 1.0) < 1e-3
    step(120)  # ...and it persists hands-off (1 s)
    report("goal-persist")
    check("exactness: seated shelf + served platter -> success() and score == 1.0, stable",
          good and bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 1.0) < 1e-3)

    # ========================= 11. deploying the decoy side REVOKES success ==================
    set_brace(-side, math.radians(90.0))
    set_leaf(-side, 0.0)
    step(240)
    report("both-open")
    check("both-open: deploying the GREY side too revokes success (it must stay folded); "
          "latched credit remains",
          not bool(scene.success()[0])
          and deg(float(scene.decoy_leaf_angle()[0])) > -10.0  # the decoy really is deployed
          and 0.50 < float(scene.score()[0]) < 0.70)

    # ========================= 12. the same structure on the WRONG side scores ~0 ============
    reset(PROBE_SEED)
    side = float(scene.side[0])
    set_brace(-side, math.radians(90.0))
    set_leaf(-side, 0.0)
    step(240)
    drop_platter(-side, c.plat_seat_y)
    report("wrong-side")
    check("wrong-side: full structure + platter on the GREY side — no latches, score ~0, "
          "no success (wrong place is not partial credit)",
          deg(float(scene.decoy_leaf_angle()[0])) > -10.0  # decoy shelf really carries it
          and not bool(scene.platter_on_shelf()[0])
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 13. near miss: platter past the shelf edge ====================
    reset(PROBE_SEED)
    side = float(scene.side[0])
    set_brace(side, math.radians(90.0))
    set_leaf(side, 0.0)
    step(240)
    drop_platter(side, 0.26)  # past the leaf's outer edge — tumbles off
    step(120)
    report("near-miss")
    check("near-miss: platter dropped past the shelf edge is NOT on the shelf, no success",
          not bool(scene.platter_on_shelf()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.60 and finite_all())

    # ========================= 14. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.serving_shelf")
        print(f"[smoke] saved {arr.shape} -> {os.path.abspath(args.out)}", flush=True)
    check("video frames recorded", len(frames) >= 20)

    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        bad = [nm for nm, ok in checks if not ok]
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)} — failing: {bad}", flush=True)
    # Kit teardown hangs are routine: watchdog then hard exit.
    t = threading.Timer(10.0, lambda: os._exit(0 if all_ok else 1))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    finally:
        os._exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
