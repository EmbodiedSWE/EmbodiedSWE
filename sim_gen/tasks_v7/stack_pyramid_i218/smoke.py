"""Smoke / rubric-REJECTION battery for PyramidFreightScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the latched
credit is monotone along a real trajectory). This battery proves the rubric REJECTS
wrong outcomes, and that the two physical claims the task rests on — the roof makes
build-inside impossible, and transport is stability-limited (a hard shove throws the
free-riding bridge cube off) — are load-bearing physics, not fiat. Every probe is
CONSTRUCTED as a settled state (teleport, real physics steps, judge) —
instrumentation, never a solution: no probe here reaches success().

Checks:
   1. settle/baseline    — seeded reset settles finite; cubes scattered on the
                           ground, cart out on the rails; score 0, no success;
   2. randomization      — two seeded resets: READBACK depot xy + yaw, cart start
                           depot-x, red-cube xy all differ;
   3. slot permutation   — over 8 resets the cube-to-slot assignment takes >= 3
                           distinct permutations — which cube lies where must be
                           perceived, not memorized;
   4. null-policy        — 240 idle steps: score ~0, no success;
   5. SEED STRATEGY      — the seed's plan verbatim ("build the pyramid in place at
                           the goal"): a real settled pyramid CONSTRUCTED on the
                           GROUND inside the depot pocket — every cube-on-cart
                           clause refuses, score ~0 (and no gripper could have done
                           it: see check 10);
   6. built-not-deliv    — a real pyramid built on the cart, cart still out on the
                           rails: seated+built latch (~0.45) but no delivery credit,
                           no success — building is not enough;
   7. empty-cart-deliv   — the EMPTY cart parked inside the pocket: delivered_ok
                           geometry holds but the delivery latch (conditioned on the
                           pyramid riding the cart) refuses, score ~0;
   8. wrong-cube-on-top  — red+blue seated in the tray, GREEN bridging on top: the
                           color-role clauses refuse (base = red AND green), score ~0;
   9. near-miss span     — blue resting entirely on ONE base cube (centred on red,
                           21 mm off the pair midpoint > span_tol 10 mm): seated
                           credit only, pyramid refused;
  10. ROOF certificate   — blue RELEASED from above the pocket falls onto the roof
                           and settles up there (z > 0.12) — a cube physically
                           cannot be brought down onto a stack inside, so
                           build-then-deliver is forced by geometry;
  11. SHOVE certificate  — a real pyramid on the cart, then a 12 N slam toward the
                           depot (30 m/s^2 >> the 8.8 m/s^2 slip threshold): the
                           cart reaches the pocket but the blue cube is thrown off —
                           pyramid broken, delivery latch refuses, no success. The
                           probe asserts the cart actually arrived (actuation is not
                           vacuous): gentle transport is load-bearing physics;
  12. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.stack_pyramid_i218.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=12)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version on some GPUs and silently rejects RTX
# -> the annotator returns EMPTY frames. Disable the check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()

try:
    from . import scene as task_scene
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene

_qz = task_scene._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if rec and _REC["i"] % args.record_every == 0:
            for _f in range(3):  # flush accumulated history (ghosting fix)
                env.sim.render()
            arr = np.asarray(_REC["annot"].get_data())
            if arr.size:
                _REC["frames"].append(arr[..., :3].astype(np.uint8).copy())
        _REC["i"] += 1


def _refresh() -> None:
    _ENV.iscene.update(0.0)


def _all_ids():
    return torch.arange(_ENV.num_envs, device=_ENV.device)


def _write_state(body, st: torch.Tensor) -> None:
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    loc = scene.cubes_sled_local()[0]
    sx = scene.sled_fix_local()[0]
    print(f"[smoke] {tag:18s} | sled depot=({float(sx[0]):+.3f},{float(sx[1]):+.3f}) "
          f"red_z={float(loc[0, 2]):+.3f} grn_z={float(loc[1, 2]):+.3f} "
          f"blu=({float(loc[2, 0]):+.3f},{float(loc[2, 1]):+.3f},{float(loc[2, 2]):+.3f}) "
          f"seated={bool(scene._seated[0])} built={bool(scene._built[0])} "
          f"deliv={bool(scene._deliv[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pyramid_freight")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.25, -1.00, 0.80)) + o),
                                tuple(np.array((0.18, 0.00, 0.05)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        _REC["annot"] = annot if warm.size else None
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    # ----- pose constructors ----------------------------------------------------------------------
    from isaaclab.utils.math import quat_apply

    def depot_pose(x_loc: float, y_loc: float, z_loc: float,
                   extra_yaw: float = 0.0) -> torch.Tensor:
        """Root state at depot-frame (x, y, z), yawed with the depot (+extra)."""
        psi = scene.fix_yaw
        cosp, sinp = torch.cos(psi), torch.sin(psi)
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = scene.fix_xy[:, 0] + cosp * x_loc - sinp * y_loc
        st[:, 1] = scene.fix_xy[:, 1] + sinp * x_loc + cosp * y_loc
        st[:, 2] = scene.env_origins[:, 2] + z_loc
        st[:, 3:7] = _qz(psi + extra_yaw)
        return st

    def cart_pose(x_loc: float, y_loc: float, z_loc: float) -> torch.Tensor:
        """Root state at cart-local (x, y, z), square to the cart (a release pose)."""
        st = torch.zeros(n, 13, device=device)
        lp = torch.tensor([x_loc, y_loc, z_loc], device=device).expand(n, 3)
        st[:, 0:3] = scene.sled.data.root_pos_w \
            + quat_apply(scene.sled.data.root_quat_w, lp)
        st[:, 3:7] = scene.sled.data.root_quat_w
        return st

    z_cell = c.deck_top + c.cube_s / 2 + 0.004

    def build_pyramid_on_cart() -> None:
        """CONSTRUCT the real built state the solve produces: drop red+green into
        the tray cells, then blue bridging the settled pair — all real physics."""
        _write_state(scene.cubes["cube_red"], cart_pose(c.tray_cx + c.cell_dx, 0.0, z_cell))
        _step(60)
        _write_state(scene.cubes["cube_green"], cart_pose(c.tray_cx - c.cell_dx, 0.0, z_cell))
        _step(90)
        loc = scene.cubes_sled_local()[0]
        mid = (loc[0] + loc[1]) / 2
        _write_state(scene.cubes["cube_blue"],
                     cart_pose(float(mid[0]), float(mid[1]),
                               float(mid[2]) + c.cube_s + 0.003))
        _step(120)

    # ================= 1. settle / baseline =======================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(120)
    _report("settle")
    _REC["on"] = False
    pos = torch.stack([scene.cubes[nm].data.root_pos_w[0] for nm in scene.CUBE_NAMES])
    z_ground = (pos[:, 2] - scene.env_origins[0, 2] - c.cube_s / 2).abs().max()
    sx = float(scene.sled_fix_local()[0, 0])
    check("settle/baseline: layout settles finite; cubes on the ground, cart out on "
          "the rails; score 0, no success",
          bool(scene._finite()[0]) and float(z_ground) < 0.01
          and 0.25 < sx < 0.45
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (scene.garage.data.root_pos_w[0, 0:2].clone(),
                yaw_of(scene.garage.data.root_quat_w[0]),
                float(scene.sled_fix_local()[0, 0]),
                scene.cubes["cube_red"].data.root_pos_w[0, 0:2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_g, a_yaw, a_sx, a_red = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_g, b_yaw, b_sx, b_red = readback()
    d_g = float((a_g - b_g).norm())
    d_yawv = dyaw(a_yaw, b_yaw)
    d_sx = abs(a_sx - b_sx)
    d_red = float((a_red - b_red).norm())
    print(f"[smoke] randomization deltas: depot_xy={d_g * 1000:.1f}mm "
          f"depot_yaw={d_yawv:.1f}deg sled_x={d_sx * 1000:.1f}mm "
          f"red_xy={d_red * 1000:.1f}mm", flush=True)
    check("randomization-is-real: depot xy + yaw, cart start depot-x, red-cube xy "
          "readback differ between seeds",
          d_g > 0.003 and d_yawv > 2.0 and d_sx > 0.002 and d_red > 0.003)

    # ================= 3. slot-permutation coverage ===============================================
    perms = set()
    for s in range(8):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        perms.add(tuple(scene.slot_perm[0].tolist()))
    print(f"[smoke] over 8 resets: {len(perms)} distinct cube-slot permutations "
          f"{sorted(perms)}", flush=True)
    check("slot permutation: >= 3 distinct cube-to-slot assignments over 8 resets — "
          "which cube lies where must be perceived, not memorized",
          len(perms) >= 3)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED'S OWN STRATEGY =======================================
    # "Build the pyramid in place at the goal" — the seed's whole plan. CONSTRUCT a
    # REAL settled pyramid on the GROUND inside the depot pocket (drops, physics):
    # every clause of the rubric judges cubes ON THE CART, so this scores ~0.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_state(scene.cubes["cube_red"], depot_pose(0.028, 0.0, c.cube_s / 2 + 0.004))
    _step(50)
    _write_state(scene.cubes["cube_green"], depot_pose(0.072, 0.0, c.cube_s / 2 + 0.004))
    _step(50)
    _write_state(scene.cubes["cube_blue"], depot_pose(0.050, 0.0, c.cube_s * 1.5 + 0.004))
    _step(120)
    _report("seed-strategy")
    blue_z = float(scene.cubes["cube_blue"].data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    blue_dep = scene.point_fix_local(scene.cubes["cube_blue"].data.root_pos_w[:, 0:2])[0]
    ground_pyramid = abs(blue_z - c.cube_s * 1.5) < 0.012 \
        and 0.0 < float(blue_dep[0]) < 0.115
    check("negative (SEED strategy): a real settled pyramid on the GROUND inside "
          "the depot pocket — cube-on-cart clauses refuse, no success, score <= 0.01",
          ground_pyramid and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.01)
    _REC["on"] = False

    # ================= 6. built but not delivered =================================================
    # The mid-solve state: real pyramid on the cart, cart still out on the rails.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    build_pyramid_on_cart()
    _report("built-not-deliv")
    s6 = float(scene.score()[0])
    check("built-not-delivered: real pyramid on the cart out on the rails — seated+"
          "built latch (score ~0.45) but no delivery credit, no success",
          bool(scene.pyramid_ok()[0]) and bool(scene._built[0])
          and not bool(scene.delivered_ok()[0]) and not bool(scene.success()[0])
          and 0.44 <= s6 <= 0.46)
    _REC["on"] = False

    # ================= 7. empty cart delivered ====================================================
    # The cart parked inside the pocket with NO cargo: the delivered_ok geometry
    # holds, but the delivery latch is conditioned on the pyramid riding the cart.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_state(scene.sled, depot_pose(0.082, 0.0, 0.002, extra_yaw=math.pi))
    _step(120)
    _report("empty-cart")
    check("empty-cart-delivered: bare cart rests inside the pocket (delivered_ok "
          "geometry true) but the delivery latch refuses without the pyramid — "
          "no success, score <= 0.01",
          bool(scene.delivered_ok()[0]) and not bool(scene._deliv[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.01)

    # ================= 8. wrong cube on top =======================================================
    # Red + BLUE seated in the tray, GREEN bridging on top: same shape, wrong color
    # roles (the seed names blue as the bridge).
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_state(scene.cubes["cube_red"], cart_pose(c.tray_cx + c.cell_dx, 0.0, z_cell))
    _step(60)
    _write_state(scene.cubes["cube_blue"], cart_pose(c.tray_cx - c.cell_dx, 0.0, z_cell))
    _step(90)
    _write_state(scene.cubes["cube_green"],
                 cart_pose(c.tray_cx, 0.0, c.deck_top + 1.5 * c.cube_s + 0.003))
    _step(120)
    _report("wrong-top")
    grn_z = float(scene.cubes_sled_local()[0, 1, 2])
    check("wrong-cube-on-top: red+blue seated, GREEN bridging on top (a real "
          "settled pyramid, wrong roles) — base/bridge color clauses refuse, "
          "no success, score <= 0.01",
          abs(grn_z - (c.deck_top + 1.5 * c.cube_s)) < c.z_tol
          and not bool(scene.pyramid_ok()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.01)

    # ================= 9. near-miss: blue on ONE base cube ========================================
    # Blue parked square on RED alone (same height as the true bridge, 21 mm off the
    # pair midpoint > span_tol 10 mm): seated credit only.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_state(scene.cubes["cube_red"], cart_pose(c.tray_cx + c.cell_dx, 0.0, z_cell))
    _step(60)
    _write_state(scene.cubes["cube_green"], cart_pose(c.tray_cx - c.cell_dx, 0.0, z_cell))
    _step(90)
    loc = scene.cubes_sled_local()[0]
    _write_state(scene.cubes["cube_blue"],
                 cart_pose(float(loc[0, 0]), float(loc[0, 1]),
                           float(loc[0, 2]) + c.cube_s + 0.003))
    _step(120)
    _report("one-cube-top")
    blu = scene.cubes_sled_local()[0, 2]
    mid_x = float((scene.cubes_sled_local()[0, 0, 0] + scene.cubes_sled_local()[0, 1, 0]) / 2)
    off = abs(float(blu[0]) - mid_x)
    s9 = float(scene.score()[0])
    print(f"[smoke] one-cube-top: blue x-offset from pair midpoint {off * 1000:.1f}mm "
          f"(span_tol {c.span_tol * 1000:.0f}mm)", flush=True)
    check("near-miss span: blue rests at bridge height but entirely on ONE base "
          "cube (offset > span_tol) — pyramid refused, seated credit only, "
          "no success, score <= 0.21",
          off > c.span_tol and bool(scene._seated[0])
          and not bool(scene.pyramid_ok()[0]) and not bool(scene.success()[0])
          and s9 <= c.w_seated + 0.01)

    # ================= 10. ROOF certificate (build-inside is impossible) ==========================
    # Release blue from open air above the pocket: it lands on the ROOF and stays
    # there — the depot is closed from above, so no cube can be brought down onto a
    # stack inside. This is the geometric fact that forces build-then-deliver.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_state(scene.cubes["cube_blue"], depot_pose(0.055, 0.0, 0.22))
    _step(150)
    _report("roof-drop")
    blue_z = float(scene.cubes["cube_blue"].data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    blue_dep = scene.point_fix_local(scene.cubes["cube_blue"].data.root_pos_w[:, 0:2])[0]
    print(f"[smoke] roof-drop: blue settled at depot=({float(blue_dep[0]):+.3f},"
          f"{float(blue_dep[1]):+.3f}) z={blue_z:.3f} (roof top "
          f"{c.wall_h + 0.012:.3f})", flush=True)
    check("ROOF certificate: a cube released above the pocket lands ON the roof "
          "(settles z > 0.12, never inside) — building inside the depot is "
          "physically impossible, no credit",
          blue_z > 0.12 and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 11. SHOVE certificate (gentle transport is load-bearing) ===================
    # Real pyramid on the cart, then slam it: 12 N on the 0.40 kg load = ~30 m/s^2,
    # far over the mu*g ~ 8.8 m/s^2 slip threshold of the free-riding blue cube.
    # The cart arrives at the pocket (asserted — the probe actuates for real), but
    # the pyramid does not: no delivery latch, no success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    build_pyramid_on_cart()
    assert bool(scene.pyramid_ok()[0]), "shove probe needs the built pyramid first"
    door_w = torch.stack([-torch.cos(scene.fix_yaw), -torch.sin(scene.fix_yaw),
                          torch.zeros_like(scene.fix_yaw)], dim=-1)
    f = (12.0 * door_w).unsqueeze(1)
    tau0 = torch.zeros(n, 1, 3, device=device)
    for i in range(240):
        x_dep = float(scene.sled_fix_local()[0, 0])
        if x_dep < 0.18:
            break
        scene.sled.set_external_force_and_torque(f, tau0, env_ids=_all_ids(),
                                                 is_global=True)
        _step(1)
    scene.sled.set_external_force_and_torque(torch.zeros_like(f), tau0,
                                             env_ids=_all_ids(), is_global=True)
    _step(240)
    _report("shove")
    x_dep = float(scene.sled_fix_local()[0, 0])
    s11 = float(scene.score()[0])
    check("SHOVE certificate: 12 N slam — the cart itself reaches the depot mouth "
          "(depot-x < 0.15: the probe really actuates) but the blue cube is thrown "
          "off, pyramid broken, no delivery latch, no success, score <= 0.46",
          x_dep < 0.15 and not bool(scene.pyramid_ok()[0])
          and not bool(scene._deliv[0]) and not bool(scene.success()[0])
          and s11 <= 0.46)
    _REC["on"] = False

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.pyramid_freight")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    check("video: frames captured and saved to frames.npz", len(_REC["frames"]) > 10)

    n_pass = sum(ok for _n, ok in checks)
    n_tot = len(checks)
    if n_pass == n_tot:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{n_tot}", flush=True)
        code = 0
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{n_tot}", flush=True)
        code = 1

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
    except Exception as exc:  # noqa: BLE001 — fail fast, don't hang until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({exc})", flush=True)
        os._exit(1)
