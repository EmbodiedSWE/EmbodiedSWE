"""Smoke / rubric-REJECTION battery for CarouselCabinetScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the latched
credit is monotone along a real crank-driven trajectory). This battery proves the
rubric REJECTS wrong outcomes, and that the geometric claims the task rests on — the
roof makes the window the only way in, the divider walls make sector membership
rotation-only — are physically load-bearing. Every probe is CONSTRUCTED as a settled
state (teleport, real physics steps, judge) — instrumentation, never a solution: no
probe here reaches success().

Checks:
   1. settle/no-NaN      — seeded reset settles finite; green sector 40..150 deg away
                           from the window, bottles outside; score 0, no success;
   2. randomization      — two seeded resets: READBACK carousel start angle and
                           ketchup xy differ;
   3. side swap          — over 10 resets the ketchup occupies BOTH ground slots;
   4. null-policy        — 240 idle steps: the carousel holds its angle, score ~0,
                           no success;
   5. ROOF INTERLOCK     — the bottle dropped from ABOVE the roof hole falls onto the
                           roof/shaft and NEVER enters the housing (the `inserted`
                           latch never fires): the window really is the only way in;
   6. DIVIDER INTERLOCK  — bottle seated in the green sector (at the window), crank
                           held by an angle servo (a hand on the crank), bottle
                           shoved tangentially at 3x its weight for 1.5 s: it slides
                           into the divider but NEVER leaves the green wedge —
                           sector membership cannot be changed without rotating;
   7. SEED STRATEGY      — the seed's plan ("drop the bottle into the open
                           receptacle"): ketchup dropped into whatever sector faces
                           the window WITHOUT aligning first -> lands in a non-green
                           sector, no success, score <= 0.25;
   8. wrong sector       — ketchup seated in a NON-green sector constructed to face
                           the BACK (a tidy episode of the wrong sector) -> no
                           success;
   9. wrong object       — MUSTARD stowed in the green sector at the back, ketchup
                           left outside -> no success, score ~0;
  10. near-miss stow     — ketchup seated in the green sector but the sector left
                           40 deg short of the back (outside `stow_tol_deg`) -> no
                           success, score <= 0.85;
  11. tipped bottle      — ketchup lying on its SIDE on the green mat at the back ->
                           upright clause refuses, no success;
  12. smuggled mustard   — ketchup PERFECTLY stowed in the green sector at the back,
                           but the mustard placed inside another sector -> no success
                           (the mustard-outside clause is load-bearing);
  13. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.<task>.smoke --headless
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

_qy, _qz, _wrap_deg = task_scene._qy, task_scene._qz, task_scene._wrap_deg

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

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


def _write_body(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    az = float(scene.green_az_deg()[0])
    kk = (scene.ketchup.data.root_pos_w - scene.env_origins)[0]
    print(f"[smoke] {tag:18s} | green_az={az:+7.1f}deg "
          f"ketchup=({float(kk[0]):+.3f},{float(kk[1]):+.3f},{float(kk[2]):+.3f}) "
          f"in_green={bool(scene.in_green_sector()[0])} stowed={bool(scene.stowed()[0])} "
          f"must_out={bool(scene.mustard_outside()[0])} "
          f"lat(a/i/s)={int(scene._aligned[0])}{int(scene._inserted[0])}"
          f"{int(scene._seated[0])} prog={float(scene._prog[0]):.2f} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


def _push(body, force_w: torch.Tensor, steps: int, *, hold_az: float | None = None) -> None:
    """Apply a world-frame force at the body's CoM for `steps` steps, then clear.
    If `hold_az` is given, an angle-hold servo on the crank buffer keeps the carousel
    at that azimuth throughout (a hand holding the crank still)."""
    env = _ENV
    scene = env.scene
    n = env.num_envs
    zero = torch.zeros(n, 1, 3, device=env.device)
    f = force_w.view(1, 1, 3).expand(n, 1, 3).contiguous()
    for _ in range(steps):
        if hold_az is not None:
            err = math.radians(float(_wrap_deg(torch.tensor(
                [hold_az - float(scene.green_az_deg()[0])]))[0]))
            w = float(scene.carousel.data.root_ang_vel_w[0, 2])
            scene.crank_tau[:] = max(-0.6, min(0.6, 1.0 * err + 0.3 * (0.0 - w)))
        body.set_external_force_and_torque(f, zero, env_ids=_all_ids(), is_global=True)
        _step(1)
    body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    scene.crank_tau[:] = 0.0


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.carousel_cabinet")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.62, -0.95, 0.72)) + o),
                                tuple(np.array((0.62, 0.00, 0.18)) + o),
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

    def az() -> float:
        _refresh()
        return float(scene.green_az_deg()[0])

    def az_local(body) -> float:
        """Body azimuth in the CAROUSEL frame (0 = green sector centre), degrees."""
        _refresh()
        rel = scene._hub_rel(body)
        a = math.degrees(math.atan2(float(rel[0, 1]), float(rel[0, 0])))
        return float(_wrap_deg(torch.tensor([a - az()]))[0])

    def set_carousel(az_deg: float) -> None:
        """CONSTRUCT the carousel at azimuth `az_deg` (a rotation about the very
        bearing axis the joint constrains — consistent with the world anchor)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = c.hub_pos[0]
        st[:, 1] = c.hub_pos[1]
        st[:, 3:7] = _qz(torch.full((n,), math.radians(az_deg), device=device))
        st[:, 0:3] += scene.env_origins
        scene.carousel.write_root_state_to_sim(st, _all_ids())
        _refresh()

    def bottle_at(body, world_az_deg: float, *, r: float = 0.095,
                  tipped: bool = False, hover: float = 0.012) -> None:
        """Drop `body` at housing azimuth `world_az_deg`, radius `r`, onto the
        platform (upright, or on its side if `tipped`)."""
        rad = math.radians(world_az_deg)
        p = torch.zeros(n, 3, device=device)
        p[:, 0] = c.hub_pos[0] + r * math.cos(rad)
        p[:, 1] = c.hub_pos[1] + r * math.sin(rad)
        p[:, 2] = c.z_plat + c.mat_t + (c.bot_r if tipped else c.bot_h / 2) + hover
        p += scene.env_origins
        q = _qy(torch.full((n,), math.pi / 2, device=device)) if tipped else None
        _write_body(body, p, q)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    a0 = abs(az())
    check("settle/no-NaN: layout settles finite; green sector 40..150 deg from the "
          "window, bottles outside; score 0, no success",
          bool(scene._finite()[0]) and c.yaw_min_deg - 8.0 <= a0 <= c.yaw_max_deg + 8.0
          and not bool(scene._inside(scene.ketchup)[0])
          and bool(scene.mustard_outside()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return az(), scene.ketchup.data.root_pos_w[0, :2].clone()

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_az, a_kp = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_az, b_kp = readback()
    d_az = abs(float(_wrap_deg(torch.tensor([a_az - b_az]))[0]))
    d_kp = float((a_kp - b_kp).norm())
    print(f"[smoke] randomization deltas: green_az={d_az:.1f}deg "
          f"ketchup_xy={d_kp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: carousel start angle and ketchup xy readback differ",
          d_az > 2.0 and d_kp > 0.003)

    # ================= 3. bottle side swap ========================================================
    sides = set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        sides.add("+" if float(scene.ketchup_side[0]) > 0 else "-")
    print(f"[smoke] over 10 resets: ketchup slots {sorted(sides)}", flush=True)
    check("side swap: the ketchup occupies BOTH ground slots over 10 resets",
          sides == {"+", "-"})

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(10)
    az_start = az()
    _step(240)
    _report("null-policy")
    drift = abs(float(_wrap_deg(torch.tensor([az() - az_start]))[0]))
    check("null-policy-fails: 240 idle steps, the carousel holds its angle "
          f"(drift {drift:.2f}deg), score ~0, no success",
          drift < 2.0 and float(scene.score()[0]) <= 0.05
          and not bool(scene.success()[0]))

    # ================= 5. ROOF INTERLOCK: the window is the only way in ===========================
    # Drop the ketchup from ABOVE the roof's centre hole (offset so it meets the hole
    # edge, not the crank arm). The hole (84 mm square) minus the 30 mm shaft leaves
    # a 27 mm gap — a 60 mm bottle cannot pass. The `inserted` LATCH doubles as a
    # transient detector: it must never fire.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    p = torch.zeros(n, 3, device=device)
    p[:, 0] = c.hub_pos[0] + 0.045
    p[:, 1] = c.hub_pos[1]
    p[:, 2] = c.roof_z + c.roof_t + c.bot_h / 2 + 0.12
    p += scene.env_origins
    z_start = float(p[0, 2])
    _write_body(scene.ketchup, p, None)
    _step(240)
    _report("roof-drop")
    _REC["on"] = False
    rel = scene._hub_rel(scene.ketchup)[0]
    z_end = float(scene.ketchup.data.root_pos_w[0, 2])
    print(f"[smoke] roof drop: fell {1000 * (z_start - z_end):.0f}mm, ended at "
          f"r={float(rel[:2].norm()) * 1000:.0f}mm z={float(rel[2]) * 1000:.0f}mm",
          flush=True)
    check("ROOF INTERLOCK: bottle dropped from above the roof hole falls (probe "
          "moved) but NEVER enters the housing — the inserted latch never fires",
          z_start - z_end > 0.05 and not bool(scene._inserted[0])
          and not bool(scene._inside(scene.ketchup)[0])
          and not bool(scene.success()[0]))

    # ================= 6. DIVIDER INTERLOCK: sectors are rotation-only ============================
    # Bottle seated in the green sector at the window; the crank held still by an
    # angle servo (a hand on the crank); the bottle shoved TANGENTIALLY at 3x its
    # weight for 1.5 s. It topples against the divider — but its carousel-frame
    # azimuth never leaves the green wedge (dividers at +/-60 deg).
    torch.manual_seed(100)
    env.reset()
    _step(30)
    set_carousel(0.0)
    bottle_at(scene.ketchup, 0.0)
    _step(90)
    _refresh()
    assert bool(scene.in_green_sector()[0]), "probe setup: bottle must seat in green"
    p_seat = scene.ketchup.data.root_pos_w[0].clone()
    _REC["on"] = True
    f = torch.tensor([0.0, 3.0 * c.bot_mass * 9.81, 0.0], device=device)
    for _ in range(9):  # 9 x 20 = 180 steps = 1.5 s of shoving
        _push(scene.ketchup, f, 20, hold_az=0.0)
    _step(60)
    _report("divider-shove")
    _REC["on"] = False
    al = az_local(scene.ketchup)
    disp = float((scene.ketchup.data.root_pos_w[0, :2] - p_seat[:2]).norm())
    print(f"[smoke] divider shove: bottle carousel-frame az={al:+.1f}deg "
          f"(wedge +/-60), displaced {disp * 1000:.0f}mm into the divider", flush=True)
    check("DIVIDER INTERLOCK: 3x-weight tangential shove for 1.5 s drives the "
          "bottle into the divider (probe moved) but it NEVER leaves the green wedge",
          disp > 0.03 and abs(al) < 60.0 and bool(scene._inside(scene.ketchup)[0])
          and not bool(scene.success()[0]))

    # ================= 7. negative: the SEED'S OWN STRATEGY =======================================
    # "Put the bottle into the open receptacle" — drop the ketchup into whatever
    # sector faces the window WITHOUT aligning first. It lands on a bare (non-green)
    # floor: inserted-latch credit only, no seat, no success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    set_carousel(130.0)  # green well away; the facing sector is bare
    _REC["on"] = True
    bottle_at(scene.ketchup, 0.0)
    _step(120)
    _report("seed-strategy")
    _REC["on"] = False
    check("negative (SEED strategy): bottle dropped into the sector facing the "
          "window without aligning — a non-green floor: no success, score <= 0.25",
          bool(scene._inside(scene.ketchup)[0])
          and not bool(scene.in_green_sector()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.25)

    # ================= 8. negative: wrong sector stowed ===========================================
    # A tidy full episode — of the WRONG sector: bottle seated in a bare sector and
    # THAT sector rotated to the back (green ends up off-back).
    torch.manual_seed(100)
    env.reset()
    _step(30)
    set_carousel(60.0)  # sector centred at carousel-local -120 now faces the back
    bottle_at(scene.ketchup, 180.0)
    _step(120)
    _report("wrong-sector")
    check("negative (wrong sector): ketchup seated in a non-green sector at the "
          "BACK — no success, score <= 0.25",
          bool(scene._inside(scene.ketchup)[0])
          and not bool(scene.in_green_sector()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.25)

    # ================= 9. negative: wrong object ==================================================
    # The MUSTARD stowed in the green sector at the back, ketchup left outside — a
    # complete, tidy episode of the wrong bottle.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    set_carousel(180.0)
    bottle_at(scene.mustard, 180.0)
    _step(120)
    _report("wrong-object")
    check("negative (wrong object): MUSTARD stowed in the green sector at the back, "
          "ketchup outside — no success, score ~0",
          bool(scene._inside(scene.mustard)[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.05)

    # ================= 10. near-miss: stowed 40 deg short =========================================
    # Ketchup PERFECTLY seated in the green sector — but the sector parked 40 deg
    # short of the back (outside `stow_tol_deg`): the one failed clause.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    set_carousel(140.0)
    bottle_at(scene.ketchup, 140.0)
    _step(120)
    _report("near-miss-stow")
    adist = 180.0 - abs(az())
    print(f"[smoke] near-miss stow: green sector {adist:.1f}deg short of the back "
          f"(tol {c.stow_tol_deg:.0f}deg)", flush=True)
    check("near-miss (stow): ketchup seated in the green sector left 40 deg short "
          "of the back — no success, score <= 0.85",
          bool(scene.in_green_sector()[0]) and adist > c.stow_tol_deg
          and not bool(scene.stowed()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.85)

    # ================= 11. negative: tipped bottle ================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    set_carousel(180.0)
    bottle_at(scene.ketchup, 180.0, tipped=True)
    _step(120)
    _report("tipped")
    check("negative (tipped): ketchup lying on its SIDE on the green mat at the "
          "back — upright clause refuses, no success",
          bool(scene._inside(scene.ketchup)[0])
          and not bool(scene.in_green_sector()[0])
          and not bool(scene.success()[0]))

    # ================= 12. negative: smuggled mustard =============================================
    # Ketchup PERFECTLY stowed — but the mustard smuggled into another sector: the
    # mustard-outside clause is load-bearing.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    set_carousel(180.0)
    bottle_at(scene.ketchup, 180.0)
    _REC["on"] = True
    bottle_at(scene.mustard, 60.0)
    _step(120)
    _report("smuggled")
    _REC["on"] = False
    check("negative (smuggled mustard): ketchup perfectly stowed in the green "
          "sector at the back BUT the mustard inside another sector — no success",
          bool(scene.in_green_sector()[0]) and bool(scene.stowed()[0])
          and not bool(scene.mustard_outside()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.85)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.carousel_cabinet")
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
    main()
