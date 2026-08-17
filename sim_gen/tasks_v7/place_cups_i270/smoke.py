"""Smoke / rubric-REJECTION battery for CargoShuttleScene (sim_gen task
`place_cups_i270`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — seat red/green/blue by gravity drops into their
color-matched sockets, then force-push the loaded shuttle under the canopy to the end
stop — is the acceptance evidence that the rubric ACCEPTS a correct outcome; it passes
on seeds 0/1). Every teleport here is instrumentation that CONSTRUCTS a wrong (or
partial) outcome as a settled state and asserts the rubric REJECTS it — including the
canopy-reality probe that proves the load-first ORDER is geometry-enforced (a canister
dropped over a docked socket lands ON the roof). No probe in this battery ever reaches
success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN     — reset layout settles finite: canisters upright on the
                           floor, shuttle at rest in the start band, score ~0;
  3-4. randomization     — READBACK over 6 seeded resets: canister masses vary (PhysX
                           view, independent of the reset cache), the shuttle start x
                           varies, the ground-slot permutation varies, xy jitter real;
  5.  null policy        — 240 idle steps -> score ~0, no success;
  6.  seed-analog        — the seed family's end state (each object resting at its own
                           dedicated STATIC location): three canisters in a neat row on
                           the static bay roof, shuttle untouched -> nothing seated,
                           NOT success;
  7.  loaded, not docked — all three seated in their matched sockets but the shuttle
                           still in the open loading zone -> all_seated() True yet NOT
                           success, latched score == 0.50 (placement is only half);
  8.  docked, empty      — shuttle teleported to the end stop with every canister left
                           on the floor -> docked() True yet NOT success, score <= 0.15;
  9.  wrong colors       — the three canisters physically seated but cyclically
                           MISMATCHED (red in green socket, ...): in-pocket z readback,
                           yet seated() False for all three, score ~0;
  10. stacked            — red dropped ON TOP of the seated green: rides at local
                           z ~0.100, above the socket-floor window -> rejected;
  11. toppled in socket  — blue lying on its side INSIDE the blue socket: z readback
                           inside the seat window (same height as seated!) yet the
                           upright cone rejects it;
  12. near-dock miss     — fully loaded shuttle placed 40 mm short of the dock line:
                           all_seated() True, docked() False -> NOT success, no dock
                           credit;
  13. latched credit     — removing a seated canister from that shuttle leaves the
                           latched score unchanged while all_seated() drops;
  14. canopy reality     — shuttle docked EMPTY, a canister dropped from above the
                           blue socket: it lands ON the roof (world-z readback), local
                           xy IS over the socket yet the z window rejects it — loading
                           after docking is physically impossible;
  15. settle gate        — the full goal state (loaded + docked) judged 2 steps after
                           construction: still_count below the persistence gate -> NOT
                           success; a canister is removed before the gate can fill;
  16. rejection audit    — success() was never True at ANY judged point;
  17. final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.place_cups_i270.smoke --headless
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cargo_shuttle")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.95, -1.10, 0.80)) + o),
                                tuple(np.array((0.00, -0.05, 0.10)) + o),
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

    def sx() -> float:
        return float(scene.shuttle_x()[0])

    def report(tag: str) -> None:
        s, ok = judge()
        bits = []
        for nm in c.cup_names:
            loc = scene._cup_local(nm)[0]
            bits.append(f"{nm}=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
                        f"{float(loc[2]):+.3f})s in={bool(scene.seated(nm)[0])}")
        print(f"[smoke] {tag:16s} | shuttle_x={sx():+.3f} docked={bool(scene.docked()[0])}"
              f" | " + " ".join(bits) + f" score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def write_cup(nm: str, world: torch.Tensor, quat: torch.Tensor,
                  settle_steps: int = 60) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = world
        st[:, 3:7] = quat
        scene.cups[nm].write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def drop_cup(nm: str, socket_i: int, z_extra: float = 0.020,
                 settle_steps: int = 150) -> None:
        """Teleport `nm` above socket `socket_i`'s floor (LIVE shuttle pose) and let
        contact seat it — the same transport-only move solve.py uses."""
        from isaaclab.utils.math import quat_apply

        sp = scene.shuttle.data.root_pos_w[0]
        sq = scene.shuttle.data.root_quat_w[0]
        local = torch.tensor([c.socket_xs[socket_i], 0.0, c.seat_rest_z + z_extra],
                             device=device)
        world = sp + quat_apply(sq.unsqueeze(0), local.unsqueeze(0))[0]
        write_cup(nm, world, sq, settle_steps)

    def to_floor(nm: str, x: float, y: float, settle_steps: int = 60) -> None:
        world = torch.tensor([x, y, c.cup_h / 2 + 0.003], device=device) \
            + scene.env_origins[0]
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        write_cup(nm, world, q, settle_steps)

    def shift_assembly(target_x: float, settle_steps: int = 60,
                       cups_aboard: bool = True, vx: float = 0.0) -> None:
        """Translate the shuttle (and, coherently, every canister riding it) along the
        channel to body x = `target_x` — write the WHOLE linkage in one frame
        (teleporting one body of a resting group gets depenetrated back). A nonzero
        `vx` constructs a JUST-ARRIVING assembly (real motion resets the stillness
        counter — a zero-velocity teleport would inherit the pre-teleport still
        streak)."""
        dx = target_x - sx()
        bodies = [scene.shuttle] + ([scene.cups[nm] for nm in c.cup_names]
                                    if cups_aboard else [])
        for b in bodies:
            st = b.data.root_state_w.clone()
            st[:, 0] += dx
            st[:, 7:13] = 0.0
            st[:, 7] = vx
            b.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    bodies = (scene.shuttle, *scene.cups.values())
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    z_ok = all(abs(float(scene.cups[nm].data.root_pos_w[0, 2]) - c.cup_h / 2) < 0.01
               for nm in c.cup_names)
    x0 = sx()
    check("settle: all states finite, canisters upright on the floor, shuttle at rest "
          f"in the start band (x={x0:+.3f} in [{c.shuttle_x0[0]:.2f},"
          f"{c.shuttle_x0[1]:.2f}])",
          fin and z_ok and c.shuttle_x0[0] - 0.02 <= x0 <= c.shuttle_x0[1] + 0.02
          and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(20)
        m_cache = scene.cup_mass[0]
        # readback mass through the view AGAIN (independent of the reset-path cache)
        m_view = [float(scene.cups[nm].root_physx_view.get_masses().reshape(-1)[0])
                  for nm in c.cup_names]
        xs = [float((scene.cups[nm].data.root_pos_w - scene.env_origins)[0, 0])
              for nm in c.cup_names]
        slot = tuple(int(np.argmin([abs(x - sxx) for sxx in c.slot_xs])) for x in xs)
        reads.append((m_view, slot, xs[0], sx(),
                      max(abs(float(m_cache[i]) - m_view[i]) for i in range(3))))
        print(f"[smoke] seed {sd}: masses=({m_view[0]:.4f},{m_view[1]:.4f},"
              f"{m_view[2]:.4f}) slots={slot} red_x={xs[0]:+.3f} "
              f"shuttle_x={sx():+.3f} cache_err={reads[-1][4]:.2e}", flush=True)
    m_spread = [max(r[0][i] for r in reads) - min(r[0][i] for r in reads)
                for i in range(3)]
    cache_ok = all(r[4] < 1e-5 for r in reads)
    check("randomization: canister masses vary across seeded resets (VIEW readback "
          f"spreads r/g/b = {m_spread[0] * 1000:.1f}/{m_spread[1] * 1000:.1f}/"
          f"{m_spread[2] * 1000:.1f} g) and the reset cache matches the view",
          all(sp > 0.005 for sp in m_spread) and cache_ok)
    slots = {r[1] for r in reads}
    sx_spread = max(r[3] for r in reads) - min(r[3] for r in reads)
    rx_spread = max(r[2] for r in reads) - min(r[2] for r in reads)
    check("randomization: shuttle start x varies (spread "
          f"{sx_spread * 1000:.0f} mm), ground-slot permutation varies (readback: "
          f"{len(slots)} distinct / 6), xy jitter real (red x spread "
          f"{rx_spread * 1000:.0f} mm)",
          sx_spread > 0.008 and len(slots) >= 2 and rx_spread > 0.008)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed-analog: a row on the static structure =============
    # The seed family's end state: each object resting at its own dedicated STATIC
    # location. Nearest analog here: the three canisters set down in a neat row on the
    # static bay roof, shuttle untouched in the loading zone.
    env.reset(seed=41)
    step(30)
    roof_top = c.roof_bot + 0.020  # roof slab is 20 mm thick
    q0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    for i, nm in enumerate(c.cup_names):
        world = torch.tensor([0.20 + 0.08 * i, 0.0, roof_top + c.cup_h / 2 + 0.003],
                             device=device) + scene.env_origins[0]
        write_cup(nm, world, q0, settle_steps=30)
    step(90)
    report("seed-analog")
    s, ok = judge()
    none_seated = not any(bool(scene.seated(nm)[0]) for nm in c.cup_names)
    z_roof = float(scene.cups["green"].data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    check("seed-analog (each canister at its own static spot: a neat row ON the bay "
          f"roof, green z={z_roof:+.3f}): nothing seated, NOT success, score ~0",
          none_seated and z_roof > c.roof_bot and not ok and s <= 0.02)

    # =========================== 7. loaded but NOT docked ===================================
    env.reset(seed=51)
    step(30)
    for i, nm in enumerate(c.cup_names):
        drop_cup(nm, i)
    report("loaded-undocked")
    s, ok = judge()
    check("loaded, not docked: all three seated in their matched sockets "
          f"(all_seated True) but the shuttle sits at x={sx():+.3f} < "
          f"dock_x={c.dock_x:.3f} -> NOT success, latched score == 0.50",
          bool(scene.all_seated()[0]) and not bool(scene.docked()[0]) and not ok
          and 0.45 <= s <= 0.55)

    # =========================== 8. docked but EMPTY ========================================
    env.reset(seed=61)
    step(30)
    shift_assembly(c.dock_rest_x - 0.004, settle_steps=90, cups_aboard=False)
    report("docked-empty")
    s, ok = judge()
    check("docked, empty: shuttle pressed to the end stop (docked True, "
          f"x={sx():+.3f}) with every canister on the floor -> NOT success, "
          "score <= 0.15",
          bool(scene.docked()[0]) and not bool(scene.all_seated()[0]) and not ok
          and s <= 0.15)

    # =========================== 9. wrong colors ============================================
    env.reset(seed=71)
    step(30)
    # cyclic mismatch: red -> green socket, green -> blue socket, blue -> red socket
    for nm, si in (("red", 1), ("green", 2), ("blue", 0)):
        drop_cup(nm, si)
    report("wrong-colors")
    s, ok = judge()
    in_pocket = all(c.seat_z_lo <= float(scene._cup_local(nm)[0, 2]) <= c.seat_z_hi
                    for nm in c.cup_names)
    none_seated = not any(bool(scene.seated(nm)[0]) for nm in c.cup_names)
    check("wrong colors (cyclic mismatch): all three physically IN pockets (z "
          "readback inside the seat window) yet seated() False for every canister "
          "-> score ~0, NOT success",
          in_pocket and none_seated and not ok and s <= 0.02)

    # =========================== 10. stacked ================================================
    env.reset(seed=81)
    step(30)
    drop_cup("green", 1)
    # red ON TOP of the seated green (flat-on-flat): rides ~ one canister too high
    drop_cup("red", 1, z_extra=c.cup_h + 0.008, settle_steps=180)
    report("stacked")
    z_red = float(scene._cup_local("red")[0, 2])
    s, ok = judge()
    check("stacked: red dropped on top of the seated green rides at local "
          f"z={z_red:+.3f} (window [{c.seat_z_lo:.3f},{c.seat_z_hi:.3f}]) -> "
          "rejected, green alone still seated, score <= 0.15",
          z_red > c.seat_z_hi + 0.02 and not bool(scene.seated("red")[0])
          and bool(scene.seated("green")[0]) and not ok and s <= 0.15)

    # =========================== 11. toppled in the socket ==================================
    env.reset(seed=91)
    step(30)
    from isaaclab.utils.math import quat_apply, quat_mul

    sp = scene.shuttle.data.root_pos_w[0]
    sq = scene.shuttle.data.root_quat_w[0]
    # blue lying on its SIDE inside the blue socket (axis along the channel; the
    # 60 mm length fits the 78 mm pocket): center rests at r above the socket floor
    local = torch.tensor([c.socket_xs[2], 0.0, c.deck_t / 2 + c.cup_r + 0.004],
                         device=device)
    world = sp + quat_apply(sq.unsqueeze(0), local.unsqueeze(0))[0]
    rot_y90 = torch.tensor([0.70710678, 0.0, 0.70710678, 0.0], device=device)
    q_lying = quat_mul(sq.unsqueeze(0), rot_y90.unsqueeze(0))[0]
    write_cup("blue", world, q_lying, settle_steps=150)
    report("toppled")
    loc = scene._cup_local("blue")[0]
    updot = float(scene._cup_updot("blue")[0])
    s, ok = judge()
    check("toppled in socket: blue lying on its side INSIDE the blue socket — z "
          f"readback {float(loc[2]):+.3f} is INSIDE the seat window (same height as "
          f"seated!) yet the upright cone rejects it (updot={updot:+.2f} < "
          f"{c.upright_min_dot:.2f}) -> NOT seated, score ~0",
          c.seat_z_lo <= float(loc[2]) <= c.seat_z_hi and abs(loc[0] - c.socket_xs[2]) <= c.socket_tol
          and updot < c.upright_min_dot and not bool(scene.seated("blue")[0])
          and not ok and s <= 0.02)

    # =========================== 12. near-dock miss =========================================
    env.reset(seed=101)
    step(30)
    for i, nm in enumerate(c.cup_names):
        drop_cup(nm, i)
    shift_assembly(c.dock_x - 0.040, settle_steps=90)
    report("near-dock")
    s, ok = judge()
    check("near-dock miss: fully loaded shuttle 40 mm short of the dock line "
          f"(x={sx():+.3f} < {c.dock_x:.3f}): all_seated True, docked False -> "
          "NOT success, no dock credit (score <= 0.55)",
          bool(scene.all_seated()[0]) and not bool(scene.docked()[0]) and not ok
          and 0.45 <= s <= 0.55)

    # =========================== 13. latched credit survives removal ========================
    s_before, _ = judge()
    to_floor("green", 0.0, -0.35, settle_steps=90)
    report("removed")
    s_after, ok = judge()
    check("latched credit: lifting green off the near-docked shuttle leaves the "
          f"latched score unchanged ({s_before:.2f} -> {s_after:.2f}) while "
          "all_seated() drops",
          abs(s_after - s_before) < 1e-3 and not bool(scene.all_seated()[0]) and not ok)

    # =========================== 14. canopy reality (order is geometry-enforced) ============
    env.reset(seed=111)
    step(30)
    shift_assembly(c.dock_rest_x - 0.004, settle_steps=60, cups_aboard=False)
    # drop blue from ABOVE the docked blue socket: it can only land ON the roof
    world = torch.tensor([sx() + c.socket_xs[2], 0.0, roof_top + c.cup_h / 2 + 0.030],
                         device=device) + scene.env_origins[0]
    write_cup("blue", world, q0, settle_steps=150)
    report("canopy")
    z_blue = float(scene.cups["blue"].data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    loc = scene._cup_local("blue")[0]
    s, ok = judge()
    check("canopy reality: canister dropped over the DOCKED blue socket lands ON the "
          f"roof (world z={z_blue:+.3f} > roof underside {c.roof_bot:.3f}; local xy IS "
          f"over the socket, dx={float(loc[0]) - c.socket_xs[2]:+.3f}) yet the z window "
          "rejects it — loading after docking is physically impossible",
          z_blue > c.roof_bot and abs(float(loc[0]) - c.socket_xs[2]) <= 2 * c.socket_tol
          and not bool(scene.seated("blue")[0]) and not ok and s <= 0.15)

    # =========================== 15. settle gate ============================================
    env.reset(seed=121)
    step(30)
    for i, nm in enumerate(c.cup_names):
        drop_cup(nm, i)
    # construct the FULL goal state ARRIVING (real forward velocity — the stillness
    # counter resets and cannot have filled), judged only 2 steps in -> NOT success
    shift_assembly(c.dock_rest_x - 0.004, settle_steps=2, vx=0.15)
    sc_now = int(scene.still_count[0])
    s, ok = judge()
    gate_ok = (bool(scene.all_seated()[0]) and bool(scene.docked()[0])
               and not bool(scene.settled()[0]) and not ok)
    report("settle-gate")
    # remove a canister BEFORE the persistence gate can fill (the battery must
    # never reach success)
    to_floor("blue", 0.0, -0.35, settle_steps=90)
    report("gate-defused")
    check("settle gate: the full goal state (loaded + docked) judged 2 steps after "
          f"construction — still_count={sc_now} < {c.settle_steps_min} -> settled "
          "False, NOT success; canister removed before the gate filled",
          gate_ok and sc_now < c.settle_steps_min and not bool(scene.success()[0]))

    # =========================== 16-17. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.cargo_shuttle")
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
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
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
