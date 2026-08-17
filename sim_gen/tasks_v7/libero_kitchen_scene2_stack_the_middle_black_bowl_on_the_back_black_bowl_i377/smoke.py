"""Smoke / rubric-REJECTION battery for GimbalServiceScene (sim_gen task
libero_kitchen_scene2_stack_the_middle_black_bowl_on_the_back_black_bowl_i377) —
NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — the hover-release build of the ring-seated
gimbal stack — is the acceptance evidence that the rubric ACCEPTS a correct
outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong (or
partially-right) outcome as a settled state and asserts the rubric's verdict on
it. No probe below constructs full success (the battery tops out at the latched
0.50).

  1-2.  settle/no-NaN     — authored layout settles finite; pitch readback inside
                            the sampled band; nothing ringed/nested/served; score 0;
  3.    determinism       — the same seed twice -> identical layout readback;
  4-5.  randomization     — READBACK over 8 seeded resets: shelf pitch/yaw/centre
                            spread, park sides take both values, parks jitter;
  6.    null policy       — 240 idle steps -> score ~0, no success;
  7.    bare-slab shed    — the cube set ON the open slick slab (clear of the ring)
                            slides off downhill by itself: the shelf really is
                            unusable bare; nothing latches;
  8.    tilted socket     — socket bowl REALLY seated (ringed, 0.20), cube laid
                            directly into it: pair friction holds it, and it rests
                            AT THE SHELF ANGLE (> 12 deg) — the 10 deg level gate
                            fails; the gimbal is necessary, not decorative;
  9.    seed strategy     — the seed's whole plan (stack bowl-in-bowl + cube, as a
                            free stack) built OFF the shelf on the flat counter:
                            b_in_a and cube_in_b are geometrically TRUE (tolerance
                            twins pass — the stack is level on a flat table!) but
                            the ring clause gates every latch -> score 0;
  10.   declared clause   — the gimbal bowl set STRAIGHT into the ring (no socket)
                            + cube: physically stable and level, cube_in_b True,
                            but `nested` demands the socket-frame bands -> score 0;
  11.   inverted socket   — socket bowl upside-down over the seat: alignment/z
                            gates reject; never ringed;
  12.   uphill arrest     — socket bowl released on the slab UPHILL of the ring:
                            it slides downhill (slick) and the curb's OUTER face
                            arrests it short of the seat — moved but never ringed;
  13-14. real stages + latch survival — socket seated + gimbal nested by real
                            hover-drops (score 0.50), then the gimbal teleported
                            away: latched 0.50 survives, no success;
  15.   finite at the end.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=12)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the pod driver version and silently rejects RTX -> the
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

# Kit teardown hangs leave GPU zombies: global watchdog long before the forge timeout.
_wd = threading.Timer(1350.0, lambda: os._exit(4))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.gimbal_service")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    origin = env.iscene.env_origins
    z0 = c.surface_z

    from isaaclab.utils.math import quat_apply, quat_mul

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.15, -1.05, 0.95)) + o),
                                tuple(np.array((0.05, 0.02, 0.28)) + o),
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

    def settle(max_steps: int = 500, poll: int = 10) -> None:
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if bool(scene.settled()[0]):
                return

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def cube_level_val() -> float:
        cq = scene.cube.data.root_quat_w
        best = 0.0
        for ax in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
            v = torch.tensor(ax, device=cq.device).expand(n, 3)
            best = max(best, float(quat_apply(cq, v)[0, 2].abs()))
        return best

    def report(tag: str) -> None:
        print(f"[smoke] {tag:18s} | pitch={float(scene.station_pitch_deg[0]):5.2f} "
              f"ringed={bool(scene.a_in_ring()[0])} nested={bool(scene.b_in_a()[0])} "
              f"served={bool(scene.cube_in_b()[0])} "
              f"R/N/S={int(scene.ever_ringed[0])}{int(scene.ever_nested[0])}"
              f"{int(scene.ever_served[0])} cube_up={cube_level_val():.3f} "
              f"settled={bool(scene.settled()[0])} score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    def place_w(body, pos_w: torch.Tensor, quat_w: torch.Tensor,
                settle_steps: int = 30) -> None:
        """Probe placement (instrumentation, not a solution) + REAL physics steps
        before judging."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    level_q = torch.zeros(n, 4, device=device)
    level_q[:, 0] = 1.0

    def st_place(body, x: float, y: float, z: float, *, align: bool = True,
                 settle_steps: int = 30) -> None:
        """Place a body at STATION-frame (x, y, z), aligned to the shelf or level."""
        loc = torch.tensor([x, y, z], device=device).expand(n, 3)
        pos = scene.station_pos + quat_apply(scene.station_quat, loc)
        place_w(body, pos, scene.station_quat if align else level_q, settle_steps)

    def bench_place(body, x: float, y: float, z: float, settle_steps: int = 30) -> None:
        pos = origin + torch.tensor([x, y, z], device=device).expand(n, 3)
        place_w(body, pos, level_q, settle_steps)

    def drop_a_into_seat() -> bool:
        """The solve's stage-1 move: hover-release the socket bowl 22 mm above the
        ring seat, shelf-aligned, and let the curb arrest the slide."""
        nudge = quat_apply(scene.station_quat,
                           torch.tensor([0.0, 0.0, 0.022], device=device).expand(n, 3))
        place_w(scene.bowl_a, scene.seat_center_w() + nudge, scene.station_quat,
                settle_steps=0)
        waited = 0
        while waited < 600:
            step(10)
            waited += 10
            if bool(scene.a_in_ring()[0]) and bool(scene.settled()[0]):
                break
        settle(max_steps=240)
        return bool(scene.a_in_ring()[0]) and bool(scene.settled()[0])

    def drop_b_into_a() -> bool:
        """The solve's stage-2 move: hover-release the gimbal WORLD-LEVEL over the
        socket mouth and let the ball seat + the ballast level the cup."""
        above = quat_apply(scene.bowl_a.data.root_quat_w,
                           torch.tensor([0.0, 0.0, 0.115], device=device).expand(n, 3))
        place_w(scene.bowl_b, scene.bowl_a.data.root_pos_w + above, level_q,
                settle_steps=0)
        settle(max_steps=500)
        return (bool(scene.b_in_a()[0]) and bool(scene.a_in_ring()[0])
                and bool(scene.settled()[0]))

    def layout_readback() -> tuple:
        sp = scene.station_pos[0] - origin[0]
        a = scene.bowl_a.data.root_pos_w[0] - origin[0]
        b = scene.bowl_b.data.root_pos_w[0] - origin[0]
        cu = scene.cube.data.root_pos_w[0] - origin[0]
        return (float(scene.station_pitch_deg[0]), float(scene.station_yaw[0]),
                float(sp[0]), float(sp[1]), float(a[0]), float(a[1]),
                float(b[0]), float(b[1]), float(cu[0]), float(cu[1]),
                float(scene.a_side[0]), float(scene.cube_side[0]))

    def all_finite() -> bool:
        return all(bool(torch.isfinite(b.data.root_state_w).all())
                   for b in (scene.station, scene.bowl_a, scene.bowl_b, scene.cube))

    # =========================== 1-2. settle / no-NaN / clean rubric =========================
    env.reset(seed=11)
    settle()
    report("reset")
    check("settle: authored layout finite and settled; pitch readback inside the "
          "sampled band; nothing ringed/nested/served",
          all_finite() and bool(scene.settled()[0])
          and c.pitch_lo_deg - 0.01 <= float(scene.station_pitch_deg[0]) <= c.pitch_hi_deg + 0.01
          and not bool(scene.a_in_ring()[0]) and not bool(scene.b_in_a()[0])
          and not bool(scene.cube_in_b()[0]) and not bool(scene.ever_ringed[0]))
    check("rubric clean at reset: score 0, no success",
          float(scene.score()[0]) <= 0.005 and not bool(scene.success()[0]))

    # =========================== 3. determinism =============================================
    env.reset(seed=777)
    step(3)
    read_a = layout_readback()
    env.reset(seed=777)
    step(3)
    read_b = layout_readback()
    print(f"[smoke] determinism readback: {read_a} vs {read_b}", flush=True)
    check("determinism: same seed -> identical layout readback",
          all(abs(a - b) < 1e-5 for a, b in zip(read_a, read_b)))

    # =========================== 4-5. randomization is real ==================================
    reads = []
    for s in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=s)
        step(3)
        reads.append(layout_readback())
    arr = np.array(reads)
    print(f"[smoke] randomization readback (pitch, yaw, sx, sy, ax, ay, bx, by, cx, cy, "
          f"aside, cside):\n{arr}", flush=True)
    check("randomization: shelf pitch, yaw and centre all spread across seeds (readback)",
          arr[:, 0].max() - arr[:, 0].min() > 1.0
          and arr[:, 1].max() - arr[:, 1].min() > 0.3
          and max(arr[:, 2].max() - arr[:, 2].min(),
                  arr[:, 3].max() - arr[:, 3].min()) > 0.01)
    check("randomization: park sides each take both values; park centres jitter",
          arr[:, 10].max() - arr[:, 10].min() > 1.0
          and arr[:, 11].max() - arr[:, 11].min() > 1.0
          and (arr[:, 4].max() - arr[:, 4].min()) > 0.005
          and (arr[:, 8].max() - arr[:, 8].min()) > 0.005)

    # =========================== 6. null policy fails ========================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    check("null policy: score ~0, no success after 240 idle steps",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 7. the bare slab sheds the cube =============================
    # Clear lane past the ring (station-local y = 0.115): the cube set on the open
    # slick slab must slide off downhill by itself — the shelf is unusable bare.
    env.reset(seed=35)
    settle()
    st_place(scene.cube, -0.10, 0.115, c.slab_t + c.cube_size / 2 + 0.002,
             align=True, settle_steps=10)
    x0 = float(scene.st_local(scene.cube.data.root_pos_w)[0, 0])
    settle(max_steps=600)
    x1 = float(scene.st_local(scene.cube.data.root_pos_w)[0, 0])
    report("bare-slab-shed")
    check("bare slab sheds the cube: it slides downhill by itself (moved "
          f"{x1 - x0:+.3f} m along station +x); nothing latches",
          x1 - x0 > 0.05 and not bool(scene.ever_served[0])
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 8. cube straight into the TILTED socket ======================
    # The socket is REALLY seated first (real drop -> ringed latches, 0.20). A cube
    # laid directly into it rests FACE-FLAT on the tilted floor, contained by the
    # lower wall, AT THE SHELF ANGLE — failing the 10 deg level gate by geometry.
    # The gimbal is necessary.
    env.reset(seed=40)
    settle()
    seated = drop_a_into_seat() or drop_a_into_seat()
    above = quat_apply(scene.bowl_a.data.root_quat_w,
                       torch.tensor([0.0, 0.0, 0.060], device=device).expand(n, 3))
    place_w(scene.cube, scene.bowl_a.data.root_pos_w + above, level_q, settle_steps=20)
    settle(max_steps=500)
    report("cube-in-socket")
    from isaaclab.utils.math import quat_apply_inverse
    cloc = quat_apply_inverse(scene.bowl_a.data.root_quat_w,
                              scene.cube.data.root_pos_w
                              - scene.bowl_a.data.root_pos_w)[0]
    lvl = cube_level_val()
    check("tilted socket: cube laid directly in the seated socket STAYS in it "
          f"(walled in) but rests at the shelf angle (up_z {lvl:.3f} < "
          f"cos 12 deg) — level gate fails, score pinned at ringed 0.20",
          seated and float(cloc[:2].norm()) < 0.045 and float(cloc[2]) < 0.055
          and lvl < math.cos(math.radians(12.0))
          and not bool(scene.ever_served[0]) and not bool(scene.ever_nested[0])
          and abs(float(scene.score()[0]) - 0.20) < 1e-3
          and not bool(scene.success()[0]))

    # =========================== 9. seed strategy: the flat stack ============================
    # The seed's whole plan — stack bowl in bowl (plus the cube) as a free stack on
    # the flat counter. b_in_a and cube_in_b are geometrically TRUE (the stack IS
    # level on a flat table), but the ring clause gates every latch.
    env.reset(seed=44)
    settle()
    bench_place(scene.bowl_a, 0.30, -0.35, z0 + 0.001, settle_steps=20)
    ap = scene.bowl_a.data.root_pos_w
    place_w(scene.bowl_b, ap + torch.tensor(
        [0.0, 0.0, c.b_rest_z_in_a + 0.004], device=device).expand(n, 3),
        level_q, settle_steps=20)
    bp = scene.bowl_b.data.root_pos_w
    place_w(scene.cube, bp + torch.tensor(
        [0.0, 0.0, c.cube_seat_z + 0.003], device=device).expand(n, 3),
        level_q, settle_steps=20)
    settle(max_steps=500)
    report("seed-strategy")
    check("seed strategy (bowl-in-bowl + cube stacked on the flat counter): b_in_a "
          "and cube_in_b geometrically TRUE, but the off-ring stack latches "
          "NOTHING — score 0, no success",
          bool(scene.b_in_a()[0]) and bool(scene.cube_in_b()[0])
          and not bool(scene.a_in_ring()[0])
          and not bool(scene.ever_ringed[0]) and not bool(scene.ever_nested[0])
          and not bool(scene.ever_served[0])
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 10. declared clause: gimbal straight into the ring ==========
    env.reset(seed=48)
    settle()
    st_place(scene.bowl_b, c.ring_x, 0.0, c.slab_t + c.b_sphere_r + 0.008,
             align=False, settle_steps=0)
    settle(max_steps=500)
    place_w(scene.cube, scene.bowl_b.data.root_pos_w + torch.tensor(
        [0.0, 0.0, 0.130], device=device).expand(n, 3), level_q, settle_steps=20)
    settle(max_steps=500)
    report("gimbal-in-ring")
    check("declared clause: the gimbal set STRAIGHT into the ring (no socket) is "
          "physically stable — cube_in_b True and level — but `nested` demands the "
          "socket-frame bands: nothing latches, score 0",
          bool(scene.cube_in_b()[0]) and not bool(scene.b_in_a()[0])
          and not bool(scene.ever_nested[0]) and not bool(scene.ever_served[0])
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 11. inverted socket over the seat ===========================
    env.reset(seed=52)
    settle()
    flip = quat_mul(scene.station_quat,
                    torch.tensor([0.0, 1.0, 0.0, 0.0], device=device).expand(n, 4))
    loc = torch.tensor([c.ring_x, 0.0, c.slab_t + c.a_h + 0.015],
                       device=device).expand(n, 3)
    place_w(scene.bowl_a, scene.station_pos + quat_apply(scene.station_quat, loc),
            flip, settle_steps=20)
    settle(max_steps=500)
    report("inverted-socket")
    check("inverted socket: the bowl dropped upside-down over the seat never counts "
          "as ringed (alignment/z gates); score 0",
          not bool(scene.a_in_ring()[0]) and not bool(scene.ever_ringed[0])
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 12. uphill release: the curb arrests OUTSIDE ================
    env.reset(seed=56)
    settle()
    st_place(scene.bowl_a, -0.12, 0.0, c.slab_t + 0.010, align=True, settle_steps=5)
    xa0 = float(scene.st_local(scene.bowl_a.data.root_pos_w)[0, 0])
    settle(max_steps=600)
    xa1 = float(scene.st_local(scene.bowl_a.data.root_pos_w)[0, 0])
    report("uphill-arrest")
    check("uphill release: the socket bowl slides downhill on the slick slab "
          f"(moved {xa1 - xa0:+.3f} m) and the curb's OUTER face arrests it short "
          "of the seat — never ringed, score 0",
          xa1 - xa0 > 0.015 and not bool(scene.a_in_ring()[0])
          and not bool(scene.ever_ringed[0])
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 13-14. real stages 1+2, then latch survival =================
    env.reset(seed=60)
    settle()
    ok_a = drop_a_into_seat() or drop_a_into_seat()
    ok_b = drop_b_into_a() or drop_b_into_a()
    report("real-stages")
    check("real stages 1+2: socket seated by the curb and gimbal self-leveled in it "
          "(hover-drops, no pose writes after release) -> ringed+nested latch, "
          "score 0.50, no success",
          ok_a and ok_b and bool(scene.ever_ringed[0]) and bool(scene.ever_nested[0])
          and abs(float(scene.score()[0]) - 0.50) < 1e-3
          and not bool(scene.success()[0]))
    bench_place(scene.bowl_b, c.a_park[0], -c.a_park[1] * float(scene.a_side[0]),
                z0 + c.b_sphere_r + 0.002, settle_steps=0)
    settle(max_steps=400)
    report("gimbal-removed")
    check("latch survival: gimbal teleported back off the stack — nested no longer "
          "holds NOW but the latched 0.50 survives (credit never evaporates); "
          "no success",
          not bool(scene.b_in_a()[0])
          and abs(float(scene.score()[0]) - 0.50) < 1e-3
          and not bool(scene.ever_served[0]) and not bool(scene.success()[0]))

    # =========================== 15. finite at the end ========================================
    check("finite: all states finite at the end", all_finite())

    # =========================== save + verdict ==============================================
    if frames:
        arrf = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arrf, env="simgen.gimbal_service")
        print(f"[smoke] saved {arrf.shape} -> {args.out}", flush=True)
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
    exit_t = threading.Timer(10.0, lambda: os._exit(code))
    exit_t.daemon = True
    exit_t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"[smoke] FATAL: unhandled {type(e).__name__}: {e}", flush=True)
        print("SIM_GEN_SMOKE: FAIL 0/0", flush=True)
        os._exit(1)
