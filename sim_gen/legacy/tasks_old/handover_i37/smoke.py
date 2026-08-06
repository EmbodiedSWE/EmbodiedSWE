"""Smoke / oracle test for RailFerryScene (sim_gen task `handover_i37`) — NullRobot,
teleport-oracle, RECORDED.

Battery (compass_crate / pen_holder smoke skeleton):
  1. settle/no-NaN      — reset layout settles finite, shuttle seated, score ~0;
  2. randomization      — READBACK across 8 seeded resets: parcel scatter, pad position,
                          shuttle start side and parcel count all move;
  3. null-policy-fails  — 240 idle steps -> score ~0, no success;
  4. oracle x3 seeds    — full ferry logistics (fetch shuttle if needed, then per parcel:
                          load -> ride across -> unload to pad -> empty return), reaching
                          success() and score 1.0 on 3 seeds;
  5. monotonicity       — the staged score after every load/cross/deliver milestone of the
                          seed-0 oracle run strictly increases to 1.0;
  6. negative A (seed)  — the seed's own strategy, a DIRECT mid-air handover: every parcel
                          carried through the air over the divide and placed PERFECTLY on
                          the pad -> every parcel violated, score <= 0.05, no success;
                          plus latch permanence: re-ferrying a violated parcel cleanly
                          (B->A->B riding the shuttle) still refuses delivery;
  7. negative B (detour)— sneaking a parcel around the END of the wall on the ground also
                          trips the violation latch (the rubric judges the plane);
  8. negative C (carry ferry) — lifting the LOADED shuttle off its rails and flying it
                          across (a handover of a bigger box) is violated too; the loaded
                          latch credit BEFORE the flight is asserted (partial progress);
  9. near-miss          — a cleanly ferried parcel set down NEXT to the pad earns only the
                          crossed credit; placing it on the pad recovers delivery credit;
 10. calibration A      — basin drop funnel: release offsets 0..60 mm, 3 directions each
                          -> published in-basin table; <=10 mm all-in, 60 mm (past the
                          inner-edge pivot) none;
 11. calibration B      — traverse-speed ride retention: constant-velocity crossings at
                          0.4 / 1.2 / 2.4 m/s, 3 runs each -> published retention table;
                          the gentle 0.4 m/s ride retains and crosses clean 3/3.

Run (forge): python -u -m simgen_tasks.handover_i37.smoke --headless
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
    from simgen_tasks.handover_i37 import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401


def _env_of(scene_or_env):
    return scene_or_env if hasattr(scene_or_env, "iscene") else scene_or_env.env


def oracle_solution(scene_or_env, step_fn=None, milestones=None, verbose: bool = True) -> bool:
    """Teleport-oracle: solve the CURRENT episode by real ferry logistics. If the shuttle
    starts at the far station, fetch it first. Then per present parcel: drop it into the
    basin at station A (a genuine drop), drive the shuttle across the bridge (kinematic
    gravity-compensated drive — the parcel rides via real contact), lift it out onto its
    pad slot, and drive the empty shuttle back. Appends score() to `milestones` after
    every load/cross/deliver stage. Returns True iff scene.success()."""
    env = _env_of(scene_or_env)
    scene = env.scene
    c = scene.cfg
    dev = env.device
    no_action = torch.empty(0, device=dev)
    all_ids = torch.arange(env.num_envs, device=dev)
    g_dt = 9.81 * env.dt

    def _step(k: int) -> None:
        if step_fn is not None:
            step_fn(k)
        else:
            for _ in range(k):
                env.step(no_action)

    def mark() -> None:
        if milestones is not None:
            milestones.append(float(scene.score()[0]))

    def shuttle_loc():
        return (scene.shuttle.data.root_pos_w - scene.env_origins)[0]

    def drive_shuttle(x_to: float, avg_v: float = 0.5) -> None:
        """Cosine-profile kinematic drive along the rail channel: rewrite the shuttle root
        state every substep (position + matching velocity + the +g*dt gravity-compensation
        term, the pen_holder held-cup pattern) so cargo rides on real floor friction; end
        with a zero-velocity re-pin + buffer refresh."""
        p0 = shuttle_loc()
        x0, z0 = float(p0[0]), float(p0[2])
        big_x = x_to - x0
        if abs(big_x) < 0.01:
            return
        big_t = max(40, int(abs(big_x) / avg_v / env.dt))
        st = torch.zeros(env.num_envs, 13, device=dev)
        for i in range(1, big_t + 1):
            f = 0.5 * (1.0 - math.cos(math.pi * i / big_t))
            df = 0.5 * math.pi / big_t * math.sin(math.pi * i / big_t)
            st.zero_()
            st[:, 0] = x0 + big_x * f
            st[:, 2] = z0
            st[:, 3] = 1.0
            st[:, 7] = big_x * df / env.dt
            st[:, 9] = g_dt
            st[:, 0:3] += scene.env_origins[all_ids]
            scene.shuttle.write_root_state_to_sim(st, all_ids)
            _step(1)
        st.zero_()
        st[:, 0] = x_to
        st[:, 2] = z0
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins[all_ids]
        scene.shuttle.write_root_state_to_sim(st, all_ids)
        env.iscene.update(0.0)

    def settle_until(pred, max_steps: int = 240, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            _step(poll)
            waited += poll
            if pred():
                return True
        return False

    def drop_into_basin(i: int) -> None:
        sp = shuttle_loc()
        st = torch.zeros(env.num_envs, 13, device=dev)
        st[:, 0] = float(sp[0])
        st[:, 1] = float(sp[1])
        st[:, 2] = float(sp[2]) + c.shuttle_floor_t / 2 + c.cargo_size / 2 + 0.045
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins[all_ids]
        scene.cargos[f"cargo_{i}"].write_root_state_to_sim(st, all_ids)
        settle_until(lambda: bool(scene.loaded[0, i]) and bool(scene.settled()[0, i]),
                     max_steps=160)

    def place_on_pad(i: int, dy: float) -> None:
        st = torch.zeros(env.num_envs, 13, device=dev)
        st[:, 0] = scene.pad_xy[all_ids, 0]
        st[:, 1] = scene.pad_xy[all_ids, 1] + dy
        st[:, 2] = c.pad_size[2] + c.cargo_size / 2 + 0.030
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins[all_ids]
        scene.cargos[f"cargo_{i}"].write_root_state_to_sim(st, all_ids)
        settle_until(lambda: bool(scene.delivered()[0, i]), max_steps=200)

    present = [i for i in range(c.n_cargo) if bool(scene.present[0, i])]
    slots = [(j - (len(present) - 1) / 2.0) * c.pad_slot_dy for j in range(len(present))]
    for t, i in enumerate(present):
        drive_shuttle(-c.station_x)  # station A (fetches the ferry on the first pass)
        drop_into_basin(i)
        mark()
        drive_shuttle(c.station_x)   # the ride across the divide
        settle_until(lambda i=i: bool(scene.crossed[0, i]), max_steps=60)
        mark()
        place_on_pad(i, slots[t])
        mark()
        if verbose:
            print(f"[oracle] parcel {i}: loaded/crossed/delivered="
                  f"{bool(scene.loaded[0, i])}/{bool(scene.crossed[0, i])}/"
                  f"{bool(scene.delivered()[0, i])} score={float(scene.score()[0]):.3f}",
                  flush=True)
    settle_until(lambda: bool(scene.success()[0]), max_steps=200)
    if milestones is not None and milestones:
        milestones[-1] = float(scene.score()[0])
    return bool(scene.success()[0])


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.rail_ferry")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    g_dt = 9.81 * env.dt

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.30, -1.30, 1.00)) + o),
                                tuple(np.array((0.0, 0.0, 0.15)) + o),
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

    def settle_until(pred, max_steps: int = 240, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def loc(body):
        return (body.data.root_pos_w - scene.env_origins)[0]

    def k_present() -> int:
        return int(scene.present[0].sum())

    def first_present() -> int:
        return [i for i in range(c.n_cargo) if bool(scene.present[0, i])][0]

    def report(tag: str) -> None:
        pres = scene.present[0].int().tolist()
        print(f"[smoke] {tag:14s} | present={pres} "
              f"loaded={scene.loaded[0].int().tolist()} "
              f"crossed={scene.crossed[0].int().tolist()} "
              f"violated={scene.violated[0].int().tolist()} "
              f"delivered={scene.delivered()[0].int().tolist()} "
              f"score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # --- staging helpers --------------------------------------------------------------------
    def write_body(body, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = env.iscene.env_origins + torch.tensor(pos, device=device)
        st[:, 3:7] = torch.tensor(quat, device=device)
        body.write_root_state_to_sim(st, all_ids)
        env.iscene.update(0.0)

    def seat_shuttle_at(x: float) -> None:
        """Instant staging write: the EMPTY shuttle may cross the plane freely."""
        write_body(scene.shuttle, (x, 0.0, c.seated_z + 0.001))

    def drive_shuttle(x_to: float, avg_v: float = 0.5, const_v: float | None = None,
                      z_hold: float | None = None) -> None:
        """Kinematic drive along x at the held height (default: current). Cosine profile
        by default; `const_v` switches to a constant-velocity profile (the calibration
        probe's inertia hammer)."""
        p0 = loc(scene.shuttle)
        x0 = float(p0[0])
        z0 = float(p0[2]) if z_hold is None else z_hold
        big_x = x_to - x0
        if abs(big_x) < 0.01:
            return
        if const_v is not None:
            big_t = max(2, int(abs(big_x) / const_v / env.dt))
        else:
            big_t = max(40, int(abs(big_x) / avg_v / env.dt))
        st = torch.zeros(n, 13, device=device)
        for i in range(1, big_t + 1):
            if const_v is None:
                f = 0.5 * (1.0 - math.cos(math.pi * i / big_t))
                df = 0.5 * math.pi / big_t * math.sin(math.pi * i / big_t)
            else:
                f = i / big_t
                df = 1.0 / big_t
            st.zero_()
            st[:, 0] = x0 + big_x * f
            st[:, 2] = z0
            st[:, 3] = 1.0
            st[:, 7] = big_x * df / env.dt
            st[:, 9] = g_dt
            st[:, 0:3] += env.iscene.env_origins
            scene.shuttle.write_root_state_to_sim(st, all_ids)
            step(1)
        st.zero_()
        st[:, 0] = x_to
        st[:, 2] = z0
        st[:, 3] = 1.0
        st[:, 0:3] += env.iscene.env_origins
        scene.shuttle.write_root_state_to_sim(st, all_ids)
        env.iscene.update(0.0)

    def lift_shuttle(dz: float, big_t: int = 50) -> float:
        """Kinematic vertical ramp (cargo rides up on floor contact). Returns final z."""
        p0 = loc(scene.shuttle)
        x0, y0, z0 = float(p0[0]), float(p0[1]), float(p0[2])
        st = torch.zeros(n, 13, device=device)
        for i in range(1, big_t + 1):
            f = 0.5 * (1.0 - math.cos(math.pi * i / big_t))
            df = 0.5 * math.pi / big_t * math.sin(math.pi * i / big_t)
            st.zero_()
            st[:, 0] = x0
            st[:, 1] = y0
            st[:, 2] = z0 + dz * f
            st[:, 3] = 1.0
            st[:, 9] = dz * df / env.dt + g_dt
            st[:, 0:3] += env.iscene.env_origins
            scene.shuttle.write_root_state_to_sim(st, all_ids)
            step(1)
        return z0 + dz

    def drop_into_basin(i: int, xy_off=(0.0, 0.0)) -> None:
        sp = loc(scene.shuttle)
        write_body(scene.cargos[f"cargo_{i}"],
                   (float(sp[0]) + xy_off[0], float(sp[1]) + xy_off[1],
                    float(sp[2]) + c.shuttle_floor_t / 2 + c.cargo_size / 2 + 0.045))
        step(50)

    def hover_place(i: int, pos) -> None:
        write_body(scene.cargos[f"cargo_{i}"], (pos[0], pos[1], pos[2] + 0.030))
        step(40)

    def carry_cargo(i: int, waypoints, step_len: float = 0.02) -> None:
        """The SEED's move: kinematic incremental carry of a parcel through free air
        (zero-velocity re-pins along the path — exactly what a gripper-to-gripper
        transfer looks like to the scene)."""
        body = scene.cargos[f"cargo_{i}"]
        p = loc(body).tolist()
        st = torch.zeros(n, 13, device=device)
        for wp in waypoints:
            d = math.dist(p, wp)
            kk = max(1, int(math.ceil(d / step_len)))
            for t in range(1, kk + 1):
                q = [p[a] + (wp[a] - p[a]) * t / kk for a in range(3)]
                st.zero_()
                st[:, 0:3] = env.iscene.env_origins + torch.tensor(q, device=device)
                st[:, 3] = 1.0
                body.write_root_state_to_sim(st, all_ids)
                step(1)
            p = wp

    # =========================== 1. settle / no-NaN =========================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    st_all = torch.cat([scene.shuttle.data.root_state_w]
                       + [b.data.root_state_w for b in scene.cargos.values()], dim=0)
    check("settle: states finite, shuttle seated, parcels at rest",
          bool(torch.isfinite(st_all).all()) and bool(scene.seated()[0])
          and bool(scene.settled()[0].all()))
    check("settle: score ~0 at reset, no success",
          float(scene.score()[0]) <= 0.005 and not bool(scene.success()[0]))

    # =========================== 2. randomization is real ===================================
    reads = []
    for s in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(s)
        env.reset()
        step(2)
        pos, _v = scene._cargo_tensors()
        locs = pos[0] - scene.env_origins[0]
        scat = float(sum(locs[i, 0] + locs[i, 1] for i in range(c.n_cargo)
                         if bool(scene.present[0, i])))
        side = 1.0 if float(loc(scene.shuttle)[0]) > 0 else -1.0
        reads.append((scat, float(scene.pad_xy[0, 0]), float(scene.pad_xy[0, 1]),
                      side, k_present()))
    arr = np.array(reads)
    print("[smoke] randomization readback (cargo_scatter_sum, pad_x, pad_y, side, k):\n"
          f"{np.round(arr, 3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: parcel scatter, pad position, shuttle side and parcel count "
          "all move (readback)",
          spread[0] > 0.02 and (spread[1] > 0.03 or spread[2] > 0.05)
          and spread[3] > 0 and spread[4] > 0)

    # =========================== 3. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: score ~0 and no success after 240 idle steps",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 4+5. oracle x3 + monotonicity ==============================
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        step(20)
        ms: list[float] = [float(scene.score()[0])]
        ok = oracle_solution(env, step_fn=step, milestones=ms)
        report(f"oracle-seed{s}")
        check(f"oracle reaches success() on seed {s} (score 1.0, k={k_present()})",
              ok and bool(scene.success()[0]) and float(scene.score()[0]) == 1.0)
        if s == 0:
            print(f"[smoke] milestones (seed 0): {[round(v, 3) for v in ms]}", flush=True)
            check("monotonicity: staged score strictly increases across "
                  "load/cross/deliver milestones to 1.0",
                  all(b > a + 1e-6 for a, b in zip(ms, ms[1:])) and ms[-1] == 1.0)

    # =========================== 6. negative A: the seed's own strategy =====================
    # mujoco_playground/handover transfers the box THROUGH THE AIR across the workspace
    # boundary (gripper-to-gripper) and sets it on the target. Express exactly that: carry
    # every parcel over the wall in free air and place it PERFECTLY on the pad. The end
    # state is indistinguishable from success by pose — and must score ~nothing.
    torch.manual_seed(51)
    env.reset()
    step(20)
    present = [i for i in range(c.n_cargo) if bool(scene.present[0, i])]
    kp = len(present)
    px, py = float(scene.pad_xy[0, 0]), float(scene.pad_xy[0, 1])
    z_pad = c.pad_size[2] + c.cargo_size / 2
    for t, i in enumerate(present):
        dy = (t - (kp - 1) / 2.0) * c.pad_slot_dy
        p0 = loc(scene.cargos[f"cargo_{i}"]).tolist()
        carry_cargo(i, [(p0[0], p0[1], 0.36), (px, py + dy, 0.36),
                        (px, py + dy, z_pad + 0.01)], step_len=0.03)
        step(30)
    settle_until(lambda: bool(scene.settled()[0].all()), max_steps=120)
    report("seed-handover")
    check("negative A (seed strategy): every air-carried parcel trips the violation latch",
          all(bool(scene.violated[0, i]) for i in present))
    check("negative A: perfect end pose on the pad, yet score <= 0.05 and no success",
          all(bool(scene.on_pad()[0, i]) for i in present)
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))
    # latch permanence: re-ferry ONE violated parcel properly (B->A->B riding the seated
    # shuttle) and re-place it — the parcel must stay refused forever.
    j = present[0]
    seat_shuttle_at(c.station_x)
    step(10)
    drop_into_basin(j)
    drive_shuttle(-c.station_x)
    step(20)
    drive_shuttle(c.station_x)
    step(20)
    hover_place(j, (px, py + (0 - (kp - 1) / 2.0) * c.pad_slot_dy, z_pad))
    settle_until(lambda: bool(scene.settled()[0, j]), max_steps=120)
    report("re-ferried")
    check("negative A permanence: cleanly re-ferried parcel is still refused "
          "(violated latch is permanent)",
          bool(scene.crossed[0, j]) and bool(scene.violated[0, j])
          and not bool(scene.delivered()[0, j]) and float(scene.score()[0]) <= 0.05)

    # =========================== 7. negative B: ground detour ===============================
    torch.manual_seed(61)
    env.reset()
    step(20)
    j = first_present()
    p0 = loc(scene.cargos[f"cargo_{j}"]).tolist()
    zg = c.cargo_size / 2 + 0.002
    # route via x=-0.55 first so the ground path clears the other parcel slots
    carry_cargo(j, [(-0.55, p0[1], zg), (-0.55, 0.62, zg), (0.30, 0.62, zg)], step_len=0.03)
    step(40)
    report("wall-detour")
    check("negative B: parcel sneaked around the wall end on the ground is violated, "
          "score capped",
          bool(scene.violated[0, j]) and float(scene.score()[0]) <= 0.03
          and not bool(scene.success()[0]))

    # =========================== 8. negative C: carry the loaded ferry ======================
    # Reduce the task back to the seed by treating the loaded shuttle as one big box and
    # handing IT across through the air: lift it off its rails, fly it over, set it down.
    # The crossing happens with the parcel in the basin but the shuttle NOT seated.
    torch.manual_seed(71)
    env.reset()
    step(20)
    j = first_present()
    seat_shuttle_at(-c.station_x)
    step(10)
    drop_into_basin(j)
    settle_until(lambda: bool(scene.loaded[0, j]), max_steps=120)
    report("loaded")
    sc_loaded = float(scene.score()[0])
    check("negative C: loaded-latch credit after loading the seated shuttle "
          "(0.15/k partial progress)",
          bool(scene.loaded[0, j]) and 0.03 <= sc_loaded <= 0.10
          and abs(sc_loaded - 0.15 / k_present()) < 0.02)
    z_air = lift_shuttle(0.12)
    drive_shuttle(c.station_x, z_hold=z_air)
    step(10)
    report("flying-ferry")
    check("negative C: carrying the LOADED shuttle through the air trips the violation "
          "latch (crossing while unseated)",
          bool(scene.violated[0, j]) and not bool(scene.crossed[0, j])
          and float(scene.score()[0]) <= 0.05)

    # =========================== 9. near-miss + recovery ====================================
    torch.manual_seed(81)
    env.reset()
    step(20)
    j = first_present()
    seat_shuttle_at(-c.station_x)
    step(10)
    drop_into_basin(j)
    drive_shuttle(c.station_x)
    step(20)
    px, py = float(scene.pad_xy[0, 0]), float(scene.pad_xy[0, 1])
    hover_place(j, (px + 0.16, py, c.cargo_size / 2 + 0.002))  # beside the pad, on the ground
    settle_until(lambda: bool(scene.settled()[0, j]), max_steps=120)
    report("near-miss")
    kp = k_present()
    sc_nm = float(scene.score()[0])
    check("near-miss: clean ferry but parcel set down NEXT to the pad -> crossed credit "
          "only (0.45/k), not delivered",
          bool(scene.crossed[0, j]) and not bool(scene.violated[0, j])
          and not bool(scene.delivered()[0, j])
          and abs(sc_nm - 0.45 / kp) < 0.05 and not bool(scene.success()[0]))
    hover_place(j, (px, py, c.pad_size[2] + c.cargo_size / 2))
    ok = settle_until(lambda: bool(scene.delivered()[0, j]), max_steps=150)
    report("recovered")
    # expected exactly 1.0/k: the recovered parcel earns full delivery credit (0.45/k ->
    # 1/k is a +0.275 jump at k=2 — assert the value, not a fuzzy jump margin)
    check("near-miss recovery: moved onto the pad -> delivery credit (score = 1/k)",
          ok and abs(float(scene.score()[0]) - 1.0 / kp) < 0.05
          and float(scene.score()[0]) > sc_nm)

    # =========================== 10. calibration A: basin drop funnel =======================
    # Geometric funnel: free capture needs the cube fully inside the 47 mm inner face
    # (clean band = basin_inner_half - cargo_half = 19.5 mm); between ~20 and ~47 mm the
    # cube lands on the wall top and tips INWARD (its center is inside the inner-edge
    # pivot), and only past ~47 mm does it tip out — so the hard cliff sits at the inner
    # face, not at the clean band. Published raw, asserted only at the extremes.
    print("[smoke] CALIBRATION A: drop offset -> in-basin rate (3 angles each)", flush=True)
    torch.manual_seed(91)
    env.reset()
    step(10)
    j = first_present()
    seat_shuttle_at(-c.station_x)
    step(10)
    funnel: dict[float, int] = {}
    for off_mm in (0.0, 10.0, 20.0, 35.0, 60.0):
        hits = 0
        for a in range(3):
            # offsets fanned +/-15 deg around +x (toward mid-bridge): per-component
            # geometry stays decisive for the square basin, and a missed drop lands on
            # the deck, safely on side A
            ang = (a - 1) * 0.26
            off = (off_mm / 1000.0 * math.cos(ang), off_mm / 1000.0 * math.sin(ang))
            drop_into_basin(j, xy_off=off)
            settle_until(lambda: bool(scene.settled()[0, j]), max_steps=100)
            hit = bool(scene.in_basin()[0, j])
            hits += int(hit)
            # clear the stage: parcel back to its side-A slot, shuttle re-seated
            write_body(scene.cargos[f"cargo_{j}"], (-0.40, -0.10, c.cargo_size / 2 + 0.002))
            seat_shuttle_at(-c.station_x)
            step(5)
        funnel[off_mm] = hits
        print(f"[smoke]   off={off_mm:.0f}mm: {hits}/3 in basin", flush=True)
    check("calibration A: drop funnel — <=10 mm offsets all captured, 60 mm (past the "
          "inner-edge pivot) never",
          funnel[0.0] == 3 and funnel[10.0] == 3 and funnel[60.0] == 0)

    # =========================== 11. calibration B: traverse-speed retention ================
    # Constant-velocity crossings hammer the parcel with the start/stop inertia jump; the
    # basin walls (35 mm) contain gentle rides. Published raw, asserted only at 0.4 m/s.
    print("[smoke] CALIBRATION B: traverse speed -> ride retention (3 runs each)", flush=True)
    speeds = (0.4, 1.2, 2.4)
    retain: dict[float, int] = {}
    clean: dict[float, int] = {}
    for v in speeds:
        r_ok = 0
        c_ok = 0
        for a in range(3):
            torch.manual_seed(100 + int(v * 10) + a)
            env.reset()
            step(10)
            j = first_present()
            seat_shuttle_at(-c.station_x)
            step(10)
            drop_into_basin(j)
            drive_shuttle(c.station_x, const_v=v)
            step(40)
            r_ok += int(bool(scene.in_basin()[0, j]))
            c_ok += int(bool(scene.crossed[0, j]) and not bool(scene.violated[0, j]))
        retain[v], clean[v] = r_ok, c_ok
        print(f"[smoke]   v={v:.1f} m/s: retained {r_ok}/3, clean crossing {c_ok}/3",
              flush=True)
    check("calibration B: gentle 0.4 m/s traverse retains the parcel and crosses "
          "clean 3/3", retain[0.4] == 3 and clean[0.4] == 3)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.rail_ferry")
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
