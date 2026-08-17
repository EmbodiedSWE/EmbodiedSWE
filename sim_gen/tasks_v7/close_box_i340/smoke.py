"""Smoke / rubric-REJECTION battery for SlamShutCourierScene — NullRobot, teleported
probes.

solve.py is the acceptance proof (one honest shove: launch force, coast, arrest flip).
This battery proves the rubric REJECTS wrong outcomes, and that the physical claims
the task rests on — the low roof jams any in-place lid closure, the lid is gravity-
bistable, a gentle delivery leaves the box open — are load-bearing. Every probe is
CONSTRUCTED (teleport, real physics steps, judge) — instrumentation, never a
solution: success() is monitored at EVERY step and must never turn True anywhere in
the battery (the audit is itself a check).

Checks:
   1. settle/no-NaN      — seeded reset settles finite; lid at its open over-center
                           rest (~-120 deg), box on the pad, cargo inside; score 0;
   2. randomization      — three seeded resets: READBACK fixture xy+yaw, box
                           alley-x and cargo box-frame xy; max-pairwise deltas real;
   3. null-policy        — 240 idle steps: score ~0, no success;
   4. SEED STRATEGY      — the seed's own plan (push the hinged lid shut): a real
                           closing force on the lid at the spawn drives it only a few
                           degrees before it JAMS on the low roof (~-110 deg, far
                           short of the `shut` latch), and it falls back to the open
                           rest when released — no credit, no success;
   5. MECHANISM: gentle  — a quasi-static CoM push delivers the box all the way to
                           the bumper; the lid never leaves the open band and the
                           delivered box is NOT a success: the slam is load-bearing;
                           entered+arrived latch (score 0.40), no more;
   6. near-miss: parked  — box constructed just short of the deliver band, settled:
                           `arrived` latches but no success; the closing sweep is
                           STILL roof-blocked there (the closure window opens only
                           at the bumper);
   7. closed-undelivered — box constructed mid-alley with the lid CLOSED (consistent
                           linkage write), settled: closed but not delivered -> no
                           success, score <= 0.70;
   8. cargo-outside      — delivered + closed constructed at the bumper but the cargo
                           first parked outside the alley -> cargo_in False, no
                           success (the contents clause is load-bearing);
   9. settle gate        — the exact success pose sliding backward at ~0.35 m/s ->
                           success refuses while anything moves (probe dismantled
                           before it can settle into a real success);
  10. rejection audit    — success() was never True at any step of this battery;
  11. score-cap audit    — score never exceeded 0.70 anywhere in this battery;
  12. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.close_box_i340.smoke --headless
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

_qapply, _qmul, _qinv = task_scene._qapply, task_scene._qmul, task_scene._qinv
_qy, encode_force = task_scene._qy, task_scene.encode_force

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
_AUD = {"on": False, "hits": 0, "smax": 0.0}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if _AUD["on"]:
            if bool(env.scene.success()[0]):
                _AUD["hits"] += 1
            _AUD["smax"] = max(_AUD["smax"], float(env.scene.score()[0]))
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


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    p = scene.box_local()[0]
    print(f"[smoke] {tag:18s} | box_x={float(p[0]):+.3f} y={float(p[1]):+.3f} "
          f"lid={math.degrees(float(scene.lid_angle()[0])):+.1f}deg "
          f"deliv={bool(scene.delivered()[0])} closed={bool(scene.closed()[0])} "
          f"cargo_in={bool(scene.cargo_in()[0])} "
          f"ent={bool(scene._entered[0])} arr={bool(scene._arrived[0])} "
          f"shut={bool(scene._shut[0])} settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.slam_shut_courier")().build(num_envs=args.num_envs,
                                                       device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.35, -0.80, 0.70)) + o),
                                tuple(np.array((0.60, 0.00, 0.10)) + o),
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

    def q_fix() -> torch.Tensor:
        _refresh()
        return scene.fixture.data.root_quat_w.clone()

    def fix_pt(loc_xyz) -> torch.Tensor:
        _refresh()
        loc = torch.tensor(list(loc_xyz), device=device).expand(n, 3)
        return scene.fixture.data.root_pos_w + _qapply(q_fix(), loc)

    def write_linkage(box_x: float, lid_deg: float,
                      lin_vel: torch.Tensor | None = None) -> None:
        """Write box + lid CONSISTENTLY (one linkage) at fixture-x `box_x` with the
        lid at `lid_deg` (0 = closed), back to back with no stepping in between."""
        qf = q_fix()
        bpos = fix_pt((box_x, 0.0, c.slab_t))
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = bpos
        st[:, 3:7] = qf
        if lin_vel is not None:
            st[:, 7:10] = lin_vel
        scene.box.write_root_state_to_sim(st, _all_ids())
        hinge = torch.tensor([-c.box_hx, 0.0, c.box_h], device=device).expand(n, 3)
        stl = torch.zeros(n, 13, device=device)
        stl[:, 0:3] = bpos + _qapply(qf, hinge)
        stl[:, 3:7] = _qmul(qf, _qy(torch.full((n,), math.radians(lid_deg),
                                               device=device)))
        if lin_vel is not None:
            stl[:, 7:10] = lin_vel
        scene.lid.write_root_state_to_sim(stl, _all_ids())
        _refresh()

    def write_cargo(pos_w: torch.Tensor, lin_vel: torch.Tensor | None = None) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3] = 1.0
        if lin_vel is not None:
            st[:, 7:10] = lin_vel
        scene.cargo.write_root_state_to_sim(st, _all_ids())
        _refresh()

    zero = torch.zeros(n, 1, 3, device=device)

    def lid_close_press(tag: str, hold: int = 240) -> tuple[float, float]:
        """Push the LID toward closed with a real horizontal force at its CoM (the
        seed's own action) while the BOX is HELD in place by a per-step root rewrite.
        The hold is essential to make this a test of IN-PLACE closure: on the slick
        alley an unheld box is simply dragged to the bumper by the press and slammed
        shut — i.e. the press performs the intended launch strategy, not the seed's.
        Returns (peak angle deg, gain deg from start); peak measured DURING the hold."""
        u3 = _qapply(q_fix(), torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(n, 3))
        box_hold = torch.zeros(n, 13, device=device)
        box_hold[:, 0:3] = scene.box.data.root_pos_w.clone()
        box_hold[:, 3:7] = scene.box.data.root_quat_w.clone()
        a0 = math.degrees(float(scene.lid_angle()[0]))
        peak = a0
        mode = 0
        q_ref = scene.lid.data.root_quat_w.clone()
        for i in range(hold):
            scene.box.write_root_state_to_sim(box_hold.clone(), _all_ids())
            f_world = torch.zeros(n, 3, device=device)
            f_world[0, :] = 2.0 * u3[0, :]
            f_arg = encode_force(mode, q_ref, scene.lid.data.root_quat_w, f_world)
            scene.lid.set_external_force_and_torque(
                f_arg.view(n, 1, 3), zero, env_ids=_all_ids(), is_global=True)
            _step(1)
            peak = max(peak, math.degrees(float(scene.lid_angle()[0])))
            if i == 59 and peak < a0 + 1.5:
                mode = 1 - mode
                print(f"[smoke] {tag}: lid not moving; force-frame mode -> {mode}",
                      flush=True)
        scene.lid.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
        print(f"[smoke] {tag}: lid pressed from {a0:.1f} to peak {peak:.1f} deg "
              f"(shut latch at {-c.shut_deg:.0f}, jam predicted "
              f"~{-math.degrees(c.alpha_cross):.0f})", flush=True)
        return peak, peak - a0

    def slow_push_to(x_target: float, tag: str) -> bool:
        """Quasi-static velocity-regulated CoM push on the BOX (same actuation family
        as solve.py but ~12x slower) until fixture-x reaches `x_target`."""
        u3 = _qapply(q_fix(), torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(n, 3))
        u_xy = u3[0, :2] / max(float(u3[0, :2].norm()), 1e-6)
        q_ref = scene.box.data.root_quat_w.clone()
        mode = 0
        floor_f = 0.8
        win_i, win_px = 0, float(scene.box_local()[0, 0])
        for i in range(2400):
            px = float(scene.box_local()[0, 0])
            if px >= x_target:
                scene.box.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
                return True
            v = scene.box.data.root_lin_vel_w[0, :2]
            v_along = float((v * u_xy).sum())
            f_along = 8.0 * (0.08 - v_along)
            if abs(v_along) < 0.02 and f_along < floor_f:
                f_along = floor_f
            f_along = min(max(f_along, -1.0), 3.0)
            f_world = torch.zeros(n, 3, device=device)
            f_world[0, :2] = u_xy * f_along
            f_arg = encode_force(mode, q_ref, scene.box.data.root_quat_w, f_world)
            scene.box.set_external_force_and_torque(
                f_arg.view(n, 1, 3), zero, env_ids=_all_ids(), is_global=True)
            _step(1)
            if i - win_i >= 45:
                px2 = float(scene.box_local()[0, 0])
                if px2 < win_px - 0.004:
                    mode = 1 - mode
                    print(f"[smoke] {tag}: moving backward; mode -> {mode}", flush=True)
                elif px2 < win_px + 0.004:
                    floor_f = min(floor_f + 0.4, 2.5)
                    print(f"[smoke] {tag}: stalled at x={px2:.3f}; floor -> "
                          f"{floor_f:.1f} N", flush=True)
                win_i, win_px = i, px2
        scene.box.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
        print(f"[smoke] {tag}: push timed out at x={float(scene.box_local()[0, 0]):.3f}",
              flush=True)
        return False

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _AUD["on"] = True
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    lid0 = math.degrees(float(scene.lid_angle()[0]))
    check("settle/no-NaN: layout settles finite; lid at its open over-center rest, "
          "box on the launch pad, cargo inside; score 0, no success",
          bool(scene._finite()[0]) and abs(lid0 + c.open_deg) < 5.0
          and abs(float(scene.box_local()[0, 0]) - c.box_spawn_x) < c.box_x_jitter + 0.01
          and bool(scene.cargo_in()[0]) and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= 2. randomization is real (3-seed max-pairwise readback) ===================
    def readback():
        _refresh()
        loc = _qapply(_qinv(scene.box.data.root_quat_w),
                      scene.cargo.data.root_pos_w - scene.box.data.root_pos_w)
        return (scene.fixture.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.fixture.data.root_quat_w[0]),
                float(scene.box_local()[0, 0]),
                loc[0, :2].clone())

    obs = []
    for s in (101, 202, 303):
        torch.manual_seed(s)
        env.reset()
        _step(10)
        obs.append(readback())
    d_f = max(float((a[0] - b[0]).norm()) for a in obs for b in obs)
    d_y = max(dyaw(a[1], b[1]) for a in obs for b in obs)
    d_bx = max(abs(a[2] - b[2]) for a in obs for b in obs)
    d_k = max(float((a[3] - b[3]).norm()) for a in obs for b in obs)
    print(f"[smoke] randomization max-pairwise deltas: fix_xy={d_f * 1000:.1f}mm "
          f"fix_yaw={d_y:.1f}deg box_x={d_bx * 1000:.1f}mm cargo_xy={d_k * 1000:.1f}mm",
          flush=True)
    check("randomization-is-real: fixture xy+yaw, box alley-x and cargo box-frame xy "
          "readback differ across seeds",
          d_f > 0.004 and d_y > 2.0 and d_bx > 0.002 and d_k > 0.003)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 4. negative: the SEED'S OWN STRATEGY (push the lid shut) ===================
    # The seed closes its box by pushing the hinged lid shut. Here the same action —
    # a real closing force on the lid, box held in place — jams the lid on the low
    # roof a few degrees off its rest, ~110 deg short of closed, and gravity returns
    # it to the open rest on release. (Unheld, the press just drags the box down the
    # slick alley and slams it — the intended strategy.) The central honesty assert.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    peak, gain = lid_close_press("seed-strategy")
    _step(120)
    back = math.degrees(float(scene.lid_angle()[0]))
    _report("seed-strategy")
    _REC["on"] = False
    check("negative (SEED strategy): a real closing press on the lid (box held in "
          "place) moves it a few degrees then JAMS on the low roof far short of the "
          "`shut` latch, and the lid falls back open on release — no latch, no "
          "success, score ~0",
          gain >= 3.0 and peak <= -100.0 and back <= -110.0
          and not bool(scene._shut[0]) and float(scene.score()[0]) <= 0.05
          and not bool(scene.success()[0]))

    # ================= 5. MECHANISM: gentle delivery leaves the box open ==========================
    # Quasi-static CoM push all the way to the bumper: the arrest is too slow to
    # carry the lid over its balance point. The check asserts the box genuinely
    # reached the deliver band (a stalled probe fails, not vacuously) and that the
    # lid never left the open band on the way.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    ok5 = slow_push_to(c.x_delivered + 0.002, "gentle")
    _step(150)
    _report("gentle-delivery")
    _REC["on"] = False
    lid5 = math.degrees(float(scene.lid_angle()[0]))
    s5 = float(scene.score()[0])
    check("MECHANISM (gentle delivery): a quasi-static push delivers the box to the "
          "bumper but the lid stays at its open rest — delivered without closed is "
          "NOT success; entered+arrived latch (score 0.40), the slam is load-bearing",
          ok5 and bool(scene.delivered()[0]) and lid5 <= -100.0
          and not bool(scene.closed()[0]) and not bool(scene._shut[0])
          and abs(s5 - (c.w_enter + c.w_arrive)) <= 1e-3
          and not bool(scene.success()[0]))

    # ================= 6. near-miss: parked just short + still roof-blocked =======================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    park_x = c.x_arrived + 0.004
    write_linkage(park_x, -c.open_deg)
    _step(90)
    _report("parked-short")
    ok6a = bool(scene._arrived[0]) and not bool(scene.delivered()[0]) \
        and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.45
    peak6, gain6 = lid_close_press("parked-press", hold=180)
    _step(90)
    _report("parked-press")
    check("near-miss (parked short): box just outside the deliver band — `arrived` "
          "latches but no success; and the closing sweep is STILL roof-blocked there "
          "(the closure window opens only at the bumper)",
          ok6a and gain6 >= 2.0 and peak6 <= -95.0 and not bool(scene._shut[0])
          and not bool(scene.success()[0]))

    # ================= 7. negative: lid closed but box undelivered ================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    write_linkage(0.30, 0.0)
    _step(120)
    _report("closed-mid-alley")
    check("negative (closed undelivered): lid constructed CLOSED mid-alley — closed "
          "without delivered is not success; score <= 0.70",
          bool(scene.closed()[0]) and not bool(scene.delivered()[0])
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.70 + 1e-4)

    # ================= 8. negative: cargo outside =================================================
    # Park the cargo OUTSIDE the alley first, then construct delivered+closed at the
    # bumper: no stepped prefix can be a genuine success (the cargo is already out).
    torch.manual_seed(100)
    env.reset()
    _step(30)
    far = torch.zeros(n, 3, device=device)
    far[:, 0] = env.iscene.env_origins[:, 0] - 0.25
    far[:, 1] = env.iscene.env_origins[:, 1] + 0.45
    far[:, 2] = env.iscene.env_origins[:, 2] + c.cargo / 2 + 0.003
    write_cargo(far)
    write_linkage(c.x_bump - c.box_hx - 0.002, 0.0)
    _step(120)
    _report("cargo-outside")
    check("negative (cargo outside): box delivered and lid closed but the cargo cube "
          "is out of the box — cargo_in False, no success (contents clause is "
          "load-bearing)",
          bool(scene.delivered()[0]) and bool(scene.closed()[0])
          and not bool(scene.cargo_in()[0]) and not bool(scene.success()[0]))

    # ================= 9. settle gate =============================================================
    # The exact success pose — delivered, closed, cargo inside — but the whole
    # linkage still sliding backward at ~0.35 m/s. success() must refuse while
    # anything moves. The probe is dismantled (transport) well before friction could
    # stop it into a real success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    qf = q_fix()
    u3 = _qapply(qf, torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(n, 3))
    vel = torch.zeros(n, 3, device=device)
    vel[:, :2] = -0.35 * u3[:, :2]
    write_linkage(c.x_bump - c.box_hx - 0.002, 0.0, lin_vel=vel)
    write_cargo(fix_pt((c.x_bump - c.box_hx - 0.002, 0.0,
                        c.slab_t + c.box_floor_t + c.cargo / 2 + 0.002)), lin_vel=vel)
    moving_ok = True
    for _ in range(3):
        _step(1)
        v = float(scene.box.data.root_lin_vel_w[0].norm())
        moving_ok = moving_ok and v > 0.1 and not bool(scene.settled()[0]) \
            and not bool(scene.success()[0])
    _report("settle-gate")
    # dismantle before it can settle into a real success
    write_linkage(0.28, -c.open_deg)
    _step(30)
    check("settle gate: the exact success pose still sliding at ~0.35 m/s is refused "
          "while anything moves",
          moving_ok)

    # ================= 10+11. audits ==============================================================
    check("rejection audit: success() was never True at any step of this battery",
          _AUD["hits"] == 0)
    print(f"[smoke] max score observed anywhere in the battery: {_AUD['smax']:.4f}",
          flush=True)
    check("score-cap audit: score never exceeded 0.70 anywhere in this battery",
          _AUD["smax"] <= 0.70 + 1e-4)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.slam_shut_courier")
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
