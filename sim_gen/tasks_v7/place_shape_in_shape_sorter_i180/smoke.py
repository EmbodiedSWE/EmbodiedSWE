"""Smoke / rubric-REJECTION battery for FeedLineCullScene — NullRobot, teleported probes.

solve.py is the acceptance proof (cull the red reject to open ground, pull the
release pin out by its knob, gravity runs the queue into the catch bin). This
battery proves the rubric REJECTS every wrong outcome and that the PIN GATE —
the task's strategic differentiator from the seed — is physically load-bearing:
the seated pin refuses a real shove on the whole queue, the knob makes the pin
one-way, and no constructed settled state reaches success() unless every blue
is genuinely banked AND the red reject is genuinely off the line AND the pin is
fully out.

Checks:
   1. settle/no-NaN     — seeded reset settles finite: queue closes up against the
                          seated pin, nothing banked; score ~0, no success;
   2. randomization     — two seeded resets: READBACK rig yaw, rig xy, pin seat,
                          red queue station (and report blue count / red slot);
   3. null-policy       — 240 idle steps: queue held on the slope, pin seated,
                          score ~0, no success;
   4. SEED STRATEGY     — "just put every shape in the sorter": pin out, ALL cubes
                          — including the RED reject — constructed settled in the
                          bin: blues bank, but success is FALSE (the cull clause
                          has teeth) and score stays <= 0.80;
   5. blue-left-out     — red culled, pin out, every blue banked EXCEPT one settled
                          on the ground beside the machine: no success;
   6. red-still-on-line — red resting on the ground UNDER the chute (inside the
                          machine footprint): red_clear is FALSE, no credit;
   7. pin-gate reality  — a real downhill shove (0.9 N per cube, 2 s) presses the
                          whole queue on the seated pin: the gate holds, nothing
                          banks, the pin does not budge (non-vacuous: the queue
                          is verified to be pressing the pin);
   8. knob one-way      — a real 4 N axial push toward the knob-less side moves the
                          pin only millimetres before the knob arrests it (non-
                          vacuous: it does reach the stop); queue still held;
   9. cull-skipped      — the solve's own extraction servo runs WITHOUT the cull:
                          the mechanism genuinely delivers (pin fully out, queue in
                          the bin) but the red reject lands in the bin too — no
                          success, score <= 0.80: order matters, culling is load-
                          bearing;
  10. top-end stop      — a real uphill shove on the queue is arrested by the top
                          end wall: nothing banks, the pin stays seated, score ~0;
  11. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.place_shape_in_shape_sorter_i180.smoke --headless
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

_qmul, _qz = task_scene._qmul, task_scene._qz
_slope_basis, _slope_point = task_scene._slope_basis, task_scene._slope_point

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
    pl = scene.pin_local()[0]
    rp = scene._rig_local(scene.red.data.root_pos_w)[0]
    bk = scene.blues_banked()[0]
    pres = scene.present[0]
    banked = "".join("B" if (pres[j] and bk[j]) else ("." if pres[j] else "_")
                     for j in range(scene.cfg.max_blue))
    print(f"[smoke] {tag:16s} | pin_y={float(pl[1]):+.4f} "
          f"red=({float(rp[0]):+.3f},{float(rp[1]):+.3f},{float(rp[2]):+.3f}) "
          f"banked=[{banked}] red_clear={bool(scene.red_clear()[0])} "
          f"pin_out={bool(scene.pin_out()[0])} settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.feed_line_cull")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.30, -1.10, 0.85)) + o),
                                tuple(np.array((0.18, 0.00, 0.18)) + o),
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

    def pin_y() -> float:
        _refresh()
        return float(scene.pin_local()[0, 1])

    zero = torch.zeros(n, 1, 3, device=device)
    ey = torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
    u3, n3 = _slope_basis(c.pitch_deg)
    u_t = torch.tensor(u3, device=device)

    def rig_vec(v3) -> torch.Tensor:
        _refresh()
        vv = torch.tensor(v3, device=device, dtype=torch.float).expand(n, 3)
        return quat_apply(scene.rig.data.root_quat_w, vv)

    def put_rig_local(body, loc_xyz, quat=None) -> None:
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        pos = scene.rig.data.root_pos_w + quat_apply(scene.rig.data.root_quat_w, loc)
        _write_body(body, pos, scene.rig.data.root_quat_w if quat is None else quat)

    def cube_s(body) -> float:
        """Along-slope station of a cube (rig-local, distance from the lip)."""
        _refresh()
        p = scene._rig_local(body.data.root_pos_w)[0]
        lip = torch.tensor([c.lip_x, 0.0, c.lip_z], device=device)
        return float(((p - lip) * u_t).sum())

    def queue_bodies():
        out = [scene.red]
        for j in range(c.max_blue):
            if bool(scene.present[0, j]):
                out.append(scene.blues[j])
        return out

    def shove_queue(direction_local, newton: float, steps: int,
                    v_cap: float = 1e9) -> None:
        """Real force on every queue cube along a rig-local direction. v_cap
        keeps the push quasi-static (memory: fast slews vault low lips far
        below the static retention angle) — force gates off above the cap."""
        bodies = queue_bodies()
        for _ in range(steps):
            d_w = rig_vec(direction_local)
            for b in bodies:
                v_al = float((b.data.root_lin_vel_w * d_w).sum(dim=-1)[0])
                f = newton if v_al < v_cap else 0.0
                f_b = quat_apply_inverse(b.data.root_quat_w, f * d_w)
                b.set_external_force_and_torque(f_b.reshape(n, 1, 3), zero)
            _step(1)
        for b in bodies:
            b.set_external_force_and_torque(zero, zero)
        _step(1)

    def pull_pin(v_des: float, stop_y: float, max_steps: int) -> None:
        """The solve's own extraction servo (same gains): axial velocity-servo
        force + gravity feedforward + weak axis-alignment torque."""
        pin_m = float(scene.pin.root_physx_view.get_masses()[0].sum())
        for _ in range(max_steps):
            _refresh()
            ey_w = quat_apply(scene.rig.data.root_quat_w, ey)
            v_ax = float((scene.pin.data.root_lin_vel_w * ey_w).sum(dim=-1)[0])
            f_ax = max(-2.0, min(8.0, 6.0 * (v_des - v_ax) + 0.5))
            q = scene.pin.data.root_quat_w
            f_w = f_ax * ey_w + pin_m * 9.81 * ez
            py_w = quat_apply(q, ey)
            t_w = 0.02 * torch.cross(py_w, ey_w, dim=-1) \
                - 0.002 * scene.pin.data.root_ang_vel_w
            scene.pin.set_external_force_and_torque(
                quat_apply_inverse(q, f_w).reshape(n, 1, 3),
                quat_apply_inverse(q, t_w).reshape(n, 1, 3))
            _step(1)
            if pin_y() >= stop_y:
                break
        scene.pin.set_external_force_and_torque(zero, zero)
        _step(1)

    bin_slots = [(-0.055, -0.038), (-0.055, 0.038), (0.005, -0.038), (0.005, 0.038)]
    z_bin = c.bin_floor_t + c.cube / 2 + 0.002

    def blues_present():
        return [j for j in range(c.max_blue) if bool(scene.present[0, j])]

    def all_present_banked() -> bool:
        _refresh()
        bk = scene.blues_banked()[0]
        return all(bool(bk[j]) for j in blues_present())

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(180)
    _report("settle")
    _REC["on"] = False
    _refresh()
    on_chute = all(float(scene._rig_local(b.data.root_pos_w)[0, 2]) > 0.20
                   for b in queue_bodies())
    front_s = min(cube_s(b) for b in queue_bodies())
    print(f"[smoke] queue front station s={front_s:.4f} "
          f"(pin gate at {c.brk_s + c.pin_r + c.cube / 2:.4f})", flush=True)
    check("settle/no-NaN: queue settles finite against the seated pin, everything "
          "on the chute, nothing banked, score ~0, no success",
          bool(scene._finite()[0]) and on_chute and abs(pin_y()) < 0.006
          and front_s < c.brk_s + c.pin_r + c.cube / 2 + 0.008
          and not bool(scene.blues_banked()[0].any())
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        ryaw = yaw_of(scene.rig.data.root_quat_w[0])
        rxy = scene.rig.data.root_pos_w[0, :2].clone()
        return (ryaw, rxy, pin_y(), cube_s(scene.red),
                int(scene.n_blue[0]), int(scene.red_slot[0]))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_xy, a_pin, a_reds, a_nb, a_rs = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_xy, b_pin, b_reds, b_nb, b_rs = readback()
    d_yaw = dyaw(a_yaw, b_yaw)
    d_xy = float((a_xy - b_xy).norm())
    d_pin = abs(a_pin - b_pin)
    d_reds = abs(a_reds - b_reds)
    print(f"[smoke] randomization deltas: rig_yaw={d_yaw:.2f}deg rig_xy={d_xy * 1000:.1f}mm "
          f"pin_seat={d_pin * 1000:.2f}mm red_station={d_reds * 1000:.1f}mm "
          f"n_blue {a_nb}->{b_nb} red_slot {a_rs}->{b_rs}", flush=True)
    check("randomization-is-real: rig yaw, rig xy, pin seat and the reject's queue "
          "station readback all differ across seeds (blue count / red slot drawn "
          "per episode)",
          d_yaw > 1.0 and d_xy > 0.003 and (d_pin > 0.0002 or d_reds > 0.001
                                            or a_nb != b_nb or a_rs != b_rs))

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    _refresh()
    on_chute = all(float(scene._rig_local(b.data.root_pos_w)[0, 2]) > 0.20
                   for b in queue_bodies())
    check("null-policy-fails: 240 idle steps — the pin holds the queue on the "
          "slope, nothing banks, score ~0, no success",
          on_chute and abs(pin_y()) < 0.008 and float(scene.score()[0]) <= 0.02
          and not bool(scene.success()[0]))

    # ================= 4. negative: the SEED'S OWN STRATEGY =======================================
    # rlbench/place_shape_in_shape_sorter: every piece goes INTO the box. Construct
    # the analogue: pin fully out, ALL cubes — including the red reject — settled
    # in the bin. The blues bank, the pin clause holds, but the cull clause fails.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    put_rig_local(scene.pin, (c.lip_x + 0.10, 0.22, c.pin_r + 0.001))
    for k, j in enumerate(blues_present()):
        put_rig_local(scene.blues[j], (*bin_slots[k], z_bin))
    put_rig_local(scene.red, (0.062, 0.0, z_bin))
    _step(180)
    _report("seed-dump-all")
    _REC["on"] = False
    check("negative (SEED strategy): pin out and EVERY cube dumped in the bin — "
          "the reject included — banks the blues yet success is FALSE (the cull "
          "clause has teeth) and score <= 0.80",
          all_present_banked() and bool(scene.red_in_bin()[0])
          and bool(scene.pin_out()[0]) and not bool(scene.red_clear()[0])
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.8000005)

    # ================= 5. near-miss: one blue left out ============================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    put_rig_local(scene.pin, (c.lip_x + 0.10, 0.22, c.pin_r + 0.001))
    pres = blues_present()
    for k, j in enumerate(pres[1:]):
        put_rig_local(scene.blues[j], (*bin_slots[k], z_bin))
    put_rig_local(scene.blues[pres[0]], (-0.05, 0.30, c.cube / 2 + 0.002))
    put_rig_local(scene.red, (0.40, -0.30, c.cube / 2 + 0.002))
    _step(180)
    _report("blue-left-out")
    _refresh()
    bk = scene.blues_banked()[0]
    check("near-miss (blue left out): red culled, pin out, every blue banked "
          "EXCEPT one settled on the ground beside the machine — no success",
          bool(scene.red_clear()[0]) and bool(scene.pin_out()[0])
          and not bool(bk[pres[0]])
          and all(bool(bk[j]) for j in pres[1:])
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.8000005)

    # ================= 6. near-miss: red on the ground but still ON the machine ===================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    put_rig_local(scene.red, (0.35, 0.10, c.cube / 2 + 0.002))
    _step(180)
    _report("red-under-chute")
    _refresh()
    rp = scene._rig_local(scene.red.data.root_pos_w)[0]
    check("near-miss (red on the line): red resting on the ground UNDER the chute, "
          "inside the machine footprint — red_clear FALSE, no cull credit, no "
          "success",
          float(rp[2]) < 0.08 and not bool(scene.red_clear()[0])
          and not bool(scene._red_latch[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.02)

    # ================= 7. pin-gate reality: a real shove cannot force the gate ====================
    torch.manual_seed(100)
    env.reset()
    _step(180)
    front0 = min(cube_s(b) for b in queue_bodies())
    _REC["on"] = True
    shove_queue((-u3[0], -u3[1], -u3[2]), 0.9, 240)  # downhill, ~2.3x slope gravity
    _step(60)
    _report("gate-shoved")
    _REC["on"] = False
    _refresh()
    front1 = min(cube_s(b) for b in queue_bodies())
    on_chute = all(float(scene._rig_local(b.data.root_pos_w)[0, 2]) > 0.20
                   for b in queue_bodies())
    print(f"[smoke] gate shove: front station {front0:.4f} -> {front1:.4f}, "
          f"pin_y={pin_y():+.4f}", flush=True)
    check("pin-gate reality: 0.9 N per cube pressed downhill for 2 s — the queue "
          "verifiably presses the gate yet nothing passes (no bank, queue on the "
          "chute, pin seated within hole slack)",
          front0 < c.brk_s + c.pin_r + c.cube / 2 + 0.008  # pressing, non-vacuous
          and front1 > c.brk_s - 0.012 and on_chute
          and not bool(scene.blues_banked()[0].any())
          and abs(pin_y()) < 0.010 and float(scene.score()[0]) <= 0.02
          and not bool(scene.success()[0]))

    # ================= 8. knob one-way stop =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(120)
    pin_m = float(scene.pin.root_physx_view.get_masses()[0].sum())
    for _ in range(180):
        _refresh()
        ey_w = quat_apply(scene.rig.data.root_quat_w, ey)
        f_w = -4.0 * ey_w + pin_m * 9.81 * ez
        q = scene.pin.data.root_quat_w
        scene.pin.set_external_force_and_torque(
            quat_apply_inverse(q, f_w).reshape(n, 1, 3), zero)
        _step(1)
    scene.pin.set_external_force_and_torque(zero, zero)
    y_pressed = pin_y()
    _step(60)
    _report("knob-stop")
    _refresh()
    on_chute = all(float(scene._rig_local(b.data.root_pos_w)[0, 2]) > 0.20
                   for b in queue_bodies())
    print(f"[smoke] knob press: pin_y={y_pressed:+.4f} (clearance "
          f"{c.knob_y - c.knob_len / 2 - c.chan_half - c.wall_t:+.4f})", flush=True)
    check("knob one-way: a real 4 N axial push toward the knob-less side moves the "
          "pin only its few-mm clearance before the knob arrests it (non-vacuous: "
          "it reaches the stop); the queue stays held",
          -0.012 < y_pressed < -0.002 and on_chute
          and not bool(scene.blues_banked()[0].any())
          and not bool(scene.success()[0]))

    # ================= 9. cull skipped: the mechanism delivers, the reject sinks the run ==========
    torch.manual_seed(100)
    env.reset()
    _step(180)
    _REC["on"] = True
    pull_pin(0.06, 0.175, 900)
    _step(600)  # gravity runs the whole line, reject included
    _report("cull-skipped")
    _REC["on"] = False
    _refresh()
    check("cull-skipped rejection: the solve's own pin extraction WITHOUT the cull "
          "genuinely delivers the queue (pin fully out, blues banked) but the red "
          "reject lands in the bin too — success FALSE, score <= 0.80",
          pin_y() > c.pin_out_y - 0.03 and all_present_banked()
          and bool(scene.red_in_bin()[0]) and not bool(scene.red_clear()[0])
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.8000005)

    # ================= 10. top-end stop ===========================================================
    torch.manual_seed(100)
    env.reset()
    _step(180)
    back0 = max(cube_s(b) for b in queue_bodies())
    shove_queue(u3, 0.6, 240, v_cap=0.35)  # uphill, quasi-static
    # peak station BEFORE release: once the force stops, gravity beats friction
    # and the whole queue slides straight back down to the pin
    back_peak = max(cube_s(b) for b in queue_bodies())
    _step(90)
    _report("top-shoved")
    _refresh()
    back1 = max(cube_s(b) for b in queue_bodies())
    print(f"[smoke] top shove: back station {back0:.4f} -> peak {back_peak:.4f} "
          f"-> settled {back1:.4f} (top wall at {c.slope_len:.3f})", flush=True)
    check("top-end stop: a real uphill shove is arrested by the end wall — the "
          "cubes verifiably ran uphill (peak station), never left the chute's "
          "top, slid back once released; nothing banks, the pin stays seated, "
          "score ~0",
          back_peak > back0 + 0.02 and back_peak < c.slope_len + 0.02
          and back1 < back_peak + 0.005
          and not bool(scene.blues_banked()[0].any()) and abs(pin_y()) < 0.010
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.feed_line_cull")
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
    except Exception as exc:  # noqa: BLE001 - Kit teardown hangs; die loudly NOW
        print(f"SIM_GEN_SMOKE: FAIL (exception: {exc!r})", flush=True)
        import traceback

        traceback.print_exc()
        os._exit(1)
