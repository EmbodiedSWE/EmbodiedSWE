"""Smoke / rubric-REJECTION battery for BayonetDockScene — NullRobot, teleported probes.

solve.py is the acceptance proof (align -> drop through the slots -> CCW twist to the
stop). This battery proves the rubric REJECTS wrong outcomes — above all that the
seed's own strategy (set the front vessel down at the goal, no orientation control)
scores far below success, and that the "locked" bit corresponds to REAL retention
physics (a locked canister resists the very pull that freely extracts an unlocked
one). Every probe is CONSTRUCTED (teleport, real physics steps, judge) —
instrumentation, never a solution.

Checks:
  1. settle/no-NaN      — seeded reset settles finite: canisters at rest in the row
                          slots, masses authored (3 x 0.12), score 0, no success;
  2. randomization      — over 8 seeded resets the dock yaw takes >= 4 distinct
                          values, the target BODY takes >= 2 values, xy jitter
                          readback differs between resets;
  3. null-policy        — 240 idle steps: no stage latches, score ~0, no success;
  4. SEED STRATEGY      — "put the front vessel on the goal object": the front
                          canister is set down on the dock with NO yaw control (lugs
                          90 degrees off the slots). It PERCHES on the catch flange,
                          ~26 mm above the collar floor: not inserted, score caps at
                          the lift stage (0.15), no success;
  5. inserted-untwisted — the aligned drop (solve phase 2) alone: seated on the
                          collar floor, twist ~0 — inserted() True but success False,
                          score exactly the insert stage (0.45);
  6. under-twist        — continuing 5: twisted CCW ~30 degrees (past the 25-degree
                          partial latch, short of the 45-degree lock gate) — score
                          caps at 0.75, no success;
  7. wrong-direction    — a seated canister driven CLOCKWISE: the back-stop pegs
                          block it at ~ -8 degrees; twist never reaches the [45, 90)
                          window, no twist credit, no success;
  8. retention contrast — a 2 N upward pull (1.7x weight) EXTRACTS a seated-but-
                          untwisted canister (rises > 50 mm), while the SAME pull on
                          a LOCKED canister is jammed by the lugs under the flange
                          (rises < 20 mm) and success survives the assault;
  9. wrong-canister     — a NON-front canister fully docked and locked (geometry
                          perfect, twist past 45): success False, score stays ~0 —
                          only the front canister counts;
 10. beside-the-dock    — the front canister set down on the counter right beside
                          the dock: upright, still, correct height band relative to
                          the counter — but not inserted, no insert credit beyond
                          the lift stage, no success;
 11. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
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
    from . import scene as task_scene  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_WD = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_WD.daemon = True
_WD.start()

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


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    t = int(scene.target[0])
    d_xy, dz = scene._rel_dock()
    tw = math.degrees(float(scene.twist()[0, t]))
    print(f"[smoke] {tag:22s} | tgt={t} d_xy={float(d_xy[0, t]) * 1000:6.1f}mm "
          f"dz={float(dz[0, t]) * 1000:+6.1f}mm twist={tw:+7.2f}deg "
          f"ins={scene.inserted()[0].tolist()} "
          f"latch=(L{int(scene._s_lift[0])},I{int(scene._s_insert[0])},"
          f"T{int(scene._s_twist[0])}) success={bool(scene.success()[0])} "
          f"score={float(scene.score()[0]):.3f} frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bayonet_dock")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.55, -0.70, 1.00)) + o),
                                tuple(np.array((0.05, 0.00, 0.44)) + o),
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

    xs = torch.tensor(c.slot_xs, device=device)

    def bowl_write(i: int, pos_w: torch.Tensor, yaw: float = 0.0) -> None:
        """Teleport canister i (upright at `yaw`, zero velocity) to a world point."""
        _refresh()
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        scene.bowls[i].write_root_state_to_sim(st, _all_ids())
        _refresh()

    def zero_wrench(i: int) -> None:
        z = torch.zeros(n, 1, 3, device=device)
        scene.bowls[i].set_external_force_and_torque(z, z)

    def dock_yaw() -> float:
        _refresh()
        return float(scene._yaw_of(scene.dock.data.root_quat_w)[0])

    def body_in_slot(s: int) -> int:
        _refresh()
        slots = [int((xs - (scene.bowls[i].data.root_pos_w[0, 0]
                            - scene.env_origins[0, 0])).abs().argmin())
                 for i in range(3)]
        return slots.index(s)

    def drop_in(i: int, yaw_off: float = 0.0) -> None:
        """Entry teleport above the collar floor (lugs above the flange ring) at the
        dock yaw + `yaw_off`, then a hands-off gravity drop + settle."""
        _refresh()
        entry = scene.dock.data.root_pos_w.clone()
        entry[:, 2] += c.base_h + 0.028
        bowl_write(i, entry, yaw=dock_yaw() + yaw_off)
        for _ in range(300):
            _step(1)
            if bool(scene.at_rest()[0, i]):
                break
        _step(30)

    def twist_probe(i: int, cut_deg: float, tau0: float = 0.03, w_cap: float = 1.2,
                    sign: float = 1.0, max_steps: int = 1500) -> float:
        """Bang-bang body-z torque on canister i until |twist| passes `cut_deg` in the
        `sign` direction or a persistent stall (a hard stop); returns the settled
        twist (deg). Mirrors solve phase 3, with escalation."""
        tau, tau_max = tau0, 0.12
        best = -math.inf
        stall = 0
        forces = torch.zeros(n, 1, 3, device=device)
        torques = torch.zeros(n, 1, 3, device=device)
        for _ in range(max_steps):
            _refresh()
            deg = sign * math.degrees(float(scene.twist()[0, i]))
            if deg >= cut_deg:
                break
            if deg > best + 0.5:
                best = deg
                stall = 0
            else:
                stall += 1
            if stall >= 90:
                if tau >= tau_max - 1e-9:
                    break  # pressed on a hard stop even at max torque
                tau = min(tau * 1.6, tau_max)
                stall = 0
            wz = sign * float(scene.bowls[i].data.root_ang_vel_w[0, 2])
            torques[:, 0, 2] = sign * (tau if wz < w_cap else 0.0)
            scene.bowls[i].set_external_force_and_torque(forces, torques)
            _step(1)
        zero_wrench(i)
        _step(60)
        _refresh()
        return math.degrees(float(scene.twist()[0, i]))

    def pull_up(i: int, newtons: float, max_steps: int, stop_rise: float) -> float:
        """Constant upward force on canister i; returns the PEAK rise (m) of its
        bottom above the start (sampled every step, force still on — gravity restores
        the state before any post-settle readback). Cuts out early past `stop_rise`."""
        _refresh()
        z_start = float(scene.bowls[i].data.root_pos_w[0, 2])
        forces = torch.zeros(n, 1, 3, device=device)
        forces[:, 0, 2] = newtons
        torques = torch.zeros(n, 1, 3, device=device)
        scene.bowls[i].set_external_force_and_torque(forces, torques)
        peak = 0.0
        for _ in range(max_steps):
            _step(1)
            _refresh()
            peak = max(peak, float(scene.bowls[i].data.root_pos_w[0, 2]) - z_start)
            if peak > stop_rise:
                break
        zero_wrench(i)
        return peak

    # ================= 1. settle / no-NaN + mass authoring ========================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    masses = [float(scene.bowls[i].root_physx_view.get_masses().reshape(-1)[0])
              for i in range(3)]
    print(f"[smoke] authored masses readback: {masses}", flush=True)
    check("settle/no-NaN: seeded reset settles finite — canisters at rest in the row "
          "slots, masses authored (3 x 0.12), score 0, no success",
          bool(scene._finite()[0]) and bool(scene.at_rest()[0].all())
          and all(abs(mv - c.mass) < 0.02 for mv in masses)
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    yaws, targets = set(), set()
    jit = []
    for s in range(8):
        torch.manual_seed(300 + 17 * s)
        env.reset()
        _step(10)
        _refresh()
        yaws.add(round(math.degrees(dock_yaw())))
        targets.add(int(scene.target[0]))
        jit.append(torch.cat([scene.dock.data.root_pos_w[0, :2],
                              scene.bowls[0].data.root_pos_w[0, :2]]).clone())
    d_jit = max(float((a - b).abs().max()) for a in jit for b in jit)
    print(f"[smoke] randomization: dock_yaws={sorted(yaws)} targets={sorted(targets)} "
          f"max_jitter_delta={d_jit * 1000:.1f}mm", flush=True)
    check("randomization-is-real: over 8 seeded resets the dock yaw takes >= 4 "
          "values, the target body takes >= 2 values, xy jitter differs",
          len(yaws) >= 4 and len(targets) >= 2 and d_jit > 0.003)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps — no stage latches, score ~0, no success",
          not bool(scene._s_lift[0]) and not bool(scene._s_insert[0])
          and not bool(scene._s_twist[0]) and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= 4. SEED STRATEGY: set it down at the goal, no yaw control ==================
    # The seed's manipulation model is "put the front vessel on the goal object" —
    # transport plus set-down, judged by an xy/z window, with NO orientation clause.
    # Executed here verbatim: the front canister is lowered onto the dock with its
    # lugs 90 degrees off the slots. The lugs land ON the catch flange and the
    # canister PERCHES ~26 mm above the collar floor: never inserted, never success.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    front = body_in_slot(0)
    _REC["on"] = True
    drop_in(front, yaw_off=math.pi / 2)
    _report("seed-strategy-perch")
    _REC["on"] = False
    _refresh()
    d_xy, dz = scene._rel_dock()
    perch_dz = float(dz[0, front])
    check("SEED STRATEGY (no yaw control): the front canister set down on the dock "
          "with lugs 90 deg off the slots perches ON the flange ~26 mm up — not "
          "inserted, score caps at the 0.15 lift stage, no success",
          perch_dz > c.insert_dz_hi + 0.004
          and not bool(scene.inserted()[0, front])
          and not bool(scene._s_insert[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.w_lift + 1e-4)

    # ================= 5. inserted but untwisted ==================================================
    # The aligned drop alone — solve's phase 2 with the twist omitted. Seated on the
    # collar floor, twist ~0: inserted() True, but success stays False and the score
    # stays exactly at the insert stage. "In the goal container" is NOT the goal.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    front = body_in_slot(0)
    _REC["on"] = True
    drop_in(front, yaw_off=0.0)
    _report("inserted-untwisted")
    _refresh()
    tw0 = abs(math.degrees(float(scene.twist()[0, front])))
    check("inserted-untwisted: the aligned drop seats the canister on the collar "
          "floor (twist ~0) — inserted True, success False, score exactly 0.45",
          bool(scene.inserted()[0, front]) and tw0 < 15.0
          and bool(scene._s_insert[0]) and not bool(scene._s_twist[0])
          and not bool(scene.success()[0])
          and 0.4499 <= float(scene.score()[0]) <= 0.4501)

    # ================= 6. under-twist =============================================================
    # Continue: twist CCW past the 25-degree partial latch but cut at 28 (speed-capped
    # low so the coast is negligible) — well short of the 45-degree lock gate.
    end_tw = twist_probe(front, cut_deg=28.0, w_cap=0.5)
    _report("under-twist")
    _REC["on"] = False
    check("under-twist: seated canister twisted CCW to ~30 deg (past the 25-deg "
          "partial latch, short of the 45-deg lock) — score caps at 0.75, no success",
          25.0 <= end_tw <= 42.0 and bool(scene._s_twist[0])
          and not bool(scene.success()[0])
          and 0.7499 <= float(scene.score()[0]) <= 0.7501)

    # ================= 7. wrong direction: clockwise into the back stops ==========================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    front = body_in_slot(0)
    drop_in(front, yaw_off=0.0)
    end_cw = twist_probe(front, cut_deg=45.0, tau0=0.05, w_cap=0.8, sign=-1.0)
    _report("wrong-direction")
    check("wrong-direction: driven CLOCKWISE the lugs hit the back-stop pegs at "
          "~ -8 deg (never past -20) — no twist credit, no success",
          -20.0 < end_cw <= -3.0 and not bool(scene._s_twist[0])
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.w_insert + 1e-4)

    # ================= 8. retention contrast: the lock is real physics ============================
    # (a) unlocked: a 2 N upward pull (1.7x the 1.2 N weight) freely EXTRACTS the
    # seated-but-untwisted canister — the actuator and force scale demonstrably move
    # this body out of the dock.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    front = body_in_slot(0)
    drop_in(front, yaw_off=0.0)
    rise_free = pull_up(front, 2.0, max_steps=120, stop_rise=0.05)
    _step(60)
    print(f"[smoke] retention: unlocked pull peak rise = {rise_free * 1000:.1f}mm",
          flush=True)
    # (b) locked: fresh episode, full dock + lock (constructed success), then the SAME
    # pull — the lugs jam on the flange bottom within ~6 mm and success survives.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    front = body_in_slot(0)
    _REC["on"] = True
    drop_in(front, yaw_off=0.0)
    end_lock = twist_probe(front, cut_deg=52.0)
    _refresh()
    locked_before = bool(scene.success()[0])
    rise_locked = pull_up(front, 2.0, max_steps=60, stop_rise=0.05)
    _step(90)
    _report("retention-locked")
    _REC["on"] = False
    print(f"[smoke] retention: locked twist={end_lock:+.1f}deg pull peak rise = "
          f"{rise_locked * 1000:.1f}mm", flush=True)
    check("retention-contrast: a 2 N pull extracts an UNTWISTED canister (rise > "
          "50 mm) but a LOCKED one jams on the flange (rise < 20 mm) and success "
          "survives the assault",
          rise_free > 0.05 and locked_before and end_lock >= 45.0
          and rise_locked < 0.020 and bool(scene.success()[0])
          and float(scene.score()[0]) >= 0.9999)

    # ================= 9. wrong canister fully locked =============================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    wrong = body_in_slot(1)   # the MIDDLE canister, not the front one
    drop_in(wrong, yaw_off=0.0)
    end_wr = twist_probe(wrong, cut_deg=52.0)
    _report("wrong-canister")
    _refresh()
    check("wrong-canister: the middle canister perfectly docked and locked (twist "
          "past 45) — success False, score stays ~0: only the FRONT canister counts",
          bool(scene.inserted()[0, wrong]) and end_wr >= 45.0
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.01)

    # ================= 10. set down beside the dock ===============================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    front = body_in_slot(0)
    _refresh()
    beside = scene.dock.data.root_pos_w.clone()
    beside[:, 0] += c.base_r + 0.055
    beside[:, 2] += 0.008
    bowl_write(front, beside, yaw=dock_yaw())
    _step(180)
    _report("beside-the-dock")
    _refresh()
    d_xy, _dz = scene._rel_dock()
    check("beside-the-dock: the front canister set down on the counter next to the "
          "dock (upright, still) — not inserted, no insert credit, no success",
          float(d_xy[0, front]) > 0.10 and not bool(scene.inserted()[0, front])
          and not bool(scene._s_insert[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.w_lift + 1e-4)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.bayonet_dock")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    check("video: frames captured and saved to frames.npz", len(_REC["frames"]) > 10)

    n_pass = sum(ok for _nm, ok in checks)
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
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({exc!r})", flush=True)
        os._exit(1)
