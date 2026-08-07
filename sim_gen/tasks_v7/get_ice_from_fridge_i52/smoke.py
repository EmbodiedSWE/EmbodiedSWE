"""Smoke battery for IceDoserScene (sim_gen task `get_ice_from_fridge_i52`) —
REJECTION-ONLY: every check either verifies basic health/randomization or
executes/constructs a wrong strategy and asserts the rubric refuses it.
success() must never fire anywhere in the battery.

 1. settle/no-NaN     — shuttle at its CLOSED stop, all 5 balls sealed in the
                        machine, cups empty and unstaged, score ~0, no success.
 2. randomization A   — housing yaw spans > 120 deg and xy jitters (readback).
 3. randomization B   — cup slots redraw (blue bearing spans > 90 deg), blue/red
                        stay angularly separated, radii inside the slot band.
 4. null policy       — 240 idle steps -> nothing dispensed, score ~0, no success.
 5. SEED strategy     — the seed's whole skill (hold the dispenser OPEN) run for
                        real: cup staged, shuttle pushed to the OPEN stop and
                        HELD there under constant force for 4 s — EXACTLY ONE
                        ball falls (the shuttle body seals the hopper: the
                        machine meters one per cycle, not per second of hold);
                        score <= stage+one credit, no success.
 6. unstaged dispense — a full pump cycle with NO cup under the outlet: the ball
                        is lost on the ground (retained drops to 4, blue stays
                        0) and NO credit latches — the staging prerequisite is
                        real and the loss is terminal.
 7. overfill          — THREE balls constructed inside the blue cup (past the
                        target count): success refuses; score stays capped.
 8. red-cup poison    — balls constructed into BOTH cups (blue holds the target
                        count, red holds one): every latch refuses while the red
                        cup is occupied; score ~0, no success.
 9. wrong cup         — the RED cup staged under the outlet and TWO real pump
                        cycles run: both balls land in the red cup — staged(),
                        the latches and success all refuse the decoy; score ~0.
10. shuttle left open — a would-be-perfect state (blue staged with exactly 2
                        constructed balls, 3 retained) but the shuttle parked
                        MID-STROKE: success refuses on the closed clause alone.
11. rim lean          — a ball constructed leaning against the OUTSIDE of the
                        staged blue cup's wall (58 mm off-axis): in_cup refuses
                        it — only balls physically inside the cavity count.
12. tipped cup        — the blue cup constructed ON ITS SIDE at the drop line
                        with a ball at its mouth: the upright gate refuses both
                        staged() and in_cup().
13. rejection audit   — success() observed False at every step of the battery.
14. final no-NaN      — and frames.npz written to the CWD.

Run (forge): python -u -m simgen_tasks.get_ice_from_fridge_i52.smoke --headless
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

F_MAX = 8.0
V_MAX = 0.08

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": True, "annot": None, "frames": [], "i": 0}
_AUDIT = {"saw_success": False}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        _AUDIT["saw_success"] |= bool(env.scene.success()[0])
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
    print(f"[smoke] {tag:14s} | d={float(scene.shuttle_d()[0]) * 1000:+7.2f}mm "
          f"blue={int(scene.cup_count('blue')[0])} "
          f"red={int(scene.cup_count('red')[0])} "
          f"kept={int(scene.retained().sum(dim=1)[0])} "
          f"staged={bool(scene.staged()[0])} "
          f"closed={bool(scene.shuttle_closed()[0])} "
          f"L={int(scene._l_stage[0])}{int(scene._l_one[0])}{int(scene._l_all[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ice_doser")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.90, -0.75, 0.62)) + o),
                                tuple(np.array((0.00, 0.00, 0.15)) + o),
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
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def hpq():
        return (scene.housing.data.root_pos_w.clone(),
                scene.housing.data.root_quat_w.clone())

    def housing_pt(lx: float, ly: float, lz: float) -> torch.Tensor:
        hp, hq = hpq()
        loc = torch.tensor([lx, ly, lz], device=device).expand(n, 3)
        return hp + quat_apply(hq, loc)

    def stage_cup(name: str) -> None:
        """CONSTRUCT (transport-only): write a cup upright on the drop line."""
        _, hq = hpq()
        pos = housing_pt(c.out_c, 0.0, 0.0)
        pos = pos.clone()
        pos[:, 2] = 0.002
        _write_body(scene.cups[name], pos, hq)
        _step(60)

    def ball_to(idx: int, pos_w: torch.Tensor) -> None:
        _write_body(scene.balls[idx], pos_w)

    def cup_pt(name: str, dx: float, dy: float, dz: float) -> torch.Tensor:
        cup = scene.cups[name]
        off = torch.tensor([dx, dy, dz], device=device).expand(n, 3)
        return cup.data.root_pos_w + quat_apply(cup.data.root_quat_w, off)

    # --- the pump (real actuation, frame-probe encoded like solve.py) ---
    zero = torch.zeros(n, 1, 3, device=device)
    force = torch.zeros(n, 1, 3, device=device)
    enc_mode = {"m": 0, "flips": 0}

    def encode(fw: torch.Tensor) -> torch.Tensor:
        if enc_mode["m"] == 0:
            return quat_apply_inverse(scene.shuttle.data.root_quat_w,
                                      fw.unsqueeze(0).expand(n, 3))
        return fw.unsqueeze(0).expand(n, 3)

    def push_shuttle(tgt: float, pred, max_steps: int, hold_steps: int = 0) -> bool:
        """PD-push the shuttle toward `tgt` until `pred()` (then optionally KEEP
        PRESSING `hold_steps` more), release, settle hands-off."""
        yaw = float(scene.housing_yaw[0])
        dir_w = torch.tensor([math.cos(yaw), math.sin(yaw), 0.0], device=device)
        done = False
        d0 = float(scene.shuttle_d()[0])
        i = 0
        while i < max_steps + (hold_steps if done else 0):
            d = float(scene.shuttle_d()[0])
            v = float(scene.shuttle.data.root_lin_vel_w[0].dot(dir_w))
            v_des = max(-V_MAX, min(V_MAX, 4.0 * (tgt - d)))
            f = max(-F_MAX, min(F_MAX, 25.0 * (v_des - v)))
            force[:, 0, :] = encode(f * dir_w)
            scene.shuttle.set_external_force_and_torque(force, zero)
            _step(1)
            i += 1
            if not done and pred():
                done = True
                if hold_steps == 0:
                    break
                max_steps = i  # switch the loop budget to the hold
            if not done and i in (90, 240) and enc_mode["flips"] < 2:
                moved = float(scene.shuttle_d()[0]) - d0
                want = tgt - d0
                if abs(want) > 0.010 and moved * math.copysign(1.0, want) < 0.002:
                    enc_mode["m"] ^= 1
                    enc_mode["flips"] += 1
                    d0 = float(scene.shuttle_d()[0])
                    print(f"[smoke] force-frame probe FLIPPED encoding to mode "
                          f"{enc_mode['m']}", flush=True)
        scene.shuttle.set_external_force_and_torque(zero, zero)
        _step(90)
        return done

    def pump_cycle() -> bool:
        """One full close->open->close cycle by contact (no dispensing predicate:
        strokes are judged purely on shuttle travel)."""
        ok = push_shuttle(c.stroke + 0.012,
                          lambda: float(scene.shuttle_d()[0]) >= c.stroke - 0.006, 1200)
        ok &= push_shuttle(-0.012,
                           lambda: float(scene.shuttle_d()[0]) <= 0.006, 1200)
        return ok

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(180)
    _report("reset")
    check("settle/no-NaN: shuttle CLOSED, 5 balls sealed in the machine, cups empty "
          "and unstaged, score ~0, no success",
          bool(scene._finite()[0]) and bool(scene.shuttle_closed()[0])
          and int(scene.retained().sum(dim=1)[0]) == c.n_balls
          and int(scene.cup_count("blue")[0]) == 0
          and int(scene.cup_count("red")[0]) == 0
          and not bool(scene.staged()[0])
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 2+3. randomization readback ================================================
    yaws, xys, slots_b, seps, radii = [], [], [], [], []
    for k in range(8):
        torch.manual_seed(20 + k)
        env.reset()
        _step(6)
        _refresh()
        yaws.append(yaw_of(scene.housing.data.root_quat_w[0]))
        hp = (scene.housing.data.root_pos_w[0] - scene.env_origins[0])[:2]
        xys.append(hp.tolist())
        ab = float(scene.slot_ang[0, 0])
        ar = float(scene.slot_ang[0, 1])
        slots_b.append(math.degrees(ab))
        seps.append(math.degrees(abs((ar - ab + math.pi) % (2 * math.pi) - math.pi)))
        for name in ("blue", "red"):
            d = (scene.cups[name].data.root_pos_w[0] - scene.env_origins[0])[:2]
            radii.append(float((d - hp).norm()))
    yspan = max(yaws) - min(yaws)
    xystd = float(np.std(np.asarray(xys), axis=0).mean())
    bspan = max(slots_b) - min(slots_b)
    print(f"[smoke] readback: yaws={[f'{y:+.0f}' for y in yaws]} span={yspan:.0f} "
          f"xystd={xystd:.3f} blue_slots={[f'{b:+.0f}' for b in slots_b]} "
          f"seps={[f'{s:.0f}' for s in seps]} "
          f"radii=[{min(radii):.3f},{max(radii):.3f}]", flush=True)
    check("randomization A: housing yaw spans > 120 deg and xy jitters across resets",
          yspan > 120.0 and xystd > 0.008)
    check("randomization B: blue slot bearing spans > 90 deg, blue/red separation "
          ">= the configured minimum, radii inside the slot band",
          bspan > 90.0 and all(s >= c.slot_sep_deg - 1.0 for s in seps)
          and all(c.slot_r[0] - 0.01 <= r <= c.slot_r[1] + 0.01 for r in radii))

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> nothing dispensed, score ~0, no success",
          int(scene.retained().sum(dim=1)[0]) == c.n_balls
          and int(scene.cup_count("blue")[0]) == 0
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 5. SEED strategy (hold the dispenser open, for real) =======================
    torch.manual_seed(41)
    env.reset()
    _step(150)
    stage_cup("blue")
    ok5 = push_shuttle(c.stroke + 0.012,
                       lambda: float(scene.shuttle_d()[0]) >= c.stroke - 0.006,
                       1200, hold_steps=480)  # 4 s pressed against the OPEN stop
    _report("seed-hold")
    check("SEED strategy: shuttle held at the OPEN stop under force for 4 s — "
          "EXACTLY ONE ball falls (the shuttle seals the hopper; the machine "
          "meters per cycle, not per second), score <= stage+one, no success",
          ok5 and int(scene.cup_count("blue")[0]) == 1
          and int(scene.cup_count("red")[0]) == 0
          and int(scene.retained().sum(dim=1)[0]) == c.n_balls - 1
          and float(scene.score()[0]) <= c.w_stage + c.w_one + 0.01
          and not bool(scene.success()[0]))

    # ================= 6. unstaged dispense (terminal loss) =======================================
    torch.manual_seed(51)
    env.reset()
    _step(150)
    ok6 = pump_cycle()   # NO cup staged: the metered ball hits open ground
    _step(120)
    _report("unstaged")
    check("unstaged dispense: a full pump cycle with no cup under the outlet loses "
          "the ball on the ground — retained 4, blue 0, NO credit latches",
          ok6 and int(scene.retained().sum(dim=1)[0]) == c.n_balls - 1
          and int(scene.cup_count("blue")[0]) == 0
          and int(scene.cup_count("red")[0]) == 0
          and not bool(scene._l_stage[0]) and not bool(scene._l_one[0])
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 7. overfill ================================================================
    torch.manual_seed(61)
    env.reset()
    _step(150)
    stage_cup("blue")
    ball_to(2, cup_pt("blue", 0.012, 0.0, c.cup_floor_t + c.ball_r + 0.002))
    ball_to(3, cup_pt("blue", -0.012, 0.0, c.cup_floor_t + c.ball_r + 0.002))
    ball_to(4, cup_pt("blue", 0.0, 0.0, c.cup_floor_t + 3 * c.ball_r + 0.004))
    _step(120)
    _report("overfill")
    check("overfill: THREE balls constructed in the blue cup (past the target of "
          "2) — success refuses, score stays capped below 1",
          int(scene.cup_count("blue")[0]) == 3
          and float(scene.score()[0]) <= 0.70 + 0.01
          and not bool(scene.success()[0]))

    # ================= 8. red-cup poison ==========================================================
    torch.manual_seed(71)
    env.reset()
    _step(150)
    ball_to(3, cup_pt("blue", 0.012, 0.0, c.cup_floor_t + c.ball_r + 0.002))
    ball_to(4, cup_pt("blue", -0.012, 0.0, c.cup_floor_t + c.ball_r + 0.002))
    ball_to(2, cup_pt("red", 0.0, 0.0, c.cup_floor_t + c.ball_r + 0.002))
    _step(120)
    _report("red-poison")
    check("red-cup poison: blue holds the target count but red holds one — every "
          "latch refuses while the red cup is occupied; score ~0, no success",
          int(scene.cup_count("blue")[0]) == 2
          and int(scene.cup_count("red")[0]) == 1
          and not bool(scene._l_one[0]) and not bool(scene._l_all[0])
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 9. wrong cup staged ========================================================
    torch.manual_seed(81)
    env.reset()
    _step(150)
    stage_cup("red")
    ok9 = pump_cycle() and pump_cycle()
    _step(120)
    _report("wrong-cup")
    check("wrong cup: RED cup staged and two real cycles pumped — both balls land "
          "in the decoy; staged(), latches and success all refuse; score ~0",
          ok9 and int(scene.cup_count("red")[0]) == 2
          and int(scene.cup_count("blue")[0]) == 0
          and not bool(scene.staged()[0]) and not bool(scene._l_stage[0])
          and not bool(scene._l_one[0])
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 10. shuttle left open ======================================================
    torch.manual_seed(91)
    env.reset()
    _step(150)
    stage_cup("blue")
    ball_to(3, cup_pt("blue", 0.012, 0.0, c.cup_floor_t + c.ball_r + 0.002))
    ball_to(4, cup_pt("blue", -0.012, 0.0, c.cup_floor_t + c.ball_r + 0.002))
    # park the shuttle MID-STROKE (0.030: pocket sealed off both holes) and carry
    # its pocket ball + restack the throat ball so the construct is penetration-free
    _, hq10 = hpq()
    _write_body(scene.shuttle, housing_pt(0.030, 0.0, c.floor_z1 + c.sh_hz + 0.001), hq10)
    ball_to(0, housing_pt(0.030, 0.0, c.floor_z1 + c.ball_r + 0.001))
    ball_to(1, housing_pt(0.003, 0.0, c.floor_z1 + 2 * c.sh_hz + c.ball_r + 0.001))
    _step(120)
    _report("left-open")
    check("shuttle left open: blue staged with exactly 2 balls, 3 retained, but "
          "the shuttle parked mid-stroke — success refuses on the CLOSED clause "
          "alone",
          int(scene.cup_count("blue")[0]) == 2
          and int(scene.cup_count("red")[0]) == 0
          and int(scene.retained().sum(dim=1)[0]) == c.n_balls - 2
          and not bool(scene.shuttle_closed()[0])
          and float(scene.score()[0]) <= 0.70 + 0.01
          and not bool(scene.success()[0]))

    # ================= 11. rim lean ===============================================================
    torch.manual_seed(101)
    env.reset()
    _step(150)
    stage_cup("blue")
    lean = c.cup_in_r + c.cup_wall_t + c.ball_r   # 58 mm off-axis: OUTSIDE the wall
    ball_to(4, cup_pt("blue", lean, 0.0, c.ball_r + 0.001))
    _step(120)
    _report("rim-lean")
    check("rim lean: a ball resting against the OUTSIDE of the staged cup's wall "
          "(58 mm off-axis) is refused by in_cup — no count, no success",
          int(scene.cup_count("blue")[0]) == 0 and not bool(scene._l_one[0])
          and float(scene.score()[0]) <= c.w_stage + 0.01
          and not bool(scene.success()[0]))

    # ================= 12. tipped cup =============================================================
    torch.manual_seed(111)
    env.reset()
    _step(150)
    hp12, hq12 = hpq()
    qx90 = torch.zeros(n, 4, device=device)
    qx90[:, 0] = math.cos(math.pi / 4)
    qx90[:, 1] = math.sin(math.pi / 4)
    q_side = quat_mul(hq12, qx90)   # cup rolled 90 deg: axis horizontal
    pos12 = housing_pt(c.out_c, 0.0, 0.0).clone()
    pos12[:, 2] = c.cup_in_r + c.cup_wall_t + 0.002
    _write_body(scene.cups["blue"], pos12, q_side)
    mouth = scene.cups["blue"].data.root_pos_w + quat_apply(
        q_side, torch.tensor([0.0, 0.0, 0.030], device=device).expand(n, 3))
    _write_body(scene.balls[4], mouth)
    _refresh()
    geo_in = int(scene.cup_count("blue")[0])   # upright gate must already refuse
    _step(120)
    _report("tipped-cup")
    check("tipped cup: the blue cup on its side at the drop line with a ball at "
          "its mouth — the upright gate refuses staged() and in_cup()",
          geo_in == 0 and int(scene.cup_count("blue")[0]) == 0
          and not bool(scene.staged()[0]) and not bool(scene._l_stage[0])
          and not bool(scene._l_one[0]) and not bool(scene.success()[0]))

    # ================= 13 + 14. audit, video, verdict =============================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ice_doser")
        print(f"[smoke] wrote {args.out}: {arr.shape}", flush=True)
    check("video: >10 frames recorded", len(_REC["frames"]) > 10)

    n_pass = sum(1 for _, ok in checks if ok)
    if n_pass == len(checks):
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
        code = 0
    else:
        for nm, ok in checks:
            if not ok:
                print(f"[smoke] FAILED: {nm}", flush=True)
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        code = 1
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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
