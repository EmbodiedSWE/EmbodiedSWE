"""Smoke battery for WedgeJackScene (sim_gen task `two_robot_pick_cube_i315`) —
REJECTION-ONLY: every check either verifies basic health/randomization or CONSTRUCTS
a settled wrong outcome and asserts the rubric refuses it. success() must never fire
anywhere in the battery.

 1. settle/no-NaN     — car resting at q=0 on the guide stop (positive gap to the
                        ram), crate on its apron slot, score ~0, no success.
 2. randomization A   — station yaw spans > 90 deg, xy jitters, the sampled ram
                        start varies AND reads back from the wedge pose.
 3. randomization B   — crate side takes BOTH signs, slot x varies, and the crate
                        settles at its recorded slot (position READBACK).
 4. null policy       — 240 idle steps -> score ~0, no success.
 5. SEED strategy     — the seed's whole skill (hold the cargo at the elevated goal
                        pose) executed literally: the crate HELD at the delivery
                        level above the deck earns ~0 while held (aboard's z clause
                        refuses altitude); released, it lands on the DOWN deck —
                        aboard latch only, platform never rose, no success.
 6. hoist cheat       — car + crate teleport-HELD at q = 54 mm with the ram still
                        out: supported() is False throughout (height inconsistent
                        with the insertion), the lift latch never pays, and on
                        release the car falls straight back to q ~ 0; no success.
 7. empty jack        — the ram REALLY driven to full insertion (sanctioned drive,
                        actuation verified: q >= q_req readback) with the crate left
                        on the slab: aboard-gated score stays ~0, the raised-empty +
                        crate-on-slab settled state is refused, and the self-locking
                        wedge HOLDS the lift for 2 s hands-off (physical check).
 8. low near-miss     — crate aboard, ram driven to q ~ 36 mm (14 mm SHORT of the
                        delivery level), everything settled and supported: the q
                        clause alone refuses; latched credit stays under the cap.
 9. latched credit    — continuing from 8: the crate stolen back to the slab — the
                        aboard + partial-lift latches survive, success does not.
10. rejection audit   — success() observed False at every step of the battery.
11. final no-NaN.
12. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.two_robot_pick_cube_i315.smoke --headless
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
    print(f"[smoke] {tag:18s} | q={float(scene.q()[0]) * 1000:+7.2f}mm "
          f"xw={float(scene.xw()[0]) * 1000:+7.1f}mm "
          f"q_ram={float(scene.q_from_ram()[0]) * 1000:+7.2f}mm "
          f"aboard={bool(scene.aboard()[0])} sup={bool(scene.supported()[0])} "
          f"latch(ab)={int(scene._aboard[0])} lift={float(scene._lift[0]):.2f} "
          f"settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.wedge_jack")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.05, -0.95, 0.70)) + o),
                                tuple(np.array((0.08, 0.00, 0.10)) + o),
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

    def succ() -> bool:
        s = bool(scene.success()[0])
        _AUDIT["saw_success"] |= s
        return s

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def station_world(local_xyz) -> torch.Tensor:
        loc = torch.tensor(local_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.station.data.root_pos_w \
            + quat_apply(scene.station.data.root_quat_w, loc)

    def station_local(pos_w: torch.Tensor) -> torch.Tensor:
        return quat_apply_inverse(scene.station.data.root_quat_w,
                                  pos_w - scene.station.data.root_pos_w)

    def load_crate() -> None:
        """CONSTRUCT (transport-only): crate released 4 cm above the deck centre;
        the drop and seating are gravity + deck/rim contact."""
        car_p = scene.car.data.root_pos_w
        car_q = scene.car.data.root_quat_w
        loc = torch.tensor([0.0, 0.0, c.crate_size / 2 + 0.04],
                           device=device).expand(n, 3)
        _write_body(scene.crate, car_p + quat_apply(car_q, loc), car_q)
        _step(150)

    def drive_to_q(q_target: float, max_steps: int = 2400) -> None:
        """REAL actuation: the sanctioned ram_drive force servos the insertion until
        the car lift reaches `q_target` (same plant + gains as solve), then hands
        off and settles — the wedge must self-lock."""
        q0c, tan_t = scene.q_at_coeff()
        xw_goal = (q0c - q_target) / tan_t
        f_ff = 8.0
        xw_ckpt = scene.xw().clone()
        for i in range(max_steps):
            rem = scene.xw() - xw_goal
            v_des = (2.0 * rem).clamp(0.008, 0.05)
            scene.ram_drive[:] = (f_ff + 60.0 * (v_des - scene.ins_rate)) \
                .clamp(0.0, c.f_max - 1.0)
            _step(1)
            if bool((scene.q() >= q_target).all()):
                break
            if i % 180 == 179:
                xw_now = scene.xw()
                if float((xw_ckpt - xw_now).min()) < 0.002:
                    f_ff = min(f_ff * 1.5, 18.0)
                xw_ckpt = xw_now.clone()
        scene.ram_drive[:] = 0.0
        _step(120)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(150)
    _report("reset")
    loc1 = station_local(scene.crate.data.root_pos_w)[0]
    check("settle/no-NaN: car at q=0 on the guide stop with a positive gap to the "
          "ram, crate settled on its apron slot, score ~0, no success",
          bool(scene._finite()[0]) and bool(scene.settled()[0])
          and abs(float(scene.q()[0])) < 0.003
          and float(scene.q_from_ram()[0]) < 1e-6
          and abs(float(loc1[0]) - float(scene.crate_slot[0, 0])) < 0.02
          and abs(float(loc1[1]) - float(scene.crate_slot[0, 1])) < 0.02
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2+3. randomization readback ================================================
    yaws, xys, xw0s, sides, cxs = [], [], [], [], []
    ram_rb_ok, slot_rb_ok = True, True
    for k in range(10):
        torch.manual_seed(20 + k)
        env.reset()
        _step(60)
        _refresh()
        yaws.append(yaw_of(scene.station.data.root_quat_w[0]))
        xys.append((scene.station.data.root_pos_w[0]
                    - scene.env_origins[0])[:2].tolist())
        xw0s.append(float(scene.xw0[0]))
        sides.append(1.0 if float(scene.crate_slot[0, 1]) > 0 else -1.0)
        cxs.append(float(scene.crate_slot[0, 0]))
        # ram READBACK: the wedge pose realizes the sampled start
        ram_rb_ok = ram_rb_ok and abs(float(scene.xw()[0]) - xw0s[-1]) < 0.004
        # slot READBACK: the crate settled where the sample put it
        loc = station_local(scene.crate.data.root_pos_w)[0]
        slot_rb_ok = slot_rb_ok \
            and abs(float(loc[0]) - float(scene.crate_slot[0, 0])) < 0.02 \
            and abs(float(loc[1]) - float(scene.crate_slot[0, 1])) < 0.02
    yspan = max(yaws) - min(yaws)
    xystd = float(np.std(np.asarray(xys), axis=0).mean())
    xw0span = max(xw0s) - min(xw0s)
    cxspan = max(cxs) - min(cxs)
    print(f"[smoke] readback: yaws={[f'{y:+.0f}' for y in yaws]} span={yspan:.0f} "
          f"xystd={xystd:.3f} xw0s={[f'{x * 1000:.0f}' for x in xw0s]}mm "
          f"span={xw0span * 1000:.0f}mm sides={sides} cxspan={cxspan * 1000:.0f}mm "
          f"ram_rb={ram_rb_ok} slot_rb={slot_rb_ok}", flush=True)
    check("randomization A: station yaw spans > 90 deg, xy jitters, ram start "
          "varies and reads back from the wedge pose",
          yspan > 90.0 and xystd > 0.008 and xw0span > 0.012 and ram_rb_ok)
    check("randomization B: crate side takes both signs, slot x varies, crate "
          "settles at its recorded slot",
          (min(sides) < 0 < max(sides)) and cxspan > 0.02 and slot_rb_ok)

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 5. SEED strategy (hold the cargo at the elevated goal pose) ===============
    torch.manual_seed(41)
    env.reset()
    _step(120)
    hold_p = station_world([0.0, 0.0, c.beam_z0 + c.crate_size / 2 + 0.002])
    hold_q = scene.car.data.root_quat_w.clone()
    held_ok = True
    for _ in range(90):
        _write_body(scene.crate, hold_p, hold_q)
        _step(1)
        held_ok = held_ok and float(scene.score()[0]) <= 0.05 and not succ()
    _step(180)  # release: it falls onto the DOWN deck and seats there
    _report("seed-hold")
    s5 = float(scene.score()[0])
    check("SEED strategy: the crate HELD at the delivery level above the deck earns "
          "~0 the whole hold (altitude without the platform is refused); released, "
          "it seats on the DOWN deck — aboard latch only, no success",
          held_ok and bool(scene.aboard()[0]) and abs(float(scene.q()[0])) < 0.003
          and 0.29 <= s5 <= 0.36 and not succ())

    # ================= 6. hoist cheat (platform lifted by hand, ram untouched) ===================
    torch.manual_seed(51)
    env.reset()
    _step(120)
    load_crate()  # legitimate aboard latch (0.30) — the cheat must add NOTHING
    q_h = 0.054
    sup_seen, s6max = False, 0.0
    car_hold = station_world([0.0, 0.0, c.deck_z_lo + q_h])
    crate_hold = station_world([0.0, 0.0, c.deck_z_lo + q_h + c.crate_size / 2])
    hold_q = scene.station.data.root_quat_w.clone()
    for _ in range(90):
        _write_body(scene.car, car_hold, hold_q)
        _write_body(scene.crate, crate_hold, hold_q)
        _step(1)
        sup_seen = sup_seen or bool(scene.supported()[0])
        s6max = max(s6max, float(scene.score()[0]))
        succ()
    scene.resync_rate()
    _step(240)  # release: the car must fall straight back to the guide stop
    _report("hoist-release")
    s6 = float(scene.score()[0])
    check("hoist cheat: car + crate teleport-held at q = 54 mm with the ram out — "
          "supported() False throughout, the lift latch never pays (score stays at "
          "the aboard latch + support-tol crumb), and on release the car falls back "
          "to q ~ 0; no success",
          (not sup_seen) and s6max <= 0.36 and abs(float(scene.q()[0])) < 0.004
          and s6 <= 0.36 and not succ())

    # ================= 7. empty jack (aboard-gated lift + self-lock physical check) ===============
    torch.manual_seed(61)
    env.reset()
    _step(120)
    drive_to_q(c.q_req + 0.002)
    _report("empty-jacked")
    q7 = float(scene.q()[0])
    reached7 = q7 >= c.q_req  # actuation verified: the car REALLY rose
    s7 = float(scene.score()[0])
    q_hold0 = scene.q().clone()
    _step(240)  # 2 s hands-off: the wedge must self-lock under the car's weight
    drift7 = float((scene.q() - q_hold0).abs().max())
    loc7 = station_local(scene.crate.data.root_pos_w)[0]
    on_slab7 = abs(float(loc7[2]) - c.crate_size / 2) < 0.012
    _report("self-lock")
    check("empty jack: ram REALLY driven to full insertion (q >= q_req readback) "
          "with the crate on the slab — aboard-gated score stays ~0, the settled "
          "raised-empty + crate-on-slab state is refused, and the self-locking "
          "wedge holds the lift hands-off (drift < 2 mm over 2 s)",
          reached7 and s7 <= 0.05 and drift7 < 0.002 and on_slab7
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 8. low near-miss (everything right except the level) =======================
    torch.manual_seed(71)
    env.reset()
    _step(120)
    load_crate()
    drive_to_q(c.q_req - 0.014)
    _report("low-miss")
    q8 = float(scene.q()[0])
    s8 = float(scene.score()[0])
    check("low near-miss: crate aboard, ram driven to ~36 mm — 14 mm short of the "
          "delivery level — settled and supported: the q clause alone refuses; "
          "latched credit stays under the 0.75 cap",
          bool(scene.aboard()[0]) and bool(scene.supported()[0])
          and bool(scene.settled()[0]) and 0.030 <= q8 <= c.q_req - 0.006
          and 0.50 <= s8 <= c.score_cap + 1e-5 and not succ())

    # ================= 9. latched credit survives theft ==========================================
    steal_p = station_world([0.50, 0.15, c.crate_size / 2 + 0.003])
    _write_body(scene.crate, steal_p, scene.station.data.root_quat_w)
    _step(180)
    _report("theft")
    s9 = float(scene.score()[0])
    check("latched credit: the crate stolen back to the slab — the aboard + "
          "partial-lift latches survive, success does not",
          not bool(scene.aboard()[0]) and 0.50 <= s9 <= c.score_cap + 1e-5
          and abs(s9 - s8) < 0.02 and not succ())

    # ================= 10-12. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.wedge_jack")
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
