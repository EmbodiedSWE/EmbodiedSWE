"""Smoke battery for TransferCarouselScene (sim_gen task `franka_handover_i115`) —
REJECTION-ONLY: every check either verifies basic health/randomization or CONSTRUCTS
a settled wrong outcome and asserts the rubric refuses it. success() must never fire
anywhere in the battery.

 1. settle/no-NaN     — parcels on their apron slots, bay open-side, score ~0,
                        no success.
 2. randomization A   — station yaw spans > 90 deg, xy jitters, drum theta0 varies
                        across resets.
 3. randomization B   — target COLOUR varies, colour->slot permutation varies, the
                        pedestal tile matches the target, and every parcel settles
                        on the slot its permutation names (position READBACK).
 4. null policy       — 240 idle steps -> score ~0, no success.
 5. SEED strategy     — the seed's whole skill (carry the object to the
                        destination) executed as a construct: the target parcel
                        released in free air above the alcove centre. It lands ON
                        the canopy roof — never in the bay; score stays ~0.
 6. load interlock    — the EMPTY bay rotated into the alcove by REAL drive torque
                        (actuation verified: the drum demonstrably reached the
                        sector), then the parcel dropped from above the bay's
                        position: the 14 mm canopy slot rejects it — it rests on
                        the roof; empty rotation earned ~0 (progress is
                        aboard-gated); no success.
 7. wrong parcel      — a DISTRACTOR loaded in the bay and rotated to the alcove
                        (real drive): distractor-in-bay + target-missing refuse;
                        score ~0 (the load latch is target-only).
 8. angle near-miss   — target loaded, rotated to 26 deg SHORT of alcove centre
                        (outside the 20 deg sector): no success; latched partial
                        credit stays under the 0.75 cap.
 9. containment miss  — drum posed at the alcove, target parcel resting on the
                        PLATTER beside the bay (outside the bay box): in-bay
                        refuses; no success.
10. latched credit    — target loaded (latch pays 0.35) then stolen back off the
                        drum: the latch survives, success does not.
11. rejection audit   — success() observed False at every step of the battery.
12. final no-NaN.
13. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.franka_handover_i115.smoke --headless
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
    tgt_in, dis_in = scene._bay_flags()
    print(f"[smoke] {tag:18s} | theta={math.degrees(float(scene.theta()[0])):+7.1f} "
          f"err={math.degrees(float(scene.sector_err()[0])):6.1f}deg "
          f"rate={float(scene.rate_fd[0]):+.3f} tgt={int(scene.target_idx[0])} "
          f"tgt_in={bool(tgt_in[0])} dis_in={bool(dis_in[0])} "
          f"latch(load)={int(scene._load[0])} prog={float(scene._prog[0]):.2f} "
          f"settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.transfer_carousel")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.95, -0.85, 0.75)) + o),
                                tuple(np.array((0.00, 0.00, 0.08)) + o),
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

    def tgt_k() -> int:
        return int(scene.target_idx[0])

    def drop_parcel(k: int) -> None:
        """CONSTRUCT (transport-only): parcel k released 5 cm above the bay; the
        drop and seating are gravity + bay-wall contact."""
        drum_p = scene.drum.data.root_pos_w
        drum_q = scene.drum.data.root_quat_w
        loc = torch.tensor([c.bay_r, 0.0, c.platter_z1 + c.parcel_size / 2 + 0.05],
                           device=device).expand(n, 3)
        _write_body(scene.parcels[k], drum_p + quat_apply(drum_q, loc), drum_q)
        _step(120)

    def drive_to(goal: float, max_steps: int = 2400,
                 stop_err: float = 0.06, stop_rate: float = 0.08) -> None:
        """REAL actuation: the sanctioned drum_drive torque servos the drum to
        `goal` (same cascaded gains as solve), then hands off and settles."""
        for _ in range(max_steps):
            err = task_scene._wrap(
                torch.full((n,), goal, device=device) - scene.theta())
            w_des = (1.5 * err).clamp(-1.2, 1.2)
            scene.drum_drive[:] = (0.8 * (w_des - scene.rate_fd)).clamp(-1.1, 1.1)
            _step(1)
            if bool((err.abs() < stop_err).all()) \
                    and bool((scene.rate_fd.abs() < stop_rate).all()):
                break
        scene.drum_drive[:] = 0.0
        _step(120)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(150)
    _report("reset")
    check("settle/no-NaN: parcels on their apron slots, bay open-side, score ~0, "
          "no success",
          bool(scene._finite()[0]) and bool(scene.settled()[0])
          and float(scene.sector_err()[0]) > 1.5
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2+3. randomization readback ================================================
    yaws, xys, th0s, tgts, perms = [], [], [], [], []
    tiles_ok, slots_ok = True, True
    for k in range(8):
        torch.manual_seed(20 + k)
        env.reset()
        _step(60)
        _refresh()
        yaws.append(yaw_of(scene.station.data.root_quat_w[0]))
        xys.append((scene.station.data.root_pos_w[0]
                    - scene.env_origins[0])[:2].tolist())
        th0s.append(float(scene.theta0[0]))
        tgts.append(tgt_k())
        perms.append(tuple(scene.slot_of[0].tolist()))
        # tile READBACK: the target's tile stands on the pedestal, the others don't
        px, py = c.ped_center
        for j in range(3):
            loc = station_local(scene.tiles[j].data.root_pos_w)[0]
            on_ped = (abs(float(loc[0]) - px) < 0.03 and abs(float(loc[1]) - py) < 0.03
                      and abs(float(loc[2]) - c.ped_h) < 0.03)
            tiles_ok = tiles_ok and (on_ped == (j == tgts[-1]))
        # slot READBACK: every parcel settled on the slot its permutation names
        for j in range(3):
            az = math.radians(c.slot_az_deg[int(scene.slot_of[0, j])])
            loc = station_local(scene.parcels[j].data.root_pos_w)[0]
            slots_ok = slots_ok \
                and abs(float(loc[0]) - c.slot_r * math.cos(az)) < 0.035 \
                and abs(float(loc[1]) - c.slot_r * math.sin(az)) < 0.035
    yspan = max(yaws) - min(yaws)
    xystd = float(np.std(np.asarray(xys), axis=0).mean())
    th0span = math.degrees(max(th0s) - min(th0s))
    print(f"[smoke] readback: yaws={[f'{y:+.0f}' for y in yaws]} span={yspan:.0f} "
          f"xystd={xystd:.3f} th0span={th0span:.0f}deg tgts={tgts} "
          f"perms={len(set(perms))} tiles_ok={tiles_ok} slots_ok={slots_ok}",
          flush=True)
    check("randomization A: station yaw spans > 90 deg, xy jitters, theta0 varies",
          yspan > 90.0 and xystd > 0.008 and th0span > 20.0)
    check("randomization B: target colour varies, slot permutation varies, the "
          "pedestal tile matches the target, parcels settle on their named slots",
          len(set(tgts)) >= 2 and len(set(perms)) >= 2 and tiles_ok and slots_ok)

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 5. SEED strategy (direct carry to the destination) =========================
    torch.manual_seed(41)
    env.reset()
    _step(120)
    _write_body(scene.parcels[tgt_k()],
                station_world([-c.bay_r, 0.0, c.roof_z1 + c.parcel_size / 2 + 0.04]),
                scene.station.data.root_quat_w)
    _step(300)
    _report("seed-carry")
    loc5 = station_local(scene.parcels[tgt_k()].data.root_pos_w)[0]
    tgt_in5, _ = scene._bay_flags()
    check("SEED strategy: the parcel carried straight to the alcove centre lands ON "
          "the canopy roof — never in the bay; score stays ~0, no success",
          float(loc5[2]) > c.roof_z0 + 0.010 and not bool(tgt_in5[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 6. load interlock (empty bay in the alcove cannot be loaded) ===============
    torch.manual_seed(51)
    env.reset()
    _step(120)
    drive_to(math.pi)
    _report("empty-rotated")
    reached6 = float(scene.sector_err()[0]) < c.sector_tol  # actuation verified
    s6 = float(scene.score()[0])
    bay_p = scene.drum.data.root_pos_w + quat_apply(
        scene.drum.data.root_quat_w,
        torch.tensor([c.bay_r, 0.0, c.roof_z1 + c.parcel_size / 2 + 0.04],
                     device=device).expand(n, 3))
    _write_body(scene.parcels[tgt_k()], bay_p, scene.drum.data.root_quat_w)
    _step(300)
    _report("interlock-drop")
    loc6 = station_local(scene.parcels[tgt_k()].data.root_pos_w)[0]
    tgt_in6, _ = scene._bay_flags()
    check("load interlock: the empty bay REALLY rotated into the alcove (drive "
          "torque, verified) earned ~0 (aboard-gated), and a parcel dropped onto "
          "the bay's position rests on the canopy roof, not in the bay; no success",
          reached6 and s6 <= 0.05 and float(loc6[2]) > c.roof_z0 + 0.010
          and not bool(tgt_in6[0]) and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 7. wrong parcel ============================================================
    torch.manual_seed(61)
    env.reset()
    _step(120)
    dis = (tgt_k() + 1) % 3
    drop_parcel(dis)
    drive_to(math.pi)
    _report("wrong-parcel")
    tgt_in7, dis_in7 = scene._bay_flags()
    check("wrong parcel: a DISTRACTOR loaded and rotated to the alcove — "
          "distractor-in-bay + target-missing refuse; score ~0 (target-only latch)",
          bool(dis_in7[0]) and not bool(tgt_in7[0])
          and float(scene.sector_err()[0]) < c.sector_tol
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 8. angle near-miss =========================================================
    torch.manual_seed(71)
    env.reset()
    _step(120)
    drop_parcel(tgt_k())
    drive_to(math.pi - 0.45)
    _report("angle-miss")
    tgt_in8, _ = scene._bay_flags()
    s8 = float(scene.score()[0])
    check("angle near-miss: target loaded, rotated to 26 deg short of alcove "
          "centre (sector is 20 deg) — no success; latched credit under the cap",
          bool(tgt_in8[0]) and float(scene.sector_err()[0]) > c.sector_tol + 0.02
          and 0.34 <= s8 <= c.score_cap + 1e-5 and not succ())

    # ================= 9. containment near-miss ===================================================
    torch.manual_seed(81)
    env.reset()
    _step(120)
    # pose the drum at the alcove (no interpenetration at any angle: clearances are
    # rotation-invariant), re-anchor the FD rate, then rest the target parcel on the
    # PLATTER beside the bay — inside the alcove but OUTSIDE the bay box
    q_pi = task_scene._qmul(scene.station.data.root_quat_w,
                            task_scene._qz(torch.full((n,), math.pi, device=device)))
    _write_body(scene.drum, scene.station.data.root_pos_w.clone(), q_pi)
    _step(1)
    scene.resync_rate()
    beside = scene.drum.data.root_pos_w + quat_apply(
        scene.drum.data.root_quat_w,
        torch.tensor([c.bay_r, c.bay_in + c.bay_t + c.parcel_size / 2 + 0.004,
                      c.platter_z1 + c.parcel_size / 2 + 0.003],
                     device=device).expand(n, 3))
    _write_body(scene.parcels[tgt_k()], beside, scene.drum.data.root_quat_w)
    _step(240)
    _report("containment-miss")
    tgt_in9, _ = scene._bay_flags()
    check("containment near-miss: target resting on the platter BESIDE the bay at "
          "the alcove — the in-bay box refuses; no success",
          not bool(tgt_in9[0]) and float(scene.sector_err()[0]) < c.sector_tol
          and not succ())

    # ================= 10. latched credit =========================================================
    torch.manual_seed(91)
    env.reset()
    _step(120)
    drop_parcel(tgt_k())
    latched = bool(scene._load[0])
    _write_body(scene.parcels[tgt_k()],
                (scene.env_origins[0:1]
                 + torch.tensor([0.9, 0.9, c.parcel_size / 2 + 0.002], device=device)
                 ).expand(n, 3))
    _step(180)
    _report("latch")
    tgt_in10, _ = scene._bay_flags()
    s10 = float(scene.score()[0])
    check("latched credit: target loaded (latch pays) then stolen off the drum — "
          "the latch survives, success does not",
          latched and not bool(tgt_in10[0]) and 0.34 <= s10 <= 0.40 and not succ())

    # ================= 11-13. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.transfer_carousel")
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
