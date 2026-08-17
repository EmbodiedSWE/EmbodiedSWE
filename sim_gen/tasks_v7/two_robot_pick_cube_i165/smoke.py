"""Smoke battery for SortingTowerScene (sim_gen task `two_robot_pick_cube_i165`) —
REJECTION-ONLY: every check either verifies basic health/randomization or CONSTRUCTS
a settled wrong outcome and asserts the rubric refuses it. success() must never fire
anywhere in the battery.

 1. settle/no-NaN     — discs on their slab slots, shaft empty, score ~0, no success.
 2. randomization A   — tower yaw spans > 90 deg, xy jitters across resets.
 3. randomization B   — slot permutation varies, WHICH size the red disc is varies,
                        and position READBACK: every green disc settles on the slot
                        its permutation names, the active red on the fourth slot,
                        the two inactive reds in the ground depot.
 4. null policy       — 240 idle steps -> score ~0, no success.
 5. SEED strategy     — the seed's whole skill (carry each object to the
                        destination, no size reasoning) executed with REAL drops in
                        DESCENDING order: the large disc seats, then the mid and
                        small discs land on top of it and STRAND far above their
                        bands; score stays ~one seat (0.22), no success.
 6. order interlock   — mid disc dropped first (REAL drop, seats at its own band,
                        seating verified), then the small disc dropped: the seated
                        mid disc wedges the shaft and the small disc perches on it,
                        outside its band; seated = (F, T, F), no success.
 7. cleanliness       — all three greens CONSTRUCTED falling 6 mm onto their exact
                        seats AND the red disc resting on top inside the shaft: all
                        three seat latches pay (score = 0.66 cap) yet success is
                        refused by the red-inside clause alone.
 8. red poison        — the ACTIVE red disc really dropped through the mouth: it
                        seats at its size's band (red_inside), worth ~0; the
                        same-size GREEN dropped after it perches one disc height
                        above the band (the tightest z near-miss) — not seated,
                        score ~0, no success.
 9. wrong place       — a green disc resting flat on the SLAB beside the tower at
                        exactly its seat height (z and flatness both inside
                        tolerance): the xy clause alone refuses.
10. latched credit    — small disc really dropped (latch pays 0.22) then stolen
                        back out of the tower: the latch survives, success does not.
11. rejection audit   — success() observed False at every step of the battery.
12. final no-NaN.
13. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.two_robot_pick_cube_i165.smoke --headless
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
    seat = scene.seated()[0]
    zs = [float(scene._tower_local(scene.cargo[k].data.root_pos_w)[0, 2])
          for k in range(3)]
    print(f"[smoke] {tag:18s} | seated={[bool(v) for v in seat]} "
          f"z={[f'{z:+.3f}' for z in zs]} "
          f"red={int(scene.active_red[0])} red_in={bool(scene.red_inside()[0])} "
          f"latch={[int(v) for v in scene._seat[0]]} "
          f"settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.sorting_tower")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.95, -0.85, 0.75)) + o),
                                tuple(np.array((0.00, 0.00, 0.10)) + o),
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

    def tower_local(pos_w: torch.Tensor) -> torch.Tensor:
        return scene._tower_local(pos_w)

    def drop(body, h: float, steps: int = 360) -> None:
        """CONSTRUCT (transport-only): the disc released in free air 3 cm above the
        mouth rim, axis vertical; descent and seating are gravity + funnel/tube
        contact — exactly the sanctioned insert move."""
        _write_body(body, scene.tower_world([0.0, 0.0, c.mz1 + 0.03 + h / 2]),
                    scene.tower.data.root_quat_w)
        _step(steps)

    def score0() -> float:
        return float(scene.score()[0])

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(150)
    _report("reset")
    rads = [float(tower_local(scene.cargo[k].data.root_pos_w)[0, :2].norm())
            for k in range(3)]
    check("settle/no-NaN: discs settled on their slab slots, shaft empty, score ~0, "
          "no success",
          bool(scene._finite()[0]) and bool(scene.settled()[0])
          and all(r > c.bm + c.wall_t for r in rads)
          and score0() <= 0.05 and not succ())

    # ================= 2+3. randomization readback ================================================
    yaws, xys, perms, reds = [], [], [], []
    slots_ok, depot_ok = True, True
    for k in range(8):
        torch.manual_seed(20 + k)
        env.reset()
        _step(60)
        _refresh()
        yaws.append(yaw_of(scene.tower.data.root_quat_w[0]))
        xys.append((scene.tower.data.root_pos_w[0] - scene.env_origins[0])[:2].tolist())
        perms.append(tuple(scene.slot_of[0].tolist()))
        reds.append(int(scene.active_red[0]))
        # slot READBACK: every green disc settled on the slot its permutation names,
        # the ACTIVE red on the fourth slot, the parked reds in the ground depot
        for j in range(3):
            az = math.radians(c.slot_az_deg[int(scene.slot_of[0, j])])
            loc = tower_local(scene.cargo[j].data.root_pos_w)[0]
            slots_ok = slots_ok \
                and abs(float(loc[0]) - c.slot_r * math.cos(az)) < 0.040 \
                and abs(float(loc[1]) - c.slot_r * math.sin(az)) < 0.040
        az = math.radians(c.slot_az_deg[int(scene.slot_of[0, 3])])
        loc = tower_local(scene.reds[reds[-1]].data.root_pos_w)[0]
        slots_ok = slots_ok \
            and abs(float(loc[0]) - c.slot_r * math.cos(az)) < 0.040 \
            and abs(float(loc[1]) - c.slot_r * math.sin(az)) < 0.040
        for j in range(3):
            if j == reds[-1]:
                continue
            p = (scene.reds[j].data.root_pos_w[0] - scene.env_origins[0])
            dx, dy = c.depot_xy[j % 2]
            depot_ok = depot_ok and abs(float(p[0]) - dx) < 0.05 \
                and abs(float(p[1]) - (dy + 0.12 * (j // 2))) < 0.05
    yspan = max(yaws) - min(yaws)
    xystd = float(np.std(np.asarray(xys), axis=0).mean())
    print(f"[smoke] readback: yaws={[f'{y:+.0f}' for y in yaws]} span={yspan:.0f} "
          f"xystd={xystd:.3f} perms={len(set(perms))} reds={reds} "
          f"slots_ok={slots_ok} depot_ok={depot_ok}", flush=True)
    check("randomization A: tower yaw spans > 90 deg, xy jitters",
          yspan > 90.0 and xystd > 0.008)
    check("randomization B: slot permutation varies, red size varies, discs settle "
          "on their named slots, parked reds in the depot (position READBACK)",
          len(set(perms)) >= 2 and len(set(reds)) >= 2 and slots_ok and depot_ok)

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> score ~0, no success",
          score0() <= 0.05 and not succ())

    # ================= 5. SEED strategy (carry to destination, no size reasoning) =================
    torch.manual_seed(41)
    env.reset()
    _step(120)
    drop(scene.cargo[2], c.disc_h[2])   # largest first — the order-blind carry
    drop(scene.cargo[1], c.disc_h[1])
    drop(scene.cargo[0], c.disc_h[0])
    _report("seed-descend")
    seat5 = scene.seated()[0]
    z5 = [float(tower_local(scene.cargo[k].data.root_pos_w)[0, 2]) for k in range(3)]
    check("SEED strategy: descending-order drops — the large disc seats, the mid "
          "and small discs strand ON TOP far above their bands; one seat of credit "
          "(~0.22), no success",
          bool(seat5[2]) and not bool(seat5[0]) and not bool(seat5[1])
          and z5[1] > scene.z_seat_t[1] + 3 * c.z_tol
          and z5[0] > scene.z_seat_t[0] + 3 * c.z_tol
          and 0.20 <= score0() <= c.w_seat + 0.015 and not succ())

    # ================= 6. order interlock (mid first wedges the shaft) ============================
    torch.manual_seed(51)
    env.reset()
    _step(120)
    drop(scene.cargo[1], c.disc_h[1])
    seat6a = bool(scene.seated()[0, 1])  # the mid disc DID seat (real descent verified)
    drop(scene.cargo[0], c.disc_h[0])
    _report("interlock")
    seat6 = scene.seated()[0]
    z6 = float(tower_local(scene.cargo[0].data.root_pos_w)[0, 2])
    check("order interlock: mid disc first (really seated), then the small disc — "
          "the seated mid disc wedges the shaft; the small disc perches above its "
          "band; seated=(F,T,F), no success",
          seat6a and bool(seat6[1]) and not bool(seat6[0])
          and z6 > scene.z_seat_t[0] + 3 * c.z_tol and not succ())

    # ================= 7. cleanliness (all seated + red rider inside) =============================
    torch.manual_seed(61)
    env.reset()
    _step(120)
    kr = int(scene.active_red[0])
    q_t = scene.tower.data.root_quat_w
    for k in range(3):
        _write_body(scene.cargo[k],
                    scene.tower_world([0.0, 0.0, scene.z_seat_t[k] + 0.006]), q_t)
    _write_body(scene.reds[kr],
                scene.tower_world([0.0, 0.0, scene.z_seat_t[2] + c.disc_h[2] / 2
                                   + c.disc_h[kr] / 2 + 0.010]), q_t)
    _step(240)
    _report("cleanliness")
    seat7 = scene.seated()[0]
    s7 = score0()
    check("cleanliness: all three greens seated (constructed 6 mm drops, latches "
          "pay the 0.66 cap) with the red disc riding inside the shaft — success "
          "refused by the red-inside clause alone",
          bool(seat7.all()) and bool(scene.red_inside()[0])
          and c.score_cap - 0.005 <= s7 <= c.score_cap + 1e-4 and not succ())

    # ================= 8. red poison (real drop) + same-size perch near-miss ======================
    torch.manual_seed(71)
    env.reset()
    _step(120)
    kr = int(scene.active_red[0])
    drop(scene.reds[kr], c.disc_h[kr])
    _report("red-drop")
    red_in8 = bool(scene.red_inside()[0])
    s8a = score0()
    drop(scene.cargo[kr], c.disc_h[kr])
    _report("green-after-red")
    z8 = float(tower_local(scene.cargo[kr].data.root_pos_w)[0, 2])
    check("red poison: the active red really dropped seats at its band inside the "
          "shaft (worth 0); the same-size green then perches ONE disc height above "
          "its band — the tightest z near-miss — not seated, score ~0, no success",
          red_in8 and s8a <= 0.05
          and not bool(scene.seated()[0, kr])
          and z8 > scene.z_seat_t[kr] + c.z_tol
          and score0() <= 0.05 and not succ())

    # ================= 9. wrong place (xy clause alone) ===========================================
    torch.manual_seed(81)
    env.reset()
    _step(120)
    _write_body(scene.cargo[0],
                scene.tower_world([0.10, 0.0, c.disc_h[0] / 2 + 0.002]),
                scene.tower.data.root_quat_w)
    _step(180)
    _report("wrong-place")
    loc9 = tower_local(scene.cargo[0].data.root_pos_w)[0]
    check("wrong place: the small disc resting FLAT on the slab beside the tower at "
          "exactly its seat height — z and tilt in tolerance, the xy clause alone "
          "refuses; no success",
          abs(float(loc9[2]) - scene.z_seat_t[0]) < c.z_tol
          and float(scene._up_z(scene.cargo[0])[0]) > 0.97
          and float(loc9[:2].norm()) > 2 * c.xy_tol
          and not bool(scene.seated()[0, 0]) and score0() <= 0.05 and not succ())

    # ================= 10. latched credit =========================================================
    torch.manual_seed(91)
    env.reset()
    _step(120)
    drop(scene.cargo[0], c.disc_h[0])
    latched = bool(scene._seat[0, 0])
    _write_body(scene.cargo[0],
                (scene.env_origins[0:1]
                 + torch.tensor([0.9, -0.9, c.disc_h[0] / 2 + 0.002], device=device)
                 ).expand(n, 3))
    _step(180)
    _report("latch-theft")
    check("latched credit: small disc really seated (latch pays ~0.22) then stolen "
          "out of the tower — the latch survives, success does not",
          latched and not bool(scene.seated()[0, 0])
          and 0.20 <= score0() <= c.w_seat + 0.015 and not succ())

    # ================= 11-13. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.sorting_tower")
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
