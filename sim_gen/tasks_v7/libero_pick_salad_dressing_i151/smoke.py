"""Smoke / rubric-REJECTION battery for FlatPackCrateScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a real panels->bottle->lid assembly
with monotone latched credit, two+ seeds). This battery proves the rubric REJECTS wrong
outcomes, and that the mechanism claims the task rests on — the funnel channel guides and
seats a dropped panel, a seated lid physically blocks every late entry — are physics, not
fiat. Every probe is CONSTRUCTED (teleport, real physics steps, judge) — instrumentation,
never a solution; drop/force probes assert the probe actually moved (no vacuous
rejections), and every constructed pose is collision-feasible before the write.

Checks:
   1. settle/no-NaN      — seeded reset settles finite; slots empty, both bottles
                           upright on the open ground, lid off; score ~0;
   2. randomization      — two seeded resets: READBACK base yaw/xy, panel and lid
                           poses all differ;
   3. bottle swap        — over 10 resets the amber bottle occupies BOTH stations;
   4. null-policy        — 240 idle steps: nothing seats, score ~0, no success;
   5. panel near-miss    — a panel standing free on the pad (NOT in a channel)
                           earns nothing;
   6. channel mechanism  — a panel dropped in free air above the funnel is captured,
                           guided by the rails and SEATED by gravity alone (0.12);
   7. SEED STRATEGY      — the seed's plan ("place the bottle in the receptacle"):
                           bottle stood on the pad of the UNBUILT crate = partial
                           credit only (<= 0.17), never success;
   8. lid blocks entry   — with the lid seated: a bottle dropped from above lands
                           ON the plate (not inside), a panel dropped over its slot
                           mouth cannot enter (both probes moved);
   9. anti-teleport      — a panel WRITTEN into its covered slot under a seated lid
                           -> breach latch: success permanently blocked, credit
                           frozen;
  10. tipped bottle      — the bottle lying on its side on the pad interior earns
                           no placed credit;
  11. lid ajar           — a 45deg-yawed lid dropped onto the crate rests on top
                           but does NOT register: no lid credit, no success;
  12. legal assembly     — aligned lid re-drop on the full crate -> success (judged
                           only after the consecutive-still settle);
  13. live + score cap   — lid lifted OFF the succeeded crate: success flips OFF
                           and the score falls to the latched cap (0.60); re-seated
                           by gravity: success returns;
  14. wrong object       — the RED decoy sealed inside instead of the amber bottle:
                           no success, credit <= 0.45;
  15. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_pick_salad_dressing_i151.smoke --headless
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
    from . import scene as task_scene  # noqa: F401 - importing registers the scene/env
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

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
    dl = scene._base_local(scene.dressing.data.root_pos_w)[0]
    ll = scene._base_local(scene.lid.data.root_pos_w)[0]
    print(f"[smoke] {tag:16s} | "
          f"fs={bool(scene.slot_filled('s')[0])} fw={bool(scene.slot_filled('w')[0])} "
          f"in={bool(scene.bottle_inside(scene.dressing)[0])} "
          f"lid={bool(scene.lid_seated()[0])} "
          f"dress=({float(dl[0]):+.3f},{float(dl[1]):+.3f},{float(dl[2]):+.3f}) "
          f"lid_z={float(ll[2]):+.3f} "
          f"latch=[{int(scene._filled_s[0])}{int(scene._filled_w[0])}"
          f"{int(scene._placed[0])}{int(scene._lidded[0])}] "
          f"breach={bool(scene._breach[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.flat_pack_crate")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.45, 0.95, 0.68)) + o),
                                tuple(np.array((0.42, 0.00, 0.10)) + o),
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

    def base_q() -> torch.Tensor:
        _refresh()
        return scene.base.data.root_quat_w

    def base_world(loc_xyz) -> torch.Tensor:
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.base.data.root_pos_w + quat_apply(base_q(), loc)

    def qz_local(deg: float) -> torch.Tensor:
        h = math.radians(deg) / 2.0
        q = torch.zeros(n, 4, device=device)
        q[:, 0], q[:, 3] = math.cos(h), math.sin(h)
        return q

    def qy_local(deg: float) -> torch.Tensor:
        h = math.radians(deg) / 2.0
        q = torch.zeros(n, 4, device=device)
        q[:, 0], q[:, 2] = math.cos(h), math.sin(h)
        return q

    def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        aw, ax, ay, az = a.unbind(-1)
        bw, bx, by, bz = b.unbind(-1)
        return torch.stack([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ], dim=-1)

    def seat_panel(panel, slot: str) -> None:
        """CONSTRUCT a seated panel (instrumentation write into the collision-free
        seat pose: 16 mm panel centred in the 22 mm gap, bottom at the pad top)."""
        if slot == "s":
            pos = base_world((-c.slot_center, 0.0, 0.1055))
            q = base_q().clone()
        else:
            pos = base_world((0.0, -c.slot_center, 0.1055))
            q = qmul(base_q(), qz_local(90.0))
        _write_body(panel, pos, q)
        _step(30)

    def drop_lid(yaw_deg: float, z: float = 0.205, settle: int = 240) -> None:
        """Teleport the lid to free air over the crate (lip hangs inside the wall
        opening, touching nothing at z=0.205; use z>=0.24 for yawed drops where the
        lip cannot enter) and let gravity seat it."""
        q = base_q() if yaw_deg == 0.0 else qmul(base_q(), qz_local(yaw_deg))
        _write_body(scene.lid, base_world((0.0, 0.0, z)), q)
        _step(settle)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    up_d = float(scene._axis_w(scene.dressing, (0.0, 0.0, 1.0))[0, 2])
    up_k = float(scene._axis_w(scene.decoy, (0.0, 0.0, 1.0))[0, 2])
    check("settle/no-NaN: layout settles finite; slots empty, both bottles upright "
          "on the open ground, lid off the crate; score ~0, no success",
          bool(scene._finite()[0])
          and not bool(scene.slot_filled("s")[0]) and not bool(scene.slot_filled("w")[0])
          and not bool(scene.bottle_inside(scene.dressing)[0])
          and bool(scene.decoy_excluded()[0]) and not bool(scene.lid_seated()[0])
          and up_d > 0.95 and up_k > 0.95
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene.base.data.root_quat_w[0]),
                scene.base.data.root_pos_w[0, :2].clone(),
                scene.panel_a.data.root_pos_w[0, :2].clone(),
                scene.lid.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.lid.data.root_quat_w[0]))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_bp, a_pp, a_lp, a_ly = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_bp, b_pp, b_lp, b_ly = readback()
    d_yaw, d_bp = dyaw(a_yaw, b_yaw), float((a_bp - b_bp).norm())
    d_pp, d_lp, d_ly = float((a_pp - b_pp).norm()), float((a_lp - b_lp).norm()), \
        dyaw(a_ly, b_ly)
    print(f"[smoke] randomization deltas: base_yaw={d_yaw:.1f}deg "
          f"base_xy={d_bp * 1000:.1f}mm panelA_xy={d_pp * 1000:.1f}mm "
          f"lid_xy={d_lp * 1000:.1f}mm lid_yaw={d_ly:.1f}deg", flush=True)
    check("randomization-is-real: base yaw, base xy, panel and lid pose readback "
          "all differ across seeds",
          d_yaw > 1.0 and d_bp > 0.002 and d_pp > 0.005 and d_lp > 0.005 and d_ly > 3.0)

    # ================= 3. bottle station swap =====================================================
    stations = set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        stations.add(int(scene.dressing_station[0]))
    print(f"[smoke] over 10 resets: dressing stations {sorted(stations)}", flush=True)
    check("bottle swap: the amber bottle occupies BOTH scatter stations over 10 resets",
          stations == {1, 2})

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, nothing seats, score ~0, no success",
          not bool(scene.slot_filled("s")[0]) and not bool(scene.slot_filled("w")[0])
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5+6. panel near-miss, then the channel mechanism is real ===================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    # 5: a panel standing free on the pad interior — upright, wall-height, but NOT
    # in any channel (feasible pose: interior is empty). It may stay or topple;
    # either way it earns nothing.
    _write_body(scene.panel_a, base_world((0.01, 0.01, 0.107)), base_q().clone())
    _step(180)
    _report("panel-freestand")
    check("panel near-miss: a panel standing free on the pad (not in a channel) "
          "earns no slot credit",
          not bool(scene.slot_filled("s")[0]) and not bool(scene.slot_filled("w")[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))
    # 6: the legit insert — hover in free air above the funnel, gravity does the rest
    _write_body(scene.panel_a, base_world((-c.slot_center, 0.0, 0.185)), base_q().clone())
    z0 = float(scene._base_local(scene.panel_a.data.root_pos_w)[0][2])
    for _ in range(300):
        _step(1)
        if bool(scene._panel_seated(scene.panel_a, "s")[0]):
            break
    _step(90)
    _report("panel-dropped")
    _REC["on"] = False
    z1 = float(scene._base_local(scene.panel_a.data.root_pos_w)[0][2])
    check("channel mechanism: a panel dropped in free air above the funnel is "
          "captured, guided and SEATED by gravity alone (probe fell 80 mm), "
          "latching the slot credit (0.12)",
          z0 - z1 > 0.05 and bool(scene._panel_seated(scene.panel_a, "s")[0])
          and bool(scene._filled_s[0])
          and abs(float(scene.score()[0]) - c.w_slot) < 0.01)

    # ================= 7. negative: the SEED'S OWN STRATEGY =======================================
    # The seed's plan is "pick the bottle and place it in the receptacle". Here the
    # receptacle does not exist yet: the bottle stood on the bare pad of the UNBUILT
    # crate is a partial subgoal at best, never success.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _write_body(scene.dressing, base_world((0.0, 0.0, 0.10)))
    _step(150)
    _report("seed-strategy")
    check("negative (SEED strategy): bottle placed on the pad of the unbuilt crate "
          "-> partial credit only (<= 0.17), no walls, no lid, no success",
          bool(scene.bottle_inside(scene.dressing)[0]) and bool(scene._placed[0])
          and float(scene.score()[0]) <= 0.17 and not bool(scene.success()[0]))

    # ================= 8+9. a seated lid physically blocks entry; writes are breach ===============
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    seat_panel(scene.panel_a, "s")
    drop_lid(0.0)  # plate rests level on the two fixed walls + the seated panel
    _report("lid-early")
    assert bool(scene.lid_seated()[0]), "probe setup: early lid must seat"
    # 8a: bottle dropped from above the sealed top lands ON the plate, never inside
    _write_body(scene.dressing, base_world((0.0, 0.0, 0.32)))
    _step(240)
    zb = float(scene._base_local(scene.dressing.data.root_pos_w)[0][2])
    bottle_blocked = (not bool(scene.bottle_inside(scene.dressing)[0])) and zb < 0.30
    # 8b: panel dropped in free air over its covered slot mouth cannot enter
    _write_body(scene.panel_b, base_world((0.0, -c.slot_center, 0.30)),
                qmul(base_q(), qz_local(90.0)))
    _step(240)
    _report("lid-blocks")
    zp = float(scene._base_local(scene.panel_b.data.root_pos_w)[0][2])
    check("lid blocks entry: with the lid seated, a dropped bottle lands ON the "
          "plate (not inside) and a panel dropped over its slot mouth cannot "
          "enter (both probes fell and were rejected by contact)",
          bottle_blocked and zp < 0.29
          and not bool(scene.slot_filled("w")[0]) and not bool(scene._breach[0])
          and not bool(scene.success()[0]))
    # 9: WRITE the panel into the covered slot -> the only way in is a teleport
    seat_panel(scene.panel_b, "w")
    _step(120)
    _report("breach")
    _REC["on"] = False
    check("anti-teleport: a panel written into its covered slot under a seated lid "
          "latches breach — success permanently blocked, credit frozen (<= 0.33)",
          bool(scene._breach[0]) and bool(scene.slot_filled("w")[0])
          and not bool(scene._filled_w[0])
          and float(scene.score()[0]) <= 0.33 and not bool(scene.success()[0]))

    # ================= 10..13. full assembly: tipped bottle, ajar lid, success, live cap ==========
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    seat_panel(scene.panel_a, "s")
    seat_panel(scene.panel_b, "w")
    # 10: bottle LYING on the pad, along the NE diagonal (feasible: ends clear the
    # fixed walls by 10 mm and the flare shelves by ~1.5 mm)
    _write_body(scene.dressing, base_world((0.005, 0.005, 0.048)),
                qmul(base_q(), qmul(qz_local(45.0), qy_local(90.0))))
    _step(150)
    _report("bottle-tipped")
    check("tipped bottle: the bottle lying on its side on the pad interior earns "
          "no placed credit",
          not bool(scene.bottle_inside(scene.dressing)[0])
          and not bool(scene._placed[0])
          and abs(float(scene.score()[0]) - 2 * c.w_slot) < 0.01)
    # stand it up properly (free vertical corridor through the open top)
    _write_body(scene.dressing, base_world((0.0, 0.0, 0.10)))
    _step(120)
    assert bool(scene._placed[0]), "probe setup: upright bottle must latch placed"
    # 11: the 45deg-yawed lid rests ON the crate (its lip half-diagonal exceeds the
    # interior) but does not register
    drop_lid(45.0, z=0.26)
    _report("lid-ajar")
    zl = float(scene._base_local(scene.lid.data.root_pos_w)[0][2])
    check("lid ajar: a 45deg-yawed lid dropped onto the crate rests on top "
          "(probe landed at wall-top height) but does NOT register: no lid "
          "credit, no success",
          zl > 0.17 and not bool(scene.lid_seated()[0]) and not bool(scene._lidded[0])
          and abs(float(scene.score()[0]) - (2 * c.w_slot + c.w_bottle)) < 0.01
          and not bool(scene.success()[0]))
    # 12: aligned re-drop -> the lip registers, the crate seals -> success. Success
    # may only appear after the consecutive-still settle window.
    early_success = False
    drop_lid(0.0, settle=0)
    for j in range(300):
        _step(1)
        _refresh()
        if bool(scene.success()[0]) and j < c.still_steps - 5:
            early_success = True
    _report("assembled")
    check("legal assembly: the aligned lid seats by gravity on the full crate -> "
          "success (and success never fired before the still window elapsed)",
          bool(scene.lid_seated()[0]) and bool(scene.success()[0])
          and not early_success and float(scene.score()[0]) >= 0.999)
    # 13: live clause + score cap: lift the lid OFF -> success flips off and the
    # score falls to the latched cap (all four credits = 0.60); re-seat -> returns
    _write_body(scene.lid, base_world((0.40, -0.40, 0.017)), base_q().clone())
    _step(120)
    _report("lid-off")
    s_off = float(scene.score()[0])
    went_off = (not bool(scene.success()[0])) and 0.59 <= s_off <= 0.6001
    drop_lid(0.0, settle=300)
    _report("lid-back")
    _REC["on"] = False
    check("live + cap: lid lifted off the succeeded crate -> success OFF and score "
          "= latched cap 0.60; re-seated by gravity -> success returns",
          went_off and bool(scene.success()[0]) and float(scene.score()[0]) >= 0.999)

    # ================= 14. negative: wrong object sealed in ======================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    seat_panel(scene.panel_a, "s")
    seat_panel(scene.panel_b, "w")
    _write_body(scene.decoy, base_world((0.0, 0.0, 0.10)))
    _step(120)
    drop_lid(0.0)
    _report("wrong-object")
    _REC["on"] = False
    check("negative (wrong object): the RED decoy sealed inside instead of the "
          "amber bottle -> decoy-exclusion clause fails, amber still outside, "
          "no success, credit <= 0.45",
          bool(scene.lid_seated()[0]) and not bool(scene.decoy_excluded()[0])
          and not bool(scene.bottle_inside(scene.dressing)[0])
          and float(scene.score()[0]) <= 0.45 and not bool(scene.success()[0]))

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.flat_pack_crate")
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
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(2)
