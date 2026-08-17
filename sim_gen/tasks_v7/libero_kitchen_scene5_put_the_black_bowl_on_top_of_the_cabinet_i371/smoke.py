"""Smoke battery for RoofShuttleScene (sim_gen task
`libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet_i371`) —
REJECTION-ONLY: every check either verifies basic health/randomization or
CONSTRUCTS a settled wrong outcome and asserts the rubric refuses it. success()
must never fire anywhere in the battery.

 1. settle/no-NaN     — bowl sealed in the roofed bay on its pedestal, panel at
                        its sampled start offset, score ~0, no success.
 2. randomization     — across 8 resets: bowl yaw spans > 90 deg, bowl and
                        pedestal xy jitter, panel start offset varies AND the
                        measured offset matches the stored readback every time.
 3. null policy       — 240 idle steps -> score ~0, no success.
 4. SEED-strategy trap— the seed's move ("put the bowl on top of the cabinet")
                        aimed at the only open top area: the bowl is released
                        over the topless REAR shaft — it falls 30 cm onto the
                        deep-red shaft floor; no seat credit, no success.
 5. roofed extraction — the interlock probed with REAL force: a ~5 N velocity-
                        servo pull hoists the bowl off its pedestal until it
                        presses the joint-locked roof panel from below for ~2 s.
                        The bowl demonstrably rose (probe moved) yet never left
                        the bay; panel stays parked; score stays ~0.
 6. inverted on panel — panel slid back (construct), bowl placed UPSIDE-DOWN on
                        its top face: the upright clause refuses; partial only.
 7. never extracted   — panel slid back but the bowl left standing in the bay
                        on its pedestal: out/seat refuse; slide+open credit only.
 8. wrong place       — panel slid back, bowl stood upright on the GROUND beside
                        the cabinet: the seat clause refuses; partial only.
 9. lying on panel    — bowl laid on its SIDE on the panel top: upright + rest-
                        band refuse; no success.
10. latched credit    — panel slid back and bowl taken out (latches fire), then
                        everything RESTORED to the start pose: slide/open/out
                        latches survive, seat does not, no success.
11. rejection audit   — success() observed False at every step of the battery.
12. final no-NaN.
13. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet_i371.smoke --headless
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
    print(f"[smoke] {tag:18s} | off={float(scene.slab_off()[0]):+.4f} "
          f"uncov={float(scene.uncovered()[0]):+.4f} "
          f"in_bay={bool(scene.bowl_in_bay()[0])} "
          f"on_slab={bool(scene.bowl_on_slab()[0])} "
          f"latch(s/o/x/t)=({float(scene._slide_f[0]):.2f},{int(scene._open_l[0])},"
          f"{int(scene._out_l[0])},{int(scene._seat_l[0])}) "
          f"settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.roof_shuttle")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-1.05, -1.05, 0.85)) + o),
                                tuple(np.array((0.00, 0.00, 0.18)) + o),
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

    origin = scene.env_origins
    zero = torch.zeros(n, 1, 3, device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)

    def world(x: float, y: float, z: float) -> torch.Tensor:
        return (origin + torch.tensor([x, y, z], device=device)).expand(n, 3)

    def q_roll_x(deg: float) -> torch.Tensor:
        half = math.radians(deg) / 2
        return torch.tensor([math.cos(half), math.sin(half), 0.0, 0.0],
                            device=device).expand(n, 4)

    def slab_to(off: float) -> None:
        """CONSTRUCT: the panel written along its own joint DOF (the one motion
        the rail allows), everything else at the joint's rest values."""
        _write_body(scene.slab, world(0.0, scene.slab_front_center() + off,
                                      scene.slab_zc()))

    def bowl_yaw() -> float:
        q = scene.bowl.data.root_quat_w[0]
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    def pull_bowl_up(steps: int, v_des: float = 0.25, kv: float = 20.0):
        """REAL actuation: a force-limited velocity-servo hoist on the bowl (what
        a firm grasp pulling straight up does), xy held at the start point, with
        a small righting torque. Returns (peak rise, bay containment held)."""
        p0 = scene.bowl.data.root_pos_w.clone()
        peak, held = 0.0, True
        for _ in range(steps):
            p = scene.bowl.data.root_pos_w
            v = scene.bowl.data.root_lin_vel_w
            w = scene.bowl.data.root_ang_vel_w
            q = scene.bowl.data.root_quat_w
            f_w = torch.zeros(n, 3, device=device)
            f_w[:, 0] = 30.0 * (p0[:, 0] - p[:, 0]) - 4.0 * v[:, 0]
            f_w[:, 1] = 30.0 * (p0[:, 1] - p[:, 1]) - 4.0 * v[:, 1]
            f_w[:, 2] = c.bowl_mass * 9.81 + (kv * (v_des - v[:, 2])).clamp(-5.0, 5.0)
            f_b = quat_apply_inverse(q, f_w)
            t_w = 0.10 * torch.cross(quat_apply(q, ez), ez, dim=-1) - 0.02 * w
            t_b = quat_apply_inverse(q, t_w)
            scene.bowl.set_external_force_and_torque(f_b.reshape(n, 1, 3),
                                                     t_b.reshape(n, 1, 3))
            _step(1)
            peak = max(peak, float((scene.bowl.data.root_pos_w - p0)[0, 2]))
            held = held and bool(scene.bowl_in_bay()[0])
        scene.bowl.set_external_force_and_torque(zero, zero)
        _step(120)
        return peak, held

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(120)
    _report("reset")
    check("settle/no-NaN: bowl sealed in the roofed bay on its pedestal, panel at "
          "its start offset, score ~0, no success",
          bool(scene._finite()[0]) and bool(scene.bowl_in_bay()[0])
          and float(scene.uncovered()[0]) < c.open_pass / 2
          and abs(float(scene.slab_off()[0]) - float(scene.slab_off0[0])) < 0.006
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2. randomization readback ==================================================
    yaws, bxys, pxys, offs, ok_off = [], [], [], [], True
    for k in range(8):
        torch.manual_seed(20 + k)
        env.reset()
        _step(60)
        _refresh()
        yaws.append(bowl_yaw())
        bxys.append((scene.bowl.data.root_pos_w[0] - origin[0])[:2].tolist())
        pxys.append((scene.pedestal.data.root_pos_w[0] - origin[0])[:2].tolist())
        offs.append(float(scene.slab_off0[0]))
        ok_off = ok_off and \
            abs(float(scene.slab_off()[0]) - float(scene.slab_off0[0])) < 0.006
    yspan = max(yaws) - min(yaws)
    bstd = float(np.std(np.asarray(bxys), axis=0).mean())
    pstd = float(np.std(np.asarray(pxys), axis=0).mean())
    ospan = max(offs) - min(offs)
    print(f"[smoke] readback: yaws={[f'{y:+.0f}' for y in yaws]} span={yspan:.0f} "
          f"bowl_xystd={bstd:.4f} ped_xystd={pstd:.4f} "
          f"offs={[f'{o:.4f}' for o in offs]} span={ospan:.4f} match={ok_off}",
          flush=True)
    check("randomization: bowl yaw spans > 90 deg, bowl and pedestal xy jitter, "
          "panel start offset varies and matches its stored readback every reset",
          yspan > 90.0 and bstd > 0.005 and pstd > 0.005
          and ospan > 0.003 and ok_off)

    # ================= 3. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 4. SEED-strategy trap (drop over the topless rear shaft) ==================
    torch.manual_seed(41)
    env.reset()
    _step(120)
    # The seed's move aimed at the only place that LOOKS like "on top of the
    # cabinet" without opening anything: release the bowl over the open rear
    # shaft, just above the rim. It falls 30 cm onto the deep-red shaft floor.
    shaft_y = (c.part_hw + c.cab_hy - c.wall_t) / 2
    _write_body(scene.bowl, world(0.0, shaft_y, c.cab_h + c.bowl_org + 0.03))
    _step(420)
    _report("shaft-trap")
    z4 = float((scene.bowl.data.root_pos_w - origin)[0, 2])
    check("SEED-strategy trap: bowl released over the topless rear half falls "
          "into the shaft — no seat credit, no success",
          z4 < c.cab_h - 0.10 and not bool(scene.bowl_on_slab()[0])
          and float(scene.score()[0]) <= 0.32 and not succ())

    # ================= 5. roofed extraction (real force against the interlock) ===================
    torch.manual_seed(51)
    env.reset()
    _step(120)
    off5_0 = float(scene.slab_off()[0])
    peak, held = pull_bowl_up(300)
    _report("roofed-pull")
    check("roofed extraction: a ~5 N hoist pressed the bowl against the closed "
          "roof for ~2 s — the bowl rose (probe moved) but never left the bay; "
          "panel stayed parked; score ~0, no success",
          peak >= 0.04 and held and bool(scene.bowl_in_bay()[0])
          and abs(float(scene.slab_off()[0]) - off5_0) < 0.02
          and not bool(scene._out_l[0]) and not bool(scene._open_l[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 6. inverted on panel =======================================================
    torch.manual_seed(61)
    env.reset()
    _step(120)
    slab_to(0.27)
    _step(60)
    _write_body(scene.bowl,
                world(0.0, scene.slab_front_center() + 0.27 + 0.05,
                      scene.slab_top_z() + (c.bowl_h - c.bowl_org) + 0.004),
                q_roll_x(180.0))
    _step(300)
    _report("inverted")
    s6 = float(scene.score()[0])
    check("inverted on panel: bowl placed upside-down on the panel top — the "
          "upright clause refuses; slide/open/out partial credit only",
          not bool(scene.bowl_on_slab()[0])
          and float(scene._bowl_up()[0, 2]) < 0.0
          and 0.55 <= s6 <= 0.70 and not succ())

    # ================= 7. never extracted (panel open, bowl untouched in the bay) =================
    torch.manual_seed(71)
    env.reset()
    _step(120)
    slab_to(0.27)
    _step(300)
    _report("never-out")
    s7 = float(scene.score()[0])
    check("never extracted: panel slid back but the bowl left standing in the "
          "bay on its pedestal — out/seat refuse; slide+open credit only",
          bool(scene.bowl_in_bay()[0]) and not bool(scene._out_l[0])
          and not bool(scene._seat_l[0])
          and 0.30 <= s7 <= 0.45 and not succ())

    # ================= 8. wrong place (bowl upright on the ground) ================================
    torch.manual_seed(81)
    env.reset()
    _step(120)
    slab_to(0.27)
    _step(60)
    _write_body(scene.bowl, world(0.55, -0.55, c.bowl_org + 0.003))
    _step(300)
    _report("wrong-place")
    s8 = float(scene.score()[0])
    check("wrong place: bowl stood upright on the GROUND beside the cabinet — "
          "the seat clause refuses; partial credit only",
          not bool(scene.bowl_on_slab()[0]) and not bool(scene._seat_l[0])
          and 0.55 <= s8 <= 0.70 and not succ())

    # ================= 9. lying on panel ==========================================================
    torch.manual_seed(91)
    env.reset()
    _step(120)
    slab_to(0.27)
    _step(60)
    _write_body(scene.bowl,
                world(0.0, scene.slab_front_center() + 0.27 + 0.05,
                      scene.slab_top_z() + c.bowl_r + 0.004),
                q_roll_x(90.0))
    _step(300)
    _report("lying")
    check("lying on panel: bowl laid on its side on the panel top — upright and "
          "rest-band clauses refuse; no success",
          not bool(scene.bowl_on_slab()[0]) and not bool(scene._seat_l[0])
          and abs(float(scene._bowl_up()[0, 2])) < math.cos(
              math.radians(c.upright_max_deg)) and not succ())

    # ================= 10. latched credit =========================================================
    torch.manual_seed(101)
    env.reset()
    _step(120)
    ped10 = (scene.pedestal.data.root_pos_w - origin)[0].clone()
    slab_to(0.27)
    _step(60)
    _write_body(scene.bowl, world(0.55, -0.55, c.bowl_org + 0.003))
    _step(120)
    latched = bool(scene._open_l[0]) and bool(scene._out_l[0]) \
        and float(scene._slide_f[0]) > 0.9
    # restore EVERYTHING to the start pose: latches must survive, seat must not
    slab_to(0.004)
    _write_body(scene.bowl,
                world(float(ped10[0]), float(ped10[1]),
                      c.ped_top + c.bowl_org + 0.002))
    _step(180)
    _report("latch")
    s10 = float(scene.score()[0])
    check("latched credit: panel opened and bowl taken out (latches fire), then "
          "everything restored to the start pose — slide/open/out latches "
          "survive, seat does not, no success",
          latched and bool(scene.bowl_in_bay()[0])
          and bool(scene._open_l[0]) and bool(scene._out_l[0])
          and not bool(scene._seat_l[0])
          and 0.55 <= s10 <= 0.70 and not succ())

    # ================= 11-13. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.roof_shuttle")
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
