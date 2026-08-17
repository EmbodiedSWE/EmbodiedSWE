"""Smoke battery for DrawerRefitScene (sim_gen task
`libero_kitchen_scene4_close_the_bottom_drawer_of_the_cabinet_i102`) —
REJECTION-ONLY: every check either verifies basic health/randomization or
CONSTRUCTS a settled wrong outcome and asserts the rubric refuses it. success()
must never fire anywhere in the battery.

 1. settle/no-NaN     — drawer and block on the ground, both bays empty,
                        score ~0, no success.
 2. randomization A   — sideboard yaw spans > 90 deg and xy jitters across resets.
 3. randomization B   — WHICH bay is tagged varies across resets, the tag chip's
                        height tracks the tagged bay every time, and the
                        drawer/block ground poses vary.
 4. null policy       — 240 idle steps -> score ~0, no success.
 5. SEED strategy     — the seed's whole skill (push the drawer toward "closed")
                        executed for REAL: an 8 N PD push on the grounded drawer
                        toward the face. It slides into the plinth and stops on
                        the ground — the bays are elevated; score stays ~0.
 6. wrong bay         — loaded drawer seated FLUSH in the UNTAGGED bay
                        (constructed, settled): block credit only, no insertion
                        credit, no success.
 7. inverted seat     — drawer seated flush in the TARGET bay but UPSIDE DOWN:
                        the upright clause refuses; ~0.
 8. near-miss proud   — loaded drawer left 25 mm short of flush in the target
                        bay: honest partial credit, below the 0.80 cap, no
                        success.
 9. flush but empty   — drawer flush in the target bay, block on the ground:
                        drawer_seated() True, success False (containment clause).
10. block loose in bay— block dumped on the target bay's floor, drawer on the
                        ground: cube-in-DRAWER refuses; ~0.
11. sealed front      — drawer flush+empty (constructed), then a REAL force-held
                        carry presses the block against the closed face for 2 s
                        (the post-closure loading attempt): the flush plate seals
                        the opening — the block never gets in; no success.
12. latched credit    — block loaded into the grounded drawer (latch fires), then
                        stolen back to the ground: block credit survives, success
                        does not.
13. rejection audit   — success() observed False at every step of the battery.
14. final no-NaN.
15. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene4_close_the_bottom_drawer_of_the_cabinet_i102.smoke --headless
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
    print(f"[smoke] {tag:18s} | gap={float(scene.plate_gap()[0]):+.4f} "
          f"bay={int(scene.target_bay[0])} "
          f"gates={bool(scene._gates()[0])} "
          f"seated={bool(scene.drawer_seated()[0])} "
          f"cube_in={bool(scene.cube_in_drawer()[0])} "
          f"latch(c/e)=({int(scene._cube_l[0])},{int(scene._eng_l[0])}) "
          f"ins_f={float(scene._ins_f[0]):.2f} "
          f"settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.drawer_refit")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-1.10, -1.10, 0.90)) + o),
                                tuple(np.array((0.00, 0.00, 0.22)) + o),
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

    def board_world(local_xyz) -> torch.Tensor:
        loc = torch.tensor(local_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.board.data.root_pos_w + quat_apply(scene.board.data.root_quat_w, loc)

    def board_x_axis() -> torch.Tensor:
        ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
        return quat_apply(scene.board.data.root_quat_w, ex)

    def ground(dx: float, dy: float, h: float) -> torch.Tensor:
        return (scene.env_origins[0:1]
                + torch.tensor([dx, dy, h], device=device)).expand(n, 3)

    def q_board_mul(axis: str, deg: float) -> torch.Tensor:
        """Board orientation composed with a rotation about a board-local axis."""
        half = math.radians(deg) / 2
        v = {"x": (math.sin(half), 0.0, 0.0), "y": (0.0, math.sin(half), 0.0)}[axis]
        qe = torch.tensor([math.cos(half), *v], device=device).expand(n, 4)
        return task_scene._qmul(scene.board.data.root_quat_w, qe)

    def seat_drawer(bay: int, gap: float, inverted: bool = False) -> None:
        """CONSTRUCT: the drawer written into bay `bay` with the given plate gap
        (0 = flush), upright or upside down, then left to physics."""
        zk = float(c.bay_floors[bay])
        x = -c.box_l / 2 + gap
        if inverted:
            _write_body(scene.drawer, board_world([x, 0.0, zk + c.box_h + 0.002]),
                        q_board_mul("x", 180.0))
        else:
            _write_body(scene.drawer, board_world([x, 0.0, zk + 0.002]),
                        scene.board.data.root_quat_w)

    def load_cube_into_drawer() -> None:
        """CONSTRUCT: the block released just above the drawer's basin; it drops in."""
        dq = scene.drawer.data.root_quat_w
        loc = torch.tensor([0.0, 0.0, c.floor_t + c.cube_s / 2 + 0.004],
                           device=device).expand(n, 3)
        _write_body(scene.cube, scene.drawer.data.root_pos_w + quat_apply(dq, loc), dq)

    def push_drawer_to_face(steps: int, clamp: float = 8.0) -> None:
        """REAL actuation: a PD push on the grounded drawer toward the face plane
        (the seed's whole skill), force-limited to what a hand would apply."""
        zero = torch.zeros(n, 1, 3, device=device)
        for _ in range(steps):
            axis = board_x_axis()
            gap = scene.plate_gap()
            v = (scene.drawer.data.root_lin_vel_w * axis).sum(-1)
            f_mag = (60.0 * (0.0 - gap) - 10.0 * v).clamp(-clamp, clamp)
            fw = axis * f_mag.unsqueeze(-1)
            fb = quat_apply_inverse(scene.drawer.data.root_quat_w, fw)
            scene.drawer.set_external_force_and_torque(fb.reshape(n, 1, 3), zero)
            _step(1)
        scene.drawer.set_external_force_and_torque(zero, zero)
        _step(60)

    def press_cube_at(target_local, mass: float, steps: int, clamp: float = 8.0) -> None:
        """REAL actuation: a force-limited PD carry (gravity feedforward — a firm
        grasp) that tries to bring the block to a board-local target."""
        zero = torch.zeros(n, 1, 3, device=device)
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
        for _ in range(steps):
            p = scene.cube.data.root_pos_w
            v = scene.cube.data.root_lin_vel_w
            f_w = mass * 9.81 * ez + 40.0 * (board_world(target_local) - p) - 5.0 * v
            f_norm = f_w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            f_w = f_w * (f_norm.clamp(max=clamp) / f_norm)
            fb = quat_apply_inverse(scene.cube.data.root_quat_w, f_w)
            scene.cube.set_external_force_and_torque(fb.reshape(n, 1, 3), zero)
            _step(1)
        scene.cube.set_external_force_and_torque(zero, zero)
        _step(90)

    def cube_local_x() -> float:
        return float(scene._board_local(scene.cube.data.root_pos_w)[0, 0])

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(150)
    _report("reset")
    check("settle/no-NaN: drawer and block on the ground, both bays empty, "
          "score ~0, no success",
          bool(scene._finite()[0]) and not bool(scene.cube_in_drawer()[0])
          and not bool(scene.drawer_seated()[0])
          and float(scene.plate_gap()[0]) > 0.25
          and float(scene.drawer.data.root_pos_w[0, 2]
                    - scene.env_origins[0, 2]) < 0.05
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2+3. randomization readback ================================================
    yaws, xys, bays, dposes, ok_tag = [], [], [], [], True
    for k in range(8):
        torch.manual_seed(20 + k)
        env.reset()
        _step(30)
        _refresh()
        yaws.append(yaw_of(scene.board.data.root_quat_w[0]))
        xys.append((scene.board.data.root_pos_w[0]
                    - scene.env_origins[0])[:2].tolist())
        b = int(scene.target_bay[0])
        bays.append(b)
        tag_loc = scene._board_local(scene.tag.data.root_pos_w)[0]
        ok_tag = ok_tag and abs(float(tag_loc[2])
                                - (c.bay_floors[b] + c.open_h + c.tag_dz)) < 0.01
        dloc = scene._board_local(scene.drawer.data.root_pos_w)[0]
        cloc = scene._board_local(scene.cube.data.root_pos_w)[0]
        dposes.append([float(dloc[0]), float(dloc[1]), float(cloc[0]), float(cloc[1])])
    yspan = max(yaws) - min(yaws)
    xystd = float(np.std(np.asarray(xys), axis=0).mean())
    dstd = float(np.std(np.asarray(dposes), axis=0).mean())
    print(f"[smoke] readback: yaws={[f'{y:+.0f}' for y in yaws]} span={yspan:.0f} "
          f"xystd={xystd:.3f} bays={bays} tag_tracks={ok_tag} "
          f"drawer/cube_std={dstd:.3f}", flush=True)
    check("randomization A: sideboard yaw spans > 90 deg and xy jitters across "
          "resets",
          yspan > 90.0 and xystd > 0.008)
    check("randomization B: WHICH bay is tagged varies across resets, the tag "
          "chip's height tracks the tagged bay every time, and the drawer/block "
          "ground poses vary",
          len(set(bays)) >= 2 and ok_tag and dstd > 0.008)

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 5. SEED strategy (real push on the grounded drawer) =======================
    torch.manual_seed(41)
    env.reset()
    _step(150)
    gap_before = float(scene.plate_gap()[0])
    push_drawer_to_face(360)   # 3 s of an 8 N hand-push toward the face plane
    _report("seed-skill")
    gap_after = float(scene.plate_gap()[0])
    check("SEED strategy: the seed's whole skill (push the drawer toward closed) "
          "for real — 8 N for 3 s: the drawer slides into the plinth and stops "
          "ON THE GROUND (the bays are elevated); it really moved, score ~0",
          gap_after < gap_before - 0.10 and gap_after > 0.06
          and float(scene.drawer.data.root_pos_w[0, 2]
                    - scene.env_origins[0, 2]) < 0.05
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 6. wrong bay ===============================================================
    torch.manual_seed(51)
    env.reset()
    _step(120)
    wrong = 1 - int(scene.target_bay[0])
    seat_drawer(wrong, 0.002)
    _step(60)
    load_cube_into_drawer()
    _step(240)
    _report("wrong-bay")
    up6, _ = scene._axes_dots()
    s6 = float(scene.score()[0])
    check("wrong bay: loaded drawer seated FLUSH in the UNTAGGED bay (verified "
          "flush, upright, in-lane) — block credit only, no insertion credit, "
          "no success",
          float(scene.plate_gap()[0]) < c.closed_tol
          and abs(float(scene._board_local(scene.drawer.data.root_pos_w)[0, 1]))
          < c.lane_tol
          and float(up6[0]) > 0.95 and bool(scene.cube_in_drawer()[0])
          and not bool(scene._gates()[0])
          and 0.10 <= s6 <= 0.20 and not succ())

    # ================= 7. inverted seat ===========================================================
    torch.manual_seed(61)
    env.reset()
    _step(120)
    seat_drawer(int(scene.target_bay[0]), 0.002, inverted=True)
    _step(240)
    _report("inverted")
    up7, _ = scene._axes_dots()
    check("inverted seat: drawer flush in the TARGET bay but UPSIDE DOWN — the "
          "upright clause refuses; ~0",
          float(scene.plate_gap()[0]) < c.closed_tol + 0.004
          and float(up7[0]) < -0.90
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 8. near-miss proud =========================================================
    torch.manual_seed(71)
    env.reset()
    _step(120)
    seat_drawer(int(scene.target_bay[0]), 0.025)
    _step(60)
    load_cube_into_drawer()
    _step(240)
    _report("near-miss")
    s8 = float(scene.score()[0])
    check("near-miss proud: loaded drawer left 25 mm short of flush in the "
          "target bay — honest partial credit, below the 0.80 cap, no success",
          0.015 <= float(scene.plate_gap()[0]) <= 0.035
          and not bool(scene.drawer_seated()[0])
          and 0.60 <= s8 <= 0.801 and not succ())

    # ================= 9. flush but empty =========================================================
    torch.manual_seed(81)
    env.reset()
    _step(120)
    seat_drawer(int(scene.target_bay[0]), 0.002)
    _step(240)
    _report("flush-empty")
    s9 = float(scene.score()[0])
    check("flush but empty: drawer flush in the target bay, block on the ground "
          "— drawer_seated() True, containment clause refuses success",
          bool(scene.drawer_seated()[0]) and not bool(scene.cube_in_drawer()[0])
          and 0.55 <= s9 <= 0.70 and not succ())

    # ================= 10. block loose in the bay =================================================
    torch.manual_seed(91)
    env.reset()
    _step(120)
    zk = float(c.bay_floors[int(scene.target_bay[0])])
    _write_body(scene.cube, board_world([-c.bay_d + c.cube_s / 2 + 0.006, 0.0,
                                         zk + c.cube_s / 2 + 0.004]),
                scene.board.data.root_quat_w)
    _step(240)
    _report("loose-in-bay")
    check("block loose in bay: block dumped on the target bay's floor, drawer on "
          "the ground — cube-in-DRAWER refuses; ~0",
          float(scene._board_local(scene.cube.data.root_pos_w)[0, 2]) > zk - 0.02
          and not bool(scene.cube_in_drawer()[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 11. sealed front (post-closure loading attempt) ============================
    torch.manual_seed(101)
    env.reset()
    _step(120)
    kbay = int(scene.target_bay[0])
    zk = float(c.bay_floors[kbay])
    seat_drawer(kbay, 0.002)
    _step(180)
    _write_body(scene.cube, board_world([0.30, 0.0, c.cube_s / 2 + 0.003]),
                scene.board.data.root_quat_w)
    _step(60)
    x_before = cube_local_x()
    # a firm force-held carry presses the block straight at the closed face for 2 s
    press_cube_at([-0.05, 0.0, zk + c.cube_s / 2 + 0.006], c.cube_mass, 240)
    _report("sealed-front")
    x_after = cube_local_x()
    s11 = float(scene.score()[0])
    check("sealed front: with the drawer home, a real 8 N held press cannot get "
          "the block past the flush plate — it really pressed (moved >= 15 cm) "
          "and stayed outside the face plane; no block credit, no success",
          x_before > 0.25 and x_after < x_before - 0.15 and x_after > 0.005
          and not bool(scene.cube_in_drawer()[0]) and not bool(scene._cube_l[0])
          and s11 <= 0.70 and not succ())

    # ================= 12. latched credit =========================================================
    torch.manual_seed(111)
    env.reset()
    _step(120)
    load_cube_into_drawer()
    _step(180)
    latched = bool(scene._cube_l[0])
    _write_body(scene.cube, ground(0.90, -0.90, c.cube_s / 2 + 0.002))
    _step(120)
    _report("latch-theft")
    s12 = float(scene.score()[0])
    check("latched credit: block loaded into the grounded drawer (latch fires), "
          "then stolen back to the ground — block credit survives, success does "
          "not",
          latched and not bool(scene.cube_in_drawer()[0])
          and 0.14 <= s12 <= 0.20 and not succ())

    # ================= 13-15. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.drawer_refit")
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
