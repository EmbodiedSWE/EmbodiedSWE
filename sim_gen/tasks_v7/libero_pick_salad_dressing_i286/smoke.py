"""Smoke / rubric-REJECTION battery for RailHangerScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a real thread->hook->slide->release
trajectory with monotone latched credit, two+ seeds). This battery proves the rubric
REJECTS wrong outcomes, and that the mechanism the task rests on — a bottle suspended in
mid-air purely by its cap flange on the rail tops — is physics, not fiat. Every probe is
CONSTRUCTED (teleport, real physics steps, judge) — instrumentation, never a solution;
drop probes assert the probe actually moved (no vacuous rejections), and every
constructed pose is collision-feasible before the write. Goal-state constructs are
DROPPED from 20 mm above the rails (not written in place) so the consecutive-still
counter restarts and the no-premature-success assertion is real.

Checks:
   1. settle/no-NaN     — seeded reset settles finite; both bottles upright on the
                          ground, nothing hanging, decoy off the rack; score ~0;
   2. randomization     — two seeded resets: READBACK rack yaw/xy, dressing xy and
                          basket xy all differ;
   3. bottle swap       — over 10 resets the amber bottle occupies BOTH stations;
   4. null-policy       — 240 idle steps: nothing hangs, score ~0, no success;
   5. SEED STRATEGY     — the seed's plan ("place the bottle in the receptacle"):
                          the amber bottle dropped INTO the basket = zero credit,
                          never success;
   6. under-rack stand  — the bottle standing on the rack slab directly under the
                          slot earns nothing (hanging is a height band, not
                          proximity);
   7. hang mechanism    — the bottle placed in the hang pose near the OPEN mouth
                          stays SUSPENDED in mid-air (cap-on-rails is real support),
                          latches `hooked` (0.25) — and does NOT self-slide to the
                          seat: partial credit only, no success;
   8. off-slot drop     — the bottle released at hang height but OUTSIDE the slot
                          falls to the ground: the hang band alone earns nothing
                          without the rails;
   9. lying across rails— the bottle dropped HORIZONTAL onto the rail tops rests on
                          top of (or falls off) the rack but never hangs: no credit;
  10. wrong object      — the RED decoy hung at the seat instead: decoy-exclusion
                          fails, no success, no credit for the amber bottle;
  11. live + score cap  — the amber bottle dropped into the seat -> success fires
                          only AFTER the still window; lifted back OFF the rack ->
                          success flips OFF and the score falls to the latched cap
                          (0.60); re-hung -> success returns;
  12. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_pick_salad_dressing_i286.smoke --headless
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
    dl = scene._rack_local(scene.dressing.data.root_pos_w)[0]
    print(f"[smoke] {tag:16s} | "
          f"dress=({float(dl[0]):+.3f},{float(dl[1]):+.3f},{float(dl[2]):+.3f}) "
          f"up_z={float(scene._up_z(scene.dressing)[0]):+.2f} "
          f"hang={bool(scene.hanging(scene.dressing)[0])} "
          f"seat={bool(scene.seated(scene.dressing)[0])} "
          f"latch=[{int(scene._hooked[0])}{int(scene._seated[0])}] "
          f"decoy_off={bool(scene.decoy_off_rack()[0])} "
          f"settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.rail_hanger")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.55, 1.05, 0.75)) + o),
                                tuple(np.array((0.45, 0.00, 0.20)) + o),
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

    def rack_q() -> torch.Tensor:
        _refresh()
        return scene.rack.data.root_quat_w

    def rack_world(loc_xyz) -> torch.Tensor:
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.rack.data.root_pos_w + quat_apply(rack_q(), loc)

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

    def drop_hang(body, x_loc: float, z_loc: float = 0.230, settle: int = 240) -> None:
        """CONSTRUCT a hang state by DROP: upright, neck between the rails (6 mm
        lateral clearance), cap underside 20 mm above the rail tops — falls onto
        them and hangs. The fall breaks the still counter, so success (if any) can
        only fire after a fresh still window."""
        _write_body(body, rack_world((x_loc, 0.0, z_loc)), rack_q().clone())
        _step(settle)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    up_d = float(scene._up_z(scene.dressing)[0])
    up_k = float(scene._up_z(scene.decoy)[0])
    check("settle/no-NaN: layout settles finite; both bottles upright on the ground, "
          "nothing hanging, decoy off the rack; score ~0, no success",
          bool(scene._finite()[0])
          and not bool(scene.hanging(scene.dressing)[0])
          and not bool(scene.hanging(scene.decoy)[0])
          and bool(scene.decoy_off_rack()[0])
          and up_d > 0.95 and up_k > 0.95
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene.rack.data.root_quat_w[0]),
                scene.rack.data.root_pos_w[0, :2].clone(),
                scene.dressing.data.root_pos_w[0, :2].clone(),
                scene.basket.data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_rp, a_dp, a_bp = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_rp, b_dp, b_bp = readback()
    d_yaw = dyaw(a_yaw, b_yaw)
    d_rp = float((a_rp - b_rp).norm())
    d_dp = float((a_dp - b_dp).norm())
    d_bp = float((a_bp - b_bp).norm())
    print(f"[smoke] randomization deltas: rack_yaw={d_yaw:.1f}deg "
          f"rack_xy={d_rp * 1000:.1f}mm dressing_xy={d_dp * 1000:.1f}mm "
          f"basket_xy={d_bp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: rack yaw, rack xy, dressing xy and basket xy "
          "readback all differ across seeds",
          d_yaw > 1.0 and d_rp > 0.002 and d_dp > 0.005 and d_bp > 0.002)

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
    check("null-policy-fails: 240 idle steps, nothing hangs, score ~0, no success",
          not bool(scene.hanging(scene.dressing)[0]) and not bool(scene._hooked[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED strategy =============================================
    # The seed task's plan: pick the dressing and place it IN THE BASKET. Here that
    # earns exactly nothing.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    bpos = scene.basket.data.root_pos_w.clone()
    drop = bpos + torch.tensor([[0.0, 0.0, 0.160]], device=device)
    _write_body(scene.dressing, drop)
    z0 = float(scene.dressing.data.root_pos_w[0, 2])
    _step(240)
    _report("seed-strategy")
    _REC["on"] = False
    z1 = float(scene.dressing.data.root_pos_w[0, 2])
    d_xy = float((scene.dressing.data.root_pos_w[0, :2] - bpos[0, :2]).norm())
    check("negative (SEED strategy): the amber bottle dropped into the basket "
          "(probe fell and stayed at the basket) earns ZERO credit and no success",
          z0 - z1 > 0.05 and d_xy < 0.10
          and not bool(scene.hanging(scene.dressing)[0]) and not bool(scene._hooked[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 6. near-miss: standing on the slab under the slot ==========================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    # feasible pose: upright on the slab top (z=0.020), under the rails, clear of
    # the column (x=-0.05; column face at -0.13)
    _write_body(scene.dressing, rack_world((-0.05, 0.0, 0.092)), rack_q().clone())
    _step(180)
    _report("slab-stand")
    check("under-rack stand: the bottle standing on the rack slab directly under "
          "the seat earns nothing (hang band is height, not proximity)",
          float(scene._up_z(scene.dressing)[0]) > 0.9
          and not bool(scene.hanging(scene.dressing)[0]) and not bool(scene._hooked[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 7. the hang mechanism is real + partial credit only ========================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    drop_hang(scene.dressing, 0.10, settle=180)  # near the OPEN mouth
    _report("mouth-hang")
    _REC["on"] = False
    loc = scene._rack_local(scene.dressing.data.root_pos_w)[0]
    s7 = float(scene.score()[0])
    check("hang mechanism: the bottle dropped into the slot near the mouth stays "
          "SUSPENDED mid-air by cap-on-rails contact (root in the hang band), "
          "latches hooked (0.25) — and does NOT self-slide to the seat: partial "
          "credit only, no success",
          bool(scene.hanging(scene.dressing)[0]) and bool(scene._hooked[0])
          and c.hang_z[0] < float(loc[2]) < c.hang_z[1]
          and float(loc[0]) > 0.03  # no free ride toward the seat
          and not bool(scene.seated(scene.dressing)[0]) and not bool(scene._seated[0])
          and 0.24 <= s7 <= 0.26 and not bool(scene.success()[0]))

    # ================= 8. hang band without the rails earns nothing ===============================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    # released at hang height but fully OUTSIDE the slot (y=0.12; rail outer face
    # at y=0.036): free air -> falls to the ground
    _write_body(scene.dressing, rack_world((0.05, 0.12, 0.211)), rack_q().clone())
    _step(240)
    _report("off-slot-drop")
    loc = scene._rack_local(scene.dressing.data.root_pos_w)[0]
    check("off-slot drop: released at hang height but outside the slot, the bottle "
          "falls (no rails, no support) and earns nothing",
          float(loc[2]) < 0.15 and not bool(scene._hooked[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 9. near-miss: lying across the rail tops ===================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    # horizontal (axis along the slot), 30+ mm above the rail tops, over the slot
    _write_body(scene.dressing, rack_world((0.03, 0.0, 0.380)),
                qmul(rack_q(), qy_local(90.0)))
    z0 = float(scene._rack_local(scene.dressing.data.root_pos_w)[0][2])
    _step(240)
    _report("across-rails")
    _REC["on"] = False
    z1 = float(scene._rack_local(scene.dressing.data.root_pos_w)[0][2])
    check("lying across rails: the bottle dropped horizontal onto the rail tops "
          "(probe fell) rests ON the rack or falls off — never hangs, no credit",
          z0 - z1 > 0.02 and not bool(scene.hanging(scene.dressing)[0])
          and not bool(scene._hooked[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 10. negative: wrong object hung at the seat ================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    drop_hang(scene.decoy, -0.100, settle=300)
    _report("wrong-object")
    _REC["on"] = False
    dloc = scene._rack_local(scene.decoy.data.root_pos_w)[0]
    check("negative (wrong object): the RED decoy hung at the seat (it does hang — "
          "the mechanism is object-agnostic) -> decoy-exclusion clause fails: no "
          "success and no credit for the untouched amber bottle",
          c.hang_z[0] < float(dloc[2]) < c.hang_z[1]  # decoy really hangs
          and not bool(scene.decoy_off_rack()[0])
          and not bool(scene._hooked[0]) and not bool(scene._seated[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 11. goal by drop: no premature success, live clause, score cap =============
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    drop_hang(scene.dressing, -0.100, settle=0)
    early_success = False
    fired = -1
    for j in range(300):
        _step(1)
        _refresh()
        if bool(scene.success()[0]):
            if fired < 0:
                fired = j
            if j < c.still_steps - 5:
                early_success = True
    _report("seat-drop")
    ok_success = bool(scene.success()[0]) and not early_success \
        and float(scene.score()[0]) >= 0.999 and bool(scene._seated[0])
    print(f"[smoke] success first fired at post-drop step {fired}", flush=True)
    # live clause + cap: lift the bottle OFF the rack, stand it on the open ground
    _write_body(scene.dressing, rack_world((0.35, -0.35, 0.072)), rack_q().clone())
    _step(120)
    _report("lifted-off")
    s_off = float(scene.score()[0])
    went_off = (not bool(scene.success()[0])) and 0.59 <= s_off <= 0.6001
    # re-hang at the seat by drop -> success returns
    drop_hang(scene.dressing, -0.100, settle=300)
    _report("re-hung")
    _REC["on"] = False
    check("live + cap: the dropped seat-hang succeeds only AFTER the still window; "
          "lifting the bottle off the rack flips success OFF and the score falls "
          "to the latched cap 0.60; re-hanging restores success",
          ok_success and went_off
          and bool(scene.success()[0]) and float(scene.score()[0]) >= 0.999)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.rail_hanger")
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
