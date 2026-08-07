"""Smoke / rubric-REJECTION battery for PistonJarScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome, credit
monotone along a real trajectory). This battery proves the rubric REJECTS wrong
outcomes and that the physics the task rests on — the seal that holds against
gravity, carrying, inversion and sub-threshold pulls but shears past its 10 N
breakForce; the press-to-eject presentation — is load-bearing. Every probe is
CONSTRUCTED (teleport, real physics steps, judge) — instrumentation, never a
solution: success() is monitored at EVERY step and must never turn True anywhere
in this battery (the audit is itself a check).

ORDER MATTERS: the seal is a PhysX breakable joint and a shear is IRREVERSIBLE
across reset (reset cannot re-weld). Checks 1-7 need the INTACT seal and run
first; check 8 breaks it deliberately; everything after runs on the broken weld
(where the lid is just a loose plate that reset re-seats).

Checks:
   1. settle/no-NaN      — seeded reset settles finite; sealed (covered), ball
                           captive, spike outside, stack coherent; score 0;
   2. randomization      — two seeded resets: READBACK pedestal xy+yaw, jar
                           xy+yaw, dish xy+yaw all differ;
   3. null-policy        — 240 idle steps: score ~0, no success;
   4. INVERSION dump     — the sealed jar rested UPSIDE DOWN and hopped twice:
                           the weld holds, the lid stays sealing the mouth, the
                           ball stays inside — you cannot shake it open;
   5. MECHANISM sub-crit — the sealed jar parked ON the spike hands-off: it
                           sleeves down and hangs by the seal (own weight ~4.4 N
                           < 10 N) — engaged yes, but NOT opened: resting on the
                           spike is not enough, score <= engage credit;
   6. bypass sealed      — the ball teleported into the dish PAST the sealed jar:
                           ball_in_dish holds yet no success — the open-mouth
                           clause is load-bearing;
   7. seal holds 6 N     — a 6 N upward pull on the lid lifts the WHOLE stack
                           hanging by the weld (~5.6 N): HOLDS, lid stays seated;
   8. seal shears 15 N   — a 15 N pull loads it ~14 N: the weld BREAKS and the
                           lid comes away — threshold bracketed from both sides;
   9. SEED STRATEGY      — the seed's own outcome ("lid off the jar"): lid clear,
                           ball still captive inside — opened credit only, no
                           success (the delivery is the goal);
  10. near-miss dish     — freed ball on the ground BESIDE the dish: no success;
  11. wrong object       — the LID laid in the dish, ball still in the jar: no
                           success;
  12. presented-only     — the full press end-state (jar bottomed on the spike,
                           ball riding above the rim, lid clear) with NO
                           delivery: engaged+opened+presented latch, score capped
                           at 0.60, no success;
  13. settle gate        — the exact success geometry with the ball still
                           ROLLING in the dish: every geometric clause holds for
                           45 sampled steps yet success refuses (consecutive-
                           still counter); dismantled before it can settle;
  14. latched credit     — after 13's dismantle (ball carried away) the live
                           state is broken but the latched 0.60 survives;
  15. rejection audit    — success() was never True at any step of this battery;
  16. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.open_jar_i65.smoke --headless
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

_qx, _qmul, _qapply = task_scene._qx, task_scene._qmul, task_scene._qapply

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
_AUD = {"on": False, "hits": 0}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if _AUD["on"] and bool(env.scene.success()[0]):
            _AUD["hits"] += 1
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


def _place(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None,
           lin_vel: torch.Tensor | None = None) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    if lin_vel is not None:
        st[:, 7:10] = lin_vel
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    tip_l = scene._jar_local(scene.spike_tip_w())[0]
    ball_l = scene._jar_local(scene.ball.data.root_pos_w)[0]
    lid_l = scene._jar_local(scene.lid.data.root_pos_w)[0]
    print(f"[smoke] {tag:16s} | tip_lz={float(tip_l[2]):+.3f} "
          f"ball_lz={float(ball_l[2]):+.3f} "
          f"lid_l=({float(lid_l[0]):+.3f},{float(lid_l[1]):+.3f},{float(lid_l[2]):+.3f}) "
          f"eng={bool(scene.engaged()[0])} cov={bool(scene.covered()[0])} "
          f"pres={bool(scene.presented()[0])} in_jar={bool(scene.ball_in_jar()[0])} "
          f"in_dish={bool(scene.ball_in_dish()[0])} still={bool(scene.still()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.piston_jar")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.25, -0.55, 0.55)) + o),
                                tuple(np.array((0.30, 0.05, 0.08)) + o),
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

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def tip_w() -> torch.Tensor:
        _refresh()
        return scene.spike_tip_w()

    def jar_q() -> torch.Tensor:
        _refresh()
        return scene.jar.data.root_quat_w.clone()

    def lid_lz() -> float:
        _refresh()
        return float(scene._jar_local(scene.lid.data.root_pos_w)[0, 2])

    CLEAR = (-0.10, 0.00)   # an empty patch of ground, clear of all fixtures

    def pt(x: float, y: float, z: float) -> torch.Tensor:
        p = torch.tensor([x, y, z], device=device).expand(n, 3).clone()
        return p + env.iscene.env_origins

    def write_stalled_on_spike() -> None:
        """The sealed stack parked ON the spike at the contact chain (piston on
        the tip, ball touching the welded lid), joint-consistent, ~0.3 mm slack."""
        t = tip_w()
        qj = jar_q()
        jz = t[:, 2:3] - (c.lid_seat_z - c.lid_t / 2 - 2 * c.ball_r - c.plate_t) + 0.0003
        _place(scene.jar, torch.cat([t[:, :2], jz], dim=-1), qj)
        _place(scene.piston, torch.cat([t[:, :2], t[:, 2:3] + 0.0003], dim=-1), qj)
        _place(scene.ball, torch.cat(
            [t[:, :2], t[:, 2:3] + 0.0003 + c.plate_t + c.ball_r], dim=-1))
        _place(scene.lid, torch.cat([t[:, :2], jz + c.lid_seat_z], dim=-1), qj)

    def write_pressed_open(lid_xy=CLEAR) -> None:
        """The press END-state: jar bottomed on the pedestal base around the
        spike, piston riding the tip, ball presented above the rim, lid on the
        ground far away (usable only once the weld is broken)."""
        t = tip_w()
        qj = jar_q()
        ped_top = scene.pedestal.data.root_pos_w[:, 2:3] - \
            env.iscene.env_origins[:, 2:3] + c.base_h
        jz = ped_top + 0.002 + env.iscene.env_origins[:, 2:3]
        _place(scene.jar, torch.cat([t[:, :2], jz], dim=-1), qj)
        _place(scene.piston, torch.cat([t[:, :2], t[:, 2:3] + 0.0005], dim=-1), qj)
        _place(scene.ball, torch.cat(
            [t[:, :2], t[:, 2:3] + 0.0005 + c.plate_t + c.ball_r], dim=-1))
        _place(scene.lid, pt(lid_xy[0], lid_xy[1], c.lid_t / 2 + 0.002))

    # ================= 1. settle / no-NaN (SEAL INTACT) ===========================================
    torch.manual_seed(100)
    env.reset()
    _AUD["on"] = True
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    _refresh()
    d_pj = float((scene.piston.data.root_pos_w[0, :2]
                  - scene.jar.data.root_pos_w[0, :2]).norm())
    d_lj = float((scene.lid.data.root_pos_w[0, :2]
                  - scene.jar.data.root_pos_w[0, :2]).norm())
    check("settle/no-NaN: seeded reset settles finite — sealed (covered), ball "
          "captive, spike outside, piston/lid riding the jar axis, score 0",
          bool(scene._finite()[0]) and bool(scene.covered()[0])
          and bool(scene.ball_in_jar()[0]) and not bool(scene.engaged()[0])
          and d_pj < 0.004 and d_lj < 0.004
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (scene.pedestal.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.pedestal.data.root_quat_w[0]),
                scene.jar.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.jar.data.root_quat_w[0]),
                scene.dish.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.dish.data.root_quat_w[0]))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b = readback()
    d_pp, d_py = float((a[0] - b[0]).norm()), dyaw(a[1], b[1])
    d_jp, d_jy = float((a[2] - b[2]).norm()), dyaw(a[3], b[3])
    d_dp, d_dy = float((a[4] - b[4]).norm()), dyaw(a[5], b[5])
    print(f"[smoke] randomization deltas: ped_xy={d_pp * 1000:.1f}mm "
          f"ped_yaw={d_py:.1f}deg jar_xy={d_jp * 1000:.1f}mm jar_yaw={d_jy:.1f}deg "
          f"dish_xy={d_dp * 1000:.1f}mm dish_yaw={d_dy:.1f}deg", flush=True)
    check("randomization-is-real: pedestal xy+yaw, jar xy+yaw, dish xy+yaw "
          "readback differ between seeds",
          d_pp > 0.003 and d_py > 1.5 and d_jp > 0.003 and d_jy > 3.0
          and d_dp > 0.003 and d_dy > 3.0)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 4. INVERSION: you cannot shake it open =====================================
    # The sealed stack rested upside down on its rim (the DAMPED captive piston
    # slides its stroke without a free-fall hammer), then hopped twice (0.5 m/s up
    # written coherently to every stack body, land on the rim): the 10 N weld
    # carries it all — the mouth stays covered, the ball stays inside.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _refresh()
    qflip = _qx(torch.full((n,), math.pi, device=device))
    pj = scene.jar.data.root_pos_w.clone()
    target = pj.clone()
    target[:, 2] = env.iscene.env_origins[:, 2] + c.rim_z + 0.002
    for body in (scene.jar, scene.piston, scene.lid, scene.ball):
        off = body.data.root_pos_w - pj
        _place(body, target + _qapply(qflip, off), _qmul(qflip, body.data.root_quat_w))
    _step(120)
    inv_ok = bool(scene.covered()[0])
    for hop in range(2):
        _refresh()
        kick = torch.zeros(n, 3, device=device)
        kick[:, 2] = 0.5
        for body in (scene.jar, scene.piston, scene.lid, scene.ball):
            _place(body, body.data.root_pos_w.clone(),
                   body.data.root_quat_w.clone(), lin_vel=kick)
        _step(120)
        inv_ok = inv_ok and bool(scene.covered()[0])
    _report("inversion")
    _REC["on"] = False
    ball_l = scene._jar_local(scene.ball.data.root_pos_w)[0]
    check("INVERSION dump rejected: upside down + two hops, the weld holds — "
          "mouth still covered, ball still inside the jar, score 0, no success",
          inv_ok and bool(scene.covered()[0])
          and abs(lid_lz() - c.lid_seat_z) < 0.003
          and float(ball_l[0].abs()) < c.bore_half
          and float(ball_l[1].abs()) < c.bore_half
          and 0.0 < float(ball_l[2]) < c.rim_z
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 5. MECHANISM: resting on the spike is NOT enough ===========================
    # The sealed stack parked on the spike hands-off: it hangs by the seal (jar
    # weight ~4.4 N < 10 N breakForce). Engaged latches — nothing else does.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    write_stalled_on_spike()
    _step(240)
    _report("parked-on-spike")
    _REC["on"] = False
    check("MECHANISM sub-critical: parked on the spike hands-off the seal HOLDS "
          "(covered, ball captive below the shoulder) — engaged only, "
          "score <= engage credit, no success",
          bool(scene.engaged()[0]) and bool(scene.covered()[0])
          and abs(lid_lz() - c.lid_seat_z) < 0.003
          and not bool(scene.presented()[0])
          and float(scene.score()[0]) <= c.w_engage + 1e-6
          and not bool(scene.success()[0]))

    # ================= 6. bypass: ball in the dish PAST the sealed jar ============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _refresh()
    dq = scene.dish.data.root_quat_w.clone()
    hov = torch.zeros(n, 3, device=device)
    hov[:, 2] = 0.055
    _place(scene.ball, scene.dish.data.root_pos_w + _qapply(dq, hov))
    _step(180)
    _report("bypass-sealed")
    check("bypass rejected: the ball teleported into the dish while the jar is "
          "still SEALED — ball_in_dish holds yet no success (open-mouth clause), "
          "score ~0",
          bool(scene.ball_in_dish()[0]) and bool(scene.covered()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.01)

    # ================= 7+8. the seal threshold, bracketed from both sides =========================
    # 6 N up on the lid lifts the whole 0.58 kg stack off the ground hanging by
    # the weld (weld load ~5.6 N < 10 N: HOLDS; rise-limited so the drop back is
    # gentle). 15 N -> ~14 N > 10 N: SHEARS. (This is the battery's first shear —
    # everything after runs on the broken weld.)
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _refresh()
    zero = torch.zeros(n, 1, 3, device=device)
    z0 = float(scene.jar.data.root_pos_w[0, 2])
    held, lifted = True, False
    f = torch.zeros(n, 1, 3, device=device)
    f[:, 0, 2] = 6.0
    for _ in range(60):
        scene.lid.set_external_force_and_torque(f, zero, env_ids=_all_ids(),
                                                is_global=True)
        _step(1)
        held = held and bool(scene.covered()[0]) and abs(lid_lz() - c.lid_seat_z) < 0.004
        if float(scene.jar.data.root_pos_w[0, 2]) - z0 > 0.020:
            lifted = True   # the whole stack came up hanging by the weld
            break
    scene.lid.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    _step(150)  # fall back, land, settle
    _report("pull-6N")
    check("seal holds 6 N: a sub-threshold pull on the lid lifts the WHOLE "
          "stack hanging by the weld — the lid never leaves its seat",
          held and lifted and bool(scene.covered()[0])
          and abs(lid_lz() - c.lid_seat_z) < 0.004
          and not bool(scene.success()[0]))

    _REC["on"] = True
    f[:, 0, 2] = 15.0
    broke = False
    for i in range(120):
        scene.lid.set_external_force_and_torque(f, zero, env_ids=_all_ids(),
                                                is_global=True)
        _step(1)
        if abs(lid_lz() - c.lid_seat_z) > 0.015:  # the lid came away from its seat
            broke = True
            break
    scene.lid.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    print(f"[smoke] 15 N pull: weld {'SHEARED' if broke else 'held'} "
          f"after {i + 1} steps, lid_lz={lid_lz() * 1000:.1f}mm", flush=True)
    # tidy up: park the flying lid on clear ground, let everything land
    _place(scene.lid, pt(CLEAR[0], CLEAR[1], c.lid_t / 2 + 0.002))
    _step(150)
    _report("pull-15N")
    _REC["on"] = False
    check("seal shears 15 N: a super-threshold pull breaks the weld and the lid "
          "comes away — the 10 N threshold is bracketed from both sides",
          broke and bool(scene.jar_open()[0]))

    # ================= 9. negative: the SEED'S OWN OUTCOME ========================================
    # rlbench/open_jar's outcome is "the lid off the jar". Reproduced here — lid
    # clear on the ground, the ball still captive inside — it earns the opened
    # credit only: the GOAL of this task is the delivery.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _place(scene.lid, pt(CLEAR[0], CLEAR[1], c.lid_t / 2 + 0.002))
    _step(120)
    _report("seed-strategy")
    check("negative (SEED strategy): lid off, ball still captive in the jar — "
          "opened credit only (score <= 0.25), no success",
          bool(scene.jar_open()[0]) and bool(scene.ball_in_jar()[0])
          and float(scene.score()[0]) <= c.w_open + 1e-6
          and not bool(scene.success()[0]))

    # ================= 10. near-miss: ball BESIDE the dish ========================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _place(scene.lid, pt(CLEAR[0], CLEAR[1], c.lid_t / 2 + 0.002))
    _step(30)
    _refresh()
    beside = torch.zeros(n, 3, device=device)
    beside[:, 0] = c.dish_half + 0.030   # on the ground just past the dish wall
    beside[:, 2] = c.ball_r + 0.002
    dpos = scene.dish.data.root_pos_w.clone()
    dpos[:, 2] = env.iscene.env_origins[:, 2]
    _place(scene.ball, dpos + _qapply(scene.dish.data.root_quat_w, beside))
    _step(120)
    _report("near-miss-dish")
    check("near-miss (dish): the freed ball at rest on the ground BESIDE the "
          "dish — ball_in_dish False, no success",
          not bool(scene.ball_in_dish()[0]) and bool(scene.jar_open()[0])
          and not bool(scene.success()[0]))

    # ================= 11. wrong object: the LID in the dish ======================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _refresh()
    inpocket = torch.zeros(n, 3, device=device)
    inpocket[:, 2] = c.dish_t + c.lid_t / 2 + 0.004
    _place(scene.lid, scene.dish.data.root_pos_w + _qapply(
        scene.dish.data.root_quat_w, inpocket), scene.dish.data.root_quat_w.clone())
    _step(120)
    _report("wrong-object")
    lid_dl = scene._dish_local(scene.lid.data.root_pos_w)[0]
    check("negative (wrong object): the LID laid inside the dish, ball still in "
          "the jar — no success",
          float(lid_dl[0].abs()) < c.dish_xy and float(lid_dl[1].abs()) < c.dish_xy
          and bool(scene.ball_in_jar()[0]) and not bool(scene.ball_in_dish()[0])
          and not bool(scene.success()[0]))

    # ================= 12. presented-only: the 0.60 cap ===========================================
    # The full press end-state — jar bottomed around the spike, ball riding above
    # the rim, lid clear — but NO delivery: all three latches fire and the score
    # caps at 0.60. Also re-verifies the press-presentation geometry statically.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    write_pressed_open()
    _step(240)
    _report("presented-only")
    _REC["on"] = False
    check("presented-only: press end-state with no delivery — engaged+opened+"
          "presented latch, score == 0.60 cap, no success",
          bool(scene.engaged()[0]) and bool(scene.jar_open()[0])
          and bool(scene.presented()[0]) and not bool(scene.ball_in_dish()[0])
          and abs(float(scene.score()[0]) - 0.60) < 1e-5
          and not bool(scene.success()[0]))

    # ================= 13+14. settle gate / latched credit ========================================
    # From the press end-state, the ball is set INSIDE the dish pocket with a
    # 0.4 m/s lateral kick: every geometric success clause holds while it rolls,
    # yet success() refuses at every sampled step (consecutive-still counter).
    # The probe is dismantled (ball carried away) before it can settle into a
    # real success; the latched credit must survive.
    _REC["on"] = True
    _refresh()
    dq = scene.dish.data.root_quat_w.clone()
    inpk = torch.zeros(n, 3, device=device)
    inpk[:, 2] = c.dish_t + c.ball_r + 0.001
    kick = torch.zeros(n, 3, device=device)
    kick[:, 0] = 0.4
    _place(scene.ball, scene.dish.data.root_pos_w + _qapply(dq, inpk),
           None, lin_vel=_qapply(dq, kick))
    geom_ok, gate_ok = True, True
    for _ in range(45):
        _step(1)
        geom_ok = geom_ok and bool(scene.ball_in_dish()[0]) \
            and bool(scene.jar_open()[0])
        gate_ok = gate_ok and not bool(scene.still()[0]) \
            and not bool(scene.success()[0])
    _report("settle-gate")
    check("settle gate: the exact success geometry with the ball still rolling "
          "in the dish — geometric clauses hold for 45 steps yet success "
          "refuses (consecutive-still counter)",
          geom_ok and gate_ok)
    # dismantle before it can settle into a real success: carry the ball away
    _place(scene.ball, pt(CLEAR[0], CLEAR[1] - 0.10, c.ball_r + 0.002))
    _step(90)
    _report("dismantled")
    _REC["on"] = False
    check("latched credit: after the dismantle the live state is broken (ball "
          "gone) but the latched credit survives: score 0.60, no success",
          not bool(scene.ball_in_dish()[0]) and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.60) < 1e-5)

    # ================= 15. rejection audit ========================================================
    check("rejection audit: success() was never True at any step of this battery",
          _AUD["hits"] == 0)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.piston_jar")
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
    main()
