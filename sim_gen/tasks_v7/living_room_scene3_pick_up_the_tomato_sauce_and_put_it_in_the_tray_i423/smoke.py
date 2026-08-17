"""Smoke battery for InertiaDerbyScene (sim_gen task
`living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray_i423`) —
REJECTION-ONLY: every check either verifies basic health / hidden-state /
randomization or CONSTRUCTS/EXECUTES a wrong outcome and asserts the rubric
refuses it. success() must never fire anywhere in the battery.

 1. settle/no-NaN     — orbs nested in their cradles per the sampled
                        permutation, bar seated in its pockets, tray upright on
                        the bench, EQUAL masses read back, hidden inertia read
                        back (exactly one solid k~0.40, two hollow k~2/3),
                        score ~0, no success.
 2. randomization     — across 8 seeds: genuine_idx varies, the orb->cradle
                        permutation varies, the tray x and yaw vary; the orbs
                        settle in their cradles every time and the physx
                        inertia readback matches the sampled genuine_idx.
 3. null policy       — 240 idle steps -> score ~0, no success.
 4. gate sanity       — an orb staged in a lane against the RESTING bar stays
                        blocked uphill of the gate (no raced latch, score 0):
                        the race physically requires lifting the bar.
 5. SEED strategy     — the seed's plain pick-and-place executed for real: the
                        genuine orb (hidden truth peeked BY THE SMOKE) dropped
                        straight into the tray without any race -> genuine_in
                        holds but race_done is False: no success, score ~0.
 6. single-race       — only ONE orb rolled through the runout, then the
                        genuine delivered: the >=2-raced rule refuses (score
                        <= half race credit + delivery gated off).
 7. contamination     — FLAGSHIP end-state-identical rejection: two orbs raced
                        for real, then a COUNTERFEIT set in the tray until the
                        foul latches (score clamps <= 0.15), then the fake
                        removed and the genuine delivered cleanly. The final
                        state is IDENTICAL to a success state — genuine seated,
                        tray clean, race done, all settled — but the foul is
                        permanent: never success, score stays <= 0.15.
 8. near-miss         — race done, genuine orb resting on the bench BESIDE the
                        tray (touching distance): no success, no delivery.
 9. tipped tray       — race done, tray tipped on its side, genuine dropped at
                        its footprint: tray_ok False -> never success.
10. raced persistence — two orbs raced, then carried BACK to their cradles:
                        the raced credit (0.15) survives (latched), delivery
                        stays off, no success.
11. rejection audit   — success() observed False at every step of the battery.
12. final no-NaN.
13. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray_i423.smoke --headless
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

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
    xs = [float(scene._local(b)[0, 0]) for b in scene.balls]
    print(f"[smoke] {tag:14s} | x=({xs[0]:+.3f},{xs[1]:+.3f},{xs[2]:+.3f}) "
          f"raced={scene._raced[0].tolist()} gin={bool(scene.genuine_in()[0])} "
          f"fin={bool(scene.fake_in()[0])} foul={bool(scene._foul[0])} "
          f"del={bool(scene._delivered[0])} score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.inertia_derby")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    th = math.radians(c.theta_deg)

    def surf(x: float) -> float:
        return c.bench_z + c.lip + (c.ramp_x1 - x) * math.tan(th)

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.85, -1.05, 0.85)) + o),
                                tuple(np.array((-0.05, 0.05, 0.12)) + o),
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

    def origin(dx: float, dy: float, dz: float) -> torch.Tensor:
        return (scene.env_origins
                + torch.tensor([dx, dy, dz], device=device)).expand(n, 3)

    def qx(theta_rad: float) -> torch.Tensor:
        q = torch.zeros(n, 4, device=device)
        q[:, 0] = math.cos(theta_rad / 2)
        q[:, 1] = math.sin(theta_rad / 2)
        return q

    def truth() -> int:
        return int(scene.genuine_idx[0])

    def roll_orbs(bs: list[int], steps: int = 320) -> None:
        """REAL race physics: hover-drop the orbs onto the ramp DOWNHILL of the
        (still seated) start bar, one per lane, and let gravity roll them
        through the runout to the finish wall. (The bar is not disturbed; the
        raced latch is earned by the actual roll.)"""
        lanes = (+c.lane_y, -c.lane_y)
        for i, b in enumerate(bs[:2]):
            hz = surf(-0.26) + c.ball_r / math.cos(th) + 0.006
            _write_body(scene.balls[b], origin(-0.26, lanes[i], hz))
        _step(steps)

    def deliver(b: int, settle: int = 200) -> None:
        """Transport the orb to hover over the tray centre and DROP; gravity
        seats it on the tray floor."""
        p = scene.tray.data.root_pos_w[0]
        pos = torch.zeros(n, 3, device=device)
        pos[:, 0] = p[0]
        pos[:, 1] = p[1]
        pos[:, 2] = p[2] + c.tray_t + c.ball_r + 0.030
        _write_body(scene.balls[b], pos)
        _step(settle)

    def park(b: int, spot: int = 0) -> None:
        """Remove an orb to the open ground beyond the bench."""
        _write_body(scene.balls[b],
                    origin(0.55, 0.28 - 0.18 * spot, c.ball_r + 0.002))
        _step(30)

    zc = c.bench_z + c.curb_h + math.sqrt(c.ball_r ** 2 - c.cradle_in ** 2)

    # ================= 1. settle / no-NaN / masses / hidden inertia ==============================
    torch.manual_seed(11)
    env.reset()
    _step(150)
    _report("reset")
    masses = [float(b.root_physx_view.get_masses().view(-1)[0]) for b in scene.balls]
    ks = scene.ball_k()[0]
    perm = scene.perm[0].tolist()
    print(f"[smoke] masses={['%.3f' % m for m in masses]} "
          f"k={['%.3f' % float(v) for v in ks]} genuine={truth()} perm={perm}",
          flush=True)
    cradled = all(
        abs(float(scene._local(scene.balls[b])[0, 0]) - c.slots_x[perm[b]]) < 0.012
        and abs(float(scene._local(scene.balls[b])[0, 1]) - c.slot_y) < 0.012
        for b in range(3))
    bar_z = float(scene._local(scene.bar)[0, 2])
    bar_seated = abs(bar_z - (surf(c.gate_x) + c.bar_gap + c.bar_lx / 2)) < 0.006
    n_solid = int(((ks - c.k_true).abs() < 0.05).sum())
    n_hollow = int(((ks - c.k_fake).abs() < 0.05).sum())
    check("settle/no-NaN: orbs cradled per the sampled permutation, bar seated "
          "in its pockets, tray upright, EQUAL masses read back, hidden "
          "inertia read back (one solid, two hollow), score ~0, no success",
          bool(scene._finite()[0]) and cradled and bar_seated
          and bool(scene.tray_ok()[0])
          and all(abs(m - c.ball_m) < 0.02 * c.ball_m for m in masses)
          and n_solid == 1 and n_hollow == 2
          and float(ks[truth()]) < 0.5
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2. randomization readback ==================================================
    gens, perms, txs, tyaws, ok_all = [], [], [], [], True
    for k in range(8):
        torch.manual_seed(20 + k)
        env.reset()
        _step(100)
        _refresh()
        gens.append(truth())
        perms.append(tuple(scene.perm[0].tolist()))
        tp = scene._local(scene.tray)[0]
        q = scene.tray.data.root_quat_w[0]
        tyaws.append(math.degrees(2 * math.atan2(float(q[3]), float(q[0]))))
        txs.append(float(tp[0]))
        ks = scene.ball_k()[0]
        ok_all = ok_all and bool(scene._finite()[0]) \
            and int(torch.argmin(ks)) == gens[-1] \
            and all(abs(float(scene._local(scene.balls[b])[0, 1]) - c.slot_y) < 0.012
                    for b in range(3)) \
            and not succ()
    tx_span = max(txs) - min(txs)
    yaw_span = max(tyaws) - min(tyaws)
    print(f"[smoke] readback: genuine={gens} perms={len(set(perms))} distinct "
          f"tx_span={tx_span * 1000:.0f}mm yaw_span={yaw_span:.1f}deg ok={ok_all}",
          flush=True)
    check("randomization: genuine_idx varies, the orb->cradle permutation "
          "varies, tray x and yaw vary; orbs settle cradled every seed and the "
          "physx inertia readback tracks the sampled genuine_idx",
          len(set(gens)) >= 2 and len(set(perms)) >= 3
          and tx_span > 0.04 and yaw_span > 6.0 and ok_all)

    # ================= 3. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 4. gate sanity: the resting bar really blocks =============================
    torch.manual_seed(41)
    env.reset()
    _step(100)
    hz = surf(-0.345) + c.ball_r / math.cos(th) + 0.008
    _write_body(scene.balls[0], origin(-0.345, c.lane_y, hz))
    _step(300)
    _report("gate-block")
    x0 = float(scene._local(scene.balls[0])[0, 0])
    check("gate sanity: an orb staged in a lane rolls against the RESTING bar "
          "and stays blocked uphill of the gate — no raced latch, score 0: "
          "the race physically requires lifting the bar",
          x0 < c.gate_x and not bool(scene._raced[0, 0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 5. SEED strategy: deliver without racing ==================================
    torch.manual_seed(51)
    env.reset()
    _step(120)
    deliver(truth(), settle=260)
    _report("seed-skill")
    check("SEED strategy: the genuine orb (truth peeked by the smoke) dropped "
          "straight into the tray with NO race — genuine_in holds, race_done "
          "False: delivery gated off, score ~0, never success",
          bool(scene.genuine_in()[0]) and not bool(scene.race_done()[0])
          and not bool(scene._delivered[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 6. single-race delivery ===================================================
    torch.manual_seed(61)
    env.reset()
    _step(120)
    roll_orbs([truth()])          # only ONE orb rolled through the runout
    deliver(truth(), settle=260)
    _report("single-race")
    check("single-race: only ONE orb raced, then the genuine delivered — the "
          ">=2-raced rule refuses: no success, score <= half race credit",
          int(scene._raced[0].sum()) == 1 and bool(scene.genuine_in()[0])
          and not bool(scene.race_done()[0]) and not bool(scene._delivered[0])
          and float(scene.score()[0]) <= c.w_race / 2 + 1e-3 and not succ())

    # ================= 7. contamination FLAGSHIP (end-state-identical) ===========================
    torch.manual_seed(71)
    env.reset()
    _step(120)
    g = truth()
    fakes = [b for b in range(3) if b != g]
    roll_orbs(fakes)              # real race: both counterfeits earn the latch
    s_raced = float(scene.score()[0])
    deliver(fakes[0], settle=120)  # counterfeit SETTLES in the tray -> foul latches
    _report("fouled")
    foul_hit = bool(scene._foul[0])
    s_foul = float(scene.score()[0])
    park(fakes[0], spot=0)         # undo the visible state: remove the fake...
    deliver(g, settle=260)         # ...and deliver the genuine CLEANLY
    _step(200)
    _report("flagship-end")
    end_identical = bool(scene.genuine_in()[0]) and not bool(scene.fake_in()[0]) \
        and bool(scene.race_done()[0]) and bool(scene.settled_core()[0]) \
        and bool(scene.tray_ok()[0])
    check("contamination flagship: two orbs raced (0.15), a counterfeit "
          "settled in the tray -> permanent foul (score clamps <= 0.15), fake "
          "removed, genuine delivered cleanly — the end state is IDENTICAL to "
          "success (genuine seated, tray clean, race done, settled) but the "
          "foul is forever: never success, score stays <= 0.15",
          abs(s_raced - c.w_race) < 0.02 and foul_hit
          and s_foul <= 0.15 + 1e-3 and end_identical
          and float(scene.score()[0]) <= 0.15 + 1e-3 and not succ())

    # ================= 8. near-miss: genuine BESIDE the tray =====================================
    torch.manual_seed(81)
    env.reset()
    _step(120)
    g = truth()
    roll_orbs([b for b in range(3) if b != g])
    p = scene.tray.data.root_pos_w[0]
    pos = torch.zeros(n, 3, device=device)
    pos[:, 0] = p[0] + c.tray_out / 2 + c.ball_r + 0.012
    pos[:, 1] = p[1]
    pos[:, 2] = float(p[2]) + c.ball_r + 0.004
    _write_body(scene.balls[g], pos)
    _step(220)
    _report("near-miss")
    check("near-miss: race done, the genuine orb resting on the bench BESIDE "
          "the tray (touching distance) — not inside: no delivery, no success",
          bool(scene.race_done()[0]) and not bool(scene.genuine_in()[0])
          and not bool(scene._delivered[0])
          and float(scene.score()[0]) <= c.w_race + 1e-3 and not succ())

    # ================= 9. tipped tray =============================================================
    torch.manual_seed(91)
    env.reset()
    _step(120)
    g = truth()
    roll_orbs([b for b in range(3) if b != g])
    p = scene.tray.data.root_pos_w[0]
    pos = torch.zeros(n, 3, device=device)
    pos[:, 0] = p[0]
    pos[:, 1] = p[1]
    pos[:, 2] = c.bench_z + c.tray_out / 2 + 0.01 + float(scene.env_origins[0, 2])
    _write_body(scene.tray, pos, qx(math.pi / 2))   # on its side
    _step(120)
    pos2 = scene.tray.data.root_pos_w.clone()
    pos2[:, 2] += c.tray_out / 2 + c.ball_r
    _write_body(scene.balls[g], pos2)
    _step(220)
    _report("tipped-tray")
    check("tipped tray: race done, tray tipped on its side, genuine dropped at "
          "its footprint — tray_ok False: never success",
          bool(scene.race_done()[0]) and not bool(scene.tray_ok()[0])
          and not bool(scene.genuine_in()[0])
          and float(scene.score()[0]) <= c.w_race + 1e-3 and not succ())

    # ================= 10. raced-latch persistence ===============================================
    torch.manual_seed(101)
    env.reset()
    _step(120)
    g = truth()
    fakes = [b for b in range(3) if b != g]
    roll_orbs(fakes)
    s_after_race = float(scene.score()[0])
    perm = scene.perm[0].tolist()
    for b in fakes:               # carry the racers back to their cradles
        _write_body(scene.balls[b],
                    origin(c.slots_x[perm[b]], c.slot_y, zc + 0.004))
        _step(40)
    _step(150)
    _report("re-cradled")
    check("raced persistence: two orbs raced (0.15) then carried back to their "
          "cradles — the raced credit survives (latched), delivery stays off, "
          "no success",
          abs(s_after_race - c.w_race) < 0.02
          and float(scene.score()[0]) >= c.w_race - 1e-3
          and int(scene._raced[0].sum()) >= 2
          and not bool(scene._delivered[0]) and not succ())

    # ================= 11-13. audit, no-NaN, video ===============================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.inertia_derby")
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
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
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
