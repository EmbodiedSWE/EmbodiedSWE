"""Smoke / rubric-REJECTION battery for HanoiPlatesScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the honest 7-move relay reaches success and the
latched credit is monotone). This battery proves the rubric REJECTS wrong outcomes —
above all the trajectory-rule latches: a tower that ENDS UP looking perfect is still
worthless if a rule was ever broken on the way. Every probe is CONSTRUCTED (teleport,
real physics steps, judge) — instrumentation, never a solution.

Checks:
  1. settle/no-NaN    — seeded reset settles finite: tower threaded on the START post,
                        at rest, score 0, no success, no violation;
  2. randomization    — two seeded resets: rack xy and yaw READBACK differ; over 10
                        resets the (start, target) post pair takes >= 3 values with
                        both members varying;
  3. null-policy      — 240 idle steps: tower stays on the start post, no violation,
                        score ~0, no success;
  4. SEED STRATEGY    — the seed's plan scaled up ("pick the thing up, put it there"):
                        carry the WHOLE TOWER over in one move. The intact stack is
                        dropped over the target post — rule 1 (one plate at a time)
                        latches during the fall -> score 0, success False, forever;
  5. perfect-but-violated — two plates parked on the GROUND beside the rack (rules
                        1+2 latch), then a PHYSICALLY PERFECT tower is built on the
                        target post by legal single drops. Arrangement clauses all
                        True — success stays False and score 0: history is not
                        forgiven;
  6. big-on-small     — blue dropped on the target post, then RED dropped on top of it
                        (larger over smaller): rule 3 latches -> score 0, no success;
  7. near-miss + parking — red and yellow correctly relayed to the target post (score
                        hits 0.75), then blue set down on the PLANK midway between
                        posts: far outside the 30 mm threaded gate (>= 45 mm from
                        every post axis) -> not threaded, no success; left there it
                        becomes a rule-2 park -> score falls to 0;
  8. wrong-post relay — a fully LEGAL 7-move relay that rebuilds the tower on the
                        SPARE post (green pad ignored): no violation, tower perfect —
                        on the wrong post: no success, score stuck at the 0.15 buffer
                        credit (partial credit is anchored to the marked post);
  9. frames.npz       — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
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

PLATE = ("red", "yellow", "blue")

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


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    thr = scene.threaded()[0]
    posts = ["-" if not thr[i].any() else str(int(thr[i].float().argmax()))
             for i in range(3)]
    print(f"[smoke] {tag:20s} | plate->post R:{posts[0]} Y:{posts[1]} B:{posts[2]} "
          f"rest={scene.at_rest()[0].tolist()} violated={bool(scene._violated[0])} "
          f"s1={bool(scene._s1[0])} s2={bool(scene._s2[0])} s3={bool(scene._s3[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.hanoi_plates")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.25, -0.60, 0.55)) + o),
                                tuple(np.array((0.42, 0.00, 0.08)) + o),
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

    def write_plate(i: int, loc: torch.Tensor) -> None:
        """Write plate i to a rack-local point, flat (rack-aligned), zero velocity."""
        _refresh()
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.rack.data.root_pos_w + quat_apply(scene.rack.data.root_quat_w,
                                                             loc)
        st[:, 3:7] = scene.rack.data.root_quat_w
        scene.plates[i].write_root_state_to_sim(st, _all_ids())
        _refresh()

    def post_loc(post: int, z: float) -> torch.Tensor:
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0:2] = scene._post_xy[post]
        loc[:, 2] = z
        return loc

    def drop(i: int, post: int) -> None:
        """One legal-looking move: teleport plate i to 6 mm above the tip of `post`,
        then hands-off until it threads and rests (same transport as solve.py, but
        WITHOUT the no-violation assert — probes deliberately violate)."""
        write_plate(i, post_loc(post, c.base_h + c.post_h + c.plate_t / 2 + 0.006))
        for _ in range(360):
            _step(1)
            if bool(scene.threaded()[0, i, post]) and bool(scene.at_rest()[0, i]):
                break
        _step(20)

    def tower_perfect_on(post: int) -> bool:
        """Arrangement clauses only (no history): all three threaded on `post`, in
        size order, all at rest."""
        _refresh()
        thr = scene.threaded()[0]
        z = scene._plate_loc()[0, :, 2]
        return (bool(thr[:, post].all()) and float(z[0]) < float(z[1]) < float(z[2])
                and bool(scene.at_rest()[0].all()))

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    S = int(scene.start_post[0])
    check("settle/no-NaN: seeded reset settles finite, tower threaded at rest on the "
          "START post, score 0, no success, no violation",
          bool(scene._finite()[0]) and bool(scene.threaded()[0, :, S].all())
          and bool(scene.at_rest()[0].all()) and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]) and not bool(scene._violated[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene.rack.data.root_quat_w[0]),
                scene.rack.data.root_pos_w[0, :2].clone(),
                (int(scene.start_post[0]), int(scene.target_post[0])))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_xy, _a_pair = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_xy, _b_pair = readback()
    d_yawv = dyaw(a_yaw, b_yaw)
    d_xy = float((a_xy - b_xy).norm())
    pairs = set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        pairs.add((int(scene.start_post[0]), int(scene.target_post[0])))
    starts = {p[0] for p in pairs}
    targets = {p[1] for p in pairs}
    print(f"[smoke] randomization deltas: rack_yaw={d_yawv:.1f}deg "
          f"rack_xy={d_xy * 1000:.1f}mm pairs(10 resets)={sorted(pairs)}", flush=True)
    check("randomization-is-real: rack yaw and xy readback differ; (start,target) "
          "takes >= 3 values with both members varying over 10 resets",
          d_yawv > 2.0 and d_xy > 0.003 and len(pairs) >= 3
          and len(starts) >= 2 and len(targets) >= 2)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    S = int(scene.start_post[0])
    check("null-policy-fails: 240 idle steps — tower stays on the start post, no "
          "violation, score ~0, no success",
          bool(scene.threaded()[0, :, S].all()) and not bool(scene._violated[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 4. SEED STRATEGY: carry the whole tower over in one move ===================
    # The seed's manipulation model is "pick the thing up, put it on the goal" — here
    # that means grabbing the intact stack and dropping it over the target post. All
    # three plates are written as one stack high above the target tip in the SAME
    # instant; during the fall >= 2 plates are off-post for >= multi_off_steps, so
    # rule 1 latches BEFORE anything lands. However the plates land, the episode is
    # dead: score 0, success False, permanently.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    T = int(scene.target_post[0])
    _REC["on"] = True
    level = c.plate_t + c.boss_h
    for i in range(3):
        write_plate(i, post_loc(T, c.base_h + c.post_h + 0.10 + i * (level + 0.002)))
    _step(30)
    viol_mid_fall = bool(scene._violated[0])
    _step(240)
    _report("bulk-carry")
    _REC["on"] = False
    print(f"[smoke] bulk carry: rule latched during the fall={viol_mid_fall}", flush=True)
    check("SEED STRATEGY (bulk carry): the intact tower dropped over the target post "
          "— rule 1 latches during the fall; score 0, no success, forever",
          viol_mid_fall and bool(scene._violated[0])
          and float(scene.score()[0]) <= 0.001 and not bool(scene.success()[0]))

    # ================= 5. perfect arrangement, violated history ===================================
    # Rules are trajectory facts: park yellow AND blue on the GROUND beside the rack
    # (two plates off the posts -> rule 1; both at rest off-post -> rule 2), then
    # build a PHYSICALLY PERFECT tower on the target post with legal single drops.
    # The end state passes every arrangement clause — and is still worth nothing.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    T = int(scene.target_post[0])
    _REC["on"] = True
    for i, y in ((1, 0.22), (2, -0.22)):
        loc = torch.zeros(n, 3, device=device)
        loc[:, 1] = y
        loc[:, 2] = c.plate_t / 2 + 0.003
        write_plate(i, loc)
    _step(80)
    viol_after_park = bool(scene._violated[0])
    _report("parked-two")
    for i in (0, 1, 2):
        drop(i, T)
    _step(60)
    _report("perfect-after-viol")
    _REC["on"] = False
    check("perfect-but-violated: two plates parked on the ground (rules latch), then "
          "a physically perfect tower built on the target post — arrangement clauses "
          "all True, success still False, score 0",
          viol_after_park and tower_perfect_on(T) and bool(scene._violated[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.001)

    # ================= 6. big-on-small ============================================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    T = int(scene.target_post[0])
    _REC["on"] = True
    drop(2, T)          # blue (smallest) onto the empty target post — legal
    drop(0, T)          # RED (largest) dropped on top of it — rule 3
    _step(60)
    _report("big-on-small")
    _REC["on"] = False
    check("big-on-small: RED dropped onto the blue plate on the target post — rule 3 "
          "latches; score 0, no success",
          bool(scene._violated[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.001)

    # ================= 7. near-miss (not threaded) + rule-2 parking ===============================
    # Red and yellow are relayed correctly (credit reaches 0.75), then blue is set
    # down on the PLANK midway between two posts: >= 75 mm from every post axis, so
    # threaded() refuses it (a threaded plate can be at most ~18 mm off-axis). Not
    # success. Left there, it becomes a rule-2 park and the credit is wiped.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    T = int(scene.target_post[0])
    drop(0, T)          # red out from under the stack (yellow/blue drop a level,
    drop(1, T)          # staying threaded on the start post) — then yellow onto red
    other = 1 if T != 1 else 0
    mid_x = float((scene._post_xy[T, 0] + scene._post_xy[other, 0]) / 2)
    loc = torch.zeros(n, 3, device=device)
    loc[:, 0] = mid_x
    loc[:, 2] = c.base_h + c.plate_t / 2 + 0.003
    write_plate(2, loc)
    _step(26)
    _refresh()
    d_axes = (scene._plate_loc()[0, 2, None, :2] - scene._post_xy).norm(dim=-1)
    d_min = float(d_axes.min())
    thr_blue = bool(scene.threaded()[0, 2].any())
    s_before = float(scene.score()[0])
    ok_miss = (d_min > 0.045 and not thr_blue and not bool(scene.success()[0])
               and s_before >= 0.74)
    print(f"[smoke] near-miss: blue centre {d_min * 1000:.0f}mm from the nearest post "
          f"axis (threaded needs < {c.thread_xy_tol * 1000:.0f}mm), score={s_before:.2f}",
          flush=True)
    _step(120)
    _report("parked-one")
    check("near-miss + parking: blue on the plank between posts is NOT threaded "
          "(>= 45 mm off-axis, gate 30 mm) and no success at score 0.75; left "
          "there, rule 2 latches and the score is wiped to 0",
          ok_miss and bool(scene._violated[0]) and float(scene.score()[0]) <= 0.001
          and not bool(scene.success()[0]))

    # ================= 8. wrong-post relay (legal, perfect — wrong post) ==========================
    # A fully legal 7-move relay that rebuilds the tower on the SPARE post, using the
    # marked target post only as the buffer. No rule is ever broken, the final tower
    # is perfect — on the wrong post: no success, and the score sticks at the 0.15
    # buffer credit (s1 fires when yellow rests on the target mid-relay; s2/s3 never).
    torch.manual_seed(100)
    env.reset()
    _step(60)
    S, T = int(scene.start_post[0]), int(scene.target_post[0])
    P = 3 - S - T
    _REC["on"] = True
    for i, post in ((2, P), (1, T), (2, T), (0, P), (2, S), (1, P), (2, P)):
        drop(i, post)
    _step(60)
    _report("wrong-post")
    _REC["on"] = False
    check("wrong-post relay: legal 7-move relay onto the SPARE post — no violation, "
          "tower perfect on the wrong post: no success, score stuck at ~0.15",
          not bool(scene._violated[0]) and tower_perfect_on(P)
          and not bool(scene.success()[0])
          and 0.14 <= float(scene.score()[0]) <= 0.16)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.hanoi_plates")
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
