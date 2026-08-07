"""Smoke / rubric-REJECTION battery for HanoiRingsScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts the 7-move recursion and the
latched credit is monotone along a real trajectory). This battery proves the rubric
REJECTS wrong outcomes, and that the geometric claim the task rests on — a threaded
ring can only leave its post vertically over the tip — is physically load-bearing.
Every probe is CONSTRUCTED as a settled state (teleport, real physics steps, judge) —
instrumentation, never a solution: no probe here reaches success().

Checks:
   1. settle/no-NaN      — seeded reset settles finite; all three rings threaded on
                           the source post, none on the green post; score 0, no success;
   2. randomization      — two seeded resets: READBACK board yaw and board xy differ;
   3. slot permutation   — over 10 resets the post->slot permutation takes >= 3
                           distinct values and the GREEN post occupies >= 2 slots;
   4. null-policy        — 240 idle steps: score ~0, no success, no violation latch;
   5. POST INTERLOCK     — the threaded small ring is shoved sideways at 3x its own
                           weight for 1.5 s: its centre never escapes the shaft and it
                           re-seats threaded ("only out over the tip" is physics);
   6. SEED STRATEGY      — the seed's plan (open container, place object at target):
                           the correct L/M/S tower CONSTRUCTED standing on the board
                           BESIDE the green post — delivered, never threaded -> NO
                           success, score ~0;
   7. wrong post         — correct L/M/S tower CONSTRUCTED threaded on the spare gray
                           post -> no success, only the 0.15 first-move credit;
   8. HANOI VIOLATION    — stack CONSTRUCTED on the green post reading S/M/L bottom-up
                           (larger resting on smaller): the violation watchdog latches,
                           no success, score capped at 0.15;
   9. VIOLATION PERMANENT— same episode, the tower then REBUILT CORRECTLY (L/M/S on
                           green, which alone would score 0.80+): success still False,
                           score still capped at 0.15 — cheating first voids the run;
  10. near-miss          — L/M threaded correctly on green, the small ring lying flat
                           on the board centimetres from the pad -> no success, score
                           <= 0.80 (and >= 0.49: the fresh reset cleared the latch);
  11. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_kitchen_scene1_open_drawer_put_bowl_i57.smoke --headless
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

_qmul, _qz = task_scene._qmul, task_scene._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

GRAV = 9.81

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
    th = scene.threaded()[0]
    z = scene.ring_post_z()[0]
    letter = ("L", "M", "S")
    towers = []
    for p, ptag in zip(range(3), ("G", "g0", "g1")):
        on = [(float(z[r, p]), letter[r]) for r in range(3) if bool(th[r, p])]
        towers.append(f"{ptag}=[{''.join(s for _, s in sorted(on))}]")
    print(f"[smoke] {tag:18s} | {' '.join(towers)} "
          f"m1={bool(scene._m1[0])} lg={bool(scene._lg[0])} md={bool(scene._md[0])} "
          f"viol={bool(scene._viol[0])} score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


def _push(body, force_w: torch.Tensor, steps: int) -> None:
    """Apply a world-frame force at the body's CoM for `steps` steps, then clear."""
    n = _ENV.num_envs
    zero = torch.zeros(n, 1, 3, device=_ENV.device)
    f = force_w.view(1, 1, 3).expand(n, 1, 3).contiguous()
    for _ in range(steps):
        body.set_external_force_and_torque(f, zero, env_ids=_all_ids(), is_global=True)
        _step(1)
    body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.hanoi_rings")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply

    R, P = scene.RING_NAMES, scene.POST_NAMES

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.05, -0.75, 0.62)) + o),
                                tuple(np.array((0.36, 0.0, 0.08)) + o),
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

    def post_world(post_name: str, loc_xyz) -> torch.Tensor:
        """Post-local point -> world (per-env)."""
        _refresh()
        p = scene.posts[post_name]
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return p.data.root_pos_w + quat_apply(p.data.root_quat_w, loc)

    def post_quat(post_name: str) -> torch.Tensor:
        _refresh()
        return scene.posts[post_name].data.root_quat_w.clone()

    def build_tower(post_name: str, names_bottom_up: list[str]) -> None:
        """CONSTRUCT `names_bottom_up` threaded on `post_name`, then settle. All
        writes land before the first step so no intermediate state is judged."""
        q = post_quat(post_name)
        for j, nm in enumerate(names_bottom_up):
            z = c.pad_t + (j + 0.5) * c.ring_h + 0.003 * (j + 1)
            _write_body(scene.rings[nm], post_world(post_name, (0.0, 0.0, z)), q)
        _step(90)

    def park_far(names: list[str]) -> None:
        """Park rings flat on the ground far from the board (probe staging)."""
        oz = env.iscene.env_origins[:, 2]
        for j, nm in enumerate(names):
            p = env.iscene.env_origins.clone()
            p[:, 0] += -0.35
            p[:, 1] += -0.30 + 0.12 * j
            p[:, 2] = oz + c.ring_h / 2 + 0.002
            _write_body(scene.rings[nm], p, None)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    pos, _q, _v = scene._ring_tensors()
    th = scene.threaded()[0]
    src, tgt = scene.SOURCE, scene.TARGET
    check("settle/no-NaN: layout settles finite; all three rings threaded on the "
          "source post, none on the green post; score 0, no success",
          bool(torch.isfinite(pos).all())
          and all(bool(th[r, src]) for r in range(3))
          and int(th[:, tgt].sum()) == 0
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene.board.data.root_quat_w[0]),
                scene.board.data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_bp = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_bp = readback()
    d_yawv = dyaw(a_yaw, b_yaw)
    d_bp = float((a_bp - b_bp).norm())
    print(f"[smoke] randomization deltas: board_yaw={d_yawv:.1f}deg "
          f"board_xy={d_bp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: board yaw and board xy readback differ",
          d_yawv > 2.0 and d_bp > 0.003)

    # ================= 3. slot permutation coverage ===============================================
    perms, green_slots = set(), set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        p = tuple(scene.slot_of_post[0].tolist())
        perms.add(p)
        green_slots.add(p[0])
    print(f"[smoke] over 10 resets: post->slot perms {sorted(perms)} "
          f"green_slots {sorted(green_slots)}", flush=True)
    check("slot permutation: >= 3 distinct post->slot permutations over 10 resets "
          "and the GREEN post occupies >= 2 different slots",
          len(perms) >= 3 and len(green_slots) >= 2)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success, no violation "
          "(the legal start tower never trips the watchdog)",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0])
          and not bool(scene._viol[0]))

    # ================= 5. POST INTERLOCK: a threaded ring cannot be pushed off sideways ===========
    # "a ring can only leave or join a post by passing vertically over the tip":
    # shove the threaded SMALL ring sideways at 3x its weight for 1.5 s — the shaft
    # through its hole must hold it. This is the physics behind every Hanoi move
    # being a real thread-over-post insertion, not a free placement.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    src_name = P[scene.SOURCE]
    lat_w = quat_apply(scene.board.data.root_quat_w,
                       torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3))[0]
    f = 3.0 * float(c.ring_mass[2]) * GRAV * lat_w
    xy_max = -1.0
    ring_s = scene.rings["ring_small"]
    si = 2
    pi = scene.SOURCE
    for _ in range(9):  # 9 x 20 = 180 steps = 1.5 s of shoving
        _push(ring_s, f, 20)
        _refresh()
        xy_max = max(xy_max, float(
            scene._post_local()[0, si, pi, 0:2].norm()))
    _step(90)  # hands-off: fall back flat onto the stack
    _report("post-interlock")
    print(f"[smoke] lateral shove: max post-frame |xy|={xy_max * 1000:.1f}mm "
          f"(hole apothem {c.hole_r * 1000:.0f}mm, shaft r {c.shaft_r * 1000:.0f}mm)",
          flush=True)
    check("POST INTERLOCK: 3x-weight sideways shove for 1.5 s never gets the small "
          "ring's centre off the shaft, and it re-seats threaded",
          xy_max < 0.020 and bool(scene.threaded()[0, si, pi])
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6. negative: the SEED'S OWN STRATEGY =======================================
    # The seed's plan is "open the container, place the object at the target" — one
    # delivery, no threading, no ordering constraint. CONSTRUCT that kind of end
    # state: the correct L/M/S tower standing flat ON THE BOARD right beside the
    # green post's pad. Delivered to the target area, correct order — but never
    # threaded on any post -> must NOT be success, and no credit latches.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    park_far(list(R))
    _step(30)
    for j, nm in enumerate(["ring_large", "ring_mid", "ring_small"]):
        # z from the PAD TOP: the tower partially overlaps the pad footprint, so
        # spawn above it and let it settle leaning on the pad edge / board.
        z = c.pad_t + (j + 0.5) * c.ring_h + 0.003 * (j + 1)
        _write_body(scene.rings[nm], post_world(P[tgt], (0.0, 0.075, z)), None)
    _step(120)
    _report("seed-strategy")
    th6 = scene.threaded()[0]
    check("negative (SEED strategy): correct L/M/S tower standing on the board "
          "BESIDE the green post — threaded nowhere, NO success, score ~0",
          int(th6.sum()) == 0 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.05)
    _REC["on"] = False

    # ================= 7. negative: right tower, wrong post =======================================
    # The correct L/M/S tower threaded on the SPARE gray post. Everything about the
    # stack is right except the destination: no success, and the only credit is the
    # 0.15 first-move latch (small threaded on a non-source post).
    torch.manual_seed(100)
    env.reset()
    _step(30)
    park_far(list(R))
    _step(30)
    build_tower(P[2], ["ring_large", "ring_mid", "ring_small"])
    _report("wrong-post")
    check("negative (wrong post): correct L/M/S tower threaded on the spare gray "
          "post — no success, no large/mid credit, score <= 0.16",
          not bool(scene.success()[0]) and not bool(scene._lg[0])
          and float(scene.score()[0]) <= 0.16)

    # ================= 8. negative: HANOI VIOLATION latches =======================================
    # The green post loaded S/M/L bottom-up — the same three rings, INVERTED: the
    # large ring rests on the mid, the mid on the small. The watchdog must latch and
    # cap the score at 0.15 (m1 credit only), success must refuse.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    park_far(list(R))
    _step(30)
    build_tower(P[tgt], ["ring_small", "ring_mid", "ring_large"])
    _report("violation")
    check("negative (HANOI VIOLATION): green post stacked S/M/L (larger resting on "
          "smaller) — violation latches, no success, score capped at 0.15",
          bool(scene._viol[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.16)

    # ================= 9. VIOLATION IS PERMANENT ==================================================
    # Same episode: rebuild the tower CORRECTLY (L/M/S threaded on green — the exact
    # success configuration, which alone latches 0.80 of credit). The latched
    # violation must still void it: success False, score still capped at 0.15.
    park_far(list(R))
    _step(30)
    build_tower(P[tgt], ["ring_large", "ring_mid", "ring_small"])
    _step(60)
    _report("viol-permanent")
    check("VIOLATION PERMANENT: after the latch, the CORRECT L/M/S tower on the "
          "green post still scores <= 0.16 and success stays False",
          bool(scene._viol[0]) and bool(scene._lg[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.16)
    _REC["on"] = False

    # ================= 10. near-miss: small ring centimetres from home ============================
    # Fresh reset (latch must clear). L/M threaded correctly on green; the small
    # ring lies flat on the board a few centimetres from the pad. Two of three
    # placed perfectly -> 0.80 latched credit, but no success and never 1.0.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    park_far(list(R))
    _step(30)
    build_tower(P[tgt], ["ring_large", "ring_mid"])
    zsm = c.pad_t + 0.5 * c.ring_h + 0.002  # above the pad top (overlaps its footprint)
    _write_body(scene.rings["ring_small"], post_world(P[tgt], (0.0, 0.075, zsm)), None)
    _step(120)
    _report("near-miss")
    sc10 = float(scene.score()[0])
    check("near-miss: L/M correct on green, small ring flat on the board "
          "centimetres away — no success, 0.49 <= score <= 0.801 (fresh reset also "
          "cleared the violation latch)",
          not bool(scene.success()[0]) and not bool(scene._viol[0])
          and not bool(scene.threaded()[0, 2, tgt]) and 0.49 <= sc10 <= 0.801)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.hanoi_rings")
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
