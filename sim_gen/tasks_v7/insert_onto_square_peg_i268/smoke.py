"""Smoke / rubric-REJECTION battery for RingBalanceScene — NullRobot, teleported probes.

solve.py is the acceptance proof (force-fetch each gold ring, contact-guided descent
over the stepped post, hands-off equilibrium). This battery proves the rubric REJECTS
wrong outcomes and that the strategic differentiator from the seed is load-bearing:
the seed (rlbench/insert_onto_square_peg) is satisfied by threading A ring onto A peg
— here that exact end state (one ring on the free post, count ignored) leaves the
beam pressed on its ballast stop and earns only partial credit. Every probe is
CONSTRUCTED as a settled state (teleport, real physics steps, judge); constructed
states may earn latched partial credit but none may reach success() unless the beam
genuinely floats level with the exact gold count and the ballast intact.

Checks:
   1. settle/no-NaN    — seeded reset settles finite: beam pressed on the ballast-
                         side stop, k in [1,4] black rings on that post, gold flat
                         on the floor, score ~0, no success;
   2. randomization    — two seeded resets: READBACK stand xy, stand yaw and the
                         gold scatter all differ;
   3. flags read back  — across seeds BOTH ballast sides occur and >= 3 distinct k
                         values; each side flag matches the pressed-angle sign AND
                         a black-ring-on-that-post readback;
   4. null-policy      — 600 idle steps: beam stays pressed, rings stay put,
                         score ~0, no success;
   5. SEED STRATEGY    — the seed's end state ("a ring is on the peg", count
                         ignored): ONE gold ring threaded on a k=2 episode leaves
                         the beam PRESSED on the ballast stop — no success, only
                         partial credit;
   6. over-load        — k+1 gold rings threaded: the beam tips onto the OPPOSITE
                         stop — no success, score capped at 0.60 (over-insertion
                         is a real failure mode, not "more is better");
   7. wrong-post       — a gold ring threaded on top of the BLACK stack (ballast
                         post): zero load credit, beam stays pressed, no success;
   8. ballast cheat    — "unload the ballast instead of counterweighting it":
                         k-1 black rings removed + one gold threaded -> the beam
                         floats LEVEL, settled — and still no success (ballast
                         gate), score stays at the partial cap;
   9. streak gate +    — k gold rings CONSTRUCTED on the free post: while the beam
      positive control   swings up through the level band success stays False (the
                         1 s settled streak rejects a beam swinging through level);
                         hands-off it then settles level -> genuine success 1.0;
  10. revocation       — from that success, one gold ring yanked off: success
                         reads False again, the beam re-tips onto the ballast
                         stop, the score falls to the latched 0.60 cap;
  11. wrong-object     — the two PARKED black rings threaded on the free post of
                         a k=2 episode: the beam floats LEVEL (weight is weight)
                         but n_free counts GOLD only -> no success, score ~0;
  12. frames.npz      — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.insert_onto_square_peg_i268.smoke --headless
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

_qmul, _qy, _qz = task_scene._qmul, task_scene._qy, task_scene._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

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
    print(f"[smoke] {tag:16s} | ang={float(scene.beam_angle_deg()[0]):+7.2f}deg "
          f"n_free={int(scene.n_free()[0])} k={int(scene.k_count[0])} "
          f"ballast_ok={bool(scene.ballast_intact()[0])} "
          f"settled={bool(scene._settled()[0])} streak={int(scene._streak[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ring_balance")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.30, -1.10, 0.85)) + o),
                                tuple(np.array((0.0, 0.0, 0.30)) + o),
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

    def ang() -> float:
        _refresh()
        return float(scene.beam_angle_deg()[0])

    def sgn() -> float:
        return float(scene.ballast_sign[0])

    def kk() -> int:
        return int(scene.k_count[0])

    def score() -> float:
        _refresh()
        return float(scene.score()[0])

    def succ() -> bool:
        _refresh()
        return bool(scene.success()[0])

    def z_seat(j: int) -> float:
        return -c.pan_drop + c.ring_t / 2 + 0.001 + j * (c.ring_t + c.stack_gap)

    def seat_ring(body, side: float, j: int) -> None:
        """Teleport `body` to the j-th seat of the post on beam-local x = side*arm,
        aligned with the CURRENT beam pose (the beam itself is never re-posed —
        rings are free bodies, the jointed stand+beam linkage stays untouched)."""
        _refresh()
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = side * c.arm
        loc[:, 2] = z_seat(j)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.beam.data.root_pos_w + quat_apply(scene.beam.data.root_quat_w, loc)
        st[:, 3:7] = scene.beam.data.root_quat_w
        body.write_root_state_to_sim(st, _all_ids())
        _refresh()

    def floor_ring(body, x: float, y: float) -> None:
        """Lay `body` flat on the ground at env-local (x, y)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = x
        st[:, 1] = y
        st[:, 2] = c.ring_t / 2 + 0.002
        st[:, 0:3] += env.iscene.env_origins
        st[:, 3] = 1.0
        body.write_root_state_to_sim(st, _all_ids())
        _refresh()

    def reseed(sd: int) -> None:
        torch.manual_seed(sd)
        env.reset()
        _refresh()

    # a k=2 seed serves checks 5-11 (needs k >= 2 for imbalance probes AND
    # exactly 2 parked black rings for the wrong-object probe)
    seed_k2 = None
    for sd in range(500, 560):
        reseed(sd)
        if kk() == 2:
            seed_k2 = sd
            break
    print(f"[smoke] scanned seeds 500..: k=2 episode at seed {seed_k2}", flush=True)
    assert seed_k2 is not None, "no k=2 seed found in 60 draws"

    # ================= 1. settle / no-NaN =========================================================
    reseed(100)
    _REC["on"] = True
    _step(180)
    _report("settle")
    _REC["on"] = False
    gz = max(float(b.data.root_pos_w[0, 2] - env.iscene.env_origins[0, 2])
             for b in scene.gold)
    check("settle/no-NaN: reset settles finite — beam pressed on the ballast-side "
          "stop, k in [1,4], gold flat on the floor, score ~0, no success",
          bool(scene._finite()[0]) and ang() * sgn() >= 10.0
          and abs(ang()) <= c.stop_deg + 1.0 and 1 <= kk() <= 4
          and int(scene.n_free()[0]) == 0 and gz < 0.05
          and score() <= 0.02 and not succ())

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        q = scene.stand.data.root_quat_w[0]
        yaw = math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))
        sp = (scene.stand.data.root_pos_w[0, :2] - env.iscene.env_origins[0, :2]).clone()
        gp = (scene.gold[0].data.root_pos_w[0, :2] - env.iscene.env_origins[0, :2]).clone()
        return yaw, sp, gp

    reseed(101)
    _step(10)
    a_yaw, a_sp, a_gp = readback()
    reseed(202)
    _step(10)
    b_yaw, b_sp, b_gp = readback()
    d_yaw = abs(a_yaw - b_yaw) % 360.0
    d_yaw = min(d_yaw, 360.0 - d_yaw)
    d_sp = float((a_sp - b_sp).norm())
    d_gp = float((a_gp - b_gp).norm())
    print(f"[smoke] randomization deltas: stand_yaw={d_yaw:.1f}deg "
          f"stand_xy={d_sp * 1000:.1f}mm gold0_xy={d_gp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: stand yaw, stand xy and the gold scatter "
          "readback all differ across seeds",
          d_yaw > 2.0 and d_sp > 0.005 and d_gp > 0.01)

    # ================= 3. randomized flags occur and read back ====================================
    seen_side = {1.0: 0, -1.0: 0}
    seen_k: set[int] = set()
    consistent = True
    for sd in range(300, 312):
        reseed(sd)
        _step(30)
        seen_side[sgn()] += 1
        seen_k.add(kk())
        # side flag must match physics readback: beam pressed toward the ballast
        # side and the first black ring captured on THAT post
        consistent &= ang() * sgn() > 8.0
        consistent &= bool(scene._on_post(scene.black[0], scene.ballast_sign)[0])
    print(f"[smoke] flags over 12 seeds: side+={seen_side[1.0]} side-={seen_side[-1.0]} "
          f"k_values={sorted(seen_k)} consistent={consistent}", flush=True)
    check("randomization (flags): both ballast sides occur, >= 3 distinct k, and "
          "each side flag matches the pressed-angle + black-on-post readback",
          all(v > 0 for v in seen_side.values()) and len(seen_k) >= 3 and consistent)

    # ================= 4. null policy fails =======================================================
    reseed(100)
    _step(60)
    g0 = torch.stack([b.data.root_pos_w[0] for b in scene.gold])
    _step(600)
    _report("null-policy")
    drift = float((torch.stack([b.data.root_pos_w[0] for b in scene.gold]) - g0)
                  .norm(dim=-1).max())
    check("null-policy-fails: 600 idle steps — beam stays pressed on its stop, "
          "gold rings stay put (drift < 2 cm), score ~0, no success",
          ang() * sgn() >= 10.0 and drift < 0.02 and score() <= 0.02 and not succ())

    # ================= 5. negative: the SEED'S OWN STRATEGY =======================================
    # rlbench/insert_onto_square_peg is satisfied by "a ring is on the peg" — count
    # ignored. Here: ONE gold ring on the free post of a k=2 episode. The beam must
    # stay PRESSED on the ballast stop: partial credit only, never success.
    reseed(seed_k2)
    _step(60)
    _REC["on"] = True
    seat_ring(scene.gold[0], -sgn(), 0)
    _step(420)
    _report("seed-strategy")
    _REC["on"] = False
    check("negative (SEED strategy): one ring threaded with the count ignored "
          "(k=2) leaves the beam pressed on the ballast stop — no success, "
          "score only the lift+half-load partials",
          ang() * sgn() >= 8.0 and int(scene.n_free()[0]) == 1
          and not succ() and 0.30 <= score() <= 0.40)

    # ================= 6. over-load: one too many tips the OPPOSITE stop ==========================
    reseed(seed_k2)
    _step(60)
    _REC["on"] = True
    for j in range(kk() + 1):
        seat_ring(scene.gold[j], -sgn(), j)
        _step(30)
    _step(540)
    _report("over-load")
    _REC["on"] = False
    check("over-load: k+1 gold rings tip the beam onto the OPPOSITE stop — no "
          "success, score capped at 0.60 (over-insertion never pays)",
          ang() * sgn() <= -10.0 and not succ() and score() <= 0.601)

    # ================= 7. wrong-post: gold on the BLACK stack earns nothing =======================
    reseed(seed_k2)
    _step(60)
    _REC["on"] = True
    seat_ring(scene.gold[0], sgn(), kk())  # on TOP of the ballast stack
    _step(360)
    _report("wrong-post")
    _REC["on"] = False
    check("wrong-post: a gold ring threaded on top of the BLACK stack — zero load "
          "credit (n_free 0), beam stays pressed, no success",
          int(scene.n_free()[0]) == 0 and ang() * sgn() >= 8.0
          and not succ() and score() <= 0.12)

    # ================= 8. ballast cheat: unloading instead of counterweighting ====================
    # Remove k-1 black rings and thread ONE gold: the beam genuinely floats level
    # (1 v 1) and settles — and success must STILL be False (ballast gate).
    reseed(seed_k2)
    _step(60)
    _REC["on"] = True
    for i in range(1, kk()):
        floor_ring(scene.black[i], -1.0 - 0.15 * i, -1.0)
    seat_ring(scene.gold[0], -sgn(), 0)
    _step(900)
    _report("ballast-cheat")
    _REC["on"] = False
    check("ballast cheat: k-1 black rings removed + one gold threaded — the beam "
          "floats LEVEL and settled, yet success is False (ballast stack broken) "
          "and the score stays partial",
          abs(ang()) <= c.succ_tol_deg and bool(scene._settled()[0])
          and not bool(scene.ballast_intact()[0]) and not succ()
          and score() <= 0.601)

    # ================= 9. streak gate + positive control ==========================================
    # Construct the exact-count answer: k gold rings seated on the free post of the
    # pressed beam. As the beam swings up it passes through the level band — success
    # must read False there (streak gate). Hands-off it then settles level: the
    # constructed exact count is GENUINE success (the rubric is satisfiable).
    reseed(seed_k2)
    _step(60)
    _REC["on"] = True
    for j in range(kk()):
        seat_ring(scene.gold[j], -sgn(), j)
        _step(20)
    saw_inband_nosucc = False
    settled_success = False
    for i in range(2400):
        _step(1)
        a = abs(ang())
        if a <= c.succ_tol_deg and int(scene._streak[0]) < c.succ_streak and not succ():
            saw_inband_nosucc = True
        if succ():
            settled_success = True
            break
    _step(120)  # a little more hands-off margin
    _report("exact-count")
    _REC["on"] = False
    check("streak gate: while the beam swings up THROUGH the level band success "
          "stays False (settled-streak gate)", saw_inband_nosucc)
    check("positive control: the constructed exact count settles level hands-off "
          "— genuine success, score 1.0",
          settled_success and succ() and score() >= 0.999
          and bool(scene.ballast_intact()[0]))

    # ================= 10. revocation: success is judged live =====================================
    ok_pre = succ()
    _REC["on"] = True
    floor_ring(scene.gold[kk() - 1], -1.0, 1.0)  # yank the top gold ring off
    _step(420)
    _report("revoked")
    _REC["on"] = False
    check("revocation: one gold ring yanked off — success reads False again, the "
          "beam re-tips onto the ballast stop, score falls to the latched 0.60",
          ok_pre and not succ() and ang() * sgn() >= 8.0
          and 0.599 <= score() <= 0.601)

    # ================= 11. wrong-object: black rings don't count ==================================
    # Thread the two PARKED black rings onto the free post (k=2): weight is weight,
    # the beam floats level — but n_free counts GOLD only, so no success and no
    # credit (nothing gold ever moved).
    reseed(seed_k2)
    _step(60)
    _REC["on"] = True
    for j, i in enumerate(range(kk(), c.n_black)):
        seat_ring(scene.black[i], -sgn(), j)
        _step(20)
    _step(900)
    _report("wrong-object")
    _REC["on"] = False
    check("wrong-object: two parked BLACK rings threaded on the free post — the "
          "beam floats LEVEL yet n_free (gold only) is 0: no success, score ~0",
          abs(ang()) <= c.succ_tol_deg and bool(scene._settled()[0])
          and int(scene.n_free()[0]) == 0 and not succ() and score() <= 0.02)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ring_balance")
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
    except Exception as exc:  # noqa: BLE001 - Kit teardown hangs; die loudly NOW
        print(f"SIM_GEN_SMOKE: FAIL (exception: {exc!r})", flush=True)
        import traceback

        traceback.print_exc()
        os._exit(1)
