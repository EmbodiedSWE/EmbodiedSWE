"""Smoke / rubric-REJECTION battery for QuarryScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts the real excavate-and-enthrone
trajectory and the latched credit is monotone along it). This battery proves the
rubric REJECTS wrong outcomes, and that the structural claims the task rests on —
the prize starts genuinely buried, the containment invariant is load-bearing, only
the gold prize seated in the pocket counts — hold. Every probe is CONSTRUCTED as a
settled state (teleport, real physics steps, judge) — instrumentation, never a
solution: no probe here reaches success().

Checks:
   1. settle/no-NaN      — seeded reset settles finite; prize on the pit floor,
                           COVERED (buried), all six stones inside the pit; score ~0,
                           no success;
   2. randomization      — two seeded resets: READBACK prize spawn xy moves > 5 mm
                           and the stone->pile-slot permutation differs;
   3. burial robustness  — 8 seeded resets: the prize is covered after settling in
                           ALL of them (the occlusion is not a lottery), and >= 6
                           distinct prize positions;
   4. null-policy        — 240 idle steps: score ~0, no success, prize never moves;
   5. KEEP-IN GATE       — one stone tossed OUTSIDE the pit first, the rest cleared,
                           the prize then seated PERFECTLY in the pocket: seated
                           reads True, but success is False and every latch stayed
                           dark (score ~0) — the containment invariant is load-
                           bearing, not decorative;
   6. WRONG OBJECT       — pile cleared legitimately, then a GRAY STONE seated in
                           the pocket with the prize left in the pit: no success,
                           only legitimate clear+expose credit (<= 0.36);
   7. NEAR-MISS PLACE    — pile cleared, prize set on the ground BESIDE the pedestal
                           (~10 cm from the pocket): no success, score < 1;
   8. BACK-IN-PIT        — the exposed prize returned to the pit floor: no success
                           (the pocket, not "out of the pit", is the goal);
   9. frames.npz         — video captured and saved to the CWD.

The seed's own end state (a cube riding aloft on a hand-held spatula blade) is not a
settleable state under this hands-off judge; its nearest settled analogs are checks
7 and 8 (prize out of the burial but not enthroned), both rejected.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.scoop_with_spatula_i387.smoke --headless
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

assert task_scene is not None

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


def _write_body(body, pos_local, quat: torch.Tensor | None = None) -> None:
    """Teleport `body` to an env-LOCAL position (zero velocity)."""
    n = _ENV.num_envs
    st = torch.zeros(n, 13, device=_ENV.device)
    st[:, 0] = float(pos_local[0])
    st[:, 1] = float(pos_local[1])
    st[:, 2] = float(pos_local[2])
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    st[:, 0:3] += _ENV.iscene.env_origins
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    tl = scene._token_local()[0]
    print(f"[smoke] {tag:16s} | token=({float(tl[0]):+.3f},{float(tl[1]):+.3f},"
          f"{float(tl[2]):+.3f}) covered={bool(scene.covered()[0])} "
          f"all_in={bool(scene.all_in_pit()[0])} seated={bool(scene.token_seated()[0])} "
          f"clear={bool(scene._clear[0])} expose={bool(scene._expose[0])} "
          f"lift={bool(scene._lift[0])} near={bool(scene._near[0])} "
          f"settled={bool(scene.settled()[0])} score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.quarry")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.30, -0.75, 0.85)) + o),
                                tuple(np.array((0.28, 0.10, 0.06)) + o),
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

    def settle(max_steps: int = 480, tail: int = 60) -> None:
        for i in range(max_steps):
            _step(1)
            if i > 30 and bool(scene.settled()[0]):
                break
        _step(tail)

    seat_z = c.ped_h + c.token_size / 2

    def clear_stones(names, slots) -> None:
        """CONSTRUCT: relocate the named stones, top-down by current height, to the
        given east-half drop slots (hover + hands-off fall + settle each)."""
        order = list(names)
        _refresh()
        loc, _v = scene._rubble_tensors()
        order.sort(key=lambda nm: -float(loc[0, scene.RUBBLE_NAMES.index(nm), 2]))
        for k, nm in enumerate(order):
            sx, sy = slots[k]
            _write_body(scene.rubble[nm], (sx, sy, c.wall_h + c.rubble_size / 2 + 0.045))
            settle(300, 30)

    # ================= 1. settle / no-NaN / buried ================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(240)
    _report("settle")
    _REC["on"] = False
    loc, _v = scene._rubble_tensors()
    tl = scene._token_local()[0]
    check("settle/no-NaN: layout settles finite; prize on the pit floor and COVERED, "
          "all six stones inside the pit; score ~0, no success",
          bool(torch.isfinite(loc).all()) and bool(scene.covered()[0])
          and bool(scene.all_in_pit()[0])
          and abs(float(tl[2]) - c.token_size / 2) < 0.012
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (scene.token_spawn[0].clone(), tuple(scene.pile_slot[0].tolist()))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_xy, a_perm = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_xy, b_perm = readback()
    d_xy = float((a_xy - b_xy).norm())
    print(f"[smoke] randomization: token_spawn moved {d_xy * 1000:.1f}mm, "
          f"perm {a_perm} -> {b_perm}", flush=True)
    check("randomization-is-real: prize spawn xy readback moves > 5 mm and the "
          "stone->pile-slot permutation differs",
          d_xy > 0.005 and a_perm != b_perm)

    # ================= 3. burial robustness across seeds ==========================================
    buried_all = True
    positions = set()
    for s in range(8):
        torch.manual_seed(300 + s)
        env.reset()
        _step(200)
        _refresh()
        cov = bool(scene.covered()[0])
        buried_all = buried_all and cov and bool(scene.all_in_pit()[0])
        t = scene.token_spawn[0]
        positions.add((round(float(t[0]), 3), round(float(t[1]), 3)))
        if not cov:
            print(f"[smoke] seed {300 + s}: prize NOT covered after settle", flush=True)
    print(f"[smoke] burial: covered in all 8 seeds = {buried_all}, "
          f"{len(positions)} distinct prize positions", flush=True)
    check("burial robustness: prize covered after settling in all 8 seeded resets, "
          ">= 6 distinct prize positions",
          buried_all and len(positions) >= 6)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _refresh()
    t0 = scene._token_local()[0].clone()
    _step(240)
    _refresh()
    drift = float((scene._token_local()[0] - t0).norm())
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success, prize never moves",
          float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0])
          and drift < 0.01 and bool(scene.covered()[0]))

    # ================= 5. KEEP-IN GATE (containment invariant is load-bearing) ====================
    # One stone tossed OUTSIDE the pit FIRST, the rest cleared, the prize then seated
    # PERFECTLY in the pocket. `seated` must read True — yet success must be False and
    # every latch must have stayed dark: all credit is gated on all-six-in-pit.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    _write_body(scene.rubble["rubble_3"], (0.75, -0.30, 0.06))
    settle(240, 30)
    rest = [nm for nm in scene.RUBBLE_NAMES if nm != "rubble_3"]
    clear_stones(rest, c.clear_slots[:5])
    _write_body(scene.token, (c.ped_pos[0], c.ped_pos[1], seat_z + 0.055))
    settle(300, 60)
    _report("keep-in-gate")
    _REC["on"] = False
    check("KEEP-IN GATE: prize seated in the pocket but one stone outside the pit — "
          "seated True, success False, all latches dark (score ~0)",
          bool(scene.token_seated()[0]) and not bool(scene.all_in_pit()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.01)

    # ================= 6. WRONG OBJECT (only the gold prize counts) ===============================
    # Pile cleared legitimately (clear+expose latch — honest credit), then a GRAY
    # STONE is seated in the pocket while the prize stays in the pit. No success;
    # nothing beyond the legitimate clearing credit.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    clear_stones(list(scene.RUBBLE_NAMES), c.clear_slots)
    settle(240, 30)
    s_clear = float(scene.score()[0])
    _write_body(scene.rubble["rubble_0"], (c.ped_pos[0], c.ped_pos[1], seat_z + 0.055))
    settle(300, 60)
    _report("wrong-object")
    _REC["on"] = False
    check("WRONG OBJECT: a gray stone seated in the pocket, prize left in the pit — "
          "no success, only legitimate clear+expose credit (<= 0.36)",
          s_clear <= 0.36 and not bool(scene.token_seated()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.36)

    # ================= 7. NEAR-MISS PLACE (beside the pedestal is not the pocket) =================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    clear_stones(list(scene.RUBBLE_NAMES), c.clear_slots)
    _write_body(scene.token, (c.ped_pos[0] + 0.10, c.ped_pos[1], c.token_size / 2 + 0.005))
    settle(300, 60)
    _report("near-miss")
    check("NEAR-MISS PLACE: prize settled on the ground beside the pedestal — "
          "not seated, no success, score < 1",
          not bool(scene.token_seated()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.70)

    # ================= 8. BACK-IN-PIT (out of the burial is not the goal) =========================
    # Same episode: the exposed prize returned to the (now empty) burial zone floor.
    _write_body(scene.token, (float(scene.token_spawn[0, 0]),
                              float(scene.token_spawn[0, 1]),
                              c.token_size / 2 + 0.005))
    settle(300, 60)
    _report("back-in-pit")
    _REC["on"] = False
    check("BACK-IN-PIT: exposed prize resting on the pit floor — not seated, "
          "no success, score < 1",
          not bool(scene.token_seated()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.70)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.quarry")
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
