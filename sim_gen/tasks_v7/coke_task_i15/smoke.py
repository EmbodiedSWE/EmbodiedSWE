"""Smoke / rubric-REJECTION battery for CanBalanceScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts the constructed equilibrium and the
latched credit is monotone along a real trajectory). This battery proves the rubric
REJECTS wrong outcomes, and that every success() clause is load-bearing — including the
cheats that make the beam PHYSICALLY LEVEL by the wrong means. Every probe is
CONSTRUCTED as a state (teleport, real physics steps, judge) — instrumentation, never a
solution.

Checks:
   1. settle/no-NaN     — seeded reset settles finite; the beam SEATED on its knife
                          edge and tipped HARD onto its stop by the preload (doing
                          nothing = visibly tilted balance); score 0, no success;
   2. randomization     — READBACK across seeded resets: stand yaw / stand xy /
                          can_red spawn all differ, and the PRELOAD SUBSET (the
                          episode's hidden target mass) takes >= 3 distinct values;
   3. null-policy       — 300 idle steps: beam stays on its stop, score ~0;
   4. UNLOAD-PRELOAD    — the preload plates teleported off the pan: the pendulum
                          levels the empty beam — a LEVEL, seated, settled beam with
                          no cans scores ~0 and is NOT success (preload_ok and
                          a-can-on-pan are load-bearing, level alone is nothing);
   5. SEED STRATEGY     — the seed task's plan is grasp-and-lift / transport a can:
                          all three cans delivered BESIDE the balance on the floor —
                          transport without weighing changes nothing; score ~0;
   6. WRONG SUBSET      — a can subset 100 g off the target placed on the empty pan:
                          the beam rests HARD TILTED against a stop (the statics
                          gate), seated, and success never fires; score <= 0.65;
   7. SPARE-PLATE cheat — preload {400 g}: spare plate_200 + can_blue (200+200 g) on
                          the empty pan make the beam PHYSICALLY LEVEL — rejected by
                          spares_ok: a level beam with a spare plate aboard is not
                          success (where the mass comes from matters);
   8. SHUFFLE cheat     — preload {100 g}: can_blue on the empty pan and can_red
                          parked ON THE LOADED PAN also level the beam — rejected by
                          cans_ok (every can must be on the EMPTY pan or clear);
   9. near-miss         — a required can left standing on the stand BASE next to the
                          pans (delivered to the balance, never weighed): inside
                          clear_r, on nothing — cans_ok fails, beam stays on stop;
  10. OFF-CRADLE        — the beam laid LEVEL on the ground, its preload plates
                          restacked on the loaded pan and the correct cans set on the
                          empty pan — every clause but seated is True and it is still
                          rejected: the measurement only counts on the knife edge;
  11. REMOVE-A-CAN      — full success CONSTRUCTED live (correct subset, beam level),
                          then one required can teleported to the floor: the beam
                          swings back onto its stop, success COLLAPSES while the
                          latched score survives at the 0.20+0.15k partial cap —
                          the equilibrium is maintained by contact, not bookkept;
  12. rejection audit   — success() never fired at any judged step during the
                          negative probes (checks 4-10);
  13. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.coke_task_i15.smoke --headless
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

_qz = task_scene._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
_AUDIT = {"on": False, "fired": False}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if _AUDIT["on"]:
            _AUDIT["fired"] |= bool(env.scene.success()[0])
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


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.can_balance")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.25, -0.95, 0.75)) + o),
                                tuple(np.array((0.28, 0.00, 0.14)) + o),
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

    def status():
        return scene._status()

    def tilt() -> float:
        return float(status()["tilt"][0])

    def dseat() -> float:
        _refresh()
        seat_loc = torch.tensor([0.0, 0.0, c.pivot_h], device=device).expand(n, 3)
        seat_w = scene.stand.data.root_pos_w \
            + quat_apply(scene.stand.data.root_quat_w, seat_loc)
        return float((scene.beam.data.root_pos_w - seat_w).norm(dim=-1)[0])

    def report(tag: str) -> None:
        s = status()
        print(f"[smoke] {tag:16s} | tilt={float(s['tilt'][0]):+6.2f}deg "
              f"seated={bool(s['seated'][0])} dseat={dseat():.4f} "
              f"preload_ok={bool(s['preload_ok'][0])} "
              f"spares_ok={bool(s['spares_ok'][0])} cans_ok={bool(s['cans_ok'][0])} "
              f"cans_on={s['cans_on'][0].tolist()} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(_REC['frames'])}", flush=True)

    def wait_beam(max_steps: int = 720) -> None:
        for k in range(max_steps):
            _step(1)
            if k > 60 and bool(scene._settled(scene.beam)[0]):
                return

    def beam_pose(loc_x: float, loc_y: float, loc_z: float) -> torch.Tensor:
        """(N,13) root state at BEAM-local (x,y,z), orientation = live beam quat,
        zero velocity (the release pose of a gripper working over the pan)."""
        _refresh()
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0], loc[:, 1], loc[:, 2] = loc_x, loc_y, loc_z
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.beam.data.root_pos_w \
            + quat_apply(scene.beam.data.root_quat_w, loc)
        st[:, 3:7] = scene.beam.data.root_quat_w
        return st

    def drop_on_pan(body, half_h: float, off_x: float, off_y: float,
                    side: int = +1, settle: int = 300) -> None:
        """TRANSPORT `body` to a hover 8 mm above the pan at `side`, hands-off."""
        body.write_root_state_to_sim(
            beam_pose(side * c.arm_len + off_x, off_y,
                      c.pan_floor_z + half_h + 0.008), _all_ids())
        _step(settle)

    def to_floor(body, x: float, y: float, half_h: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.env_origins
        st[:, 0] += x
        st[:, 1] += y
        st[:, 2] += half_h + 0.003
        st[:, 3] = 1.0
        body.write_root_state_to_sim(st, _all_ids())

    def required() -> list[int]:
        return [i for i in range(3) if bool(scene.preload[0, i])]

    def offsets(lst: list[int]) -> dict[int, tuple[float, float]]:
        """Zero-net-moment drop offsets (same layout the teleport solution uses)."""
        if len(lst) == 1:
            return {lst[0]: (0.0, 0.0)}
        if len(lst) == 2:
            return {lst[0]: (0.0, -0.031), lst[1]: (0.0, 0.031)}
        return {0: (0.0232, 0.0268), 1: (-0.0349, 0.0067), 2: (0.0116, -0.0335)}

    def place_subset(lst: list[int]) -> None:
        offs = offsets(lst)
        for i in lst:
            drop_on_pan(scene.cans[c.can_names[i]], c.can_h[i] / 2, *offs[i])
        wait_beam(960)
        _step(120)

    def fresh(seed: int, settle: int = 120) -> None:
        torch.manual_seed(seed)
        env.reset()
        _step(settle)
        wait_beam(600)

    def reset_until(pattern: list[bool], base: int) -> None:
        """Seeded resets until the sampled preload matches `pattern` (READBACK —
        the randomization is real, so the pattern must be hunted, not written)."""
        for t in range(40):
            torch.manual_seed(base + 1000 * t)
            env.reset()
            _step(30)
            if [bool(scene.preload[0, i]) for i in range(3)] == pattern:
                _step(90)
                wait_beam(600)
                return
        raise AssertionError(f"no seeded reset produced preload {pattern}")

    gate = c.level_tol_deg

    # ================= 1. settle / no-NaN / tipped initial condition ==============================
    fresh(100)
    _REC["on"] = True
    _step(60)
    _REC["on"] = False
    report("settle")
    s = status()
    bodies = [scene.beam, *scene.plates.values(), *scene.cans.values()]
    finite = all(bool(torch.isfinite(b.data.root_pos_w).all()) for b in bodies)
    check("settle/no-NaN: seeded reset settles finite; beam SEATED on its knife edge "
          "and tipped hard onto its stop by the preload; score 0, no success",
          finite and bool(s["seated"][0]) and abs(tilt()) > 2.0 * gate
          and bool(s["preload_ok"][0]) and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        q = scene.stand.data.root_quat_w
        yaw = math.degrees(2.0 * math.atan2(float(q[0, 3]), float(q[0, 0])))
        return (yaw, scene.stand.data.root_pos_w[0, :2].clone(),
                scene.cans["can_red"].data.root_pos_w[0, :2].clone(),
                tuple(bool(scene.preload[0, i]) for i in range(3)))

    obs = []
    for k in range(6):
        torch.manual_seed(300 + k)
        env.reset()
        _step(10)
        obs.append(readback())
    d_yaw = max(o[0] for o in obs) - min(o[0] for o in obs)
    d_sp = max(float((a[1] - b[1]).norm()) for a in obs for b in obs)
    d_can = max(float((a[2] - b[2]).norm()) for a in obs for b in obs)
    subsets = {o[3] for o in obs}
    print(f"[smoke] randomization: yaw spread {d_yaw:.1f}deg, stand xy spread "
          f"{d_sp * 1000:.0f}mm, can_red spread {d_can * 1000:.0f}mm, "
          f"preload subsets {sorted(subsets)}", flush=True)
    check("randomization-is-real: stand yaw / stand xy / can_red spawn READBACK all "
          "vary and the preload subset (the hidden target mass) takes >= 3 values "
          "across 6 seeded resets",
          d_yaw > 5.0 and d_sp > 0.005 and d_can > 0.05 and len(subsets) >= 3)

    # ================= 3. null policy fails =======================================================
    fresh(100)
    _step(300)
    report("null-policy")
    check("null-policy-fails: 300 idle steps, beam still hard on its stop, score ~0, "
          "no success",
          abs(tilt()) > 2.0 * gate and float(scene.score()[0]) <= 0.02
          and not bool(scene.success()[0]))

    _AUDIT["on"] = True  # ---- negative probes below: success must never fire ----

    # ================= 4. UNLOAD-PRELOAD: level alone is nothing ==================================
    fresh(100)
    _REC["on"] = True
    for j, i in enumerate(required()):
        to_floor(scene.plates[c.plate_names[i]], -0.75, 0.45 + 0.15 * j,
                 c.plate_t[i] / 2)
    wait_beam(960)
    _step(120)
    report("unload-preload")
    _REC["on"] = False
    s = status()
    check("UNLOAD-PRELOAD counterfactual: plates teleported off the pan, the "
          "pendulum levels the EMPTY beam — level + seated + settled with no cans "
          "is NOT success and scores ~0 (level alone is nothing)",
          abs(tilt()) < gate and bool(s["seated"][0])
          and not bool(s["preload_ok"][0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.02)

    # ================= 5. SEED STRATEGY: transport without weighing ===============================
    fresh(100)
    _REC["on"] = True
    _refresh()
    for j, nm in enumerate(c.can_names):
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = 0.42
        loc[:, 1] = -0.14 + 0.14 * j
        loc[:, 2] = c.can_h[j] / 2 + 0.004
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.stand.data.root_pos_w \
            + quat_apply(scene.stand.data.root_quat_w, loc)
        st[:, 3:7] = scene.stand.data.root_quat_w
        scene.cans[nm].write_root_state_to_sim(st, _all_ids())
    _step(300)
    report("seed-strategy")
    _REC["on"] = False
    check("negative (SEED strategy): all three cans grasped-and-delivered BESIDE the "
          "balance on the floor — transport without weighing changes nothing: beam "
          "hard on its stop, score ~0, no success",
          abs(tilt()) > 2.0 * gate and float(scene.score()[0]) <= 0.02
          and not bool(scene.success()[0]))

    # ================= 6. WRONG SUBSET: the statics gate ==========================================
    fresh(100)
    req = required()
    wrong = sorted(set(req) ^ {0}) or [1]  # +/- 100 g off the unique answer
    t_req = sum(c.masses[i] for i in req)
    t_wrong = sum(c.masses[i] for i in wrong)
    assert abs(t_wrong - t_req) > 0.05, "wrong subset must miss the target"
    print(f"[smoke] wrong subset: target {t_req * 1000:.0f} g, placing "
          f"{[c.can_names[i] for i in wrong]} = {t_wrong * 1000:.0f} g", flush=True)
    _REC["on"] = True
    place_subset(wrong)
    report("wrong-subset")
    _REC["on"] = False
    s = status()
    check("negative (WRONG SUBSET): cans 100 g off the target rest the beam HARD "
          "against a stop (>= 2x the level gate), seated — no success, score <= 0.65",
          abs(tilt()) > 2.0 * gate and bool(s["seated"][0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.65)

    # ================= 7. SPARE-PLATE cheat: level by the wrong means =============================
    # preload {400 g} -> spares are plate_100/plate_200, the correct answer is
    # can_green alone. Cheat: spare plate_200 + can_blue = 400 g on the empty pan —
    # the beam levels PHYSICALLY, and spares_ok still rejects it.
    reset_until([False, False, True], base=500)
    _REC["on"] = True
    drop_on_pan(scene.plates["plate_200"], c.plate_t[1] / 2, 0.0, 0.0)
    drop_on_pan(scene.cans["can_blue"], c.can_h[1] / 2 + c.plate_t[1], 0.0, 0.0)
    wait_beam(960)
    _step(120)
    report("spare-plate")
    _REC["on"] = False
    s = status()
    check("near-miss (SPARE-PLATE cheat): preload 400 g countered by spare "
          "plate_200 + can_blue (400 g) — the beam IS physically level, and it is "
          "still rejected (spares_ok): where the counterweight comes from matters",
          abs(tilt()) < gate and bool(s["seated"][0])
          and not bool(s["spares_ok"][0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.30)

    # ================= 8. SHUFFLE cheat: a can on the LOADED pan ==================================
    # preload {100 g} -> the correct answer is can_red alone. Cheat: can_blue on the
    # empty pan and can_red parked ON THE LOADED pan (100+100 vs 200) — level again,
    # and cans_ok rejects it.
    reset_until([True, False, False], base=700)
    _REC["on"] = True
    drop_on_pan(scene.cans["can_blue"], c.can_h[1] / 2, 0.0, 0.0)
    drop_on_pan(scene.cans["can_red"], c.can_h[0] / 2 + c.plate_t[0], 0.0, 0.0,
                side=-1)
    wait_beam(960)
    _step(120)
    report("shuffle-cheat")
    _REC["on"] = False
    s = status()
    check("near-miss (SHUFFLE cheat): can_red parked on the LOADED pan and can_blue "
          "on the empty pan level the beam physically — rejected by cans_ok (every "
          "can must rest on the EMPTY pan or stay clear)",
          abs(tilt()) < gate and bool(s["seated"][0])
          and not bool(s["cans_ok"][0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.30)

    # ================= 9. near-miss: delivered to the balance, never weighed ======================
    fresh(100)
    i0 = required()[0]
    _refresh()
    loc = torch.zeros(n, 3, device=device)
    loc[:, 0] = 0.10
    loc[:, 2] = 0.016 + c.can_h[i0] / 2 + 0.005
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.stand.data.root_pos_w \
        + quat_apply(scene.stand.data.root_quat_w, loc)
    st[:, 3:7] = scene.stand.data.root_quat_w
    scene.cans[c.can_names[i0]].write_root_state_to_sim(st, _all_ids())
    _step(300)
    report("on-stand-base")
    s = status()
    check("near-miss (on the stand base): a required can standing on the balance's "
          "base plate next to the pans — inside clear_r, on no pan — fails cans_ok; "
          "beam stays on its stop, no success",
          abs(tilt()) > 2.0 * gate and not bool(s["cans_ok"][0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.02)

    # ================= 10. OFF-CRADLE: everything but the knife edge ==============================
    fresh(100)
    req = required()
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.env_origins
    st[:, 0] += -0.10
    st[:, 1] += -0.55
    st[:, 2] += 0.070
    st[:, 3] = 1.0
    scene.beam.write_root_state_to_sim(st, _all_ids())
    _step(180)
    z_cur = 0.0
    for i in sorted(req, reverse=True):  # thickest-first restack on the grounded beam
        drop_on_pan(scene.plates[c.plate_names[i]], z_cur + c.plate_t[i] / 2,
                    0.0, 0.0, side=-1, settle=120)
        z_cur += c.plate_t[i] + 0.002
    offs = offsets(req)
    for i in req:
        drop_on_pan(scene.cans[c.can_names[i]], c.can_h[i] / 2, *offs[i], settle=180)
    _step(300)
    report("off-cradle")
    s = status()
    check("near-miss (OFF-CRADLE): the beam laid LEVEL on the ground, preload "
          "restacked on its loaded pan, the CORRECT cans on its empty pan — every "
          "clause but seated holds and it is still rejected: the measurement only "
          "counts on the knife edge",
          abs(tilt()) < gate and bool(s["preload_ok"][0])
          and bool(s["cans_on"][0].any()) and dseat() > c.seat_tol
          and not bool(s["seated"][0]) and not bool(scene.success()[0]))

    _AUDIT["on"] = False  # ---- end of negative probes ----

    # ================= 11. REMOVE-A-CAN: success collapses, latches survive =======================
    fresh(100)
    req = required()
    _REC["on"] = True
    place_subset(req)
    report("full-solution")
    built = bool(scene.success()[0])
    score_before = float(scene.score()[0])
    to_floor(scene.cans[c.can_names[req[0]]], -0.75, -0.55, c.can_h[req[0]] / 2)
    wait_beam(960)
    _step(120)
    report("can-removed")
    _REC["on"] = False
    latched = min(0.65, c.w_first + c.w_each * len(req) + c.w_near)
    score_after = float(scene.score()[0])
    check("REMOVE-A-CAN: constructed success is live (correct subset, beam level, "
          "score 1.0), then one required can teleported away -> the beam swings "
          "back onto its stop, success COLLAPSES while the latched score survives "
          "at the partial cap",
          built and score_before == 1.0 and abs(tilt()) > 2.0 * gate
          and not bool(scene.success()[0])
          and abs(score_after - latched) < 0.011 and score_after < 0.9)

    # ================= 12. rejection audit ========================================================
    check("rejection audit: success() never fired at any step of the negative "
          "probes (checks 4-10)", not _AUDIT["fired"])

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.can_balance")
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
    except BaseException as e:  # noqa: BLE001 - die fast, never idle until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL (exception: {e})", flush=True)
        os._exit(2)
