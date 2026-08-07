"""Smoke / rubric-REJECTION battery for MugHookScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a real force-threaded,
gravity-hung mug and the latched credit is monotone along that trajectory). This
battery proves the rubric REJECTS wrong outcomes — above all the SEED task's own
end state (the mug set down on the floor at the "right spot") — and that the
geometric claims the task rests on (the 30 mm thread gate, the 25 mm tip margin,
the 0.10 m airborne gate, the closed loop actually carrying load) are physically
load-bearing. Every probe is CONSTRUCTED as a settled state (teleport, real
physics steps, judge) — instrumentation, never a solution: no probe here reaches
success() (the one real hang constructed is on the WRONG peg).

Checks:
   1. settle/no-NaN      — seeded reset settles finite, mug and beaker at rest on
                           the floor, score 0, no success;
   2. randomization      — READBACK across seeds: rack xy, mug xy, beaker xy all
                           move; over 10 resets the BLUE peg occupies >= 2 distinct
                           slots (the color->slot permutation is real) and its
                           mounting height spans > 5 mm;
   3. null-policy        — 240 idle steps: score ~0, no success (nothing moves);
   4. SEED STRATEGY      — the seed's plan ("carry the object to the right spot and
                           set it down"): the mug placed on the FLOOR directly under
                           the blue peg, settled -> NO success, score ~0 (suspension,
                           not placement, is the goal);
   5. wrong object       — the amber BEAKER (no handle) pushed onto the blue peg by
                           its mouth, the mug untouched -> no success, score 0;
   6. WRONG PEG          — a REAL hang constructed on the RED peg (threaded, airborne,
                           settled — everything but the color) -> NO success, score
                           caps at lift credit;
   7. LOOP INTERLOCK     — physics claim: that red-peg hang is load-bearing — a 3x-
                           weight downward shove for 1.5 s cannot tear the closed
                           loop off the peg (still threaded-on-red, still airborne);
   8. mouth-hook         — the mug perched on the BLUE peg by its CUP OPENING (the
                           plausible near-miss): hangs, and its settled tilt can put
                           the aperture center near the distance gate — but the
                           aperture stays EDGE-ON to the axis, so the alignment
                           clause rejects it: threaded() False, no latch, no success;
   9. off-tip near-miss  — the solve's own STAGING pose (aperture aligned on the
                           axis, 10 mm past the tip): NOT threaded (the 25 mm tip
                           margin holds — teleported staging can't latch), and on
                           release the unsupported mug simply FALLS to the floor ->
                           no success (suspension is gravity + contact, not pose);
  10. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_kitchen_scene7_put_the_white_bowl_to_the_right_of_the_plate_i24.smoke --headless
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
    from . import scene as scene_mod
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod

_ = scene_mod  # imported for its registrations

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
    mz = float((scene.mug.data.root_pos_w - scene.env_origins)[0, 2])
    d_b = float(scene.dist_to_peg("blue")[0])
    d_r = float(scene.dist_to_peg("red")[0])
    print(f"[smoke] {tag:18s} | mug_z={mz:.3f} d_blue={d_b * 1000:5.1f}mm "
          f"d_red={d_r * 1000:5.1f}mm lift={bool(scene._lift[0])} "
          f"thread={bool(scene._thread[0])} threaded_now={bool(scene.threaded()[0])} "
          f"airborne={bool(scene.airborne()[0])} settled={bool(scene.mug_settled()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


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
    from isaaclab.utils.math import quat_apply, quat_from_matrix

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.mug_hook")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.10, -0.70, 0.55)) + o),
                                tuple(np.array((0.35, 0.05, 0.15)) + o),
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

    up = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)

    def peg_axes(color: str):
        """(root, dir, tip) of peg `color` (world), frozen tensors."""
        _refresh()
        root, dirw = scene.peg_frame(color)
        return root.clone(), dirw.clone(), (root + c.peg_len * dirw).clone()

    def hang_frame(dirw: torch.Tensor) -> torch.Tensor:
        """Mug orientation for threading: local +y along -dir, local +z up-ish."""
        that = -dirw
        z_ax = up - (up * that).sum(-1, keepdim=True) * that
        z_ax = z_ax / z_ax.norm(dim=-1, keepdim=True)
        x_ax = torch.cross(that, z_ax, dim=-1)
        return quat_from_matrix(torch.stack([x_ax, that, z_ax], dim=-1))

    def place_aperture(color: str, s_ax: float) -> torch.Tensor:
        """CONSTRUCT: aperture-aligned mug, aperture center at axial coord `s_ax`
        of peg `color`. Returns the orientation used."""
        root, dirw, _tip = peg_axes(color)
        q = hang_frame(dirw)
        ap = root + s_ax * dirw
        pos = ap - quat_apply(q, scene._ap_local.expand(n, 3))
        _write_body(scene.mug, pos, q)
        return q

    def mouth_hook(body, color: str, gap: float) -> None:
        """CONSTRUCT: vessel slid over peg `color` by its CUP OPENING — cup axis
        along the peg, opening toward the panel, bottom `gap` past the tip."""
        root, dirw, tip = peg_axes(color)
        z_c = -dirw  # cup opening (+z local) faces the panel
        h = up
        y_c = torch.cross(z_c, h, dim=-1)
        y_c = y_c / y_c.norm(dim=-1, keepdim=True)
        x_c = torch.cross(y_c, z_c, dim=-1)
        q = quat_from_matrix(torch.stack([x_c, y_c, z_c], dim=-1))
        _write_body(body, tip + gap * dirw, q)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(150)
    _report("settle")
    _REC["on"] = False
    mz = float((scene.mug.data.root_pos_w - scene.env_origins)[0, 2])
    bz = float((scene.beaker.data.root_pos_w - scene.env_origins)[0, 2])
    check("settle/no-NaN: layout settles finite; mug and beaker at rest on the "
          "floor; score 0, no success",
          bool(torch.isfinite(scene.mug.data.root_state_w).all())
          and mz < 0.05 and bz < 0.05 and bool(scene.mug_settled()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (scene.rack.data.root_pos_w[0, :2].clone(),
                scene.mug.data.root_pos_w[0, :2].clone(),
                scene.beaker.data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_r, a_m, a_b = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_r, b_m, b_b = readback()
    d_r, d_m, d_b = (float((a_r - b_r).norm()), float((a_m - b_m).norm()),
                     float((a_b - b_b).norm()))
    blue_slots = set()
    blue_z = []
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        blue_slots.add(int(scene.peg_slot[0, 2]))  # peg_colors index 2 = blue
        blue_z.append(float(scene.pegs["blue"].data.root_pos_w[0, 2]))
    z_span = max(blue_z) - min(blue_z)
    print(f"[smoke] randomization deltas: rack_xy={d_r * 1000:.1f}mm "
          f"mug_xy={d_m * 1000:.1f}mm beaker_xy={d_b * 1000:.1f}mm "
          f"blue_slots={sorted(blue_slots)} blue_z_span={z_span * 1000:.1f}mm",
          flush=True)
    check("randomization-is-real: rack/mug/beaker xy readback move across seeds; "
          "the blue peg's SLOT permutes (>= 2 values in 10 resets) and its height "
          "spans > 5 mm",
          d_r > 0.003 and d_m > 0.003 and d_b > 0.003
          and len(blue_slots) >= 2 and z_span > 0.005)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 4. negative: the SEED'S OWN STRATEGY =======================================
    # "Carry the object to the right spot and SET IT DOWN" — the seed's whole plan.
    # CONSTRUCT its end state here: the mug placed upright on the floor directly
    # under the blue peg (the most on-target floor placement possible). Suspension
    # is the goal, not placement: no success, score ~0.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _root, _dirw, tip = peg_axes("blue")
    p = tip.clone()
    p[:, 2] = _ENV.iscene.env_origins[:, 2] + 0.002
    _write_body(scene.mug, p, None)
    _step(150)
    _report("seed-strategy")
    check("negative (SEED strategy): mug set down on the floor at the 'right "
          "spot' directly under the blue peg — placement is not suspension: no "
          "success, score ~0",
          not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.05
          and not bool(scene.threaded()[0]))
    _REC["on"] = False

    # ================= 5. negative: WRONG OBJECT ==================================================
    # The amber beaker has no handle — hook it on the blue peg by its mouth (the
    # only way it can hang at all); the mug never moves. Nothing about the mug's
    # goal holds: no success, score 0.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    mouth_hook(scene.beaker, "blue", 0.012)
    _step(180)
    _report("wrong-object")
    bz = float((scene.beaker.data.root_pos_w - scene.env_origins)[0, 2])
    print(f"[smoke] beaker after mouth-hook: z={bz:.3f}", flush=True)
    check("negative (wrong object): beaker hooked on the blue peg by its mouth, "
          "mug untouched — no success, score 0",
          not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.01)
    _REC["on"] = False

    # ================= 6. negative: WRONG PEG (a real hang, wrong color) ==========================
    # CONSTRUCT a genuine hang on the RED peg: aperture on the red axis mid-
    # segment, settle — gravity seats the top bar on the peg; threaded-on-red,
    # airborne, settled. Everything but the color is right -> NO success, and the
    # score caps at the lift latch (0.15): threading credit is target-only.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    place_aperture("red", 0.040)
    _step(240)
    for _ in range(8):  # wait out the pendulum swing
        if bool(scene.mug_settled()[0]):
            break
        _step(120)
    _report("wrong-peg")
    hung_on_red = (float(scene.dist_to_peg("red")[0]) < c.thread_gate
                   and bool(scene.airborne()[0]) and bool(scene.mug_settled()[0]))
    check("negative (WRONG PEG): a real, settled hang on the RED peg — threaded, "
          "airborne — but not the target: no success, score <= 0.16",
          hung_on_red and not bool(scene.success()[0])
          and not bool(scene.threaded()[0]) and float(scene.score()[0]) <= 0.16)

    # ================= 7. LOOP INTERLOCK: the closed loop carries load ============================
    # Physics claim behind the whole task: a peg through the handle loop is a real
    # suspension. Shove the red-hung mug DOWNWARD at 3x its weight for 1.5 s — the
    # closed loop cannot be torn off; the peg keeps carrying it.
    f = torch.tensor([0.0, 0.0, -3.0 * c.mug_mass * 9.81], device=device)
    for _ in range(9):  # 9 x 20 = 180 steps = 1.5 s of shoving straight down
        _push(scene.mug, f, 20)
    _step(90)
    _report("loop-interlock")
    check("LOOP INTERLOCK: 3x-weight downward shove for 1.5 s cannot tear the "
          "closed loop off the peg — still hung (threaded-on-red, airborne)",
          float(scene.dist_to_peg("red")[0]) < c.thread_gate
          and bool(scene.airborne()[0]) and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 8. near-miss: MOUTH-HOOK on the blue peg ===================================
    # The plausible cheat: perch the mug on the BLUE peg by its CUP OPENING (peg
    # inside the cup, not the handle). It hangs — and its settled tilt can bring
    # the aperture CENTER as close as ~29 mm to the axis — but the aperture NORMAL
    # stays perpendicular to the axis (cup axis along the peg), so the alignment
    # clause of threaded() rejects it: no thread latch, no success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    mouth_hook(scene.mug, "blue", 0.020)
    _step(240)
    _report("mouth-hook")
    d_now = float(scene.dist_to_peg("blue")[0])
    a_now = float(scene.loop_alignment()[0])
    print(f"[smoke] mouth-hook: aperture-to-blue-axis {d_now * 1000:.1f}mm "
          f"(gate {c.thread_gate * 1000:.0f}mm), alignment |cos|={a_now:.2f} "
          f"(min {c.align_min:.2f})", flush=True)
    check("near-miss (mouth-hook): mug perched on the BLUE peg by its cup "
          "opening hangs, but the aperture is edge-on to the axis — threaded() "
          "False, thread latch never set, no success",
          a_now < c.align_min and not bool(scene.threaded()[0])
          and not bool(scene._thread[0]) and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 9. near-miss: aligned OFF-TIP + release -> gravity says no =================
    # The solve's own staging pose: aperture on the axis, 10 mm PAST the tip —
    # perfectly aligned, not threaded. The 25 mm tip margin must hold (a teleport
    # to staging cannot latch), and with nothing through the loop the released
    # mug must simply FALL: suspension is gravity + contact, not pose.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    place_aperture("blue", c.peg_len + 0.010)
    _refresh()
    d_stage = float(scene.dist_to_peg("blue")[0])
    staged_out = (d_stage > c.thread_gate and not bool(scene._thread[0])
                  and not bool(scene.threaded()[0]))
    _step(240)  # release: nothing holds it — it falls
    _report("off-tip-release")
    mz = float((scene.mug.data.root_pos_w - scene.env_origins)[0, 2])
    print(f"[smoke] off-tip: staged dist={d_stage * 1000:.1f}mm "
          f"(gate {c.thread_gate * 1000:.0f}mm), after release mug_z={mz:.3f}",
          flush=True)
    check("near-miss (off-tip): aperture aligned 10 mm past the tip is OUTSIDE "
          "the gate (tip margin holds, no latch from staging), and the released "
          "mug falls to the floor — no success",
          staged_out and mz < c.hang_clear_z and not bool(scene.airborne()[0])
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.mug_hook")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    check("video: frames captured and saved to frames.npz", len(_REC["frames"]) > 10)

    n_pass = sum(ok for _nm, ok in checks)
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
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - die loudly, don't wait for the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL (exception: {exc!r})", flush=True)
        os._exit(2)
