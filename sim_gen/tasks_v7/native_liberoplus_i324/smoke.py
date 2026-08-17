"""Smoke / rubric-REJECTION battery for BlockMagazineScene (sim_gen task
`native_liberoplus_i324`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — the plunger driven through its stroke by a
regulated force, blocks dispensed one at a time through the port and routed by
color — is the acceptance evidence that the rubric ACCEPTS a correct outcome).
Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial)
outcome as a settled state and asserts the rubric REJECTS it; no probe in this
battery ever reaches success(), and a final audit check asserts exactly that.
Probes that need the port-transit latch earn it the honest way the geometry
defines it (the block posed inside the port passage — the only way out — then
falling free), and each multi-move probe is ordered so NO prefix of it satisfies
the goal.

  1-2.  settle/no-NaN     — reset layout settles finite: four blocks stacked inside
                            the shaft (red at its sampled depth), plunger resting
                            retracted, everything still; score ~0, no success;
  3-4.  randomization     — READBACK over 8 seeded resets: the red block's stack
                            depth G takes >= 2 distinct values in {1,2,3}; bin and
                            pad xy jitter really vary;
  5.   null policy        — 240 idle steps -> score ~0, no success;
  6.   seed strategy      — the end state the seed's plan (grasp the target, hover,
                            release over the goal) produces here: the red block
                            CANNOT be grasped, so the closest analogue — red
                            dropped from above the magazine — lands on the sealed
                            ROOF: rejected, no transit, no credit;
  7.   bypass gating      — red teleported STRAIGHT onto the pad (never through the
                            port): on-pad + still + grays accounted, but the
                            transit latch is false -> pad credit gated to 0, score
                            ~0, no success;
  8.   pad near-miss      — transit latched honestly, red settled just OFF the pad
                            centre (beyond xy tol) -> rejected;
  9.   gray beside bin    — red fully done (transit + on pad) but one dispensed
                            gray lying on the open floor beside the bin -> rejected
                            (every dispensed gray must be IN the bin);
  10.  gray on the roof   — red fully done but a gray perched on the magazine roof
                            (outside, not binned, not inside) -> rejected;
  11.  gray stuck in port — red fully done but a gray left straddling the port
                            mouth (half-dispensed) -> rejected;
  12.  wrong destination  — transit latched, red dropped INTO the discard bin
                            instead of onto the pad -> rejected;
  13.  latched credit     — a gray binned (eject+bin credit earned), then yanked
                            back out to the floor -> the latched score holds
                            (credit does not evaporate), still no success;
  14.  rejection audit    — success() was never True at ANY judged point;
  15.  final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.native_liberoplus_i324.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import os
import threading

import numpy as np
import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.block_magazine")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    mx, my = c.mag_pos

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.35, -0.85, 0.70)) + o),
                                tuple(np.array((0.42, 0.02, 0.10)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action)
            if (annot is not None and step_i % args.record_every == 0
                    and len(frames) < args.max_frames):
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    ever_success = [False]

    def judge() -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def loc(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    bodies = {"red": scene.red, "gray0": scene.grays[0], "gray1": scene.grays[1],
              "gray2": scene.grays[2]}

    def report(tag: str) -> None:
        parts = []
        for nm, b in bodies.items():
            p = loc(b)
            parts.append(f"{nm}=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):.3f})")
        s, ok = judge()
        print(f"[smoke] {tag:16s} | ins={float(scene.rod_insertion()[0]):.4f} "
              + " ".join(parts)
              + f" lat=[e{int(scene._eject[0].sum())} b{int(scene._binned[0].sum())} "
              f"t{int(scene._transit[0])} p{int(scene._pad[0])}] G={int(scene._G[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, x: float, y: float, z: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def states_finite() -> bool:
        okf = torch.isfinite(scene.rod.data.root_state_w).all()
        for b in bodies.values():
            okf = okf & torch.isfinite(b.data.root_state_w).all()
        return bool(okf)

    def all_blocks_still() -> bool:
        return all(bool(scene._still(b)[0]) for b in bodies.values())

    def red_transit_honest() -> None:
        """Earn the transit latch the way the geometry defines it: pose the red
        block inside the PORT PASSAGE (the only opening; free space, 2 mm clear of
        the stack) and let it fall out onto the floor."""
        place(scene.red, mx, my + 0.052, 0.030)
        step(60)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    report("reset")
    step(90)
    report("show")
    zs = sorted(float(loc(b)[2]) for b in bodies.values())
    in_mag = all(bool(scene.in_magazine(loc(b).unsqueeze(0))[0]) for b in bodies.values())
    check("settle: states finite, four blocks stacked inside the shaft (z ranks at the "
          "slot heights), plunger resting retracted, everything still",
          states_finite() and in_mag and all_blocks_still()
          and float(scene.rod_insertion()[0]) < 0.01
          and all(abs(zs[i] - c.slot_z[i]) < 0.012 for i in range(4)))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(2)
        b0, p0 = loc(scene.bin), loc(scene.pad)
        reads.append((int(scene._G[0]), float(b0[0]), float(b0[1]),
                      float(p0[0]), float(p0[1])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (G, bin_x, bin_y, pad_x, pad_y):\n{arr}",
          flush=True)
    check("randomization: the red block's stack depth G takes >= 2 distinct values "
          "in {1,2,3} across seeded resets (readback)",
          len(set(arr[:, 0].tolist())) >= 2
          and all(1 <= g <= 3 for g in arr[:, 0].tolist()))
    check("randomization: bin and pad xy jitter really vary across seeded resets "
          "(readback ranges > 1 cm)",
          float(arr[:, 1].max() - arr[:, 1].min()) > 0.01
          and float(arr[:, 2].max() - arr[:, 2].min()) > 0.01
          and float(arr[:, 3].max() - arr[:, 3].min()) > 0.01
          and float(arr[:, 4].max() - arr[:, 4].min()) > 0.01)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: grasp-hover-release ======================
    # The seed's plan — grasp the named target and release it over the goal — is
    # physically void here: the red block sits inside a sealed shaft. The closest
    # executable analogue, dropping the red block from above the magazine, lands it
    # on the ROOF: outside, no transit, nothing earned.
    env.reset(seed=41)
    step(10)
    roof_top = c.in_h + c.roof_t
    place(scene.red, mx, my, roof_top + c.edge / 2 + 0.06)
    step(150)
    report("seed-strategy")
    s, ok = judge()
    pz = float(loc(scene.red)[2])
    check("seed strategy: red dropped from above lands ON the sealed roof — no transit, "
          "no credit (score <= 0.02), no success",
          pz > roof_top - 0.02 and not bool(scene._transit[0]) and s <= 0.02 and not ok)

    # =========================== 7. bypass gating: straight to the pad ======================
    env.reset(seed=51)
    step(10)
    p0 = loc(scene.pad)
    place(scene.red, float(p0[0]), float(p0[1]), 0.10)
    step(150)
    report("bypass-pad")
    s, ok = judge()
    check("bypass gating: red teleported STRAIGHT onto the pad (never through the port) "
          "— on-pad and settled, grays untouched, but the transit latch is false: pad "
          "credit gated to 0, score <= 0.02, no success",
          bool(scene.red_on_pad()[0]) and not bool(scene._transit[0])
          and not bool(scene._pad[0]) and s <= 0.02 and not ok)

    # =========================== 8. pad near-miss (transit honest) ==========================
    env.reset(seed=61)
    step(10)
    red_transit_honest()
    report("post-transit")
    p0 = loc(scene.pad)
    place(scene.red, float(p0[0]) + 0.055, float(p0[1]) + 0.055, 0.08)
    step(150)
    report("pad-near-miss")
    s, ok = judge()
    dx = abs(float(loc(scene.red)[0] - loc(scene.pad)[0]))
    dy = abs(float(loc(scene.red)[1] - loc(scene.pad)[1]))
    check("pad near-miss: transit latched honestly, red settled just OFF the pad centre "
          f"(|dxy|=({dx:.3f},{dy:.3f}) > tol {c.pad_xy_tol}) — rejected, score <= 0.20",
          bool(scene._transit[0]) and (dx > c.pad_xy_tol or dy > c.pad_xy_tol)
          and not bool(scene.red_on_pad()[0]) and 0.10 <= s <= 0.20 and not ok)

    # ============ 9-11. red fully done but ONE gray mis-routed -> all rejected ==============
    # Non-vacuous: in each construct the ONLY failing condition is the gray's state.
    # Ordered so no prefix satisfies the goal: transit first, red parked far away,
    # then the offending gray placed, and only THEN red set on the pad.
    def red_done_with_gray_at(tag: str, gx: float, gy: float, gz: float) -> tuple[float, bool]:
        env.reset(seed={"beside-bin": 71, "on-roof": 81, "in-port": 91}[tag])
        step(10)
        red_transit_honest()                       # transit latched, red on the floor
        place(scene.red, 0.10, -0.32, c.edge / 2 + 0.002)   # park far from everything
        step(30)
        place(scene.grays[0], gx, gy, gz)          # the mis-routed gray (unaccounted)
        step(90)
        p0 = loc(scene.pad)
        place(scene.red, float(p0[0]), float(p0[1]), 0.08)  # NOW finish the red
        step(150)
        report(tag)
        return judge()

    b0 = loc(scene.bin)
    s, ok = red_done_with_gray_at("beside-bin", float(b0[0]) - 0.15, float(b0[1]),
                                  c.edge / 2 + 0.002)
    check("gray beside bin: red fully done (transit + settled on pad) but a dispensed "
          "gray lying on the open floor beside the bin — grays unaccounted, no success",
          bool(scene._transit[0]) and bool(scene.red_on_pad()[0])
          and not bool(scene.grays_accounted()[0]) and not ok)

    s, ok = red_done_with_gray_at("on-roof", mx, my, c.in_h + c.roof_t + c.edge / 2 + 0.02)
    gz = float(loc(scene.grays[0])[2])
    check("gray on the roof: red fully done but a gray perched on the magazine roof — "
          "outside, not binned, not inside: no success",
          bool(scene.red_on_pad()[0]) and gz > c.in_h - 0.02
          and not bool(scene.grays_accounted()[0]) and not ok)

    s, ok = red_done_with_gray_at("in-port", mx, my + 0.054, 0.030)
    gy_port = float(loc(scene.grays[0])[1]) - my
    check("gray stuck in port: red fully done but a gray left straddling the port mouth "
          f"(half-dispensed, dy={gy_port:.3f}) — neither in the magazine nor in the "
          "bin: no success",
          bool(scene.red_on_pad()[0]) and not bool(scene.grays_accounted()[0]) and not ok)

    # =========================== 12. wrong destination: red in the bin ======================
    env.reset(seed=101)
    step(10)
    red_transit_honest()
    b0 = loc(scene.bin)
    place(scene.red, float(b0[0]), float(b0[1]), 0.15)
    step(150)
    report("red-in-bin")
    s, ok = judge()
    check("wrong destination: transit latched, red dropped INTO the discard bin instead "
          "of onto the pad — rejected, score <= 0.20",
          bool(scene._transit[0]) and bool(scene.in_bin(loc(scene.red).unsqueeze(0))[0])
          and not bool(scene.red_on_pad()[0]) and s <= 0.20 and not ok)

    # =========================== 13. latched credit survives regression =====================
    env.reset(seed=111)
    step(10)
    b0 = loc(scene.bin)
    place(scene.grays[0], float(b0[0]), float(b0[1]), 0.15)   # gray into the bin
    step(150)
    s_a, ok_a = judge()
    report("gray-binned")
    place(scene.grays[0], 0.05, -0.35, c.edge / 2 + 0.002)    # yank it back out
    step(90)
    s_b, ok_b = judge()
    report("regressed")
    check("latched credit: a gray binned (eject+bin credit earned) then yanked back out "
          f"to the floor — the latched score holds ({s_a:.3f} -> {s_b:.3f}), > 0.05, "
          "still no success",
          s_a > 0.05 and abs(s_a - s_b) < 1e-3 and not ok_a and not ok_b)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", states_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.block_magazine")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(okc for _nm, okc in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, okc in checks:
            if not okc:
                print(f"[smoke]   FAILED: {nm}", flush=True)
    code = 0 if all_ok else 1
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
    except Exception:  # noqa: BLE001 — die loudly, never hang in Kit teardown
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)
