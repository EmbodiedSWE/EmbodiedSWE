"""Smoke battery for ChuteSwitchScene (sim_gen task
`libero_kitchen_scene1_open_top_drawer_i14`) — REJECTION-ONLY: every check either
verifies basic health/randomization or CONSTRUCTS a settled wrong outcome and
asserts the rubric refuses it. success() must never fire anywhere in the battery.

 1. settle/no-NaN     — gate parked at a stop, bins mated at the outlets, score ~0.
 2. randomization A   — router yaw + xy vary across resets (readback).
 3. randomization B   — bins_swapped, gate initial side and ball slot permutation
                        all vary across resets (readback).
 4. null policy       — 240 idle steps -> score ~0, no success.
 5. SEED strategy     — the seed's whole skill (slide the prismatic part) executed
                        for real, twice: force-flip the gate to the far stop and
                        back. Parks verified; score stays ~0, no success.
 6. roof holds        — a green ball dropped from above the GREEN bin lands on the
                        roof, never inside: no latch, no credit.
 7. mouth gap holds   — a ball laid at the chute/bin gap cannot enter: no credit.
 8. blocked channel   — a ball fed into the channel the gate blocks jams UPSTREAM
                        of the switch: never routed, never binned, no credit.
 9. wrong routing     — greens in the RED bin and red in the GREEN bin (constructed,
                        settled): no latch fires, no success.
10. blue decoy        — correct greens+red PLUS the blue ball inside a bin: success
                        refused (score capped below 1).
11. bin displaced     — full correct arrangement but the green bin dragged off its
                        outlet: bins_seated refuses, no success.
12. missing delivery  — one green still on the ground: no success, partial score only.
13. latched credit    — red delivered (constructed) then removed: the 0.20 latch
                        survives, success does not.
14. rejection audit   — success() observed False at every step of the battery.
15. final no-NaN      — and frames.npz written to the CWD.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_open_top_drawer_i14.smoke
             --headless
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

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

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
    ig = scene.balls_in_bin("green")[0]
    ir = scene.balls_in_bin("red")[0]
    print(f"[smoke] {tag:18s} | gate_side={int(scene.gate_side()[0]):+d} "
          f"gate_y={float(scene.gate_y()[0]):+.3f} "
          f"green_bin={[i for i in range(4) if bool(ig[i])]} "
          f"red_bin={[i for i in range(4) if bool(ir[i])]} "
          f"seated={bool(scene.bins_seated()[0])} settled={bool(scene.settled()[0])} "
          f"L=r{int(scene._route[0])}g{int(scene._g1[0])}{int(scene._g2[0])}"
          f"r{int(scene._r1[0])} score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.chute_switch")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.90, -0.75, 0.70)) + o),
                                tuple(np.array((0.10, 0.00, 0.10)) + o),
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

    def router_world(local_xyz) -> torch.Tensor:
        loc = torch.tensor(local_xyz, device=device).expand(n, 3)
        return scene.router.data.root_pos_w + quat_apply(
            scene.router.data.root_quat_w, loc)

    def side_of(bin_name: str) -> int:
        swapped = bool(scene.bins_swapped[0])
        green_left = not swapped
        return (1 if green_left else -1) if bin_name == "green" else \
            (-1 if green_left else 1)

    def park_gate(side: int) -> None:
        """CONSTRUCT: write the gate parked on `side` (+1 blocks LEFT)."""
        pos = router_world([c.gate_x, side * c.gate_park, c.gate_size[2] / 2])
        _write_body(scene.gate, pos, scene.router.data.root_quat_w)
        _step(30)

    def drop_at_inlet(ball: str, side: int) -> None:
        """Hover the ball over channel `side`'s open inlet and release."""
        pos = router_world([-0.24, side * c.ch_off, c.ball_r + 0.050])
        _write_body(scene.balls[ball], pos)

    def put_in_bin(ball: str, bin_name: str, dx: float = 0.0, dy: float = 0.0) -> None:
        """CONSTRUCT: write the ball inside the (roofed) bin interior."""
        b = scene.bins[bin_name]
        loc = torch.tensor([dx, dy, c.bfloor_t + c.ball_r + 0.004],
                           device=device).expand(n, 3)
        pos = b.data.root_pos_w + quat_apply(b.data.root_quat_w, loc)
        _write_body(scene.balls[ball], pos)

    def push_gate(target: int, steps: int = 360) -> bool:
        """REAL actuation: force along the bar axis until parked at `target`."""
        ey = torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3)
        zero = torch.zeros(n, 1, 3, device=device)
        yt = target * (c.gate_park + 0.006)   # aim just past the park, into the post
        ok_frames = 0
        for _ in range(steps):
            # PD force along the bar axis (fingertip guiding the bar): ramming the
            # end post at full force wedges the tip into it, so brake on approach.
            axis = quat_apply(scene.router.data.root_quat_w, ey)
            v = (scene.gate.data.root_lin_vel_w * axis).sum(-1)
            f_mag = (150.0 * (yt - scene.gate_y()) - 12.0 * v).clamp(-5.0, 5.0)
            fw = axis * f_mag.unsqueeze(-1)
            # set_external_force_and_torque takes BODY-frame forces: convert each step.
            fb = quat_apply_inverse(scene.gate.data.root_quat_w, fw)
            scene.gate.set_external_force_and_torque(fb.reshape(n, 1, 3), zero)
            _step(1)
            parked = int(scene.gate_side()[0]) == target and abs(float(v[0])) < 0.02
            ok_frames = ok_frames + 1 if parked else 0
            if ok_frames >= 5:
                break
        scene.gate.set_external_force_and_torque(zero, zero)
        _step(60)
        return int(scene.gate_side()[0]) == target

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(120)
    _report("reset")
    check("settle/no-NaN: gate parked at a stop, bins mated, everything finite, "
          "score ~0, no success",
          bool(scene._finite()[0]) and int(scene.gate_side()[0]) != 0
          and bool(scene.bins_seated()[0]) and float(scene.score()[0]) <= 0.02
          and not bool(scene.success()[0]))

    # ================= 2+3. randomization readback ================================================
    yaws, xys, swaps, inits, perms = [], [], [], [], []
    for k in range(8):
        torch.manual_seed(20 + k)
        env.reset()
        _step(6)
        _refresh()
        yaws.append(yaw_of(scene.router.data.root_quat_w[0]))
        xys.append((scene.router.data.root_pos_w[0]
                    - scene.env_origins[0])[:2].tolist())
        swaps.append(bool(scene.bins_swapped[0]))
        inits.append(bool(scene.gate_init_left[0]))
        perms.append(tuple(scene.slot_perm[0].tolist()))
    yspan = max(yaws) - min(yaws)
    xystd = float(np.std(np.asarray(xys), axis=0).mean())
    print(f"[smoke] readback: yaws={[f'{y:+.0f}' for y in yaws]} span={yspan:.0f} "
          f"xystd={xystd:.3f} swaps={swaps} gate_left={inits} "
          f"perms={len(set(perms))} distinct", flush=True)
    check("randomization A: router yaw spans > 90 deg and xy jitters across resets",
          yspan > 90.0 and xystd > 0.008)
    check("randomization B: bin arrangement, gate initial side and ball slot "
          "permutation all vary",
          len(set(swaps)) == 2 and len(set(inits)) == 2 and len(set(perms)) >= 3)

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> score ~0, no success",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 5. SEED strategy (the prismatic slide, for real) ===========================
    torch.manual_seed(41)
    env.reset()
    _step(120)
    s5 = int(scene.gate_side()[0])
    ok_a = push_gate(-s5)
    ok_b = push_gate(s5)
    _report("seed-skill")
    check("SEED strategy: the seed's whole skill (slide the prismatic bar) executed "
          "for real, twice — parks fine, scores ~0, no success",
          ok_a and ok_b and float(scene.score()[0]) <= 0.02
          and not bool(scene.success()[0]))

    # ================= 6. roof holds ==============================================================
    gb = scene.bins["green"]
    hover = gb.data.root_pos_w + torch.tensor(
        [0.0, 0.0, c.bwall_h + c.roof_t + c.ball_r + 0.05],
        device=device).expand(n, 3)
    _write_body(scene.balls["green_0"], hover)
    _step(150)
    _report("roof-drop")
    check("roof holds: a green ball dropped from above the GREEN bin lands on the "
          "roof, never inside — no latch, no credit",
          not bool(scene.balls_in_bin("green")[0, 0]) and not bool(scene._g1[0]))

    # ================= 7. mouth gap holds =========================================================
    loc = torch.tensor([-(c.bin_d / 2 + c.ball_r + 0.002), 0.0, c.ball_r],
                       device=device).expand(n, 3)
    pos = gb.data.root_pos_w + quat_apply(gb.data.root_quat_w, loc)
    _write_body(scene.balls["green_1"], pos)
    _step(150)
    _report("mouth-gap")
    check("mouth gap holds: a ball laid at the chute/bin seam cannot enter the bin "
          "(sill + sub-ball gap)",
          not bool(scene.balls_in_bin("green")[0, 1]) and not bool(scene._g1[0]))

    # ================= 8. blocked channel =========================================================
    torch.manual_seed(51)
    env.reset()
    _step(90)
    park_gate(+1)                        # CONSTRUCT: gate blocks the LEFT channel
    drop_at_inlet("red_0", +1)           # feed exactly that channel
    _step(300)
    _report("blocked")
    lx = float(scene._router_local(scene.balls["red_0"].data.root_pos_w)[0, 0])
    check("blocked channel: a ball fed into the gated channel jams UPSTREAM of the "
          "switch — never routed, never binned",
          lx < c.past_x and not bool(scene._route[0])
          and not bool(scene.balls_in_bin("green")[0, 2])
          and not bool(scene.balls_in_bin("red")[0, 2]))

    # ================= 9. wrong routing ===========================================================
    torch.manual_seed(61)
    env.reset()
    _step(90)
    put_in_bin("green_0", "red", dx=-0.025)
    put_in_bin("green_1", "red", dx=+0.025)
    put_in_bin("red_0", "green")
    _step(120)
    _report("wrong-route")
    check("wrong routing: greens in the RED bin and red in the GREEN bin latch "
          "nothing — no success, score ~0",
          not bool(scene._g1[0]) and not bool(scene._r1[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.02)

    # ================= 10. blue decoy =============================================================
    torch.manual_seed(71)
    env.reset()
    _step(90)
    put_in_bin("blue_0", "red", dx=+0.025)   # decoy FIRST: success may never flicker on
    put_in_bin("green_0", "green", dx=-0.025)
    put_in_bin("green_1", "green", dx=+0.025)
    put_in_bin("red_0", "red", dx=-0.025)
    _step(120)
    _report("blue-decoy")
    s10 = float(scene.score()[0])
    check("blue decoy: correct greens+red PLUS the blue ball inside a bin — success "
          "refused, score capped below 1",
          bool(scene._g1[0]) and bool(scene._g2[0]) and bool(scene._r1[0])
          and not bool(scene.success()[0]) and s10 <= 0.70 + 1e-6)

    # ================= 11. bin displaced ==========================================================
    torch.manual_seed(81)
    env.reset()
    _step(90)
    gb = scene.bins["green"]
    _write_body(gb, gb.data.root_pos_w
                + torch.tensor([0.05, 0.04, 0.0], device=device).expand(n, 3),
                gb.data.root_quat_w)         # drag it off its outlet FIRST
    put_in_bin("green_0", "green", dx=-0.025)
    put_in_bin("green_1", "green", dx=+0.025)
    put_in_bin("red_0", "red")
    _step(120)
    _report("bin-off")
    check("bin displaced: full correct arrangement but the green bin dragged off "
          "its outlet — bins_seated refuses success",
          not bool(scene.bins_seated()[0]) and not bool(scene.success()[0]))

    # ================= 12. missing delivery =======================================================
    torch.manual_seed(91)
    env.reset()
    _step(90)
    put_in_bin("green_0", "green")
    put_in_bin("red_0", "red")
    _step(120)
    _report("missing-one")
    s12 = float(scene.score()[0])
    check("missing delivery: one green still on the ground — no success, partial "
          "score only",
          not bool(scene.success()[0]) and 0.35 <= s12 <= 0.45)

    # ================= 13. latched credit =========================================================
    torch.manual_seed(101)
    env.reset()
    _step(90)
    put_in_bin("red_0", "red")
    _step(60)
    latched = bool(scene._r1[0])
    ground = (scene.env_origins[0:1]
              + torch.tensor([0.9, 0.9, c.ball_r], device=device)).expand(n, 3)
    _write_body(scene.balls["red_0"], ground)
    _step(90)
    _report("latch")
    s13 = float(scene.score()[0])
    check("latched credit: red delivered (constructed) then removed — the 0.20 "
          "latch survives, success does not",
          latched and bool(scene._r1[0]) and 0.15 <= s13 <= 0.25
          and not bool(scene.success()[0]))

    # ================= 14 + 15. audit, video, verdict =============================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.chute_switch")
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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
