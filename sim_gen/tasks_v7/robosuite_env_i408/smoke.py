"""Smoke / rubric-REJECTION battery for QCChuteScene — NullRobot probes.

solve.py is the acceptance proof (probe pads in order, physics classifies each disc,
the polished one delivers itself; score staircase is monotone). This battery proves the
rubric REJECTS wrong outcomes, and that the physical mechanisms the task rests on are
load-bearing, not fiat:

  - CLASSIFICATION: friction vs the 12-deg pitch, not a script, separates the discs —
    the SAME release point holds a rough disc (check 4) and launches the polished one
    (check 8);
  - THE ROOF:       the tunnel roof physically caps a disc inside it (check 5) and the
    cup roof carries a pressed disc without letting it through (check 7);
  - IDENTITY:       the cup accepts ONLY the polished disc — a rough disc kinetically
    rammed through the whole gauntlet reaches the cup and still scores nothing, and it
    FOULS the cup against a subsequent legitimate delivery (check 6).

Force probes follow the non-vacuity rule: every "it was refused" assertion is paired
with a readback that the probe actually acted (the disc measurably moved / pressed /
traversed). Constructed states (teleports) build settled WRONG outcomes for the rubric
to refuse; no constructed state reaches success().

Checks:
   1. settle          — seeded reset settles finite; three discs on three distinct
                        pads; authored materials (0.05 vs 0.70, get_material_properties
                        readback) and masses took; score ~0, no success;
   2. randomization   — 8 seeded resets: the polished disc's pad varies (the hidden
                        bit is real), rig yaw / xy and pad jitter all spread;
   3. null-policy     — 2.5 s of nothing: score ~0, no success, discs stay put;
   4. DECOY STALL     — a rough disc released at the solve's exact probe point: it
                        demonstrably rests ON the deck (height readback) and holds
                        still (< 5 mm over 2 s) on the 12-deg slope; no credit (the
                        latches key on the polished disc); paired with checks 6/8
                        where the same release launches the polished disc;
   5. TUNNEL ROOF     — a rough disc parked mid-tunnel, pulled straight up at 4x its
                        weight: it presses into the roof (height demonstrably rises)
                        but stays capped far below escaping; released, it drops back;
                        no credit;
   6. FOULED CUP      — a rough disc servo-rammed down the ramp fast enough to shoot
                        the tunnel: it lands INSIDE the cup and scores NOTHING; a
                        subsequent legitimate polished delivery arms the latches
                        (score 0.60) but success stays REFUSED — the cup is fouled;
   7. near-miss gates — the polished disc constructed ON the cup roof (plus a 3x-
                        weight press the roof carries), BESIDE the cup, and BEYOND
                        it on the ground: every near-miss refused;
   8. success+revoke  — the real strategy inline: polished disc released at the probe
                        point, hands-off — success() with score exactly 1.0, still
                        true 1 s later; plucking it back out revokes success LIVE and
                        the score falls to the latched 0.60 — 1.0 iff success;
   9. wrong object    — a rough disc constructed alone INSIDE the cup (polished still
                        on its pad): the cup clause keys on identity — no success,
                        no credit;
  10. frames.npz      — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -u -m simgen_tasks.robosuite_env_i408.smoke --headless
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
    from . import scene as task_scene  # noqa: F401 — import registers the scene + env
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene  # noqa: F401

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


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    locs = scene.puck_locs()
    s, _, h = scene.chute_coords(locs)
    cup = [bool(scene.in_cup(locs[:, j])[0]) for j in range(3)]
    print(f"[smoke] {tag:16s} | "
          + " ".join(f"p{j}=({float(locs[0, j, 0]):+.3f},{float(locs[0, j, 1]):+.3f},"
                     f"{float(locs[0, j, 2]):+.3f})" for j in range(3))
          + f" s0={float(s[0, 0]):+.3f} h0={float(h[0, 0]):+.3f}"
          f" track={bool(scene._track[0])} tunnel={bool(scene._tunnel[0])}"
          f" in_cup={cup} success={bool(scene.success()[0])}"
          f" score={float(scene.score()[0]):.3f} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    from isaaclab.utils.math import quat_apply, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.qc_chute")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    zero = torch.zeros(n, 1, 3, device=device)
    th = math.radians(c.pitch_deg)
    q_pitch = torch.tensor([math.cos(th / 2), 0.0, math.sin(th / 2), 0.0],
                           device=device).expand(n, 4)

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.15, -0.90, 0.80)) + o),
                                tuple(np.array((0.40, -0.05, 0.08)) + o),
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

    def ramp_point(s: float, h: float) -> tuple:
        return (c.mid_x + s * math.cos(th) + h * math.sin(th), 0.0,
                c.mid_z - s * math.sin(th) + h * math.cos(th))

    def write_local(body, loc_xyz, *, pitched: bool = False) -> None:
        """Teleport a body to a rig-local pose (rig yaw [+ deck pitch]), zero velocity."""
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float32).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.rig.data.root_pos_w + quat_apply(scene.rig.data.root_quat_w, loc)
        q = scene.rig.data.root_quat_w
        st[:, 3:7] = quat_mul(q, q_pitch) if pitched else q
        body.write_root_state_to_sim(st, _all_ids())
        _refresh()

    def s_h_of(body) -> tuple:
        s, _, h = scene.chute_coords(scene.rig_local(body.data.root_pos_w))
        return float(s[0]), float(h[0])

    def push(body, dir_w: torch.Tensor, mag: float, steps: int) -> None:
        """Constant world-frame force at the CoM for `steps` substeps, then clear."""
        f = (mag * dir_w).unsqueeze(1)
        for _ in range(steps):
            body.set_external_force_and_torque(f, zero, env_ids=_all_ids(), is_global=True)
            _step(1)
        body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())

    def downslope_w() -> torch.Tensor:
        d = torch.tensor((math.cos(th), 0.0, -math.sin(th)),
                         device=device, dtype=torch.float32).expand(n, 3)
        return quat_apply(scene.rig.data.root_quat_w, d)

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    probe_pt = ramp_point(-0.10, c.puck_h / 2 + 0.004)
    pol = scene.pucks["puck_0"]

    # ================= 1. settle + authored materials / masses ====================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(120)
    _report("settle")
    _REC["on"] = False
    mats = [scene.pucks[nm].root_physx_view.get_material_properties().flatten().tolist()
            for nm in scene.PUCK_NAMES]
    m0 = float(pol.root_physx_view.get_masses().flatten()[0])
    slot = scene._slot[0].tolist()
    pads = torch.tensor(c.pad_xy, device=device)
    on_pads = all(
        float((scene.puck_locs()[0, j, :2] - pads[slot[j]]).norm()) < 0.03
        for j in range(3))
    print(f"[smoke] materials={[[round(v, 3) for v in m] for m in mats]} "
          f"mass={m0:.3f} kg (want {c.puck_mass}) slots={slot} on_pads={on_pads}",
          flush=True)
    check("settle: seeded reset settles finite, three discs on three distinct pads, "
          "authored materials (polished 0.05 vs rough 0.70) and mass took, score ~0, "
          "no success",
          bool(torch.isfinite(scene.puck_locs()).all())
          and on_pads and len(set(slot)) == 3
          and abs(mats[0][0] - c.mu_polished_s) < 0.01
          and abs(mats[1][0] - c.mu_rough_s) < 0.01
          and abs(mats[2][0] - c.mu_rough_s) < 0.01
          and abs(m0 - c.puck_mass) < 0.02
          and float(scene.score()[0]) <= 0.03 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    pol_pads: set = set()
    yaws: list[float] = []
    xys: list[torch.Tensor] = []
    jit: list[float] = []
    perm_ok = True
    for s in range(8):
        torch.manual_seed(300 + s)
        env.reset()
        _step(5)
        _refresh()
        sl = scene._slot[0].tolist()
        perm_ok = perm_ok and len(set(sl)) == 3
        pol_pads.add(int(sl[0]))
        yaws.append(yaw_of(scene.rig.data.root_quat_w[0]))
        xys.append(scene.rig.data.root_pos_w[0, :2].clone())
        jit.append(float((scene.puck_locs()[0, 0, :2] - pads[sl[0]]).norm()))
    yaw_spread = max(yaws) - min(yaws)
    xy_spread = float(max((a - b).norm() for a in xys for b in xys))
    jit_spread = max(jit) - min(jit)
    print(f"[smoke] randomization: polished pads {sorted(pol_pads)}, "
          f"yaw spread {yaw_spread:.1f}deg, rig xy spread {xy_spread * 1000:.0f}mm, "
          f"pad-jitter spread {jit_spread * 1000:.1f}mm, perms valid={perm_ok}",
          flush=True)
    check("randomization-is-real: the polished disc's pad varies over 8 resets (the "
          "hidden bit is live), every draw is a valid 3-pad permutation, rig yaw / "
          "xy and pad jitter all spread",
          len(pol_pads) >= 2 and perm_ok and yaw_spread > 5.0
          and xy_spread > 0.01 and jit_spread > 0.005)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    p_start = scene.puck_locs()[0].clone()
    _step(300)
    _report("null-policy")
    drift = float((scene.puck_locs()[0] - p_start).norm(dim=-1).max())
    check("null-policy-fails: 2.5 s of nothing — score ~0, no success, discs stay put",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0])
          and drift < 0.005)

    # ================= 4. DECOY STALL: friction classifies, latches key on identity ===============
    # A ROUGH disc released at the solve's exact probe point: contact readback proves
    # it rests ON the deck; it must hold still for 2 s; and it earns NOTHING (the
    # track/tunnel latches key on the polished disc). Checks 6/8 launch the polished
    # disc from the same point — the pairing that makes this probe non-vacuous.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    rough = scene.pucks["puck_1"]
    write_local(rough, probe_pt, pitched=True)
    _step(30)
    s_r0, h_r0 = s_h_of(rough)
    _step(240)
    _report("decoy-stall")
    s_r1, h_r1 = s_h_of(rough)
    print(f"[smoke] decoy stall: s {s_r0 * 1000:+.1f} -> {s_r1 * 1000:+.1f}mm "
          f"(ds {abs(s_r1 - s_r0) * 1000:.1f}mm), h {h_r1 * 1000:.1f}mm "
          f"(deck rest ~{c.puck_h / 2 * 1000:.0f}mm)", flush=True)
    check("DECOY STALL: a rough disc at the probe point rests ON the deck (height "
          "readback) and holds still (< 5 mm over 2 s) on the 12-deg slope; no "
          "credit, no success",
          abs(s_r1 - s_r0) < 0.005 and 0.005 < h_r1 < 0.020
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 5. TUNNEL ROOF: captivity under the cover ==================================
    # Park the rough disc mid-tunnel (a state only a kinetic shove could reach) and
    # pull straight up at 4x its weight: the roof must cap it (height rises — the pull
    # provably acts — but stays far below the underside); released, it drops back.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    rough = scene.pucks["puck_1"]
    write_local(rough, ramp_point(0.15, c.puck_h / 2 + 0.004), pitched=True)
    _step(30)
    _, h0 = s_h_of(rough)
    up = torch.zeros(n, 3, device=device)
    up[:, 2] = 1.0
    h_peak = 0.0
    for _ in range(15):  # 15 x 10 = 150 substeps = 1.25 s of sustained pull
        push(rough, up, 4.0 * c.puck_mass * 9.81, 10)
        h_peak = max(h_peak, s_h_of(rough)[1])
    _step(90)
    _report("tunnel-roof")
    _, h_back = s_h_of(rough)
    print(f"[smoke] tunnel roof: h rest {h0 * 1000:.1f} -> peak {h_peak * 1000:.1f}mm "
          f"(roof underside {c.wall_h * 1000:.0f}mm, disc-centre cap "
          f"{(c.wall_h - c.puck_h / 2) * 1000:.0f}mm) -> released {h_back * 1000:.1f}mm",
          flush=True)
    check("TUNNEL ROOF: a 4x-weight straight-up pull on a disc mid-tunnel presses it "
          "into the roof (height demonstrably rises) but cannot lift it out; "
          "released, it drops back to the deck; no credit",
          h_peak > h0 + 0.003 and h_peak < c.wall_h - 0.004
          and h_back < h0 + 0.003
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 6. FOULED CUP: kinetic bypass scores nothing and spoils the batch ==========
    # Servo-ram the rough disc down the ramp fast enough to shoot the tunnel (needs
    # ~1.3 m/s at the roof mouth; we enter at ~2 m/s). It lands INSIDE the cup — and
    # scores NOTHING. Then the legitimate polished delivery arms the latches but
    # success stays refused: the cup contains a rough disc.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    rough = scene.pucks["puck_1"]
    write_local(rough, ramp_point(-0.20, c.puck_h / 2 + 0.004), pitched=True)
    _step(20)
    d_w = downslope_w()
    kp, v_des, fmax = 8.0, 2.2, 3.0
    for _ in range(240):
        s_now, _ = s_h_of(rough)
        if s_now > c.roof_s0 - 0.05:
            break
        v_along = (rough.data.root_lin_vel_w * d_w).sum(dim=-1)
        f = (d_w * (kp * (v_des - v_along)).clamp(-fmax, fmax).unsqueeze(-1))
        rough.set_external_force_and_torque(f.unsqueeze(1), zero,
                                            env_ids=_all_ids(), is_global=True)
        _step(1)
    rough.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    _step(300)  # coast: tunnel, flight, capture — all hands-off
    _report("kinetic-bypass")
    s_end, _ = s_h_of(rough)
    rough_loc = scene.rig_local(rough.data.root_pos_w)
    rough_in = bool(scene.in_cup(rough_loc)[0])
    s_bypass = float(scene.score()[0])
    print(f"[smoke] bypass: rough disc ended at s={s_end * 1000:+.0f}mm, "
          f"rig-local=({float(rough_loc[0, 0]):+.3f},{float(rough_loc[0, 1]):+.3f},"
          f"{float(rough_loc[0, 2]):+.3f}), in_cup={rough_in}, score={s_bypass:.3f}",
          flush=True)
    bypass_ok = s_end > c.roof_s0 and rough_in and s_bypass <= 0.02 \
        and not bool(scene.success()[0])
    # now the legitimate delivery — into a fouled cup
    write_local(pol, probe_pt, pitched=True)
    _step(420)
    _report("fouled-cup")
    s_foul = float(scene.score()[0])
    check("FOULED CUP: a rough disc rammed through the tunnel lands in the cup and "
          "scores NOTHING (identity latches); the subsequent legitimate polished "
          "delivery arms track+tunnel (0.60) but success stays REFUSED — the cup "
          "is fouled",
          bypass_ok and bool(scene._track[0]) and bool(scene._tunnel[0])
          and 0.59 <= s_foul <= 0.6000002 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 7. near-miss gates: on / beside / beyond the cup ===========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    # (a) ON the cup roof — plus a 3x-weight press the roof must carry
    roof_top = c.cup_top + c.cup_roof_t
    write_local(pol, (0.30, 0.0, roof_top + c.puck_h / 2 + 0.002))
    _step(60)
    z_on = float(scene.rig_local(pol.data.root_pos_w)[0, 2])
    down = torch.zeros(n, 3, device=device)
    down[:, 2] = -1.0
    push(pol, down, 3.0 * c.puck_mass * 9.81, 120)
    _step(30)
    z_pressed = float(scene.rig_local(pol.data.root_pos_w)[0, 2])
    on_refused = (not bool(scene.in_cup(scene.rig_local(pol.data.root_pos_w))[0])
                  and not bool(scene.success()[0]))
    _report("on-roof")
    # (b) BESIDE the cup on the ground (outside the y walls)
    write_local(pol, (0.30, 0.11, c.puck_h / 2 + 0.001))
    _step(60)
    loc_b = scene.rig_local(pol.data.root_pos_w)
    beside_refused = (not bool(scene.in_cup(loc_b)[0])) and not bool(scene.success()[0])
    y_beside = float(loc_b[0, 1])
    # (c) BEYOND the cup on the ground (past the pedestal)
    write_local(pol, (0.43, 0.0, c.puck_h / 2 + 0.001))
    _step(60)
    loc_c = scene.rig_local(pol.data.root_pos_w)
    beyond_refused = (not bool(scene.in_cup(loc_c)[0])) and not bool(scene.success()[0])
    x_beyond = float(loc_c[0, 0])
    _report("near-miss")
    print(f"[smoke] near-miss: on-roof z {z_on * 1000:.1f}mm (pressed 3x-weight: "
          f"{z_pressed * 1000:.1f}mm, roof top {roof_top * 1000:.0f}mm), beside "
          f"y={y_beside * 1000:+.0f}mm, beyond x={x_beyond * 1000:+.0f}mm", flush=True)
    check("near-miss gates: the polished disc ON the cup roof (which carries a 3x-"
          "weight press without letting it through), BESIDE the cup, and BEYOND it "
          "on the ground — every near-miss refused",
          on_refused and z_on > roof_top and z_pressed > roof_top - 0.002
          and beside_refused and abs(y_beside) > c.cup_y
          and beyond_refused and x_beyond > c.cup_x1)

    # ================= 8. success is live: the real strategy, then revocation =====================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    write_local(pol, probe_pt, pitched=True)
    got = False
    for _ in range(600):
        _step(1)
        if bool(scene.success()[0]):
            got = True
            break
    _step(90)
    _report("success")
    got = got and bool(scene.success()[0]) and float(scene.score()[0]) == 1.0
    _step(120)  # 1 s hands-off: still true
    held = bool(scene.success()[0]) and float(scene.score()[0]) == 1.0
    # revocation: pluck the disc back out onto open ground — constructed state
    write_local(pol, (0.0, -0.35, c.puck_h / 2 + 0.001))
    _step(60)
    _report("revoked")
    s_rev = float(scene.score()[0])
    check("success-is-live: the polished disc released at the probe point delivers "
          "itself — success() with score exactly 1.0 (held 1 s); plucking it back "
          "out revokes success LIVE and the score falls to the latched 0.60 — "
          "1.0 iff success",
          got and held and not bool(scene.success()[0])
          and 0.59 <= s_rev <= 0.6000002)
    _REC["on"] = False

    # ================= 9. wrong object alone in the cup ===========================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    rough = scene.pucks["puck_2"]
    write_local(rough, (0.30, 0.0, c.cup_floor_z + c.puck_h / 2 + 0.002))
    _step(90)
    _report("wrong-object")
    r_loc = scene.rig_local(rough.data.root_pos_w)
    check("negative (wrong object): a rough disc constructed alone INSIDE the cup "
          "(polished still on its pad) — the cup clause keys on identity: no "
          "success, no credit",
          bool(scene.in_cup(r_loc)[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.02)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.qc_chute")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    check("video: frames captured and saved to frames.npz", len(_REC["frames"]) >= 20)

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
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 — die loudly, never idle until the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)
