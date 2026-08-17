"""Smoke / rubric-REJECTION battery for MugRackHangScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts the wrench-inserted, gravity-hung
mug and the latched credit is monotone along a real trajectory). This battery proves
the rubric REJECTS wrong outcomes, and that every success() clause is load-bearing —
including cheats that make the mug PHYSICALLY HANG by the wrong means. Every probe is
CONSTRUCTED as a state (teleport, real physics steps, judge) — instrumentation, never
a solution.

Checks:
   1. settle/no-NaN     — seeded reset settles finite; both mugs upright on the floor;
                          score 0, no success;
   2. randomization     — READBACK across seeded resets: rack yaw / rack xy / white-mug
                          spawn all vary, the target peg takes >= 2 values, and the
                          green MARKER column matches the target peg on every reset;
   3. null-policy       — 300 idle steps: mugs stay on the floor, score ~0;
   4. SEED STRATEGY (a) — white mug transported and SET DOWN standing on the rack's
                          base plate at the panel's foot (approach + carry + place,
                          the seed's delivery) — stands settled, nothing latches,
                          score ~0;
   5. SEED STRATEGY (b) — white mug HELD IN THE AIR beside the green peg by the hand
                          wrench (the seed's terminal state: object acquired and held)
                          — airborne but never threaded: only the carry latches fire,
                          score caps at 0.20, no success;
   6. WRONG OBJECT      — the BLACK decoy mug genuinely HUNG on the green peg (drop,
                          catch, slide, settle — a real hang) — rejected: the judged
                          mug is the white one; score ~0;
   7. WRONG PEG         — the white mug genuinely HUNG on the farthest UNMARKED peg —
                          a real hang, real skill — rejected by target identity: the
                          thread/hang latches never fire; score caps at the carry 0.20;
   8. RIM-HANG cheat    — the white mug hung over the green peg by its RIM (peg inside
                          the cup body, mug dangling mouth-to-panel) — genuinely
                          suspended and settled at the right peg — rejected by the
                          handle-window clause; no thread latch, no success;
   9. NO INSTANT CREDIT — the white mug teleported directly INTO the perfect hang pose
                          (peg through window, zero velocity): success() is FALSE at
                          once — the still-hang streak cannot be teleported; the mug
                          is removed before it can settle into a real hang;
  10. DISTURB/RECOVER   — full success CONSTRUCTED (drop onto the peg, settle, score
                          1.0), then the mug teleported to the floor: success
                          COLLAPSES while the latched score survives at the 0.60 cap —
                          the hang is physics, not bookkeeping;
  11. rejection audit   — success() never fired at any judged step during the negative
                          probes (checks 4-9);
  12. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.approach_grasp_ceramic_teapot_i109.smoke --headless
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
    env = ENVS.get("simgen.mug_rack_hang")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.55, -0.95, 0.75)) + o),
                                tuple(np.array((0.40, 0.00, 0.22)) + o),
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

    def mug(i: int):
        return scene.mugs[c.mug_names[i]]

    WHITE, BLACK = 0, 1

    def status():
        _refresh()
        return scene._status()

    def score() -> float:
        _refresh()
        return float(scene.score()[0])

    def success() -> bool:
        _refresh()
        return bool(scene.success()[0])

    def gi() -> int:
        return int(scene.green_idx[0])

    def idx_t(j: int) -> torch.Tensor:
        return torch.full((n,), j, dtype=torch.long, device=device)

    def peg_frame(j: int) -> tuple[torch.Tensor, torch.Tensor]:
        _refresh()
        root_w, dir_w = scene._peg_world(idx_t(j))
        return root_w[0], dir_w[0]

    def threaded_on(i: int, j: int) -> bool:
        _refresh()
        return bool(scene._threaded(mug(i), idx_t(j))[0])

    def hang_geom(i: int, j: int) -> bool:
        """Genuine hang of mug i on peg j: threaded there, airborne, settled."""
        _refresh()
        return threaded_on(i, j) and bool(scene._airborne(mug(i))[0]) \
            and bool(scene._settled(mug(i))[0])

    def mug_z(i: int) -> float:
        _refresh()
        return float((mug(i).data.root_pos_w - scene.env_origins)[0, 2])

    def report(tag: str) -> None:
        s = status()
        pw = (mug(WHITE).data.root_pos_w - scene.env_origins)[0]
        print(f"[smoke] {tag:16s} | green={gi()} thr={bool(s['threaded'][0])} "
              f"air={bool(s['airborne'][0])} still={bool(s['settled'][0])} "
              f"streak={int(scene._streak[0])} d_ap={float(s['d_aperture'][0]) * 1000:.0f}mm "
              f"white=({float(pw[0]):+.3f},{float(pw[1]):+.3f},{float(pw[2]):+.3f}) "
              f"score={score():.3f} success={success()} "
              f"frames={len(_REC['frames'])}", flush=True)

    def hang_quat(j: int) -> torch.Tensor:
        """Orientation for a handle-hang on peg j: mug +y along tip->root, upright."""
        _root, d = peg_frame(j)
        y_m = -d / d.norm()
        z_w = torch.tensor([0.0, 0.0, 1.0], device=device)
        z_m = z_w - (z_w @ y_m) * y_m
        z_m = z_m / z_m.norm()
        x_m = torch.linalg.cross(y_m, z_m)
        R = torch.stack([x_m, y_m, z_m], dim=1)
        w = math.sqrt(max(1e-9, 1.0 + float(R[0, 0] + R[1, 1] + R[2, 2]))) / 2.0
        q = torch.tensor([w,
                          float(R[2, 1] - R[1, 2]) / (4 * w),
                          float(R[0, 2] - R[2, 0]) / (4 * w),
                          float(R[1, 0] - R[0, 1]) / (4 * w)], device=device)
        return q / q.norm()

    def place_threaded(i: int, j: int, s_along: float = 0.032) -> None:
        """TRANSPORT mug i to a free-space pose with peg j's axis through the CENTER
        of its handle window at `s_along` from the root (clearance all around,
        asserted in cfg); zero velocity. The mug then FALLS ~18 mm until the top
        handle bar catches on the peg — the hang itself is contact dynamics."""
        root, d = peg_frame(j)
        q = hang_quat(j)
        ap_l = torch.tensor(c.ap_local, device=device)
        pos = (root + d * s_along) - quat_apply(q.unsqueeze(0), ap_l.unsqueeze(0))[0]
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = q
        mug(i).write_root_state_to_sim(st, _all_ids())

    def build_hang(i: int, j: int, budget: int = 900) -> bool:
        """Constructed hang: drop-in place, then hands-off until it genuinely hangs."""
        for attempt in range(4):
            place_threaded(i, j)
            for _ in range(budget // 30):
                _step(30)
                if hang_geom(i, j):
                    return True
            if mug_z(i) < 0.12:
                print(f"[smoke] build_hang attempt {attempt}: mug fell, retrying", flush=True)
                continue
        return hang_geom(i, j)

    def to_floor(i: int, x: float, y: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.env_origins
        st[:, 0] += x
        st[:, 1] += y
        st[:, 2] += c.body_h / 2 + 0.003
        st[:, 3] = 1.0
        mug(i).write_root_state_to_sim(st, _all_ids())

    def zero_wrench(i: int) -> None:
        z = torch.zeros(n, 1, 3, device=device)
        mug(i).set_external_force_and_torque(z, z.clone(), env_ids=_all_ids(),
                                             is_global=True)

    def fresh(seed: int, settle: int = 150) -> None:
        torch.manual_seed(seed)
        env.reset()
        _step(settle)

    # ================= 1. settle / no-NaN =========================================================
    fresh(100)
    _REC["on"] = True
    _step(60)
    _REC["on"] = False
    report("settle")
    s = status()
    finite = all(bool(torch.isfinite(b.data.root_pos_w).all()) for b in scene.mugs.values())
    on_floor = all(abs(mug_z(i) - c.body_h / 2) < 0.01 for i in (WHITE, BLACK))
    check("settle/no-NaN: seeded reset settles finite; both mugs upright on the "
          "floor; score 0, no success",
          finite and on_floor and score() <= 0.01 and not success())

    # ================= 2. randomization is real (readback) ========================================
    obs = []
    for k in range(6):
        torch.manual_seed(300 + k)
        env.reset()
        _step(15)
        _refresh()
        q = scene.rack.data.root_quat_w
        yaw = math.degrees(2.0 * math.atan2(float(q[0, 3]), float(q[0, 0])))
        mk_loc = quat_apply_inverse(
            q, scene.marker.data.root_pos_w - scene.rack.data.root_pos_w)[0]
        marker_matches = abs(float(mk_loc[1]) - c.peg_ys[gi()]) < 0.005
        obs.append((yaw, scene.rack.data.root_pos_w[0, :2].clone(),
                    mug(WHITE).data.root_pos_w[0, :2].clone(), gi(), marker_matches))
    d_yaw = max(o[0] for o in obs) - min(o[0] for o in obs)
    d_rp = max(float((a[1] - b[1]).norm()) for a in obs for b in obs)
    d_mug = max(float((a[2] - b[2]).norm()) for a in obs for b in obs)
    greens = {o[3] for o in obs}
    marker_ok = all(o[4] for o in obs)
    print(f"[smoke] randomization: rack yaw spread {d_yaw:.1f}deg, rack xy spread "
          f"{d_rp * 1000:.0f}mm, white spawn spread {d_mug * 1000:.0f}mm, "
          f"green pegs {sorted(greens)}, marker matches target on all resets: "
          f"{marker_ok}", flush=True)
    check("randomization-is-real: rack yaw / rack xy / white-mug spawn READBACK all "
          "vary, the target peg takes >= 2 values across 6 resets, and the green "
          "marker column matches the target peg on every reset",
          d_yaw > 5.0 and d_rp > 0.005 and d_mug > 0.05 and len(greens) >= 2
          and marker_ok)

    # ================= 3. null policy fails =======================================================
    fresh(100)
    _step(300)
    report("null-policy")
    check("null-policy-fails: 300 idle steps, both mugs still on the floor, "
          "score ~0, no success",
          abs(mug_z(WHITE) - c.body_h / 2) < 0.01 and score() <= 0.01 and not success())

    _AUDIT["on"] = True  # ---- negative probes below: success must never fire ----

    # ================= 4. seed strategy (a): carry + set down at the rack ========================
    fresh(100)
    _REC["on"] = True
    # set the white mug standing on the base plate at the foot of the panel.
    # The plate front (x=-0.080) to panel face (x=-0.015) gap is 65 mm for a 64 mm
    # cup: center the body in it (x=-0.049) and yaw the mug 90 deg so the handle
    # points along +y, clear of the panel (it would penetrate 5 cm if it faced +x).
    _refresh()
    qr = scene.rack.data.root_quat_w
    pr = scene.rack.data.root_pos_w
    loc = torch.tensor([[-0.049, 0.0,
                         c.base_size[2] + c.body_h / 2 + 0.004]], device=device)
    q90 = torch.tensor([[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]],
                       device=device).expand(n, 4)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = pr + quat_apply(qr, loc.expand(n, 3))
    st[:, 3:7] = quat_mul(qr, q90)
    mug(WHITE).write_root_state_to_sim(st, _all_ids())
    _step(300)
    report("set-down")
    _REC["on"] = False
    stands = bool(scene._settled(mug(WHITE))[0]) \
        and abs(mug_z(WHITE) - (c.base_size[2] + c.body_h / 2)) < 0.012
    check("negative (SEED STRATEGY a): white mug transported and SET DOWN standing "
          "on the rack's base plate at the panel's foot — the seed's "
          "approach-carry-place delivery — stands settled, nothing latches, "
          "score ~0, no success",
          stands and score() <= 0.01 and not success())

    # ================= 5. seed strategy (b): acquired and HELD in the air ========================
    fresh(100)
    _REC["on"] = True
    root_g, d_g = peg_frame(gi())
    peg_mid = root_g + d_g * (c.peg_len / 2)
    side = torch.linalg.cross(d_g, torch.tensor([0.0, 0.0, 1.0], device=device))
    side = side / side.norm()
    p_tgt = peg_mid + 0.06 * side
    m_mass, G = c.mug_mass, 9.81
    for k in range(300):
        _refresh()
        p = mug(WHITE).data.root_pos_w[0]
        v = mug(WHITE).data.root_lin_vel_w[0]
        w_ang = mug(WHITE).data.root_ang_vel_w[0]
        f = m_mass * (G * torch.tensor([0.0, 0.0, 1.0], device=device)
                      + 25.0 * (p_tgt - p) - 10.0 * v)
        fn = f.norm()
        if float(fn) > 8.0:
            f = f * (8.0 / fn)
        tq = -0.01 * w_ang
        mug(WHITE).set_external_force_and_torque(
            f.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            tq.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            env_ids=_all_ids(), is_global=True)
        _step(1)
    report("held-in-air")
    s = status()
    held = bool(s["airborne"][0]) \
        and float((scene._aperture_w(mug(WHITE))[0] - peg_mid).norm()) < 0.12
    sc5 = score()
    zero_wrench(WHITE)
    _REC["on"] = False
    check("negative (SEED STRATEGY b): white mug HELD in the air beside the green "
          "peg by the hand wrench — the seed's terminal 'acquired' state — never "
          "threaded, no success; only the carry latches fire (score <= 0.20)",
          held and not bool(s["threaded"][0]) and not success()
          and sc5 <= c.w_near + c.w_lift + 0.001)

    # ================= 6. wrong object: the BLACK mug genuinely hung on the green peg ============
    fresh(100)
    _REC["on"] = True
    black_hangs = build_hang(BLACK, gi())
    _step(120)
    report("black-on-green")
    _REC["on"] = False
    check("negative (WRONG OBJECT): the BLACK decoy mug genuinely HUNG on the green "
          "peg — threaded, airborne, settled, a real hang — rejected because the "
          "judged mug is the WHITE one; score ~0, no success",
          black_hangs and hang_geom(BLACK, gi()) and score() <= 0.01 and not success())

    # ================= 7. wrong peg: white hung on the farthest unmarked peg ======================
    fresh(100)
    g7 = gi()
    wrong = max((j for j in range(3) if j != g7),
                key=lambda j: (c.peg_ys[j] - c.peg_ys[g7]) ** 2
                + (c.peg_zs[j] - c.peg_zs[g7]) ** 2)
    _REC["on"] = True
    white_wrong = build_hang(WHITE, wrong)
    _step(120)
    report(f"white-on-peg{wrong}")
    _REC["on"] = False
    s = status()
    check("negative (WRONG PEG): the white mug genuinely HUNG on the farthest "
          "unmarked peg — a real hang, real skill — rejected by target identity: "
          "never threaded on the green peg, the thread/hang latches never fire, "
          "score caps at the carry credit (<= 0.20), no success",
          white_wrong and hang_geom(WHITE, wrong) and not bool(s["threaded"][0])
          and not bool(scene._threaded_ever[0]) and not bool(scene._hung_ever[0])
          and score() <= c.w_near + c.w_lift + 0.001 and not success())

    # ================= 8. rim-hang cheat: peg inside the cup, not the handle =====================
    fresh(100)
    g8 = gi()
    root8, d8 = peg_frame(g8)
    rim_ok = False
    for attempt in range(4):
        # mouth faces the panel (local +z along -d); cup cavity swallows the peg
        z_m = -d8 / d8.norm()
        a = torch.tensor([0.0, 0.0, 1.0], device=device)
        x_m = a - (a @ z_m) * z_m
        x_m = x_m / x_m.norm()
        y_m = torch.linalg.cross(z_m, x_m)
        R = torch.stack([x_m, y_m, z_m], dim=1)
        w8 = math.sqrt(max(1e-9, 1.0 + float(R[0, 0] + R[1, 1] + R[2, 2]))) / 2.0
        q8 = torch.tensor([w8,
                           float(R[2, 1] - R[1, 2]) / (4 * w8),
                           float(R[0, 2] - R[2, 0]) / (4 * w8),
                           float(R[1, 0] - R[0, 1]) / (4 * w8)], device=device)
        q8 = q8 / q8.norm()
        # body center on the peg axis, mid-cavity over the peg, tiny lift then drop
        pos8 = root8 + d8 * 0.050 + z_m * (-0.010) \
            + torch.tensor([0.0, 0.0, 0.004], device=device)
        st8 = torch.zeros(n, 13, device=device)
        st8[:, 0:3] = pos8
        st8[:, 3:7] = q8
        mug(WHITE).write_root_state_to_sim(st8, _all_ids())
        _REC["on"] = True
        _step(600)
        _REC["on"] = False
        _refresh()
        rim_ok = bool(scene._airborne(mug(WHITE))[0]) \
            and bool(scene._settled(mug(WHITE))[0]) \
            and float((scene._aperture_w(mug(WHITE))[0]
                       - (root8 + d8 * (c.peg_len / 2))).norm()) < 0.11
        if rim_ok:
            break
        print(f"[smoke] rim-hang attempt {attempt} not suspended, retrying", flush=True)
    report("rim-hang")
    s = status()
    check("near-miss (RIM-HANG cheat): the white mug dangling over the GREEN peg by "
          "its rim/body (peg inside the cup, not the handle) — genuinely suspended, "
          "settled, at the right peg — rejected by the handle-window clause: not "
          "threaded, no thread/hang latch, no success",
          rim_ok and not bool(s["threaded"][0]) and not bool(scene._threaded_ever[0])
          and not bool(scene._hung_ever[0]) and not success())

    # ================= 9. no instant credit: the streak cannot be teleported =====================
    fresh(100)
    place_threaded(WHITE, gi())  # the perfect hang pose, zero velocity, ZERO steps
    _refresh()
    s = status()
    instant = bool(s["threaded"][0]) and bool(s["airborne"][0]) and not success() \
        and int(scene._streak[0]) == 0
    to_floor(WHITE, -0.55, 0.45)  # remove before it can settle into a real hang
    _step(60)
    check("no-instant-credit: the white mug teleported INTO the perfect hang pose "
          "(threaded, airborne, zero velocity) — success() is FALSE at once: the "
          "continuous still-hang streak is earned by simulated physics, not by a "
          "state write",
          instant and not success())

    _AUDIT["on"] = False  # ---- end of negative probes ----

    # ================= 10. disturb/recover: success collapses, latches survive ===================
    fresh(100)
    _REC["on"] = True
    built = build_hang(WHITE, gi())
    for _ in range(20):
        if success():
            break
        _step(30)
    report("full-solution")
    built = built and success()
    score_before = score()
    to_floor(WHITE, -0.60, -0.45)
    _step(240)
    report("disturbed")
    _REC["on"] = False
    score_after = score()
    check("DISTURB: constructed success is live (score 1.0), then the white mug "
          "teleported to the floor — success COLLAPSES (the hang is a live physical "
          "state) while the latched score survives at the 0.60 partial cap",
          built and score_before >= 0.999 and not success()
          and abs(score_after - c.score_cap) < 0.011)

    # ================= 11. rejection audit ========================================================
    check("rejection audit: success() never fired at any step of the negative "
          "probes (checks 4-9)", not _AUDIT["fired"])

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.mug_rack_hang")
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
