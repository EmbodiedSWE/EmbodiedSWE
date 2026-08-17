"""Smoke / rubric-REJECTION battery for CooperCurveScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a real taper-steered rolling
delivery and the latched credit is monotone along it). This battery proves the rubric
REJECTS wrong outcomes, and that the physical claims the task rests on — the equal-rim
decoy cannot turn the gallery, the roof refuses lifted passage — are load-bearing.
Every probe is CONSTRUCTED as a settled state (teleport, real physics steps, judge) —
instrumentation, never a solution: no probe here reaches success().

Checks:
   1. settle/no-NaN      — seeded reset settles finite; both rollers lying in the pen;
                           score 0, no success;
   2. randomization      — three seeded resets, max-pairwise READBACK deltas: cup xy,
                           cup yaw, decoy xy all differ;
   3. slot permutation   — over 10 resets both cup/decoy slot assignments appear and
                           the readback cup azimuth matches the recorded slot;
   4. null-policy        — 240 idle steps: score ~0, no success (nothing moves itself);
   5. SEED-analog        — the seed's whole strategy is "grasp the cup and raise it /
                           place it": the cup teleported DIRECTLY into the catch bay,
                           settled, lying — the ordered chain refuses: score ~0, no
                           success (the goal position without the rolled journey is
                           worth nothing);
   6. wrong object       — the DECOY walked through staging, every checkpoint sector
                           and into the bay: no cup latch fires, score ~0, no success;
   7. DECOY CANNOT TURN  — the decoy is staged and launched with the very servo push
                           the solve uses on the cup: it rolls STRAIGHT, wedges on the
                           outer wall far short of the bay (assert it moved >= 10 cm —
                           non-vacuous — never entered the bay, and stopped well short);
   8. out-of-order       — cup teleported into the cp3 sector, then into the bay,
                           WITHOUT ever staging: chain latches all refuse, score ~0,
                           no success even though the cup rests lying in the bay;
   9. near-miss          — the full chain constructed in order (stage -> cp1 -> cp2 ->
                           cp3) but the cup stops at azimuth 84 deg, 4 deg short of
                           the bay: score capped at 0.65, no success;
  10. roof gate          — a 6 N upward pull on the cup mid-gallery: it RISES (assert
                           >= 10 mm — non-vacuous) but the roof arrests it far below
                           lifted-carry height; on release it falls back; no credit;
  11. restraint          — decoy parked IN the bay first, then the full chain run and
                           the cup settled in the bay beside it: every cup clause
                           holds, the decoy clause alone refuses: score <= 0.751,
                           no success;
  12. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.pick_up_cup_i364.smoke --headless
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
    from . import scene as task_scene  # noqa: F401  (registers the scene)
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


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    rho, phi = scene._polar(scene.cup.data.root_pos_w)
    rho_d, phi_d = scene._polar(scene.decoy.data.root_pos_w)
    print(f"[smoke] {tag:18s} | cup=(rho {float(rho[0]):.3f}, phi {float(phi[0]):+7.2f}) "
          f"decoy=(rho {float(rho_d[0]):.3f}, phi {float(phi_d[0]):+7.2f}) "
          f"staged={bool(scene._staged[0])} cp1={bool(scene._cp1[0])} "
          f"cp2={bool(scene._cp2[0])} cp3={bool(scene._cp3[0])} "
          f"bay={bool(scene._bay[0])} score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cooper_curve")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    all_ids = torch.arange(n, device=device)
    no_action = torch.empty(0, device=device)
    zero3 = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.85, -0.95, 0.90)) + o),
                                tuple(np.array((0.00, -0.05, 0.05)) + o),
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

    def dang(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def put(body, rho: float, phi_deg: float, tapered: bool,
            yaw_deg: float | None = None, settle: int = 60,
            arrest: int = 0) -> None:
        """CONSTRUCT: write the roller to a rest pose (default radial yaw) and give
        it real settle steps — probe instrumentation, never a solution. `arrest`
        adds velocity-zeroing bursts right after the set-down: the 2 mm drop jolt
        otherwise starts a roller arc-rolling tens of mm before it calms (needed
        when a placement must stay put next to tight clearances)."""
        rho_t = torch.full((n,), float(rho), device=device)
        phi_t = torch.full((n,), math.radians(phi_deg), device=device)
        yaw_t = torch.full((n,), math.radians(phi_deg if yaw_deg is None else yaw_deg),
                           device=device)
        st = scene._rest_state(rho_t, phi_t, yaw_t, tapered=tapered)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        _refresh()
        for _ in range(arrest):
            _step(5)
            body.write_root_velocity_to_sim(
                torch.zeros(n, 6, device=device), all_ids)
        _step(settle)

    def seg_of(body):
        p = (body.data.root_pos_w - scene.env_origins)[0]
        ax = scene._axis_w(body)[0]
        axn = math.hypot(float(ax[0]), float(ax[1]))
        ux, uy = ((float(ax[0]) / axn, float(ax[1]) / axn) if axn > 1e-6
                  else (1.0, 0.0))
        return float(p[0]), float(p[1]), ux, uy

    def stage_clear_phi(mover_r: float, other) -> float:
        """First staging azimuth whose radial-yaw rest pose clears `other`'s ACTUAL
        axis segment (sampled segment distance > rim sum + 8 mm)."""
        ox, oy, ux, uy = seg_of(other)
        h = c.length / 2.0
        need = mover_r + max(c.r_mouth, c.r_decoy) + 0.008
        for phi_deg in (-87.0, -84.0, -81.0, -78.0):
            ar = math.radians(phi_deg)
            cxm, cym = c.rho_stage() * math.cos(ar), c.rho_stage() * math.sin(ar)
            best = math.inf
            for f in (-1.0, -0.5, 0.0, 0.5, 1.0):
                px, py = cxm + f * h * math.cos(ar), cym + f * h * math.sin(ar)
                t = max(-h, min(h, (px - ox) * ux + (py - oy) * uy))
                best = min(best, math.hypot(px - (ox + t * ux), py - (oy + t * uy)))
                qx, qy = ox + f * h * ux, oy + f * h * uy
                t2 = max(-h, min(h, (qx - cxm) * math.cos(ar) + (qy - cym) * math.sin(ar)))
                best = min(best, math.hypot(qx - (cxm + t2 * math.cos(ar)),
                                            qy - (cym + t2 * math.sin(ar))))
            if best > need:
                return phi_deg
        return -87.0  # fall back (seed-fixed layouts below never need it)

    def servo_launch(body, v_des: float, f_cap: float, max_steps: int = 480) -> None:
        """The solve's launch servo, verbatim physics: tangential velocity servo at
        the CoM, horizontal cap, CUT (zero-wrench write) at azimuth -76.5 deg."""
        for _ in range(max_steps):
            _refresh()
            rho, phi = scene._polar(body.data.root_pos_w)
            if float(phi[0]) >= c.stage_phi_hi - 0.5:
                break
            pr = torch.deg2rad(phi)
            u_t = torch.stack([-torch.sin(pr), torch.cos(pr)], dim=-1)
            v_xy = body.data.root_lin_vel_w[:, :2]
            f_xy = c.roller_mass * 8.0 * (v_des * u_t - v_xy)
            mag = f_xy.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            f_xy = f_xy * (mag.clamp(max=f_cap) / mag)
            wrench = torch.zeros(n, 1, 3, device=device)
            wrench[:, 0, 0:2] = f_xy
            body.set_external_force_and_torque(wrench, zero3, env_ids=all_ids,
                                               is_global=True)
            _step(1)
        body.set_external_force_and_torque(zero3, zero3, env_ids=all_ids, is_global=True)
        _step(1)

    def build_chain(final_phi: float, final_rho: float,
                    arrest_final: int = 0) -> None:
        """Construct the ordered latch chain by sequential settled placements:
        stage, cp1, cp2, cp3, then the caller-chosen final pose (NEVER a pose that
        satisfies the goal mid-construction: the bay is entered only when the caller
        says so, and by then any refusal clause under test is already armed)."""
        put(scene.cup, c.rho_stage(), -83.0, tapered=True, settle=60)
        for cp in c.cp_phis:
            put(scene.cup, c.rho_stage(), float(cp), tapered=True, settle=40)
        put(scene.cup, final_rho, final_phi, tapered=True, settle=90,
            arrest=arrest_final)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(120)
    _report("settle")
    _REC["on"] = False
    rho_c, phi_c = scene._polar(scene.cup.data.root_pos_w)
    rho_d, phi_d = scene._polar(scene.decoy.data.root_pos_w)
    in_pen = (float(phi_c[0]) < c.stage_phi_lo and float(phi_d[0]) < c.stage_phi_lo
              and c.clear_in() < float(rho_c[0]) < c.clear_out()
              and c.clear_in() < float(rho_d[0]) < c.clear_out())
    check("settle/no-NaN: layout settles finite, both rollers lying in the pen, "
          "score 0, no success",
          bool(torch.isfinite(scene.cup.data.root_state_w).all())
          and bool(torch.isfinite(scene.decoy.data.root_state_w).all())
          and in_pen and bool(scene.lying()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (3-seed max-pairwise readback) ===================
    reads = []
    for s in (11, 22, 33):
        env.reset(seed=s)
        _step(10)
        _refresh()
        reads.append(((scene.cup.data.root_pos_w[0, :2] - scene.env_origins[0, :2]).clone(),
                      yaw_of(scene.cup.data.root_quat_w[0]),
                      (scene.decoy.data.root_pos_w[0, :2] - scene.env_origins[0, :2]).clone()))
    d_cup = max(float((a[0] - b[0]).norm()) for a, b in ((reads[0], reads[1]),
                (reads[0], reads[2]), (reads[1], reads[2])))
    d_yaw = max(dang(a[1], b[1]) for a, b in ((reads[0], reads[1]),
                (reads[0], reads[2]), (reads[1], reads[2])))
    d_dec = max(float((a[2] - b[2]).norm()) for a, b in ((reads[0], reads[1]),
                (reads[0], reads[2]), (reads[1], reads[2])))
    print(f"[smoke] randomization max-pairwise deltas: cup_xy={d_cup * 1000:.1f}mm "
          f"cup_yaw={d_yaw:.1f}deg decoy_xy={d_dec * 1000:.1f}mm", flush=True)
    check("randomization-is-real: cup xy, cup yaw and decoy xy readbacks differ "
          "across 3 seeds", d_cup > 0.004 and d_yaw > 5.0 and d_dec > 0.004)

    # ================= 3. slot permutation coverage ===============================================
    slots_seen, consistent = set(), True
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        slot = int(scene.cup_slot[0])
        slots_seen.add(slot)
        _r, phi_now = scene._polar(scene.cup.data.root_pos_w)
        consistent &= dang(float(phi_now[0]), c.slot_phis[slot]) < 6.0
    print(f"[smoke] over 10 resets: cup slots seen {sorted(slots_seen)} "
          f"consistent={consistent}", flush=True)
    check("slot permutation: both cup/decoy slot assignments appear over 10 resets "
          "and the cup azimuth readback matches its recorded slot",
          slots_seen == {0, 1} and consistent)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED's strategy (place at the goal) =======================
    # rlbench/pick_up_cup is solved by grasping the cup and raising/placing it. The
    # closest analog here: carry the cup STRAIGHT to the catch bay. CONSTRUCT that
    # settled end state — the ordered chain must refuse it wholesale.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    put(scene.cup, c.rho_stage(), 100.0, tapered=True, settle=120)
    _report("seed-analog")
    _REC["on"] = False
    check("negative (SEED analog): cup teleport-carried directly into the bay, "
          "settled lying — in_bay holds but chain latches refuse: score ~0, no "
          "success",
          bool(scene.in_bay(scene.cup)[0]) and bool(scene.lying()[0])
          and not bool(scene._staged[0]) and not bool(scene._bay[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 6. negative: wrong object through the whole course =========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    put(scene.decoy, c.rho_stage(), -83.0, tapered=False, settle=40)
    for cp in c.cp_phis:
        put(scene.decoy, c.rho_stage(), float(cp), tapered=False, settle=30)
    put(scene.decoy, c.rho_stage(), 100.0, tapered=False, settle=90)
    _report("wrong-object")
    check("negative (wrong object): the DECOY walked through staging, all "
          "checkpoints and into the bay — no cup latch fires, score ~0, no success",
          bool(scene.in_bay(scene.decoy)[0]) and not bool(scene._staged[0])
          and not bool(scene._cp1[0]) and not bool(scene._bay[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 7. THE DECOY CANNOT TURN (the identity claim is physics) ===================
    # Stage the decoy exactly as the solve stages the cup, launch it with the very
    # same servo push. Equal rims -> straight line -> the curved outer wall catches
    # it far short of the bay.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    put(scene.decoy, c.rho_stage(), stage_clear_phi(c.r_decoy, scene.cup),
        tapered=False, settle=60)
    start_xy = scene.decoy.data.root_pos_w[0, :2].clone()
    servo_launch(scene.decoy, 0.75, 2.2)
    saw_bay, max_phi = False, -180.0
    for _ in range(96):
        _step(10)
        _refresh()
        saw_bay = saw_bay or bool(scene.in_bay(scene.decoy)[0])
        _r, phi_now = scene._polar(scene.decoy.data.root_pos_w)
        max_phi = max(max_phi, float(phi_now[0]))
        if float(scene.decoy.data.root_lin_vel_w.norm(dim=-1)[0]) < 0.03:
            break
    _step(60)
    _report("decoy-launch")
    _REC["on"] = False
    moved = float((scene.decoy.data.root_pos_w[0, :2] - start_xy).norm())
    _r, phi_end = scene._polar(scene.decoy.data.root_pos_w)
    print(f"[smoke] decoy launch: moved {moved * 1000:.0f}mm, max phi "
          f"{max_phi:+.1f}deg, rest phi {float(phi_end[0]):+.1f}deg "
          f"(bay begins at {c.bay_phi_lo:.0f}deg)", flush=True)
    check("DECOY CANNOT TURN: launched with the solve's own push it moves >= 10 cm "
          "but rolls straight into the outer wall and never reaches the bay "
          "(stops short of +45 deg)",
          moved >= 0.10 and not saw_bay and max_phi < 45.0
          and not bool(scene.success()[0]))

    # ================= 8. negative: out-of-order (no staging, zones visited) ======================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    put(scene.cup, c.rho_stage(), float(c.cp_phis[2]), tapered=True, settle=60)
    put(scene.cup, c.rho_stage(), 100.0, tapered=True, settle=90)
    _report("out-of-order")
    check("negative (out of order): cup dropped into the cp3 sector then the bay "
          "without ever staging — every chain latch refuses, score ~0, no success",
          bool(scene.in_bay(scene.cup)[0]) and not bool(scene._staged[0])
          and not bool(scene._cp3[0]) and not bool(scene._bay[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 9. near-miss: full chain, stopped 4 deg short of the bay ===================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    build_chain(final_phi=84.0, final_rho=c.rho_stage())
    _report("near-miss")
    _REC["on"] = False
    s9 = float(scene.score()[0])
    check("near-miss: staged + all three checkpoints latched but the cup rests at "
          "84 deg, short of the 88 deg bay mouth — score capped at 0.65, no success",
          bool(scene._staged[0]) and bool(scene._cp3[0]) and not bool(scene._bay[0])
          and 0.64 <= s9 <= 0.66 and not bool(scene.success()[0]))

    # ================= 10. roof gate: lifted carry is physically refused ==========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    put(scene.cup, c.rho_stage(), 0.0, tapered=True, settle=60)
    z_rest = float(scene._rel_z(scene.cup)[0])
    up = torch.zeros(n, 1, 3, device=device)
    up[:, 0, 2] = 6.0  # ~2.4x weight, straight up, mid-gallery
    z_max = z_rest
    for _ in range(120):
        scene.cup.set_external_force_and_torque(up, zero3, env_ids=all_ids,
                                                is_global=True)
        _step(1)
        _refresh()
        z_max = max(z_max, float(scene._rel_z(scene.cup)[0]))
    scene.cup.set_external_force_and_torque(zero3, zero3, env_ids=all_ids,
                                            is_global=True)
    _step(120)
    _report("roof-gate")
    print(f"[smoke] roof gate: rest z={z_rest * 1000:.1f}mm, max lifted "
          f"z={z_max * 1000:.1f}mm (roof underside {c.roof_lo * 1000:.0f}mm, "
          f"centre-at-contact ~{(c.roof_lo - c.r_mouth) * 1000:.0f}mm)", flush=True)
    check("roof gate: a 6 N upward pull mid-gallery lifts the cup (>= 10 mm, "
          "non-vacuous) but the roof arrests it far below carry height; no credit",
          z_max >= z_rest + 0.010 and z_max <= c.roof_lo - c.r_core
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 11. restraint: the decoy clause alone refuses ==============================
    # Decoy parked IN the bay FIRST (so no prefix of the construction is a success),
    # then the full chain is run and the cup settled in the bay beside it. All cup
    # clauses hold; the decoy clause must refuse alone. Placements verified
    # clearance-positive: decoy radial (0.312, 108 deg) — DEEP in the bay, 20 deg
    # from the 88 deg mouth (a first attempt at 88.8 settle-rolled 1.4 deg back out
    # of the bay) and 7 mm clear of the 115 deg arrest wall (0.367*sin(7deg)=44.7mm
    # > r 37.5); cup radial (0.26, 93 deg), 5 deg inside the mouth vs ~2 deg settle
    # drift. Closest sphere pairs: cup base (rho .205) vs decoy inner (rho .257) at
    # dphi 15 deg -> 79 mm vs 67.5 mm contact; cup mouth vs decoy inner -> 94 mm vs
    # 82.5 mm.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    put(scene.decoy, 0.312, 108.0, tapered=False, settle=90, arrest=10)
    build_chain(final_phi=93.0, final_rho=0.260, arrest_final=10)
    _step(120)
    _report("restraint")
    _REC["on"] = False
    s11 = float(scene.score()[0])
    check("restraint: full chain + cup settled lying in the bay, but the decoy is "
          "in the bay too — success refused by the decoy clause alone, score <= "
          "0.751",
          bool(scene._bay[0]) and bool(scene.in_bay(scene.cup)[0])
          and bool(scene.lying()[0]) and bool(scene.in_bay(scene.decoy)[0])
          and not bool(scene.success()[0]) and s11 <= 0.751)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.cooper_curve")
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
    except BaseException as exc:  # noqa: BLE001 - die fast, Kit won't exit on its own
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL (exception: {exc!r})", flush=True)
        os._exit(2)
