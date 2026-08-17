"""Smoke / rubric-REJECTION battery for StoneDoorVaultScene — NullRobot, teleported probes.

solve.py is the acceptance proof along a REAL driven trajectory (monotone latched credit,
success only after the roll-and-capture). This battery proves the rubric REJECTS wrong
outcomes and that the physical claims the task rests on — the wall/rails block the seed's
push-toward-the-wall motion, the pocket is a genuine gravity detent that CAPTURES and
RETAINS the stone — are load-bearing. Probes are CONSTRUCTED settled states (teleport,
real physics steps, judge). The only probe that touches success() is the ACCEPTANCE
construct in check 6, which verifies the rubric accepts the goal state and that the
seated stone survives a sub-extraction-threshold pull.

Checks:
   1. settle/no-NaN      — seeded reset settles finite: stone upright in the channel
                           0.10-0.28 m from the pocket, decoy on the OTHER side;
                           score 0, no success;
   2. randomization      — three seeded resets: max-pairwise READBACK deltas of vault
                           yaw, vault xy and stone start distance all real;
   3. side swap          — over 10 resets the stone starts on BOTH sides of the pocket;
   4. null-policy        — 240 idle steps: score <= 0.05, no success, and the decoy
                           block does NOT wander down the track (it cannot roll);
   5. SEED STRATEGY      — the seed's only motion (push the "door" TOWARD the wall):
                           3 N at the stone's CoM toward the wall for 1.5 s. The stone
                           presses the inner rail (min-x readback proves contact) and
                           goes nowhere — no track progress, no credit, no success;
   6. capture+retention  — stone CONSTRUCTED seated in the pocket -> success accepted,
                           score 1.0; then a 0.6 N steady track drag. NOTE the bound
                           that matters for a CONSTANT force is energetic, not static:
                           the force does work across the ~12 mm pocket slack plus the
                           ~33 mm climb, so extraction needs F * 45 mm > mgh ~ 49 mJ
                           -> F >~ 1.1 N (a 2 N pull DOES extract — measured). 0.6 N
                           has ~1.8x margin: the stone rolls to the bed edge (peak-|y|
                           readback proves the drag bit) but stays captive and success
                           holds after release;
   7. near-miss short    — stone upright ON the bed 100 mm from the pocket center on
                           its OWN side (the decoy owns the other side — a probe there
                           interpenetrates and gets depenetration-shoved): still on
                           the bed, no `near`/`in` credit, no success;
   8. wrong object       — RED decoy seated in the pocket, stone untouched -> pocket
                           readback confirms the decoy is in, score ~0, no success;
   9. wrong orientation  — stone lying FLAT bridging the rails over the pocket:
                           upright clause refuses, not seated, no success;
  10. perch              — stone upright on the bed EDGE (55 mm, past the `near`
                           latch, above the seated band): perched, not in, no success;
  11. outside channel    — stone upright on the GROUND in front of the channel at the
                           pocket's y: passes the y-band, z-band and upright clauses,
                           REJECTED by the in-channel clause alone;
  12. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.close_microwave_i306.smoke --headless
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

from isaaclab.utils.math import quat_apply  # noqa: E402

_qmul, _qy = task_scene._qmul, task_scene._qy

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


def _stone_z() -> float:
    return float(_ENV.scene.stone.data.root_pos_w[0, 2] - _ENV.scene.env_origins[0, 2])


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    sl = scene._vault_local(scene.stone.data.root_pos_w)[0]
    dl = scene._vault_local(scene.decoy.data.root_pos_w)[0]
    print(f"[smoke] {tag:18s} | stone y={float(sl[1]):+.3f} x={float(sl[0]):+.3f} "
          f"z={_stone_z():.3f} decoy_y={float(dl[1]):+.3f} "
          f"seated={bool(scene.stone_seated()[0])} "
          f"m/n/i={int(scene._moved[0])}{int(scene._near[0])}{int(scene._in[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


def _push_stone(dir_w: torch.Tensor, mag: float, steps: int) -> dict:
    """Drive the stone at its CoM with a constant WORLD-space force for `steps` steps.

    IMPORTANT — wrench frame: `is_global=True` wrenches on a ROLLING body are applied
    through a stale rotation reference, so the "world" force direction rotates with the
    disc's accumulated roll angle (the solve-side root cause). The force is therefore
    re-expressed in the stone's body frame with the CURRENT quat, every step. Returns
    peak vault-local excursions sampled DURING the push (measure at the peak, not after
    force-off — gravity restores detent states before a post-settle readback)."""
    n = _ENV.num_envs
    scene = _ENV.scene
    zero = torch.zeros(n, 1, 3, device=_ENV.device)
    f_w = mag * dir_w.view(1, 3).expand(n, 3).contiguous()
    stats = {"min_x": 1e9, "max_absy": 0.0, "max_z": 0.0}
    for _ in range(steps):
        q = scene.stone.data.root_quat_w
        q_c = torch.cat([q[:, :1], -q[:, 1:]], dim=1)
        scene.stone.set_external_force_and_torque(
            quat_apply(q_c, f_w).view(n, 1, 3), zero, env_ids=_all_ids())
        _step(1)
        sl = scene._vault_local(scene.stone.data.root_pos_w)[0]
        stats["min_x"] = min(stats["min_x"], float(sl[0]))
        stats["max_absy"] = max(stats["max_absy"], abs(float(sl[1])))
        stats["max_z"] = max(stats["max_z"], _stone_z())
    scene.stone.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    return stats


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.stone_door_vault")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.45, -0.55, 0.55)) + o),
                                tuple(np.array((0.42, 0.00, 0.08)) + o),
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

    def vault_world(loc_xyz) -> torch.Tensor:
        """Vault-local point -> world (per-env)."""
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.vault.data.root_pos_w + quat_apply(scene.vault.data.root_quat_w, loc)

    def vault_quat(q_extra: torch.Tensor | None = None) -> torch.Tensor:
        _refresh()
        q = scene.vault.data.root_quat_w
        return q if q_extra is None else _qmul(q, q_extra)

    def q_disc() -> torch.Tensor:
        """Upright-stone quat: cylinder axis across the channel (vault-local x)."""
        return vault_quat(_qy(torch.full((n,), math.pi / 2, device=device)))

    def track_dir(sign: float) -> torch.Tensor:
        """World unit vector along the track (vault-local y * sign)."""
        return quat_apply(vault_quat(),
                          torch.tensor([0.0, sign, 0.0], device=device).expand(n, 3))[0]

    def stone_loc() -> torch.Tensor:
        _refresh()
        return scene._vault_local(scene.stone.data.root_pos_w)[0]

    # ================= 1. settle / no-NaN =========================================================
    env.reset(seed=100)
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    sl = stone_loc()
    dl = scene._vault_local(scene.decoy.data.root_pos_w)[0]
    check("settle/no-NaN: stone upright in the channel 0.10-0.28 m from the pocket, "
          "decoy on the OTHER side, score 0, no success",
          bool(scene._finite()[0])
          and abs(float(sl[0]) - c.chan_x) < 0.02
          and 0.10 < abs(float(sl[1])) < 0.28
          and 0.064 < _stone_z() < 0.078
          and float(sl[1]) * float(dl[1]) < 0.0
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (3-seed max-pairwise readback) ===================
    def readback():
        _refresh()
        return (yaw_of(scene.vault.data.root_quat_w[0]),
                scene.vault.data.root_pos_w[0, :2].clone(),
                abs(float(scene._vault_local(scene.stone.data.root_pos_w)[0, 1])))

    vals = []
    for s in (101, 202, 303):
        env.reset(seed=s)
        _step(10)
        vals.append(readback())
    d_yawv = max(dyaw(a[0], b[0]) for a in vals for b in vals)
    d_xy = max(float((a[1] - b[1]).norm()) for a in vals for b in vals)
    d_dist = max(abs(a[2] - b[2]) for a in vals for b in vals)
    print(f"[smoke] randomization max-pairwise deltas: vault_yaw={d_yawv:.1f}deg "
          f"vault_xy={d_xy * 1000:.1f}mm stone_dist={d_dist * 1000:.1f}mm", flush=True)
    check("randomization-is-real: vault yaw, vault xy, stone start distance readback "
          "all differ across seeds",
          d_yawv > 2.0 and d_xy > 0.003 and d_dist > 0.003)

    # ================= 3. side swap ===============================================================
    sides = set()
    for s in range(10):
        env.reset(seed=300 + s)
        _refresh()
        sides.add("+" if float(scene.stone_side[0]) > 0 else "-")
    print(f"[smoke] over 10 resets: stone sides {sorted(sides)}", flush=True)
    check("side swap: the stone starts on BOTH sides of the pocket over 10 resets",
          sides == {"+", "-"})

    # ================= 4. null policy fails (and the decoy stays put) =============================
    env.reset(seed=100)
    _refresh()
    dy0 = float(scene._vault_local(scene.decoy.data.root_pos_w)[0, 1])
    _step(240)
    _report("null-policy")
    dy1 = float(scene._vault_local(scene.decoy.data.root_pos_w)[0, 1])
    check("null-policy-fails: 240 idle steps, score <= 0.05, no success, decoy "
          "does not wander down the track",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0])
          and abs(dy1 - dy0) < 0.010)

    # ================= 5. negative: the SEED'S OWN STRATEGY =======================================
    # The seed closes its microwave by pushing the door TOWARD the frame. Here that
    # motion is a push toward the wall (vault-local -x): 3 N (~0.6 mg) at the CoM for
    # 1.5 s. The inner rail takes the load (min-x readback shows the ~4 mm travel to
    # rail contact) and the stone makes NO track progress: no credit, no success.
    env.reset(seed=100)
    _step(30)
    _REC["on"] = True
    y_before = float(stone_loc()[1])
    to_wall = quat_apply(vault_quat(),
                         torch.tensor([-1.0, 0.0, 0.0], device=device).expand(n, 3))[0]
    stats = _push_stone(to_wall, 3.0, 180)
    _step(60)
    _report("seed-strategy")
    y_after = float(stone_loc()[1])
    print(f"[smoke] seed-strategy push: min_x={stats['min_x'] * 1000:.1f}mm "
          f"(rest {c.chan_x * 1000:.0f}mm) max_z={stats['max_z'] * 1000:.0f}mm "
          f"dy={abs(y_after - y_before) * 1000:.0f}mm", flush=True)
    check("negative (SEED strategy): 3 N push toward the wall for 1.5 s presses the "
          "inner rail and achieves nothing — no track progress, score <= 0.05, "
          "no success",
          stats["min_x"] < c.chan_x - 0.0015 and stats["max_z"] < 0.10
          and abs(y_after - y_before) < 0.05
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6. ACCEPTANCE + retention: the pocket is a real detent =====================
    # Construct the goal state (stone on the pocket floor) -> the rubric must accept
    # it. Then drag 0.6 N along the track toward the stone's OWN side (away from the
    # decoy). For a CONSTANT force the extraction bound is ENERGETIC, not static: the
    # force works over the ~12 mm slack + ~33 mm climb, so escape needs
    # F * 45 mm > mgh ~ 49 mJ -> F >~ 1.1 N (a 2 N pull DOES extract — measured).
    # 0.6 N (~1.8x margin) rolls the stone to the bed edge (peak-|y| DURING the drag
    # proves it bit) but the pocket keeps it — seated and successful after release.
    env.reset(seed=100)
    _step(30)
    _REC["on"] = True
    own_side = float(scene.stone_side[0])
    _write_body(scene.stone, vault_world((c.chan_x, 0.0, c.stone_r + 0.001)), q_disc())
    _step(90)
    _report("constructed-seat")
    ok_accept = bool(scene.success()[0]) and float(scene.score()[0]) >= 0.99
    stats = _push_stone(track_dir(own_side), 0.6, 180)
    _step(90)
    _report("retention-pull")
    print(f"[smoke] retention drag: peak|y|={stats['max_absy'] * 1000:.1f}mm "
          f"(edge contact ~12mm, bed edge {c.pocket_hl * 1000:.0f}mm)", flush=True)
    check("ACCEPTANCE+retention: constructed seat -> success, score 1.0; a 0.6 N "
          "steady track drag rolls the stone to the bed edge but cannot extract it — "
          "still seated and successful after release",
          ok_accept and 0.004 < stats["max_absy"] < 0.045
          and bool(scene.stone_seated()[0]) and bool(scene.success()[0])
          and float(scene.score()[0]) >= 0.99)
    _REC["on"] = False

    # ================= 7. near-miss: short of the pocket ==========================================
    # Stone upright ON the bed, 100 mm from the pocket center ON ITS OWN SIDE (the
    # decoy owns the other side — placing there interpenetrates and gets the probe
    # depenetration-shoved): outside the `near` latch (75 mm), still at bed height.
    # No `near`/`in` credit, no success; the decoy must not have been disturbed.
    env.reset(seed=100)
    _step(30)
    own_side = float(scene.stone_side[0])
    dy0 = float(scene._vault_local(scene.decoy.data.root_pos_w)[0, 1])
    _write_body(scene.stone,
                vault_world((c.chan_x, own_side * 0.10, c.bed_h + c.stone_r + 0.0005)),
                q_disc())
    _step(90)
    _report("near-miss-short")
    sl = stone_loc()
    dy1 = float(scene._vault_local(scene.decoy.data.root_pos_w)[0, 1])
    check("near-miss (short): stone on the bed 100 mm out — still short, on the bed, "
          "score <= 0.16, no success, decoy undisturbed",
          abs(float(sl[1])) > c.near_y and _stone_z() > 0.066
          and abs(dy1 - dy0) < 0.010
          and float(scene.score()[0]) <= 0.16 and not bool(scene.success()[0]))

    # ================= 8. negative: wrong object ==================================================
    # The RED decoy block seated in the pocket, the stone untouched: a tidy episode —
    # of the wrong object. Pocket readback confirms the decoy is in; score ~0.
    env.reset(seed=100)
    _step(30)
    _REC["on"] = True
    _write_body(scene.decoy, vault_world((c.chan_x, 0.0, c.decoy_s / 2 + 0.001)),
                vault_quat())
    _step(90)
    _report("wrong-object")
    dl = scene._vault_local(scene.decoy.data.root_pos_w)[0]
    dz = float(scene.decoy.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    print(f"[smoke] decoy in pocket: y={float(dl[1]) * 1000:+.0f}mm z={dz * 1000:.0f}mm",
          flush=True)
    check("negative (wrong object): RED decoy seated in the pocket, stone untouched — "
          "score <= 0.05, no success",
          abs(float(dl[1])) < 0.020 and dz < 0.050
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 9. negative: wrong orientation =============================================
    # Stone lying FLAT (cylinder axis vertical) bridging the two rails over the
    # pocket — the disc (120 mm) cannot fit down the 48 mm channel flat, so it rests
    # on the rail tops. The upright clause refuses; not seated, no success.
    env.reset(seed=100)
    _step(30)
    _write_body(scene.stone, vault_world((0.064, 0.0, c.rail_h + c.stone_t / 2 + 0.002)),
                vault_quat())
    _step(90)
    _report("flat-stone")
    ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
    ax_z = abs(float(quat_apply(scene.stone.data.root_quat_w, ez)[0, 2]))
    print(f"[smoke] flat stone: axis z-component={ax_z:.2f} z={_stone_z() * 1000:.0f}mm",
          flush=True)
    check("negative (orientation): stone lying FLAT across the rails over the pocket "
          "— upright clause refuses, not seated, no success",
          ax_z > 0.9 and not bool(scene._upright()[0])
          and not bool(scene.stone_seated()[0]) and not bool(scene.success()[0]))

    # ================= 10. near-miss: perched on the bed edge =====================================
    # Stone upright at 55 mm — inside the `near` latch but on the bed edge, above the
    # seated band (the tipping point is ~43.5 mm, so it stays perched). Not in, no
    # success, score capped by the missing `in` latch.
    env.reset(seed=100)
    _step(30)
    _REC["on"] = True
    own_side = float(scene.stone_side[0])
    _write_body(scene.stone,
                vault_world((c.chan_x, own_side * 0.055, c.bed_h + c.stone_r + 0.0005)),
                q_disc())
    _step(90)
    _report("perch")
    sl = stone_loc()
    check("near-miss (perch): stone on the bed edge at 55 mm — stays perched above "
          "the seated band, no `in` credit, no success",
          abs(float(sl[1])) > c.pos_tol and _stone_z() > 0.066
          and float(scene.score()[0]) <= 0.41 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 11. negative: right pose, wrong place ======================================
    # Stone upright on the GROUND in front of the channel at the pocket's track
    # coordinate: |y| passes, center height passes (ground rest ~ pocket-floor rest),
    # upright passes — the in-channel clause ALONE rejects it.
    env.reset(seed=100)
    _step(30)
    _write_body(scene.stone, vault_world((0.15, 0.0, c.stone_r + 0.0005)), q_disc())
    _step(45)
    _report("outside-channel")
    sl = stone_loc()
    in_y_band = abs(float(sl[1])) < c.pos_tol
    in_z_band = c.seat_z_min < _stone_z() < c.seat_z_max
    print(f"[smoke] outside-channel: x={float(sl[0]) * 1000:.0f}mm "
          f"(chan {c.chan_x * 1000:.0f}+/-{c.chan_x_tol * 1000:.0f}mm) "
          f"y_band={in_y_band} z_band={in_z_band} upright={bool(scene._upright()[0])}",
          flush=True)
    check("negative (outside channel): stone upright on the ground at the pocket's y "
          "— y/z/upright clauses all pass, the in-channel clause alone rejects",
          in_y_band and in_z_band and bool(scene._upright()[0])
          and not bool(scene.stone_seated()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.16)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.stone_door_vault")
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
