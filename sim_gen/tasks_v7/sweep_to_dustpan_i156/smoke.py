"""Smoke / rubric-REJECTION battery for LetterboxBinScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a real contact-driven posting
trajectory and the latched credit is monotone along it). This battery proves the rubric
REJECTS wrong outcomes and that the physical claims the task rests on are load-bearing:
the letterbox flap is the ONLY way in, it is genuinely one-way, ground-level sweeping
(the seed strategy) earns nothing, the roof blocks drop-ins, and blue contamination is
an irreversible cap. Every probe is CONSTRUCTED (teleport, real physics steps, judge) —
instrumentation, never a solution: no probe here reaches success().

Checks:
   1. settle/no-NaN      — seeded reset settles finite; flap hangs closed; nothing
                           inside; score 0, no success;
   2. randomization      — two seeded resets: cube positions/yaws and the red present
                           pattern differ (world-pose readback);
   3. red-count coverage — over 15 resets the sampled present-count k covers {1,2,3}
                           and every red index is both present and parked at least once;
   4. null-policy        — 300 idle steps: flap does not drift, nothing enters, score
                           ~0, no success;
   5. SEED STRATEGY      — the seed's plan (sweep debris along the ground into the
                           receptacle): a red cube is dragged quasi-statically straight
                           at the slot — it travels freely but ends at GROUND level
                           against the blank lower face, below the sill; never inside,
                           approach never latches, score ~0;
   6. drop-in blocked    — a red cube dropped from above the bin lands ON the solid
                           roof and stays out;
   7. FLAP ONE-WAY       — the same 1.0 N force that swings the flap far inward cannot
                           move it outward past the closed stop (joint upper limit);
                           released, it falls shut again;
   8. INTERIOR RETENTION — a red cube constructed inside (k>=2 episode) and shoved at
                           the slot from within at 3x its weight visibly moves but
                           cannot get out (sill 4 cube-halves above the floor);
   9. CONTAMINATION      — all present reds + one blue constructed inside: score capped
                           at contam_cap (< the 0.45 insert credit), no success; the
                           blue teleport-EXTRACTED afterwards: the latch holds — still
                           no success, still capped (posting is irreversible);
  10. doorway near-miss  — a red cube at rest ON the sill in the slot doorway, 2 mm
                           from the flap: approach credit only, NOT inside (it has not
                           fallen below the sill), no insert latch, no success;
  11. flap-held-open     — with the flap pressed open, `flap_closed` (a success
                           conjunct) is False; released, gravity re-closes it;
  12. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.sweep_to_dustpan_i156.smoke --headless
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


def _clear(body) -> None:
    zero = torch.zeros(_ENV.num_envs, 1, 3, device=_ENV.device)
    body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.letterbox_bin")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    bx, by = c.bin_pos

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.30, -0.70, 0.55)) + o),
                                tuple(np.array((bx, by, 0.12)) + o),
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

    def flap_deg() -> float:
        _refresh()
        return float(scene.flap_open_deg()[0])

    def loc_of(body) -> torch.Tensor:
        _refresh()
        return scene._bin_local(body.data.root_pos_w)[0]

    def bin_world(loc_xyz) -> torch.Tensor:
        """Bin-local point -> world (the bin is axis-aligned at a fixed pose)."""
        p = torch.tensor([bx + loc_xyz[0], by + loc_xyz[1], loc_xyz[2]],
                         device=device).expand(n, 3).clone()
        return p + env.iscene.env_origins

    def present() -> list[bool]:
        return scene.present_red[0].tolist()

    def first_present() -> int:
        return present().index(True)

    def red_body(i: int):
        return scene.reds[scene.RED_NAMES[i]]

    def report(tag: str) -> None:
        _refresh()
        print(f"[smoke] {tag:18s} | flap={flap_deg():+6.1f}deg present={present()} "
              f"inserted={scene._inserted[0].tolist()} appr={bool(scene._appr[0])} "
              f"contam={bool(scene._contam[0])} "
              f"n_red_in={int(scene.red_inside()[0].sum())} "
              f"n_blue_in={int(scene.blue_inside()[0].sum())} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(_REC['frames'])}", flush=True)

    def push(body, force_w, steps: int, speed_cap: float | None = None) -> None:
        """World-frame force at the CoM for `steps` steps (optionally quasi-static:
        force only while slower than `speed_cap`), then clear."""
        zero = torch.zeros(n, 1, 3, device=device)
        f = torch.tensor(force_w, device=device).view(1, 1, 3).expand(n, 1, 3).contiguous()
        for _ in range(steps):
            use = f
            if speed_cap is not None:
                v = float(body.data.root_lin_vel_w[0, :2].norm())
                use = f if v < speed_cap else zero
            body.set_external_force_and_torque(use, zero, env_ids=_all_ids(),
                                               is_global=True)
            _step(1)
        _clear(body)

    def reset_with_k(min_k: int, seed0: int) -> int:
        for s in range(seed0, seed0 + 20):
            torch.manual_seed(s)
            env.reset()
            _step(30)
            if int(scene.present_red[0].sum()) >= min_k:
                return s
        raise AssertionError(f"no seed in [{seed0},{seed0 + 20}) with k >= {min_k}")

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(180)
    report("settle")
    _REC["on"] = False
    check("settle/no-NaN: seeded reset settles finite, flap hangs closed "
          f"({flap_deg():+.2f} deg), nothing inside, score 0, no success",
          bool(scene._finite()[0]) and abs(flap_deg()) < 3.0
          and int(scene.red_inside()[0].sum()) == 0
          and int(scene.blue_inside()[0].sum()) == 0
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        pos = torch.stack([b.data.root_pos_w[0, :2].clone()
                           for b in list(scene.reds.values()) + list(scene.blues.values())])
        yaws = [math.degrees(2.0 * math.atan2(float(b.data.root_quat_w[0, 3]),
                                              float(b.data.root_quat_w[0, 0])))
                for b in scene.blues.values()]
        return pos, yaws, tuple(present())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_pos, a_yaw, a_pat = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_pos, b_yaw, b_pat = readback()
    d_pos = float((a_pos - b_pos).norm(dim=1).max())
    d_yaw = max(abs((a - b + 180.0) % 360.0 - 180.0) for a, b in zip(a_yaw, b_yaw))
    print(f"[smoke] randomization deltas: max cube xy {d_pos * 1000:.1f}mm, max blue "
          f"yaw {d_yaw:.1f}deg, present {a_pat} vs {b_pat}", flush=True)
    check("randomization-is-real: cube world positions and yaws differ across seeds "
          "(readback)", d_pos > 0.030 and d_yaw > 5.0)

    # ================= 3. red-count coverage ======================================================
    ks: set[int] = set()
    seen_present = [False] * c.n_red
    seen_absent = [False] * c.n_red
    for s in range(15):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        pat = present()
        ks.add(sum(pat))
        for i, p in enumerate(pat):
            seen_present[i] |= p
            seen_absent[i] |= not p
        # parked cubes really are in the depot (world readback, not the mask)
        for i, p in enumerate(pat):
            x = float((red_body(i).data.root_pos_w - env.iscene.env_origins)[0, 0])
            assert (x > 1.0) == (not p), f"present mask vs world pos mismatch (red_{i})"
    print(f"[smoke] over 15 resets: k values {sorted(ks)}, present coverage "
          f"{seen_present}, absent coverage {seen_absent}", flush=True)
    check("red-count coverage: the sampled present-count covers {1,2,3} and every red "
          "index is both present and parked at least once (depot readback agrees)",
          ks == {1, 2, 3} and all(seen_present) and all(seen_absent))

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    f0 = flap_deg()
    _step(300)
    report("null-policy")
    check("null-policy-fails: 300 idle steps, flap drift "
          f"{abs(flap_deg() - f0):.2f} deg, nothing enters, score ~0, no success",
          abs(flap_deg() - f0) < 2.0 and int(scene.red_inside()[0].sum()) == 0
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. SEED STRATEGY: ground sweep at the receptacle ===========================
    # The seed's whole plan: sweep debris ALONG THE GROUND into the receptacle mouth.
    # Construct it: a present red cube on the ground in front of the bin, dragged
    # quasi-statically (3x weight, speed-capped — sweeping, not hurling) straight at
    # the slot. The sill is 120 mm up with a blank face below: the cube travels freely,
    # passes under the staging ledge, and ends pressed against the lower front face at
    # ground level — never inside, never even inside the approach ball.
    s5 = reset_with_k(1, 500)
    i5 = first_present()
    print(f"[smoke] seed-strategy seed {s5}: dragging red_{i5}", flush=True)
    _REC["on"] = True
    _write_body(red_body(i5), bin_world((-0.55, 0.0, c.cube_size / 2 + 0.003)))
    _step(60)
    p0 = red_body(i5).data.root_pos_w[0].clone()
    push(red_body(i5), (3.0 * c.cube_mass * 9.81, 0.0, 0.0), 480, speed_cap=0.35)
    _step(120)
    report("seed-strategy")
    l5 = loc_of(red_body(i5))
    moved5 = float((red_body(i5).data.root_pos_w[0] - p0).norm())
    print(f"[smoke] ground drag: travelled {moved5 * 1000:.0f}mm, ends at bin-frame "
          f"x={float(l5[0]) * 1000:.0f}mm z={float(l5[2]) * 1000:.0f}mm "
          f"(sill {c.sill_z * 1000:.0f}mm)", flush=True)
    check("negative (SEED strategy): a quasi-static ground drag straight at the slot "
          "travels freely but ends at ground level against the blank lower face — "
          "never inside, approach never latches, score ~0, no success",
          moved5 > 0.100 and float(l5[2]) < 0.06 and float(l5[0]) < -c.half_x
          and int(scene.red_inside()[0].sum()) == 0 and not bool(scene._appr[0])
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6. drop-in blocked: the roof is load-bearing ===============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    i6 = first_present()
    _REC["on"] = True
    _write_body(red_body(i6), bin_world((0.0, 0.0, c.roof_z + 0.25)))
    _step(180)
    report("drop-in")
    l6 = loc_of(red_body(i6))
    print(f"[smoke] drop: cube rests at z_loc={float(l6[2]) * 1000:.0f}mm "
          f"(roof top {(c.roof_z + c.wall_t) * 1000:.0f}mm)", flush=True)
    check("negative (drop-in blocked): a red cube dropped from above the bin lands ON "
          "the solid roof and stays out — no insert credit, no success",
          float(l6[2]) > c.roof_z and int(scene.red_inside()[0].sum()) == 0
          and not bool(scene._inserted[0, i6]) and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 7. FLAP ONE-WAY: inward yields, outward is a wall ==========================
    # Same magnitude force both ways (1.0 N ~ 3.4x the flap's weight), applied at the
    # flap origin (the hinge line) — push the PLATE: apply at the CoM via the body API.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    zero = torch.zeros(n, 1, 3, device=device)
    fin_ = torch.zeros(n, 1, 3, device=device)
    fin_[:, 0, 0] = 1.0  # +x = inward
    max_in = 0.0
    for _ in range(120):
        scene.flap.set_external_force_and_torque(fin_, zero, env_ids=_all_ids(),
                                                 is_global=True)
        _step(1)
        max_in = max(max_in, flap_deg())
    _clear(scene.flap)
    _step(240)
    closed_after_in = abs(flap_deg()) < c.flap_closed_deg
    fout = torch.zeros(n, 1, 3, device=device)
    fout[:, 0, 0] = -1.0  # -x = outward
    min_out = 0.0
    for _ in range(120):
        scene.flap.set_external_force_and_torque(fout, zero, env_ids=_all_ids(),
                                                 is_global=True)
        _step(1)
        min_out = min(min_out, flap_deg())
    _clear(scene.flap)
    _step(240)
    report("flap-one-way")
    print(f"[smoke] flap 1.0 N: inward max {max_in:+.1f}deg (re-closed: "
          f"{closed_after_in}), outward min {min_out:+.1f}deg", flush=True)
    check("FLAP ONE-WAY: the same 1.0 N force swings the flap far inward (probe is "
          "live) but cannot move it outward past the closed stop; released, it falls "
          "shut again",
          max_in > 25.0 and closed_after_in and min_out > -3.0
          and abs(flap_deg()) < c.flap_closed_deg)
    _REC["on"] = False

    # ================= 8. INTERIOR RETENTION: nothing comes back out ==============================
    # k>=2 so a single red inside is NOT success. Construct it settled on the bin
    # floor, then shove it AT the slot from within at 3x its weight for 1.5 s: it must
    # visibly move (pressed to the front wall) yet stay inside — the sill is 4
    # cube-half-heights above the floor and the flap is out of reach.
    s8 = reset_with_k(2, 520)
    i8 = first_present()
    print(f"[smoke] retention seed {s8} (k={sum(present())}): red_{i8} inside", flush=True)
    _REC["on"] = True
    _write_body(red_body(i8), bin_world((0.02, 0.0, 0.045)))
    _step(90)
    l8a = loc_of(red_body(i8))
    assert bool(scene.red_inside()[0, i8]), f"retention construct not inside: {l8a}"
    p0 = red_body(i8).data.root_pos_w[0].clone()
    push(red_body(i8), (-3.0 * c.cube_mass * 9.81, 0.0, 0.0), 180)
    _step(120)
    report("retention")
    l8 = loc_of(red_body(i8))
    moved8 = float((red_body(i8).data.root_pos_w[0] - p0).norm())
    print(f"[smoke] interior shove: moved {moved8 * 1000:.0f}mm, ends at "
          f"x_loc={float(l8[0]) * 1000:.0f}mm z_loc={float(l8[2]) * 1000:.0f}mm — "
          f"still inside: {bool(scene.red_inside()[0, i8])}", flush=True)
    check("INTERIOR RETENTION: a red cube shoved at the slot from inside at 3x its "
          "weight visibly moves but cannot leave the bin (one-way); k>=2 so no "
          "success, score <= 0.6001",
          moved8 > 0.030 and bool(scene.red_inside()[0, i8])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.6001)
    _REC["on"] = False

    # ================= 9. CONTAMINATION: capped and irreversible ==================================
    # All present reds constructed inside (insert credit 0.45 would be earned) PLUS one
    # blue: the cap must pull the score DOWN to contam_cap. Then the blue is teleport-
    # EXTRACTED (physically impossible — only a probe can do this): the latch must
    # hold — still capped, still no success.
    s9 = reset_with_k(2, 540)
    k9 = sum(present())
    print(f"[smoke] contamination seed {s9} (k={k9})", flush=True)
    _REC["on"] = True
    spots = ((0.02, -0.06), (0.02, 0.0), (0.02, 0.06))
    for j, i in enumerate([i for i, p in enumerate(present()) if p]):
        _write_body(red_body(i), bin_world((spots[j][0], spots[j][1], 0.045)))
        _step(30)
    _write_body(scene.blues["blue_0"], bin_world((0.06, -0.03, 0.075)))
    _step(150)
    report("contaminated")
    s_cont = float(scene.score()[0])
    ok_cap = (int(scene.red_inside()[0].sum()) == k9 and bool(scene._contam[0])
              and s_cont <= c.contam_cap + 1e-3 and not bool(scene.success()[0]))
    # extract the blue (probe-only ability) — the latch must hold
    _write_body(scene.blues["blue_0"], bin_world((-0.70, 0.40, c.cube_size / 2 + 0.003)))
    _step(120)
    report("blue-extracted")
    check("CONTAMINATION: a posted blue caps the score at "
          f"{c.contam_cap} (< the earned 0.45 insert credit) and blocks success; "
          "teleport-extracting it does NOT lift the cap or grant success (latched, "
          "irreversible)",
          ok_cap and int(scene.blue_inside()[0].sum()) == 0 and bool(scene._contam[0])
          and float(scene.score()[0]) <= c.contam_cap + 1e-3
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 10. doorway near-miss: on the sill is not inside ===========================
    # Cube at rest ON the sill in the slot doorway, its leading face 2 mm from the
    # hanging flap — the classic "almost posted" state. It is at slot height, so the
    # below-the-sill clause refuses: approach credit only.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    i10 = first_present()
    _REC["on"] = True
    _write_body(red_body(i10), bin_world((-0.125, 0.0, c.sill_z + c.cube_size / 2 + 0.002)))
    _step(150)
    report("doorway")
    l10 = loc_of(red_body(i10))
    print(f"[smoke] doorway rest: x_loc={float(l10[0]) * 1000:.0f}mm "
          f"z_loc={float(l10[2]) * 1000:.0f}mm (inside needs z < "
          f"{c.inside_z_max * 1000:.0f}mm), flap {flap_deg():+.1f}deg", flush=True)
    check("near-miss (doorway): a red cube at rest ON the sill in the slot doorway "
          "earns approach credit only — not inside (has not fallen below the sill), "
          "no insert latch, no success, score <= 0.1501",
          abs(float(l10[2]) - (c.sill_z + c.cube_size / 2)) < 0.010
          and bool(scene._appr[0]) and not bool(scene.red_inside()[0, i10])
          and not bool(scene._inserted[0, i10]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.w_appr + 1e-3)
    _REC["on"] = False

    # ================= 11. flap-held-open: the closed-flap clause is live =========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    fin_ = torch.zeros(n, 1, 3, device=device)
    fin_[:, 0, 0] = 1.0
    for _ in range(120):
        scene.flap.set_external_force_and_torque(fin_, zero, env_ids=_all_ids(),
                                                 is_global=True)
        _step(1)
    held_deg = flap_deg()
    held_not_closed = not bool(scene.flap_closed()[0])
    _clear(scene.flap)
    _step(300)
    report("flap-held-open")
    check("flap-held-open: with the flap pressed open "
          f"({held_deg:+.1f} deg) the `flap_closed` success conjunct is False; "
          "released, gravity re-closes it "
          f"({flap_deg():+.1f} deg)",
          held_deg > 20.0 and held_not_closed and bool(scene.flap_closed()[0]))
    _REC["on"] = False

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.letterbox_bin")
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
    except Exception:  # noqa: BLE001 - Kit teardown hangs; die loudly instead
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(2)
