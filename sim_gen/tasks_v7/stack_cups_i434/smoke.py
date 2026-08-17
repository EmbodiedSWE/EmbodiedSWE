"""Smoke / rubric-REJECTION battery for ShroudPostsScene (sim_gen task
`stack_cups_i434`) — NullRobot, teleported probe states + real force probes,
RECORDED.

This is NOT a solution (solve.py — unshroud the nest big->mid->small with
regulated lifts, carry each cup to its size-matched post, seat it through contact
— is the acceptance evidence that the rubric ACCEPTS a correct outcome; it passes
on seeds 0/1). Every teleport here is instrumentation that CONSTRUCTS a wrong (or
partial) outcome and asserts the rubric REJECTS it; the two captivity probes use
REAL applied forces to prove the LIFO claim physically. No probe ever reaches
success(), and a final audit asserts exactly that.

   1-2. settle/no-NaN   — reset layout settles finite: three cups shrouded
                          concentrically mouth-down at storage, four stations on
                          distinct ring slots; score ~0;
   3-4. randomization   — READBACK over 8 seeded resets: the station-slot
                          permutation varies, the ring phase rotates and the
                          radial jitter is live; every reset lies sane;
   5.  null policy      — 240 idle steps -> score ~0, no success;
   6.  captivity (lift) — a REAL 1.5x-weight lift on the shrouded SMALL cup for
                          2 s: it rises only the ~20 mm ceiling headroom and jams
                          (free flight would exceed 0.5 m), the big cup never
                          moves — an inner cup cannot be pulled out the top;
   7.  captivity (shove)— a REAL 1.0 N lateral shove on the shrouded small cup
                          for 1 s: it cannot leave the nest (relative excursion
                          bounded by the mm wall gap), clear credit never fires —
                          an inner cup cannot be dragged out the side;
   8.  seed strategy    — the seed family's outcome (a stack/nest of cups piled
                          together) re-CONSTRUCTED on the open floor -> nothing
                          capped, no success, no cap credit: nests score nothing;
   9.  radius lockout   — the SMALL cup dropped concentric over the MID post (the
                          tightest 3 mm interference): the mouth cannot pass the
                          shaft, the cup perches high on the cone -> rejected;
  10. pigeonhole        — the BIG cup dropped over the SMALL post genuinely CAPS
                          it (loose fit counts, as described — rubric anchor);
                          but then the MID cup is locked out of the BIG post ->
                          that branch can never reach success;
  11. settle gate       — a genuinely capped pair with velocity injected (the
                          sustained-stillness counter resets) is NOT capped while
                          moving;
  12. latched credit    — teleporting that cup away leaves the cap credit latched
                          while capped() correctly drops with the physical state;
  13. rejection audit   — success() was never True at ANY judged point;
  14. final no-NaN      — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.stack_cups_i434.smoke --headless
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

import math
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

SIZES = scene_mod.SIZES

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.shroud_posts")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.35, -1.10, 0.90)) + o),
                                tuple(np.array((0.00, 0.00, 0.05)) + o),
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

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def wrench(body, f_w) -> None:
        """WORLD force -> body link frame (`is_global=True` drops torques on this
        stack — the house convention). Re-set every step while pushing; ZEROED
        after every probe (the wrench buffer survives env.reset)."""
        from isaaclab.utils.math import quat_apply_inverse

        f = torch.zeros(n, 3, device=device)
        f[:] = torch.tensor([float(v) for v in f_w], device=device)
        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f).unsqueeze(1),
            torch.zeros(n, 1, 3, device=device), env_ids=all_ids)

    def report(tag: str) -> None:
        s, ok = judge()
        cap = scene.capped()[0]
        up = scene.cup_up_z()[0]
        rim = scene.rim_height_over()[0]
        ad = scene.axis_dist()[0]
        cups = " ".join(
            f"{nm}:up={float(up[i]):+.2f} rim={float(rim[i, i]) * 1000:+.0f} "
            f"ax={float(ad[i, i]) * 1000:.0f}"
            for i, (nm, _) in enumerate(SIZES))
        caps = " ".join(f"{nm}={bool(cap[i])}" for i, (nm, _) in enumerate(SIZES))
        print(f"[smoke] {tag:16s} | {cups} | capped: {caps} | score={s:.3f} "
              f"success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, xyz, quat=None, vel=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor([float(v) for v in xyz], device=device)
        if quat is None:
            st[:, 3] = 1.0
        else:
            st[:, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        if vel is not None:
            st[:, 7:10] = torch.tensor([float(v) for v in vel], device=device)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def drop_over_post(i: int, j: int, settle_steps: int = 120) -> None:
        """Hover cup i concentric above post j's cone tip and let gravity do the
        rest (mouth 15 mm above the tip — nothing teleported into contact)."""
        p = rel(scene.posts[j])
        tip_z = c.pad_h + c.post_h[j]
        h = float(scene.heights[i])
        place(scene.cups[i], (float(p[0]), float(p[1]), tip_z + 0.015 + h / 2),
              settle_steps=settle_steps)

    def station_xy() -> list[tuple[float, float]]:
        return [(float(rel(b)[0]), float(rel(b)[1]))
                for b in [*scene.posts, scene.storage]]

    def layout_sane(tag: str) -> bool:
        """Reset honesty: the three cups shrouded concentrically mouth-down at the
        storage station; four stations on distinct ring slots."""
        stor = scene.storage_xy[0]
        cup_off = [float((rel(b)[:2] - stor).norm()) for b in scene.cups]
        cup_z = [float(rel(b)[2]) for b in scene.cups]
        up = scene.cup_up_z()[0]
        st = station_xy()
        d_min = min(math.dist(st[a], st[b])
                    for a in range(4) for b in range(a + 1, 4))
        stor_body = rel(scene.storage)[:2]
        ok = (all(v < 0.010 for v in cup_off)
              and all(float(u) > 0.95 for u in up)
              and all(abs(cup_z[i] - (c.pad_h + float(scene.heights[i]) / 2)) < 0.012
                      for i in range(3))
              and d_min > 0.15
              and float((stor_body - stor).norm()) < 0.005)
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}: cup_off={cup_off} "
                  f"cup_z={cup_z} d_min={d_min:.3f}", flush=True)
        return ok

    bodies = [*scene.cups, *scene.posts, scene.storage]

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; three cups shrouded concentric mouth-down at "
          "storage, four stations on distinct ring slots", fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        st = station_xy()
        ang = [math.atan2(y, x) for x, y in st]
        slot = round(((ang[0] - ang[3]) % (2 * math.pi)) / (math.pi / 2)) % 4
        rad_dev = [math.hypot(x, y) - c.ring_r for x, y in st]
        reads.append([ang[3], float(slot), max(abs(v) for v in rad_dev)])
    arr = np.array(reads)
    print("[smoke] randomization readback (storage_angle, small_post_rel_slot, "
          f"max_radial_dev):\n{np.round(arr, 3)}", flush=True)
    slots = {int(v) for v in arr[:, 1]}
    ang_spread = float(arr[:, 0].max() - arr[:, 0].min())
    rad_spread = float(arr[:, 2].max() - arr[:, 2].min())
    check("randomization: the station-slot permutation varies across seeds "
          f"(small-post slot relative to storage takes {len(slots)} distinct values "
          f"{sorted(slots)})", len(slots) >= 2)
    check("randomization: the ring phase rotates (storage angle spread "
          f"{ang_spread:.2f} rad) and radial jitter is live (spread "
          f"{rad_spread * 1000:.1f} mm); every reset sane",
          ang_spread > 1.0 and rad_spread > 0.002 and sane)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. captivity: vertical lift jams ===========================
    # REAL force probe: pull the shrouded SMALL cup UP with 1.5x its weight for 2 s.
    # Free, that force takes it > 0.5 m; shrouded, it rises only the ~20 mm ceiling
    # headroom, presses the mid cup's ceiling (net 0.4 N << the 1.9 N weight above)
    # and jams. Peak sampled DURING the push (gravity restores the state at
    # force-off). The big cup must not move: the nest is opened outermost-first.
    env.reset(seed=41)
    step(60)
    small, big = scene.cups[0], scene.cups[2]
    z0_small, z0_big = float(rel(small)[2]), float(rel(big)[2])
    f_up = 1.5 * c.cup_mass[0] * 9.81
    peak_small = peak_big = 0.0
    for _ in range(240):
        wrench(small, (0.0, 0.0, f_up))
        env.step(no_action)
        peak_small = max(peak_small, float(rel(small)[2]) - z0_small)
        peak_big = max(peak_big, float(rel(big)[2]) - z0_big)
    wrench(small, (0.0, 0.0, 0.0))
    step(60)
    report("lift-probe")
    s, ok = judge()
    head = c.depth[1] - (c.depth[0] + c.top_t)
    check("captivity (lift): 1.5x-weight pull on the shrouded small cup for 2 s "
          f"rises only {peak_small * 1000:.1f} mm (ceiling headroom {head * 1000:.0f} mm; "
          "free flight would exceed 500 mm) and jams on the mid cup's ceiling; the "
          f"big cup moved {peak_big * 1000:.1f} mm — an inner cup cannot be pulled "
          "out the top",
          0.010 < peak_small < 0.055 and peak_big < 0.010 and not ok)

    # =========================== 7. captivity: lateral shove is caged =======================
    # REAL force probe: shove the shrouded small cup sideways with 1.0 N (heavier
    # than any xy force solve.py ever applies) for 1 s. Its excursion RELATIVE to
    # the big cup is bounded by the mm-scale wall gap; clear credit never fires.
    xy0 = (rel(small)[:2] - rel(big)[:2]).clone()
    peak_relx = 0.0
    for _ in range(120):
        wrench(small, (1.0, 0.0, 0.0))
        env.step(no_action)
        peak_relx = max(peak_relx,
                        float(((rel(small)[:2] - rel(big)[:2]) - xy0).norm()))
    wrench(small, (0.0, 0.0, 0.0))
    step(60)
    report("shove-probe")
    s, ok = judge()
    check("captivity (shove): a 1.0 N lateral shove on the shrouded small cup for "
          f"1 s moves it only {peak_relx * 1000:.1f} mm relative to the big cup "
          "(caged by the mm wall gap); clear credit never fired — an inner cup "
          "cannot be dragged out the side",
          peak_relx < 0.020 and float(scene.clear_latch[0, 0]) == 0.0
          and s <= 0.02 and not ok)

    # =========================== 8. seed strategy: a nest scores nothing ====================
    # rlbench/stack_cups succeeds by piling/nesting the cups together. Re-CONSTRUCT
    # exactly that outcome on the open floor at the ring center (small, then mid
    # shrouded over it, then big — a genuine nest, settled). Nothing is capped, no
    # cap credit; only the latched clear credits (cups did leave storage) remain.
    env.reset(seed=51)
    step(30)
    place(scene.cups[0], (0.0, 0.0, float(scene.heights[0]) / 2 + 0.002),
          settle_steps=30)
    place(scene.cups[1], (0.0, 0.0, float(scene.heights[1]) / 2 + 0.030),
          settle_steps=45)
    place(scene.cups[2], (0.0, 0.0, float(scene.heights[2]) / 2 + 0.040),
          settle_steps=90)
    report("nest-on-floor")
    s, ok = judge()
    cup_off = [float(rel(b)[:2].norm()) for b in scene.cups]
    check("seed strategy: the cups re-NESTED on the open floor (genuine shroud "
          f"stack, concentric within {max(cup_off) * 1000:.0f} mm, settled) -> "
          "nothing capped, no cap credit, no success: piling/nesting cups is "
          "worth nothing here",
          not ok and not bool(scene.capped()[0].any())
          and float(scene.cap_latch[0].sum()) == 0.0
          and s <= 3 * c.clear_credit + 0.01)

    # =========================== 9. radius lockout: small over MID post =====================
    # The tightest interference (3 mm): the small cup's 19 mm mouth cannot pass the
    # mid post's 22 mm shaft. Dropped concentric, it perches high on the cone (or
    # topples) — the rim never reaches the pad band and the pairwise concentricity
    # threshold for an impossible fit is non-positive: rejected.
    env.reset(seed=61)
    step(30)
    drop_over_post(0, 1, settle_steps=150)
    report("lockout")
    s, ok = judge()
    rim01 = float(scene.rim_height_over()[0, 0, 1])
    check("radius lockout: small cup dropped concentric over the MID post cannot "
          f"enter (3 mm interference) — rim ends {rim01 * 1000:+.0f} mm over the pad "
          f"(band < {c.rim_high * 1000:.0f} mm), pair not capped, no success",
          not bool(scene.pair_capped()[0, 0, 1]) and not bool(scene.capped()[0, 1])
          and not ok)

    # =========================== 10. pigeonhole: wrong assignment dead-ends =================
    # The described semantics, both halves: (a) the BIG cup dropped over the SMALL
    # post genuinely CAPS it (loose fit counts — rubric anchor: capped CAN read
    # True); (b) but the MID cup is then locked out of the BIG post (3 mm
    # interference), so the wrong-assignment branch can never cap all three posts.
    env.reset(seed=71)
    step(30)
    drop_over_post(2, 0, settle_steps=150)  # big cup over small post: caps
    anchor = bool(scene.capped()[0, 0])
    report("big-on-small")
    drop_over_post(1, 2, settle_steps=150)  # mid cup over big post: locked out
    report("mid-on-big")
    s, ok = judge()
    rim12 = float(scene.rim_height_over()[0, 1, 2])
    check("pigeonhole: big cup DOES cap the small post (loose drop counts, capped="
          f"{anchor} — rubric anchor) but the mid cup is locked out of the big post "
          f"(rim {rim12 * 1000:+.0f} mm over pad) -> the wrong assignment can never "
          "reach success",
          anchor and not bool(scene.pair_capped()[0, 1, 2])
          and not bool(scene.capped()[0, 2]) and not ok)

    # =========================== 11. settle gate ============================================
    # Construct ONE genuinely capped pair (small on its own post), then inject
    # velocity: while it moves, the sustained-stillness counter is zero and
    # capped() must read False. (One capped post never makes success.)
    env.reset(seed=81)
    step(30)
    drop_over_post(0, 0, settle_steps=150)
    report("capped-anchor")
    was_capped = bool(scene.capped()[0, 0])
    cu = scene.cups[0]
    st = cu.data.root_state_w.clone()
    st[:, 7] = 0.25  # lateral kick, velocity written WITH the pose
    cu.write_root_state_to_sim(st, all_ids)
    step(2)
    report("kicked")
    v_now = float(cu.data.root_lin_vel_w[0].norm())
    moving_capped = bool(scene.capped()[0, 0])
    s, ok = judge()
    check("settle gate: a genuinely capped pair (rubric anchor: capped read True "
          f"once settled) with velocity injected ({v_now:.2f} m/s) is NOT capped "
          "while moving — sustained stillness is required",
          was_capped and not moving_capped and not ok)

    # =========================== 12. latched cap credit =====================================
    place(scene.cups[0], (0.0, 0.0, float(scene.heights[0]) / 2 + 0.003),
          settle_steps=60)
    report("moved-away")
    s_after, ok = judge()
    expect = c.clear_credit * float(scene.clear_latch[0].sum()) + c.cap_credit
    check("latched credit: teleporting the small cup off its post leaves the cap "
          f"credit latched (score={s_after:.3f} ~= {expect:.3f}) while capped() "
          "correctly drops with the physical state",
          abs(s_after - expect) < 0.01 and not bool(scene.capped()[0, 0]) and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.shroud_posts")
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)
