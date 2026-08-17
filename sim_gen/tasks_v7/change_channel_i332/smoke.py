"""Smoke / rubric-REJECTION battery for ChannelFlipboardScene (sim_gen task
`change_channel_i332`) — NullRobot, teleported probe states + drive/force probes, RECORDED.

This is NOT a solution (solve.py — outer-first over-center flips through the live hinge
plant — is the acceptance evidence that the rubric ACCEPTS a correct outcome). Every
teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a judged
state and asserts the rubric REJECTS it — plus drive/force probes that prove the two
mechanical interlocks (outer-first drag lock, rider lock) are physically real. No probe
in this battery ever reaches success() at a judged point; a final audit asserts exactly
that.

  1-2. settle/no-NaN      — reset layout settles finite: cards resting ON their stops in
                            the start partition, target's guide tile in the frame, spare
                            tiles hidden in the cabinet, drive buffers zero; score ~0 at
                            rest, no success;
  3-4. randomization      — READBACK over 10 seeded resets: start jitter is physically
                            posed and varies (pre-settle angles); the (start, target)
                            sampler honors the reachability contract (t > s or t == 0),
                            BOTH flip directions occur, starts and targets vary, and the
                            guide tile matches the sampled target on every seed;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed strategy       — rlbench/change_channel's verb is grasp-the-remote + press-a-
                            button. No hand-held device or button exists here by
                            constructed premise; the closest transplant — a fingertip
                            PRESS on the visible front card's face — rocks the card
                            ~10 deg (non-vacuous) and gravity returns it to its stop:
                            channel unchanged, no credit, no success;
  7.  wrong channel       — every card flipped to the front (board settled, showing
                            channel 1, not the target): settled holds, partition does
                            not -> no success, no credit;
  8.  near-miss           — the single required mover left 5 deg SHORT of side_min on
                            its goal side (everything else perfect) -> no success (the
                            categorical-side threshold is real); removed before it can
                            fall home;
  9.  ordering interlock  — driving an INNER front card backward at the solve's own
                            torque cap (0.12 N*m): it lifts >= 5 deg (non-vacuous) then
                            STALLS far short of the apex — the outer card it drags never
                            leaves the front — so the outer-first order is physically
                            forced at fingertip force;
  10. rider lock          — a landed pile (card 1 propped with card 2 riding on it) is
                            constructed and holds; card 1 is then driven backward at
                            2.5x the solve cap for 900 steps: the rider is disturbed
                            (non-vacuous) but a CLEAN extraction (card 1 back while
                            card 2 stays front) NEVER occurs — the geometric rider lock
                            that justifies the reachable-target sampler;
  11. settle gate         — the goal partition constructed but one mover still spinning
                            (|w| > settle_omega) is NOT success (the settle gate is
                            real); partition broken before it can settle;
  12. part-way cap        — all required movers latched but the board NOT showing the
                            target: score == w_flip cap (0.55), NOT success — full
                            credit short of success is impossible;
  13. latched credit      — knocking a second latched mover back to the front leaves
                            the latched score unchanged, still no success;
  14. wrong object        — card 4 (innermost — never a required mover) flipped to the
                            back does not fake progress: no credit, no success;
  15. rejection audit     — success() was never True at ANY judged point;
  16. final no-NaN        — all task-object states finite at the end;
  17. camera              — >= 20 rgb frames captured -> frames.npz.

Run (forge): python -u -m simgen_tasks.change_channel_i332.smoke --headless
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

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)",
                                             flush=True), os._exit(3)))
_wd.daemon = True
_wd.start()

# Interlock probes (the honesty numbers): the solve's own fingertip-scale cap, and an
# elevated cap for the rider-lock probe (the lock is geometric — force cannot break it).
TAU_SOLVE = 0.12  # N*m — identical to solve.py's TAU_MAX
TAU_RIDER = 0.30  # N*m — 2.5x the solve cap; still no clean extraction
PILE = (124.0, 107.0, 84.6)  # measured landed-pile prop angles (cards 0, 1, 2)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.channel_flipboard")().build(num_envs=args.num_envs,
                                                       device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.45, -1.05, 0.90)) + o),
                                tuple(np.array((0.45, 0.0, 0.35)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)

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

    def report(tag: str) -> None:
        s, ok = judge()
        a = [f"{v:+6.1f}" for v in scene.angles_deg()[0].tolist()]
        print(f"[smoke] {tag:16s} | ang=({','.join(a)})deg "
              f"ch={int(scene.visible_channel()[0]) + 1} "
              f"s={int(scene.start_ch[0]) + 1} t={int(scene.target_ch[0]) + 1} "
              f"latch={scene.flip_latch[0].int().tolist()} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def ang_i(i: int) -> float:
        return float(scene.angles_deg()[0, i])

    def place_card(i: int, phi_deg: float, wy: float = 0.0,
                   settle_steps: int = 45) -> None:
        """Kinematic probe placement of card i at hinge angle phi (instrumentation, not
        a solution) + REAL physics steps before judging (the zero-step trap)."""
        phi = math.radians(phi_deg)
        pos, quat = c.card_pose(i, phi)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor(pos, device=device) + scene.env_origins
        st[:, 3:7] = torch.tensor(quat, device=device)
        if wy:
            # joint-consistent spin: pair the hinge rate with the CoM velocity
            # w x r (r = rc*e(phi)) or the constraint projection absorbs the spin
            rc = c.r_gap + c.card_l / 2
            st[:, 7] = wy * rc * math.cos(phi)
            st[:, 9] = -wy * rc * math.sin(phi)
            st[:, 11] = wy
        scene.cards[i].write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def bodies() -> tuple:
        return (scene.cabinet, *scene.cards, *scene.tiles)

    def layout_sane(tag: str) -> bool:
        """Reset honesty: all states finite, cards resting ON their stops in the start
        partition (cards < s back, the rest front), the TARGET's guide tile in the
        frame, spares hidden in the cabinet depot, drive buffers zero, no success."""
        fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies())
        ang = scene.angles_deg()[0]
        s0, t0 = int(scene.start_ch[0]), int(scene.target_ch[0])
        part = all((float(ang[i]) < -118.0) if i < s0 else (float(ang[i]) > 118.0)
                   for i in range(c.n_cards))
        tp = torch.tensor(c.tile_pos, device=device)
        d_tile = float((scene.tiles[t0].data.root_pos_w[0]
                        - scene.env_origins[0] - tp).norm())
        spares = True
        for i, tl in enumerate(scene.tiles):
            if i == t0:
                continue
            dp = torch.tensor((c.tile_depot[0], c.tile_depot[1] + 0.05 * i,
                               c.tile_depot[2]), device=device)
            spares = spares and \
                float((tl.data.root_pos_w[0] - scene.env_origins[0] - dp).norm()) < 0.02
        ok = (fin and part and d_tile < 0.02 and spares
              and float(scene.card_drive.abs().max()) == 0.0
              and float(scene.card_force.abs().max()) == 0.0
              and not bool(scene.success()[0]))
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}: s={s0} t={t0} "
                  f"ang={ang.tolist()} d_tile={d_tile:.3f} spares={spares}", flush=True)
        return ok

    def pick_seed(base: int, pred, tag: str) -> tuple[int, int]:
        """Deterministically scan seeds until the sampled episode matches `pred(s, t)`;
        leaves the env freshly reset + settled on that episode."""
        for sd in range(base, base + 40):
            env.reset(seed=sd)
            step(30)
            s0, t0 = int(scene.start_ch[0]), int(scene.target_ch[0])
            if pred(s0, t0):
                print(f"[smoke] {tag}: seed {sd} -> channel {s0 + 1} -> {t0 + 1}",
                      flush=True)
                return s0, t0
        raise RuntimeError(f"no seed matched for {tag}")

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    check("settle: all states finite; cards resting ON their stops in the start "
          "partition, target guide tile in the frame, spare tiles in the cabinet "
          "depot, drive buffers zero", layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    jit_abs = []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26, 27, 28, 29, 30):
        env.reset(seed=sd)
        step(1)  # refresh readback; jitter poses not yet settled onto the stops
        jit_abs.append(scene.angles_deg()[0].abs().tolist())
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        reads.append((int(scene.start_ch[0]), int(scene.target_ch[0]),
                      int(scene.k_req[0])))
    jit = np.array(jit_abs)  # (10, P) pre-settle |angles|
    jspread = (jit.max(axis=0) - jit.min(axis=0)).max()
    print(f"[smoke] randomization readback (s, t, k_req): {reads}\n"
          f"[smoke] pre-settle |angle| spread across seeds: {jspread:.2f} deg",
          flush=True)
    check("randomization: card start jitter is physically posed and varies "
          f"(pre-settle |angle| spread {jspread:.2f} deg across 10 seeds); layout "
          "sane on all seeds", jspread > 0.5 and sane)
    contract = all(t > s or t == 0 for s, t, _k in reads)
    fwd = any(t > s for s, t, _k in reads)
    back = any(t == 0 and s > 0 for s, t, _k in reads)
    starts = {s for s, _t, _k in reads}
    tgts = {t for _s, t, _k in reads}
    check("randomization: the (start, target) sampler honors the reachability "
          "contract (t > s or t == 0) on all 10 seeds, BOTH flip directions occur, "
          f"starts and targets vary (starts={sorted(x + 1 for x in starts)}, "
          f"targets={sorted(x + 1 for x in tgts)}); the guide tile matched the "
          "target every seed (layout sanity)",
          contract and fwd and back and len(starts) >= 2 and len(tgts) >= 2)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. seed strategy rejected ==================================
    # rlbench/change_channel's verb is grasp-the-remote + press-a-channel-button. No
    # hand-held device or pressable button exists here by constructed premise: the cards
    # are the channel indicator itself. The closest transplant — a fingertip PRESS on
    # the visible front card's face (0.45 N along the slab normal ~ the seed's button
    # press) — rocks the card toward the apex and gravity returns it to its stop.
    s0, t0 = pick_seed(41, lambda s, t: True, "press-probe")
    a0 = ang_i(s0)
    amin = a0
    scene.card_force[0, s0, 0] = -0.45  # body -x: into the front card's outward face
    for _i in range(180):
        step(1)
        amin = min(amin, ang_i(s0))
    scene.card_force[0, s0, 0] = 0.0
    step(90)
    report("press-probe")
    s, ok = judge()
    check("seed strategy: grasp+press is N/A by premise (no remote, no button); the "
          f"transplant press on the front card rocks it {a0 - amin:.1f} deg "
          "(>= 4, non-vacuous) but never near the apex, and it falls back onto its "
          f"stop ({ang_i(s0):+.1f} deg): channel unchanged, no credit, no success",
          a0 - amin >= 4.0 and amin > 60.0 and ang_i(s0) > 118.0
          and int(scene.visible_channel()[0]) == s0 and s <= 0.02 and not ok)

    # =========================== 7. wrong channel ===========================================
    s0, t0 = pick_seed(51, lambda s, t: s >= 1 and t > s, "wrong-channel")
    for i in range(s0):  # flip the whole back slope to the front -> board shows ch 1
        place_card(i, 124.0, settle_steps=0)
    step(60)
    report("wrong-channel")
    s, ok = judge()
    check("wrong channel: every card flipped to the front — the board settled showing "
          f"channel 1, target is channel {t0 + 1} — settled holds, the partition does "
          "not -> no success, no credit",
          int(scene.visible_channel()[0]) == 0
          and bool(scene.settled()[0]) and s <= 0.02 and not ok)

    # =========================== 8. near-miss ===============================================
    s0, t0 = pick_seed(61, lambda s, t: t == s + 1, "near-miss")
    # single required mover (card s0, front -> back); park it 5 deg SHORT of side_min
    # on the goal side, everything else already perfect
    place_card(s0, -(c.side_min_deg - 5.0), settle_steps=2)
    report("near-miss")
    a_now = ang_i(s0)
    s, ok = judge()
    miss_ok = (-c.side_min_deg < a_now < -10.0 and not ok)
    # restore it to the front BEFORE it can fall home into the goal (this battery must
    # never succeed at a judged point)
    place_card(s0, 124.0, settle_steps=45)
    check("near-miss: the single required mover parked "
          f"{c.side_min_deg - abs(a_now):.1f} deg short of side_min on its goal side "
          "(everything else perfect) is NOT success — the categorical-side threshold "
          "is real; restored before it could fall home", miss_ok)

    # =========================== 9. ordering interlock probe ================================
    # Drive an INNER front card backward at the solve's own cap: it must lift (the
    # probe is non-vacuous) then stall — dragging the outer card, not crossing.
    s0, t0 = pick_seed(71, lambda s, t: t > s and s <= 2, "ordering-interlock")
    inner = s0 + 1
    a_in0, a_out0 = ang_i(inner), ang_i(s0)
    min_in, min_out = a_in0, a_out0
    scene.card_drive[0, inner] = -TAU_SOLVE
    for _i in range(600):
        step(1)
        min_in = min(min_in, ang_i(inner))
        min_out = min(min_out, ang_i(s0))
    scene.card_drive[0, inner] = 0.0
    step(90)
    report("ordering-stall")
    s, ok = judge()
    check("ordering interlock: the inner front card driven backward at the solve's "
          f"own cap ({TAU_SOLVE} N*m) lifts {a_in0 - min_in:.1f} deg (>= 5, "
          f"non-vacuous) then STALLS at {min_in:.1f} deg — far short of the apex — "
          f"while the outer card it drags never leaves the front (min "
          f"{min_out:.1f} deg > 60); both re-settle front: outer-first order is "
          "physically forced, no credit, no success",
          a_in0 - min_in >= 5.0 and min_in > 30.0 and min_out > 60.0
          and ang_i(inner) > 100.0 and ang_i(s0) > 100.0 and s <= 0.02 and not ok)

    # =========================== 10. rider-lock probe =======================================
    # The geometric lock behind the reachable-target sampler: a card with another
    # PROPPED on it cannot be cleanly flipped back — the rider is carried. Construct a
    # landed pile (measured prop angles), verify it holds, then drive card 1 backward
    # at 2.5x the solve cap and assert a clean extraction never occurs.
    # t in {1, 2}: card 0 stays front (or the pile lands short of ANY prefix
    # partition), so no probe outcome can accidentally be the target partition
    s0, t0 = pick_seed(81, lambda s, t: t in (1, 2), "rider-lock")
    place_card(3, 124.0, settle_steps=0)
    place_card(0, PILE[0], settle_steps=0)
    place_card(1, PILE[1], settle_steps=0)
    place_card(2, PILE[2], settle_steps=0)
    step(120)
    report("pile-built")
    a1s, a2s = ang_i(1), ang_i(2)
    pile_ok = (all(ang_i(i) > 60.0 for i in range(4))
               and abs(a1s - PILE[1]) < 12.0 and abs(a2s - PILE[2]) < 12.0)
    clean = False
    rider_moved = 0.0
    min1 = a1s
    scene.card_drive[0, 1] = -TAU_RIDER
    for _i in range(900):
        step(1)
        a1, a2 = ang_i(1), ang_i(2)
        min1 = min(min1, a1)
        rider_moved = max(rider_moved, abs(a2 - a2s))
        clean = clean or (a1 < -60.0 and a2 > 60.0)
    scene.card_drive[0, 1] = 0.0
    step(120)
    report("rider-lock")
    s, ok = judge()
    check("rider lock: the landed pile (card 2 riding on card 1) holds "
          f"(pile_ok={pile_ok}); driving card 1 backward at {TAU_RIDER} N*m (2.5x "
          f"the solve cap) for 900 steps lifts it to {min1:.1f} deg and disturbs the "
          f"rider {rider_moved:.1f} deg (non-vacuous) but a CLEAN extraction (card 1 "
          "back while card 2 stays front) NEVER occurs — the sampler's reachability "
          "restriction is physically justified; no success",
          pile_ok and a1s - min1 >= 5.0 and rider_moved >= 5.0
          and not clean and not ok)

    # =========================== 11-13. settle gate + part-way cap + latched credit =========
    s0, t0 = pick_seed(91, lambda s, t: t - s >= 2, "settle-gate")
    spin_card = t0 - 1  # a required mover
    for i in range(c.n_cards):  # construct the goal partition ...
        place_card(i, -124.0 if i < t0 else 124.0,
                   wy=(2.5 if i == spin_card else 0.0), settle_steps=0)
    step(1)  # ... with one mover still spinning
    w_now = float(scene.cards[spin_card].data.root_ang_vel_w[0].norm())
    report("settle-gate")
    _s, ok = judge()
    gate_ok = (bool(scene.partition_ok()[0]) and w_now > c.settle_omega and not ok)
    # break the partition BEFORE it can settle into the goal (never succeed)
    place_card(s0, 124.0, settle_steps=45)
    check("settle gate: the goal partition constructed with one mover still spinning "
          f"(|w|={w_now:.2f} rad/s > {c.settle_omega}) is NOT success — the settle "
          "gate is real; partition broken before it could settle", gate_ok)
    report("part-way")
    s_before, ok = judge()
    latched = scene.flip_latch[0].int().tolist()
    frac_full = int(scene.flip_latch[0].sum()) == int(scene.k_req[0])
    check("part-way cap: all required movers latched during the construct "
          f"(latch={latched}, k_req={int(scene.k_req[0])}) but the board no longer "
          f"shows the target: score == w_flip cap ({s_before:.3f} ~ {c.w_flip}), NOT "
          "success — full credit short of success is impossible",
          frac_full and abs(s_before - c.w_flip) < 1e-4 and not ok)
    place_card(s0 + 1, 124.0, settle_steps=45)  # knock a second latched mover back
    report("knocked-off")
    s_after, ok = judge()
    check("latched credit: knocking a second latched mover back to the front leaves "
          f"the latched score unchanged ({s_before:.2f} -> {s_after:.2f}), still no "
          "success", abs(s_after - s_before) < 1e-3 and not ok)

    # =========================== 14. wrong object ===========================================
    s0, t0 = pick_seed(101, lambda s, t: True, "wrong-object")
    place_card(3, -124.0, settle_steps=45)  # card 4 is NEVER a required mover
    report("wrong-object")
    s, ok = judge()
    check("wrong object: card 4 (innermost — never a required mover, goal_sign 0) "
          "flipped to the back does not fake progress: no credit, no success",
          float(scene.goal_sign[0, 3]) == 0.0 and ang_i(3) < -100.0
          and s <= 0.02 and not ok)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies())
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== 17. camera + save + verdict ================================
    check(f"camera: >= 20 rgb frames captured ({len(frames)})", len(frames) >= 20)
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.channel_flipboard")
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
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
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
