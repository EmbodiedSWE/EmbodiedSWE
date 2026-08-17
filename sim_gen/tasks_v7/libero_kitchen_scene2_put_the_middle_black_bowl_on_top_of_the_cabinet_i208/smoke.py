"""Smoke / rubric-rejection battery for LetterboxCabinetScene (sim_gen task
`libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i208`) —
NullRobot, teleported/forced probe states (instrumentation, NOT a solution), RECORDED.

Battery:
  1-2. settle/no-NaN    — reset settles finite: both slabs on the seat, three bowls in a
                          row on the staging block, bin empty, score ~0, no latches;
  3. randomization      — READBACK across seeds: fixture root xy+yaw move (all
                          predicates fixture-frame), the TARGET BODY INDEX varies
                          (bowl->slot permutation is real), slab y, row y, bowl yaw move;
  4. null-policy-fails  — 240 idle steps -> score ~0, no success;
  5-7. oracle x3 seeds  — force-push the front slab: contact chain shunts the back slab
                          through the chute into the bin (0.25), then the front slab
                          (0.50); teleport the MIDDLE bowl above the port (zero credit),
                          free-fall through the port onto the cleared seat -> success
                          1.0; persists 240 steps;
  8. occupied forfeit   — after oracle 2, a DECOY teleported into the compartment flips
                          success OFF (score falls back to the 0.80 latch sum);
  9. discard forfeit    — the decoy moved into the BIN instead: success stays OFF;
  10. monotonicity      — ladder recorded during oracle 0: idle 0 < back slab chuted
                          (0.125) < back slab binned (0.25) < both out (0.50) < seated
                          (1.0); all partials < 1.0;
  11. negative (seed strategy) — with the slabs still in place, the seed's plan
                          "release the bowl above the cabinet": it falls through the
                          port and parks ON THE SLABS above the seat band (readback z)
                          -> live seat predicate FALSE, score 0;
  12. negative (teleport bypass) — slabs teleported straight into the bin + target
                          teleported onto the seat: live in-bin and seated predicates
                          TRUE (probes not vacuous) yet every latch refuses (no chute,
                          no port passage) -> score 0, no success;
  13. negative (mouth entry) — from a legitimately cleared seat (restored checkpoint,
                          0.50): the bowl enters through the FRONT MOUTH (drops inside
                          the mouth cut, pushed to the seat center by force) -> live
                          seated TRUE but the port-passage gate refuses the seated
                          latch -> score stays 0.50, no success;
  14. permanence + live bin residency — restored 0.50 checkpoint: a slab teleported
                          OUT of the bin keeps the latched 0.50; a legit port drop then
                          raises latched credit to 0.80, but success is REFUSED because
                          bin residency is judged live;
  15. negative (inverted) — restored 0.50 checkpoint: bowl placed UPSIDE-DOWN on the
                          seat (position readback inside the seat band — not vacuous)
                          -> live seat predicate refuses, score stays 0.50.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i208.smoke --headless
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
# RTX recipe: kit may mis-decode the driver version and silently reject RTX -> the
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
    from simgen_tasks.libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i208 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.letterbox_cabinet")().build(num_envs=args.num_envs,
                                                       device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(1, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.95, -1.65, 1.05)) + o),
                                tuple(np.array((-0.15, 0.0, 0.20)) + o),
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

    def settle_until(pred, max_steps: int = 600, poll: int = 10,
                     min_steps: int = 30) -> bool:
        """Poll `pred` while stepping. ALWAYS steps at least `min_steps` first: right
        after a teleport all velocities are zero, so settled()-style predicates are
        vacuously true before physics has run (and post_step latches never fire)."""
        step(min_steps)
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def sc() -> float:
        return float(scene.score()[0])

    def tgt():
        return scene.bowls[int(scene.target_idx[0])]

    def tgt_fix() -> torch.Tensor:
        return scene.target_fix()[0]

    def slug_x(i: int) -> float:
        return float(scene._slug_pos_fix()[0, i, 0])

    def report(tag: str) -> None:
        tf = tgt_fix()
        sp = scene._slug_pos_fix()[0]
        print(f"[smoke] {tag:16s} tgt={int(scene.target_idx[0])} "
              f"tgt=({tf[0]:+.3f},{tf[1]:+.3f},{tf[2]:+.3f}) "
              f"slugA=({sp[0, 0]:+.3f},{sp[0, 2]:+.3f}) "
              f"slugB=({sp[1, 0]:+.3f},{sp[1, 2]:+.3f}) "
              f"chute={scene._chute_ever[0].tolist()} "
              f"binned={scene._binned_ever[0].tolist()} "
              f"ported={bool(scene._ported_ever[0])} "
              f"seated={bool(scene._seated_ever[0])} "
              f"settled={bool(scene.settled()[0])} score={sc():.3f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul  # noqa: E402

    def fix_to_world(loc) -> torch.Tensor:
        p = quat_apply(scene.fixture.data.root_quat_w[0:1],
                       torch.tensor([loc], device=device))[0]
        return p + scene.fixture.data.root_pos_w[0]

    def fix_quat(inverted: bool = False) -> torch.Tensor:
        q = scene.fixture.data.root_quat_w[0:1]
        if inverted:
            q_roll = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)  # pi about x
            q = quat_mul(q, q_roll)
        return q[0]

    def tp(body, pos_w, quat_w) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        body.write_root_state_to_sim(st, all_ids)

    def v_along(body, axis: torch.Tensor) -> float:
        return float((body.data.root_lin_vel_w[0] * axis).sum())

    def clear_forces() -> None:
        for b in scene.slugs + scene.bowls:
            b.set_external_force_and_torque(zero3, zero3)

    def drive(body, axis: torch.Tensor, fmag: float, vmax: float, done,
              max_steps: int) -> bool:
        """Pulsed push along world `axis` with the pod-dependent frame probe
        (toggle raw-world <-> body-frame encoding if progress stalls)."""
        mode = 0
        last_probe = float((body.data.root_pos_w[0] * axis).sum())
        for i in range(max_steps):
            if done():
                clear_forces()
                return True
            f = fmag if v_along(body, axis) < vmax else 0.0
            fw = f * axis
            if mode == 1:
                fw = quat_apply_inverse(body.data.root_quat_w[0:1], fw.view(1, 3))[0]
            clear_forces()
            body.set_external_force_and_torque(fw.view(1, 1, 3), zero3)
            step(1)
            if (i + 1) % 40 == 0:
                cur = float((body.data.root_pos_w[0] * axis).sum())
                if cur - last_probe < 0.001:
                    mode ^= 1
                last_probe = cur
        clear_forces()
        return done()

    def evict_ax() -> torch.Tensor:
        return quat_apply(scene.fixture.data.root_quat_w[0:1],
                          torch.tensor([[-1.0, 0.0, 0.0]], device=device))[0]

    def evict(seed_tag: str) -> tuple[float, float, float, bool]:
        """Force-evict both slabs (chain push, then the pusher itself). Returns the
        score ladder (chuted-B, binned-B, both-out) and overall ok."""
        drive(scene.slugs[0], evict_ax(), 6.0, 0.08,
              lambda: bool(scene._chute_ever[0, 1]), 3600)
        s_chute = sc()  # 0.125: back slab inside the chute opening
        drive(scene.slugs[0], evict_ax(), 6.0, 0.08,
              lambda: slug_x(1) < -0.215, 1800)
        ok_b = settle_until(lambda: bool(scene._binned_ever[0, 1])
                            and bool(scene.slugs_in_bin()[0, 1])
                            and bool(scene.settled()[0]), max_steps=1200)
        s_b = sc()  # 0.250
        drive(scene.slugs[0], evict_ax(), 4.0, 0.08,
              lambda: slug_x(0) < -0.215, 3600)
        ok_a = settle_until(lambda: bool(scene._binned_ever[0].all())
                            and bool(scene.slugs_in_bin()[0].all())
                            and bool(scene.settled()[0]), max_steps=1200)
        s_a = sc()  # 0.500
        report(f"{seed_tag}-evicted")
        return s_chute, s_b, s_a, ok_b and ok_a

    def drop_target() -> bool:
        """Teleport the MIDDLE bowl above the port (transport, above the passage
        band), then hands off: free-fall through the port onto the seat."""
        s_before = sc()
        tp(tgt(), fix_to_world([0.03, 0.0, 0.44]), fix_quat())
        assert abs(sc() - s_before) < 1e-6, "transport teleport moved the score"
        return settle_until(lambda: bool(scene.success()[0]), max_steps=600)

    # =========================== 1-2. settle / no-NaN ============================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("settled")
    st0 = torch.cat([scene.fixture.data.root_state_w, scene.plate.data.root_state_w]
                    + [b.data.root_state_w for b in scene.bowls]
                    + [s.data.root_state_w for s in scene.slugs], dim=-1)
    bp = scene._bowl_pos_fix()[0]  # (3, 3)
    check("settle: states finite, both slabs at rest on the seat, three bowls in a row "
          "on the staging block, bin empty, everything settled",
          bool(torch.isfinite(st0).all())
          and bool(scene.slugs_in_comp()[0].all())
          and bool((bp[:, 0] > 0.35).all()) and bool((bp[:, 2] > 0.11).all())
          and bool((bp[:, 2] < 0.16).all())
          and not bool(scene.bowls_in_comp()[0].any())
          and not bool(scene.slugs_in_bin()[0].any())
          and not bool(scene.bowls_in_bin()[0].any())
          and bool(scene.settled()[0]))
    check("settle: score ~0 at reset, no success, no latches",
          sc() <= 0.005 and not bool(scene.success()[0])
          and not bool(scene._chute_ever[0].any())
          and not bool(scene._binned_ever[0].any())
          and not bool(scene._ported_ever[0]) and not bool(scene._seated_ever[0]))

    # =========================== 3. randomization is real ========================================
    reads = []
    for s in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(s)
        env.reset()
        step(4)
        fp = (scene.fixture.data.root_pos_w[0] - scene.env_origins[0])
        fq = scene.fixture.data.root_quat_w[0]
        fyaw = 2.0 * math.atan2(float(fq[3]), float(fq[0]))
        tl = tgt_fix()
        sp = scene._slug_pos_fix()[0]
        ex = quat_apply(scene.bowls[0].data.root_quat_w[0:1],
                        torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        byaw = math.atan2(float(ex[1]), float(ex[0]))
        reads.append((float(fp[0]), float(fp[1]), fyaw, float(scene.target_idx[0]),
                      float(sp[0, 1]), float(tl[1]), byaw))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (fix_x, fix_y, fix_yaw, target_idx, "
          f"slugA_y, tgt_row_y, bowl0_yaw):\n{np.round(arr, 3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    n_tgt = len(set(int(v) for v in arr[:, 3]))
    check("randomization: fixture root xy+yaw move, TARGET BODY INDEX varies (bowl->slot "
          "permutation), slab y, target row y and bowl yaw move (READBACK from sim)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.05
          and n_tgt >= 2 and spread[4] > 0.008 and spread[5] > 0.015
          and spread[6] > 0.5)

    # =========================== 4. null policy fails ============================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: score ~0, no success after 240 idle steps",
          sc() <= 0.02 and not bool(scene.success()[0]))

    # =========================== 5-7. oracle on 3 seeds ==========================================
    ladder = None
    chk = None
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        step(30)
        s_idle = sc()
        s_chute, s_b, s_a, ok_evict = evict(f"oracle{s}")
        if s == 0:
            ladder = (s_idle, s_chute, s_b, s_a)
        if s == 2:
            chk = scene.get_state(all_ids)  # cleared seat, 0.50 latched, bowls staged
        ok_drop = drop_target()
        step(240)  # persistence: success must not flicker off
        report(f"oracle{s}-final")
        check(f"oracle seed {s}: chain-push evicts both slabs through the chute into "
              f"the bin (0.25 -> 0.50), port drop seats the middle bowl = success, "
              f"score 1.0, persists 240 steps",
              s_idle <= 0.02 and ok_evict and 0.48 <= s_a <= 0.52 and ok_drop
              and bool(scene.success()[0]) and sc() == 1.0)

    # =========================== 8. occupied-compartment forfeit =================================
    # Continue from oracle 2's success state: teleport a DECOY into the compartment.
    decoy = scene.bowls[(int(scene.target_idx[0]) + 1) % 3]
    tp(decoy, fix_to_world([-0.10, 0.10, 0.30]), fix_quat())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    report("occupied")
    check("occupied forfeit: decoy bowl inside the compartment flips success OFF; "
          "score falls back to the 0.80 latch sum",
          bool(scene.decoy_in_comp()[0]) and not bool(scene.success()[0])
          and 0.78 <= sc() <= 0.82)

    # =========================== 9. discard forfeit ==============================================
    # Move the same decoy from the compartment into the BIN: still a forfeit.
    tp(decoy, fix_to_world([-0.52, 0.08, 0.10]), fix_quat())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    report("discarded")
    check("discard forfeit: decoy bowl thrown into the catch bin keeps success OFF "
          "(compartment now clear of decoys — the bin veto is doing the work)",
          bool(scene.decoy_in_bin()[0]) and not bool(scene.decoy_in_comp()[0])
          and not bool(scene.success()[0]) and 0.78 <= sc() <= 0.82)

    # =========================== 10. rubric monotonicity =========================================
    s_idle, s_chute, s_b, s_a = ladder
    print(f"[smoke] monotonicity ladder: idle={s_idle:.3f} chutedB={s_chute:.3f} "
          f"binnedB={s_b:.3f} both_out={s_a:.3f} seated=1.000", flush=True)
    check("monotonicity: idle < back slab chuted (0.125) < back slab binned (0.25) "
          "< both slabs out (0.50) < seated (1.0)",
          s_idle <= 0.02 and 0.115 <= s_chute <= 0.135 and 0.24 <= s_b <= 0.26
          and 0.48 <= s_a <= 0.52 and s_idle < s_chute < s_b < s_a < 1.0)
    check("monotonicity: partial states score < 1.0", max(s_idle, s_chute, s_b, s_a) < 1.0)

    # =========================== 11. negative: the seed's strategy ===============================
    # The seed's plan — release the bowl above the cabinet — with the slabs still in
    # place: the bowl falls through the port and parks ON THE SLABS, above the seat band.
    torch.manual_seed(51)
    env.reset()
    step(30)
    tp(tgt(), fix_to_world([0.03, 0.0, 0.44]), fix_quat())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    zf = float(tgt_fix()[2])
    report("on-slabs")
    check("negative (seed strategy): bowl released above the cabinet lands ON THE "
          "SLABS above the seat band (rest z readback) -> live seat predicate FALSE, "
          "no seated latch, score 0, no success",
          zf > float(c.seat_z_hi) and not bool(scene.target_seated()[0])
          and not bool(scene._seated_ever[0]) and sc() <= 1e-6
          and not bool(scene.success()[0]))

    # =========================== 12. negative: teleport bypass ===================================
    torch.manual_seed(61)
    env.reset()
    step(30)
    tp(scene.slugs[0], fix_to_world([-0.52, -0.08, 0.05]), fix_quat())
    tp(scene.slugs[1], fix_to_world([-0.68, 0.06, 0.05]), fix_quat())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    tp(tgt(), fix_to_world([0.03, 0.0, 0.31]), fix_quat())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    live_bin = bool(scene.slugs_in_bin()[0].all())
    live_seat = bool(scene.target_seated()[0])
    report("bypass")
    check("negative (teleport bypass): slabs placed straight in the bin + target "
          "placed on the seat — live in-bin and seated predicates TRUE (probes not "
          "vacuous) yet chute/bin/seated latches all refuse -> score 0, no success",
          live_bin and live_seat and not bool(scene._chute_ever[0].any())
          and not bool(scene._binned_ever[0].any()) and not bool(scene._seated_ever[0])
          and sc() <= 1e-6 and not bool(scene.success()[0]))

    # =========================== 13. negative: front-mouth entry =================================
    # Restore the legitimately cleared seat (0.50 latched). The bowl enters through the
    # FRONT MOUTH: set just inside the mouth at floor level (x=0.13, z=0.30 — outside
    # the port's x band AND below its z band, so nothing can fire the port latch),
    # then force-pushed to the seat center.
    scene.set_state(chk, all_ids)
    step(10)
    assert 0.48 <= sc() <= 0.52, f"checkpoint restore lost the 0.50 credit: {sc():.3f}"
    tp(tgt(), fix_to_world([0.13, 0.0, 0.30]), fix_quat())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    drive(tgt(), evict_ax(), 2.0, 0.08,
          lambda: abs(float(tgt_fix()[0]) - 0.03) < 0.03, 1500)
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    live_seat13 = bool(scene.target_seated()[0])
    report("mouth-entry")
    check("negative (mouth entry): bowl brought in through the FRONT MOUTH rests in "
          "the seat band (live predicate TRUE — probe not vacuous) yet the seated "
          "latch refuses without the port passage -> score stays 0.50, no success",
          live_seat13 and not bool(scene._ported_ever[0])
          and not bool(scene._seated_ever[0]) and 0.48 <= sc() <= 0.52
          and not bool(scene.success()[0]))

    # =========================== 14. permanence + live bin residency =============================
    scene.set_state(chk, all_ids)
    step(10)
    s_restored = sc()  # 0.50 latched
    tp(scene.slugs[1], fix_to_world([0.55, 0.45, 0.033]), fix_quat())  # out of the bin
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    s_after_removal = sc()
    ok_drop14 = drop_target()  # legit port drop: seated latch CAN fire (gates latched)
    settle_until(lambda: bool(scene._seated_ever[0]), max_steps=300)
    step(60)
    report("bin-residency")
    check("permanence + live bin residency: slab removed from the bin keeps the "
          "latched 0.50; a legit port drop raises latched credit to 0.80, but success "
          "is REFUSED because the slab is no longer in the bin",
          0.48 <= s_restored <= 0.52 and 0.48 <= s_after_removal <= 0.52
          and not ok_drop14 and bool(scene._seated_ever[0])
          and 0.78 <= sc() <= 0.82 and not bool(scene.success()[0]))

    # =========================== 15. negative: inverted on the seat ==============================
    scene.set_state(chk, all_ids)
    step(10)
    tp(tgt(), fix_to_world([0.03, 0.0, 0.30]), fix_quat(inverted=True))
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    loc = tgt_fix()
    in_band = (abs(float(loc[0]) - float(c.seat_cx)) < float(c.seat_xy_tol)
               and abs(float(loc[1])) < float(c.seat_xy_tol)
               and float(c.seat_z_lo) < float(loc[2]) < float(c.seat_z_hi))
    report("inverted")
    check("negative (inverted): bowl UPSIDE-DOWN on the seat (position readback IN "
          "the seat band — probe not vacuous) -> live seat predicate refuses, score "
          "stays 0.50, no success",
          in_band and not bool(scene.target_seated()[0])
          and not bool(scene._seated_ever[0]) and 0.48 <= sc() <= 0.52
          and not bool(scene.success()[0]))

    # =========================== save + verdict ==================================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.letterbox_cabinet")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, ok in checks:
            if not ok:
                print(f"[smoke]   FAILED: {nm}", flush=True)
    code = 0 if all_ok else 1
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()
