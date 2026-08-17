"""Smoke battery for EscrowClampScene (sim_gen task `handover_i339`) — REJECTION-ONLY:
every check either verifies basic health/randomization or CONSTRUCTS a settled wrong
outcome and asserts the rubric refuses it. success() must never fire anywhere in the
battery. All mechanism motion goes through the sanctioned drives (scene.jaw_drive /
scene.ledge_drive) — the same velocity-servo law the solve uses; physical probes
assert the actuator actually MOVED so no rejection is vacuous.

 1. settle/no-NaN     — fresh episode: jaw OPEN, ledge IN, both cylinders on the
                        apron, score ~0, no success.
 2. randomization A   — station yaw spans > 90 deg, xy jitters, the spool/decoy slot
                        permutation takes both values, jaw-opening and ledge-seat
                        bands both span, across 8 resets.
 3. randomization B   — READBACK: the spool stands in exactly the slot scene.swap
                        names and the decoy in the other; q_jaw()/q_ledge() agree
                        with the episode's q_jaw0/q_ledge0.
 4. null policy       — 240 idle steps -> score ~0, no success.
 5. SEED strategy     — the seed's whole skill (carry the object to the receiver and
                        let go): spool presented on the ledge, NOTHING else. Present
                        latch pays 0.25, never more; settled; no success.
 6. premature release — jaw OPEN, spool presented, then the ledge honestly pulled
                        OUT: no receiver holds the spool — it rides away with the
                        withdrawn shelf (slow pull; the audited shelf-clearance
                        puts it beyond the cell's |y|) or falls into the basin
                        (fast pull); score holds at 0.25; no success.
 7. clamp-first       — the jaw honestly driven CLOSED while EMPTY, then the spool
                        dropped at the capture spot: it PERCHES on the roofed prong
                        plates ABOVE the hang band (never enters the present box);
                        ledge then pulled OUT; score stays ~0 (empty close pays
                        nothing); no success.
 8. partial close     — spool presented, jaw driven only partway (q ~ 25-35 mm,
                        NOT closed), ledge pulled OUT: whether the spool half-hangs
                        at the channel entrance or slips into the basin, the jaw
                        clause refuses; clamp latch never fires; score <= 0.25.
 9. wrong object      — the DECOY presented at the capture spot, jaw driven closed
                        with the full servo: it shoves the decoy against the anvil
                        and STALLS far short of CLOSED (moved >= 20 mm from its
                        seat, yet q_jaw stays > closed + 20 mm); score ~0; no
                        success — the waistless package can never be handed over.
10. ledge-in near-miss— spool presented AND the jaw honestly clamped (both latches
                        pay, 0.55 cap) but the ledge NEVER pulled: jaw closed, spool
                        upright in the cell, everything settled — the ledge-out
                        clause ALONE refuses.
11. latched credit    — continuing: the clamped spool is STOLEN off the station;
                        score holds at the 0.55 cap, success never fires.
12. rejection audit   — success() observed False at every step of the battery.
13. final no-NaN.
14. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.handover_i339.smoke --headless
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

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
    loc = scene.spool_local()
    print(f"[smoke] {tag:18s} | q_jaw={1000 * float(scene.q_jaw()[0]):+6.1f}mm "
          f"q_ledge={1000 * float(scene.q_ledge()[0]):+6.1f}mm "
          f"spool=({float(loc[0, 0]):+.3f},{float(loc[0, 1]):+.3f},{float(loc[0, 2]):+.3f}) "
          f"up={bool(scene.spool_upright()[0])} cell={bool(scene.spool_in_cell()[0])} "
          f"latch(p)={int(scene._presented[0])} latch(c)={int(scene._clamped[0])} "
          f"settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.escrow_clamp")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.00, -0.85, 0.75)) + o),
                                tuple(np.array((0.00, 0.00, 0.15)) + o),
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

    def succ() -> bool:
        s = bool(scene.success()[0])
        _AUDIT["saw_success"] |= s
        return s

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    stand_z = c.ledge_anchor[2] + c.shelf_hz + c.spool_hh          # 0.176
    perch_z = c.jaw_anchor[2] + c.plate_hz + c.spool_hh            # 0.209

    def station_release(local_xyz, up: float) -> torch.Tensor:
        """World release point: live station-local point + `up` straight up."""
        loc = torch.tensor(local_xyz, device=device, dtype=torch.float).expand(n, 3)
        p = scene.station.data.root_pos_w \
            + quat_apply(scene.station.data.root_quat_w, loc)
        p = p.clone()
        p[:, 2] += up
        return p

    def present(body, hh: float) -> None:
        """CONSTRUCT (gravity + contact): stand `body` upright at the capture spot
        on the escrow ledge — released 25 mm up, station heading."""
        _write_body(body, station_release([c.present_xy[0], c.present_xy[1],
                                           c.ledge_anchor[2] + c.shelf_hz + hh], 0.025),
                    scene.station.data.root_quat_w)
        _step(240)

    def servo_jaw(v_des: float, q_cut: float, max_steps: int) -> None:
        """Sanctioned drive: velocity servo on the jaw slide until q <= q_cut.
        The jaw closes in -q, so K*(v_des + q̇) is the negative-feedback law."""
        for _ in range(max_steps):
            done = scene.q_jaw() <= q_cut
            if bool(done.all()):
                break
            scene.jaw_drive = torch.where(
                done, torch.zeros(n, device=device),
                15.0 * (v_des + scene.rate_jaw))
            _step(1)
        scene.jaw_drive = torch.zeros(n, device=device)
        _step(120)

    def servo_ledge_out(max_steps: int = 900) -> None:
        """Sanctioned drive: velocity servo on the ledge slide until q >= 0.100."""
        for _ in range(max_steps):
            done = scene.q_ledge() >= 0.100
            if bool(done.all()):
                break
            scene.ledge_drive = torch.where(
                done, torch.zeros(n, device=device),
                15.0 * (0.06 - scene.rate_ledge))
            _step(1)
        scene.ledge_drive = torch.zeros(n, device=device)
        _step(180)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(150)
    _report("reset")
    check("settle/no-NaN: fresh episode rests — jaw OPEN, ledge IN, cylinders on "
          "the apron, score ~0, no success",
          bool(scene._finite()[0]) and bool(scene.settled()[0])
          and float(scene.q_jaw()[0]) > c.jaw_closed_q + 0.05
          and float(scene.q_ledge()[0]) < c.ledge_out_q - 0.05
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2+3. randomization readback ================================================
    yaws, xys, swaps, qj0s, ql0s = [], [], [], [], []
    slot_ok, seat_ok = True, True
    for s in range(8):
        torch.manual_seed(20 + s)
        env.reset()
        _step(60)
        _refresh()
        yaws.append(yaw_of(scene.station.data.root_quat_w[0]))
        xys.append((scene.station.data.root_pos_w[0]
                    - scene.env_origins[0])[:2].tolist())
        swaps.append(bool(scene.swap[0]))
        qj0s.append(float(scene.q_jaw0[0]))
        ql0s.append(float(scene.q_ledge0[0]))
        # READBACK: the spool stands in exactly the slot `swap` names, decoy in the
        # other; the slide seats read back at their sampled values.
        for body, want in ((scene.spool, 1 if swaps[-1] else 0),
                           (scene.decoy, 0 if swaps[-1] else 1)):
            loc = scene._station_local(body.data.root_pos_w)[0]
            hit = -1
            for j, (sx, sy) in enumerate(c.slot_xy):
                if abs(float(loc[0]) - sx) < 0.06 and abs(float(loc[1]) - sy) < 0.06 \
                        and 0.0 < float(loc[2]) < 0.10:
                    hit = j
            slot_ok = slot_ok and (hit == want)
        seat_ok = seat_ok and abs(float(scene.q_jaw()[0]) - qj0s[-1]) < 0.006 \
            and abs(float(scene.q_ledge()[0]) - ql0s[-1]) < 0.006
    yspan = max(yaws) - min(yaws)
    xystd = float(np.std(np.asarray(xys), axis=0).mean())
    jspan = max(qj0s) - min(qj0s)
    lspan = max(ql0s) - min(ql0s)
    print(f"[smoke] readback: yaws={[f'{y:+.0f}' for y in yaws]} span={yspan:.0f} "
          f"xystd={xystd:.3f} swaps={[int(b) for b in swaps]} "
          f"jspan={1000 * jspan:.1f}mm lspan={1000 * lspan:.1f}mm "
          f"slot_ok={slot_ok} seat_ok={seat_ok}", flush=True)
    check("randomization A: station yaw spans > 90 deg, xy jitters, the slot "
          "permutation takes both values, jaw-opening and ledge-seat bands span",
          yspan > 90.0 and xystd > 0.008 and len(set(swaps)) == 2
          and jspan > 0.003 and lspan > 0.003)
    check("randomization B: spool/decoy stand in the slots scene.swap names; "
          "q_jaw()/q_ledge() agree with q_jaw0/q_ledge0",
          slot_ok and seat_ok)

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 5. SEED strategy (carry to the receiver and let go) =======================
    torch.manual_seed(41)
    env.reset()
    _step(60)
    present(scene.spool, c.spool_hh)
    _step(240)
    _report("seed-carry")
    s5 = float(scene.score()[0])
    check("SEED strategy: spool presented on the ledge and released — nothing "
          "else. Present latch pays 0.25, never more; settled; no success",
          bool(scene._presented[0]) and bool(scene.settled()[0])
          and 0.24 <= s5 <= 0.26 and not succ())

    # ================= 6. premature release (escrow order violated) ==============================
    # Continue from the presented state: jaw still OPEN — pull the ledge honestly.
    q_l_before = float(scene.q_ledge()[0])
    servo_ledge_out()
    _step(240)
    _report("premature")
    loc6 = scene.spool_local()
    s6 = float(scene.score()[0])
    # Two-branch containment: a slow pull friction-drags the spool OUT with the shelf
    # (|y| ends beyond the withdrawn shelf's audited clearance of the capture cell); a
    # fast pull leaves it behind to fall into the basin. Either way it leaves the cell
    # and no receiver holds it.
    out_of_cell6 = float(loc6[0, 2]) < 0.10 or abs(float(loc6[0, 1])) > c.hang_y_tol
    check("premature release: ledge pulled OUT with the jaw OPEN — the spool rides "
          "away with the shelf (or falls into the basin), out of the capture cell; "
          "score holds at 0.25; no success",
          float(scene.q_ledge()[0]) > q_l_before + 0.05  # the actuator moved
          and bool(scene.ledge_out()[0])
          and out_of_cell6 and not bool(scene.spool_in_cell()[0])
          and bool(scene.settled()[0])
          and 0.24 <= s6 <= 0.26 and not succ())

    # ================= 7. clamp-first blocked (roofed prongs) ====================================
    torch.manual_seed(51)
    env.reset()
    _step(60)
    q_j_before = float(scene.q_jaw()[0])
    servo_jaw(0.08, 0.004, 900)  # close the EMPTY jaw honestly
    assert float(scene.q_jaw()[0]) < q_j_before - 0.05, "jaw must have closed"
    # drop the spool at the capture spot — it can only PERCH on the closed plates
    _write_body(scene.spool,
                station_release([c.present_xy[0], c.present_xy[1], perch_z], 0.025),
                scene.station.data.root_quat_w)
    _step(240)
    servo_ledge_out()
    _step(240)
    _report("clamp-first")
    loc7 = scene.spool_local()
    s7 = float(scene.score()[0])
    check("clamp-first: jaw honestly closed EMPTY, spool dropped at the spot — it "
          "PERCHES on the roofed prongs ABOVE the band; ledge out changes nothing; "
          "empty close pays nothing; score ~0; no success",
          bool(scene.jaw_closed()[0]) and bool(scene.ledge_out()[0])
          and float(loc7[0, 2]) > c.hang_z_hi + 0.01
          and not bool(scene._presented[0]) and not bool(scene._clamped[0])
          and s7 <= 0.05 and not succ())

    # ================= 8. partial close (jaw not CLOSED) =========================================
    torch.manual_seed(61)
    env.reset()
    _step(60)
    present(scene.spool, c.spool_hh)
    servo_jaw(0.04, 0.036, 900)  # slow servo, cut early: ends q ~ 25-35 mm, NOT closed
    q8 = float(scene.q_jaw()[0])
    assert q8 > c.jaw_closed_q + 0.005, f"partial close must NOT reach CLOSED, q={q8}"
    servo_ledge_out()
    _step(300)
    _report("partial-close")
    s8 = float(scene.score()[0])
    check("partial close: jaw driven only partway, ledge pulled OUT — whether the "
          "spool half-hangs or slips into the basin, the jaw clause refuses; the "
          "clamp latch never fires; score <= 0.25",
          not bool(scene.jaw_closed()[0]) and bool(scene.ledge_out()[0])
          and not bool(scene._clamped[0])
          and s8 <= 0.26 and not succ())

    # ================= 9. wrong object (waistless decoy stalls the jaw) ==========================
    torch.manual_seed(71)
    env.reset()
    _step(60)
    present(scene.decoy, c.decoy_hh)
    q_j0 = float(scene.q_jaw()[0])
    servo_jaw(0.08, 0.004, 600)  # full closing servo — must STALL on the decoy body
    q9 = float(scene.q_jaw()[0])
    _report("wrong-object")
    s9 = float(scene.score()[0])
    check("wrong object: the DECOY presented, the full closing servo shoves it "
          "against the anvil and STALLS far short of CLOSED (moved >= 20 mm, yet "
          "q_jaw > closed + 20 mm); score ~0; no success",
          q9 < q_j0 - 0.020  # the jaw genuinely drove (not a vacuous probe)
          and q9 > c.jaw_closed_q + 0.020  # ... and stalled far short of CLOSED
          and not bool(scene.jaw_closed()[0])
          and s9 <= 0.05 and not succ())

    # ================= 10. ledge-in near-miss (every clause but one) =============================
    torch.manual_seed(81)
    env.reset()
    _step(60)
    present(scene.spool, c.spool_hh)
    servo_jaw(0.08, 0.004, 900)  # honest full clamp around the waist
    _step(240)
    _report("ledge-in")
    s10 = float(scene.score()[0])
    check("ledge-in near-miss: presented AND honestly clamped (both latches pay, "
          "0.55 cap) but the ledge never pulled — jaw closed, spool upright in the "
          "cell, settled; the ledge-out clause ALONE refuses",
          bool(scene.jaw_closed()[0]) and not bool(scene.ledge_out()[0])
          and bool(scene.spool_in_cell()[0]) and bool(scene.spool_upright()[0])
          and bool(scene.settled()[0])
          and bool(scene._presented[0]) and bool(scene._clamped[0])
          and 0.54 <= s10 <= c.score_cap + 1e-3 and not succ())

    # ================= 11. latched credit survives theft =========================================
    # Continue: steal the clamped spool clean off the station.
    _write_body(scene.spool,
                (scene.env_origins[0:1] + torch.tensor(
                    [0.9, 0.9, c.spool_hh + 0.002], device=device)).expand(n, 3))
    _step(240)
    _report("theft")
    s11 = float(scene.score()[0])
    check("latched credit: the clamped spool STOLEN off the station — score holds "
          "at the 0.55 cap, success never fires",
          not bool(scene.spool_in_cell()[0])
          and 0.54 <= s11 <= c.score_cap + 1e-3 and not succ())

    # ================= 12-14. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.escrow_clamp")
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
