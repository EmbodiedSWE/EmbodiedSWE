"""Smoke battery for RelayCascadeScene (sim_gen task `franka_handover_i356`) —
REJECTION-ONLY: every check either verifies basic health/randomization or CONSTRUCTS
a settled wrong outcome and asserts the rubric refuses it. success() must never fire
anywhere in the battery.

 1. settle/no-NaN     — cubes on their apron slots, trays at their rest stops
                        (-8/+8 deg readback), score ~0, no success.
 2. randomization A   — housing yaw spans > 90 deg and xy jitters across resets.
 3. randomization B   — target COLOUR varies, cube->slot permutation varies, the
                        pedestal tile matches the target, and every cube settles on
                        the slot its permutation names (position READBACK).
 4. null policy       — 240 idle steps -> score ~0, no success.
 5. SEED strategy     — the seed's whole skill (carry the object to the
                        destination) executed as a construct: the target cube
                        released in free air over the sealed tower (over the roof
                        plate, NOT the intake mouth). It rests ON the roof — never
                        reaches any stage; score stays ~0.
 6. order interlock   — target dropped through the intake onto tray 1, then L2
                        pressed FIRST with real drive torque (actuation verified:
                        tray 2 demonstrably reached its -35 deg stop and
                        gravity-returned): the cube stays on tray 1 — pressing L2
                        early moves an empty tray; score stays at the intake latch
                        only; no success.
 7. empty presses     — fresh episode (cubes on the apron), L1 then L2 pressed to
                        their stops with verified actuation: the aboard-gated
                        latches pay ~0; no success.
 8. wrong cube        — the DECOY relayed through the full cascade (intake, L1,
                        L2, all real): decoy ends in the bin, target still on the
                        apron — the target-only latches pay ~0 and decoy-in-bin
                        refuses success.
 9. containment miss  — the target cube settled INSIDE the tower on the floor but
                        BEHIND the bin partition (wrong side): the bin box refuses;
                        score ~0; no success.
10. latched credit    — target dropped onto tray 1 (latch pays 0.20) then stolen
                        out to open ground: the latch survives, success does not.
11. rejection audit   — success() observed False at every step of the battery.
12. final no-NaN.
13. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.franka_handover_i356.smoke --headless
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
    ph = scene.phi()[0]
    tgt_in, dec_in = scene._cube_flags()
    tgt = int(scene.target_idx[0])
    loc = scene._housing_local(scene.cubes[tgt].data.root_pos_w)[0]
    print(f"[smoke] {tag:18s} | phi=({math.degrees(float(ph[0])):+6.1f},"
          f"{math.degrees(float(ph[1])):+6.1f})deg "
          f"cube_loc=({float(loc[0]):+.3f},{float(loc[1]):+.3f},{float(loc[2]):+.3f}) "
          f"tgt={tgt} tgt_in={bool(tgt_in[0])} dec_in={bool(dec_in[0])} "
          f"latch=({int(scene._intake[0])},{int(scene._stage[0])},{int(scene._bin[0])}) "
          f"settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.relay_cascade")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.10, -0.90, 0.85)) + o),
                                tuple(np.array((0.00, 0.05, 0.25)) + o),
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

    def housing_world(local_xyz) -> torch.Tensor:
        loc = torch.tensor(local_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.housing.data.root_pos_w \
            + quat_apply(scene.housing.data.root_quat_w, loc)

    def housing_local(pos_w: torch.Tensor) -> torch.Tensor:
        return scene._housing_local(pos_w)

    def tgt_k() -> int:
        return int(scene.target_idx[0])

    def drop_cube(k: int) -> None:
        """CONSTRUCT (transport-only): cube k released 3 cm above the intake mouth;
        the fall through the mouth and the cradle on tray 1 are gravity + contact."""
        my = (c.mouth_y0 + c.mouth_y1) / 2
        _write_body(scene.cubes[k],
                    housing_world([0.0, my, c.top_z1 + c.cube_size / 2 + 0.03]),
                    scene.housing.data.root_quat_w)
        _step(180)

    def press(k: int, sign: float, hold: int = 300) -> bool:
        """REAL actuation: the sanctioned clamped `lever_drive` torque drives tray k
        to its stop; peak angle is sampled DURING the press (gravity restores the
        tray at force-off), then release + gravity-return + settle. Returns True iff
        the tray demonstrably reached the stop AND returned to rest."""
        lim = math.radians((c.t1_lim, c.t2_lim)[k][1 if sign > 0 else 0])
        rest = math.radians((c.t1_rest, c.t2_rest)[k])
        peak = torch.full((n,), math.inf if sign < 0 else -math.inf, device=device)
        scene.lever_drive[:, k] = sign * c.tau_max
        for _ in range(hold):
            _step(1)
            ph = scene.phi()[:, k]
            peak = torch.minimum(peak, ph) if sign < 0 else torch.maximum(peak, ph)
        scene.lever_drive[:, k] = 0.0
        _step(240)
        reached = bool(((peak - lim).abs() < math.radians(4.0)).all())
        returned = bool(((scene.phi()[:, k] - rest).abs() < math.radians(4.0)).all())
        return reached and returned

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(150)
    _report("reset")
    ph1 = scene.phi()[0]
    check("settle/no-NaN: cubes on the apron, trays at their rest stops, score ~0, "
          "no success",
          bool(scene._finite()[0]) and bool(scene.settled()[0])
          and abs(math.degrees(float(ph1[0])) - c.t1_rest) < 3.0
          and abs(math.degrees(float(ph1[1])) - c.t2_rest) < 3.0
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2+3. randomization readback ================================================
    yaws, xys, tgts, perms = [], [], [], []
    tiles_ok, slots_ok = True, True
    for k in range(8):
        torch.manual_seed(20 + k)
        env.reset()
        _step(60)
        _refresh()
        yaws.append(yaw_of(scene.housing.data.root_quat_w[0]))
        xys.append((scene.housing.data.root_pos_w[0]
                    - scene.env_origins[0])[:2].tolist())
        tgts.append(tgt_k())
        perms.append(tuple(scene.slot_of[0].tolist()))
        # tile READBACK: the target's tile stands on the pedestal, the other doesn't
        px, py = c.ped_center
        ped_z = c.ap_z1 + c.ped_h
        for j in range(2):
            loc = housing_local(scene.tiles[j].data.root_pos_w)[0]
            on_ped = (abs(float(loc[0]) - px) < 0.03 and abs(float(loc[1]) - py) < 0.03
                      and abs(float(loc[2]) - ped_z) < 0.03)
            tiles_ok = tiles_ok and (on_ped == (j == tgts[-1]))
        # slot READBACK: every cube settled on the slot its permutation names
        for j in range(2):
            sx, sy = c.slot_xy[int(scene.slot_of[0, j])]
            loc = housing_local(scene.cubes[j].data.root_pos_w)[0]
            slots_ok = slots_ok \
                and abs(float(loc[0]) - sx) < 0.035 and abs(float(loc[1]) - sy) < 0.035
    yspan = max(yaws) - min(yaws)
    xystd = float(np.std(np.asarray(xys), axis=0).mean())
    print(f"[smoke] readback: yaws={[f'{y:+.0f}' for y in yaws]} span={yspan:.0f} "
          f"xystd={xystd:.3f} tgts={tgts} perms={len(set(perms))} "
          f"tiles_ok={tiles_ok} slots_ok={slots_ok}", flush=True)
    check("randomization A: housing yaw spans > 90 deg, xy jitters",
          yspan > 90.0 and xystd > 0.008)
    check("randomization B: target colour varies, slot permutation varies, the "
          "pedestal tile matches the target, cubes settle on their named slots",
          len(set(tgts)) >= 2 and len(set(perms)) >= 2 and tiles_ok and slots_ok)

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 5. SEED strategy (direct carry to the destination) =========================
    torch.manual_seed(41)
    env.reset()
    _step(120)
    # released over the SEALED part of the roof (y=0.22 is roof plate, not the mouth)
    _write_body(scene.cubes[tgt_k()],
                housing_world([0.0, 0.22, c.top_z1 + c.cube_size / 2 + 0.04]),
                scene.housing.data.root_quat_w)
    _step(300)
    _report("seed-carry")
    loc5 = housing_local(scene.cubes[tgt_k()].data.root_pos_w)[0]
    tgt_in5, _ = scene._cube_flags()
    check("SEED strategy: the cube carried straight to the tower rests ON the roof "
          "plate — reaches no stage; score stays ~0, no success",
          float(loc5[2]) > c.top_z0 + 0.010 and not bool(tgt_in5[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 6. order interlock (L2 first moves an empty tray) ==========================
    torch.manual_seed(51)
    env.reset()
    _step(120)
    drop_cube(tgt_k())
    _report("intake-first")
    on_t1 = bool(scene._boxes(scene.cubes[tgt_k()].data.root_pos_w)[0][0])
    ok6 = press(1, -1.0)  # L2 FIRST — real drive, actuation verified
    _report("L2-first")
    still_t1 = bool(scene._boxes(scene.cubes[tgt_k()].data.root_pos_w)[0][0])
    s6 = float(scene.score()[0])
    check("order interlock: L2 pressed first (tray 2 verifiably reached its stop "
          "and returned) moves an EMPTY tray — the cube stays on tray 1, no "
          "stage/bin latch, score stays at the intake latch; no success",
          on_t1 and ok6 and still_t1
          and not bool(scene._stage[0]) and not bool(scene._bin[0])
          and 0.19 <= s6 <= 0.21 and not succ())

    # ================= 7. empty presses ===========================================================
    torch.manual_seed(61)
    env.reset()
    _step(120)
    ok7a = press(0, +1.0)
    ok7b = press(1, -1.0)
    _report("empty-press")
    check("empty presses: L1 then L2 driven to their stops (verified) with no cube "
          "aboard — the aboard-gated latches pay ~0; no success",
          ok7a and ok7b and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 8. wrong cube ==============================================================
    torch.manual_seed(71)
    env.reset()
    _step(120)
    dec = 1 - tgt_k()
    drop_cube(dec)
    ok8a = press(0, +1.0)
    ok8b = press(1, -1.0)
    _step(120)
    _report("wrong-cube")
    tgt_in8, dec_in8 = scene._cube_flags()
    check("wrong cube: the DECOY relayed through the full cascade (real presses, "
          "verified) ends in the bin — target-only latches pay ~0 and "
          "decoy-in-bin refuses success",
          ok8a and ok8b and bool(dec_in8[0]) and not bool(tgt_in8[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 9. containment near-miss ===================================================
    torch.manual_seed(81)
    env.reset()
    _step(120)
    # settled on the tower floor but BEHIND the partition (wrong side of the bin);
    # free air at (0,-0.09,0.08): below tray 2's sweep (min z ~0.17), above the base
    _write_body(scene.cubes[tgt_k()],
                housing_world([0.0, -0.09, 0.08]),
                scene.housing.data.root_quat_w)
    _step(240)
    _report("containment-miss")
    loc9 = housing_local(scene.cubes[tgt_k()].data.root_pos_w)[0]
    tgt_in9, _ = scene._cube_flags()
    check("containment near-miss: target settled inside the tower on the floor "
          "BEHIND the bin partition — the bin box refuses; score ~0; no success",
          float(loc9[1]) < c.part_y0 and float(loc9[2]) < c.bin_z_max
          and not bool(tgt_in9[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 10. latched credit =========================================================
    torch.manual_seed(91)
    env.reset()
    _step(120)
    drop_cube(tgt_k())
    latched = bool(scene._intake[0])
    _write_body(scene.cubes[tgt_k()],
                (scene.env_origins[0:1]
                 + torch.tensor([0.9, 0.9, c.cube_size / 2 + 0.002], device=device)
                 ).expand(n, 3))
    _step(180)
    _report("latch-steal")
    s10 = float(scene.score()[0])
    on_t1_10 = bool(scene._boxes(scene.cubes[tgt_k()].data.root_pos_w)[0][0])
    check("latched credit: target dropped onto tray 1 (latch pays 0.20) then "
          "stolen out to open ground — the latch survives, success does not",
          latched and not on_t1_10 and 0.19 <= s10 <= 0.21 and not succ())

    # ================= 11-13. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.relay_cascade")
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
