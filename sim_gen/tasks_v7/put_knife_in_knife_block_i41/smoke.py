"""Smoke battery for `kerf_chop` (put_knife_in_knife_block_i41).

Rubric REJECTION battery + mechanism probes. Checks, in order (the seam welds are
PhysX breakable joints and breaking is IRREVERSIBLE across resets, so every check
that needs intact welds runs BEFORE the deliberate chop; the post-break negatives
reuse the broken episode WITHOUT resetting):

  1.  settle/no-NaN      — layout settles finite: rod welded, spanning, outers seated.
  2.  geometry gate      — knife blade fits the kerf slot, cleaver blade does not (config).
  3.  randomization      — station yaw/xy, rod offset, knife pose differ across seeds.
  4.  tool slot swap     — the knife spawns on BOTH sides of the station over 10 resets.
  5.  null policy        — 240 idle steps: seams intact, score ~0, no success.
  6.  SEED strategy      — the seed task's plan ("put the knife into the block's slot"):
                           knife left resting in the kerf slot -> engaged latch only
                           (score 0.15), seams HOLD under the resting knife, no success.
  7.  wrong tool         — cleaver pressed down HARD over the slot: blade too thick,
                           stopped by the cover, seams intact, no success.
  8.  THE CHOP           — knife press ramp through the slot severs BOTH seams; the red
                           piece drops and settles in the well -> success, score 1.0.
  9.  severed-elsewhere  — (post-break) red piece parked on the ground outside: no success.
  10. severed-on-cover   — (post-break) red piece resting on the cover plate: no success.
  11. wrong piece        — (post-break) BEIGE piece in the well, red piece outside: no success.
  12. outer knocked off  — (post-break) red in the well but an outer off its anvil: no success.
  13. restore goal       — (post-break) proper end state re-constructed: success again.
  14. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.put_knife_in_knife_block_i41.smoke --headless
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

_qmul, _qinv, _qapply = task_scene._qmul, task_scene._qinv, task_scene._qapply

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


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    sep = scene.seam_sep()[0]
    tip = scene.blade_tip_local()[0]
    ml = scene._station_local(scene.mid.data.root_pos_w)[0]
    print(f"[smoke] {tag:16s} | tip=({float(tip[0]):+.3f},{float(tip[1]):+.3f},"
          f"{float(tip[2]):+.3f}) mid=({float(ml[0]):+.3f},{float(ml[1]):+.3f},"
          f"{float(ml[2]):+.3f}) sep=({float(sep[0]) * 1000:.1f},"
          f"{float(sep[1]) * 1000:.1f})mm eng={bool(scene._engaged[0])} "
          f"sevA={bool(scene._sev_a[0])} sevB={bool(scene._sev_b[0])} "
          f"well={bool(scene.mid_in_well()[0])} seated={bool(scene.outers_seated()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.kerf_chop")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.15, -0.60, 0.45)) + o),
                                tuple(np.array((0.45, 0.00, 0.08)) + o),
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

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def station_world(loc_xyz) -> torch.Tensor:
        """Station-local point -> world (per-env). Accepts tuple or (N,3) tensor."""
        _refresh()
        if isinstance(loc_xyz, torch.Tensor):
            loc = loc_xyz
        else:
            loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.station.data.root_pos_w + quat_apply(
            scene.station.data.root_quat_w, loc)

    def station_quat() -> torch.Tensor:
        _refresh()
        return scene.station.data.root_quat_w.clone()

    def seams_intact() -> bool:
        return float(scene.seam_sep()[0].max()) < 0.005

    def knife_slot_loc(tip_z: float) -> torch.Tensor:
        """Station-frame knife-origin location putting the blade tip at the rod
        centre-line (x = rod_off) at height tip_z."""
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = scene.rod_off - c.blade_len / 2
        loc[:, 2] = tip_z + c.blade_h
        return loc

    # held-knife emulation (identity force frame — probed before use in the chop)
    pid_int = torch.zeros(n, 3, device=device)
    enc_ref: dict = {"q": None}

    def enc(v_world: torch.Tensor) -> torch.Tensor:
        if enc_ref["q"] is None:
            return v_world
        return _qapply(_qmul(enc_ref["q"], _qinv(scene.knife.data.root_quat_w)),
                       v_world)

    def hold_wrench(fz: float, q_use: torch.Tensor,
                    xy_target: torch.Tensor | None = None) -> None:
        f_w = torch.zeros(n, 3, device=device)
        f_w[:, 2] = c.tool_mass * 9.81 - fz
        if xy_target is not None:
            dxy = xy_target - scene.knife.data.root_pos_w[:, :2]
            vxy = scene.knife.data.root_lin_vel_w[:, :2]
            f_w[:, :2] = (60.0 * dxy - 4.0 * vxy).clamp(-2.5, 2.5)
        q_err = _qmul(q_use, _qinv(scene.knife.data.root_quat_w))
        sgn = torch.where(q_err[:, 0:1] < 0, -torch.ones_like(q_err[:, 0:1]),
                          torch.ones_like(q_err[:, 0:1]))
        ax = q_err[:, 1:4] * sgn * 2.0
        w = scene.knife.data.root_ang_vel_w
        pid_int.add_(0.002 * ax).clamp_(-1.2, 1.2)
        t_w = 2.0 * ax - 0.15 * w + pid_int
        scene.knife.set_external_force_and_torque(
            enc(f_w).view(n, 1, 3), enc(t_w).view(n, 1, 3),
            env_ids=_all_ids(), is_global=True)
        _step(1)

    def clear_wrench() -> None:
        z = torch.zeros(n, 1, 3, device=device)
        scene.knife.set_external_force_and_torque(z, z, env_ids=_all_ids())

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(120)
    _report("settle")
    _REC["on"] = False
    check("settle/no-NaN: layout settles finite; rod welded (seams ~0), red middle "
          "spanning the well, outers seated, station upright; score ~0, no success",
          bool(scene._finite()[0]) and seams_intact()
          and not bool(scene.mid_in_well()[0]) and bool(scene.outers_seated()[0])
          and bool(scene.station_upright()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. geometry gate (tool discrimination is physical) =========================
    check("geometry gate: knife blade (4 mm) fits the 13 mm kerf slot with clearance; "
          "cleaver blade (20 mm) is thicker than the slot",
          c.knife_blade_t + 0.004 < 2 * c.slot_y_half
          and c.cleaver_blade_t > 2 * c.slot_y_half + 0.004)

    # ================= 3. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene.station.data.root_quat_w[0]),
                scene.station.data.root_pos_w[0, :2].clone(),
                float(scene.rod_off[0]),
                scene.knife.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.knife.data.root_quat_w[0]))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_sp, a_ro, a_kp, a_ky = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_sp, b_ro, b_kp, b_ky = readback()
    d_yaw, d_sp = dyaw(a_yaw, b_yaw), float((a_sp - b_sp).norm())
    d_ro, d_kp = abs(a_ro - b_ro), float((a_kp - b_kp).norm())
    print(f"[smoke] randomization deltas: station_yaw={d_yaw:.1f}deg "
          f"station_xy={d_sp * 1000:.1f}mm rod_off={d_ro * 1000:.1f}mm "
          f"knife_xy={d_kp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: station yaw/xy, rod offset and knife pose readback "
          "differ across seeds",
          d_yaw > 2.0 and d_sp > 0.003 and d_ro > 0.001 and d_kp > 0.005)

    # ================= 4. tool slot swap ==========================================================
    sides = set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        sides.add("+" if float(scene.knife_slot[0]) > 0 else "-")
    print(f"[smoke] over 10 resets: knife start side {sorted(sides)}", flush=True)
    check("slot swap: the knife spawns on BOTH sides of the station over 10 resets",
          sides == {"+", "-"})

    # ================= 5. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, seams intact, score ~0, no success",
          seams_intact() and float(scene.score()[0]) <= 0.05
          and not bool(scene.success()[0]))

    # ================= 6. negative: the SEED'S OWN STRATEGY =======================================
    # The seed task's plan is "pick up the knife and put it into the block's slot".
    # Here that end state — the knife left resting in the kerf slot — is worth only
    # the engaged latch (0.15). Its ~1.5 N resting weight is BELOW the seam
    # threshold: the welds must hold. No success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_body(scene.knife,
                station_world(knife_slot_loc(c.rod_top_z + 0.001)), station_quat())
    _step(240)
    _report("seed-strategy")
    _REC["on"] = False
    check("negative (SEED strategy): knife left resting in the kerf slot — engaged "
          "latch only (score 0.15), seams HOLD under its resting weight, no success",
          seams_intact() and abs(float(scene.score()[0]) - c.w_engaged) < 0.01
          and not bool(scene.mid_in_well()[0]) and not bool(scene.success()[0]))

    # ================= 7. negative: wrong tool (cleaver, pressed HARD) ============================
    # The cleaver in the same use pose over the slot, pressed with 30 N for 1.5 s:
    # its 20 mm blade cannot enter the 13 mm slot — the cover carries the whole
    # press and the seams never see it.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    cl_loc = torch.zeros(n, 3, device=device)
    cl_loc[:, 0] = scene.rod_off - c.blade_len / 2
    cl_loc[:, 2] = c.cover_top + 0.002 + c.blade_h
    _write_body(scene.cleaver, station_world(cl_loc), station_quat())
    _step(20)
    zero = torch.zeros(n, 1, 3, device=device)
    f_dn = torch.zeros(n, 1, 3, device=device)
    f_dn[:, 0, 2] = -30.0
    cl_tip_min = 1.0
    for _ in range(180):
        scene.cleaver.set_external_force_and_torque(f_dn, zero, env_ids=_all_ids(),
                                                    is_global=True)
        _step(1)
        _refresh()
        clt = scene._station_local(
            scene.cleaver.data.root_pos_w
            + quat_apply(scene.cleaver.data.root_quat_w,
                         torch.tensor([c.blade_len / 2, 0.0, -c.blade_h],
                                      device=device).expand(n, 3)))
        cl_tip_min = min(cl_tip_min, float(clt[0, 2]))
    scene.cleaver.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    _step(60)
    _report("wrong-tool")
    _REC["on"] = False
    print(f"[smoke] cleaver press: lowest blade-tip height {cl_tip_min * 1000:.1f} mm "
          f"(rod top {c.rod_top_z * 1000:.0f} mm — never reached)", flush=True)
    check("negative (wrong tool): cleaver pressed at 30 N over the slot — too thick "
          "to enter, tip stopped above the rod, seams intact, score ~0, no success",
          seams_intact() and cl_tip_min > c.rod_top_z + 0.004
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 8. THE CHOP (the demonstrated mechanism, end to end) =======================
    # Replicates the solution: stage the knife in the slot (transport through free
    # space), touch down, ramp a press through the blade until BOTH breakable seam
    # welds snap, retract, and let the red piece drop and settle in the well.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    q_use = station_quat()
    # probe the force frame at hover (identity expected on this pod)
    hover = station_world(knife_slot_loc(c.cover_top + 0.012))
    _write_body(scene.knife, hover, q_use)
    z0 = float(scene.knife.data.root_pos_w[0, 2])
    q_reset0 = scene.knife.data.root_quat_w.clone()
    for cand in (None, q_reset0):
        enc_ref["q"] = cand
        pid_int.zero_()
        for _ in range(12):
            hold_wrench(fz=-2.0, q_use=q_use)
        if float(scene.knife.data.root_pos_w[0, 2]) > z0 - 0.005:
            break
        _write_body(scene.knife, hover, q_use)
    clear_wrench()
    print(f"[smoke] force frame locked: "
          f"{'identity' if enc_ref['q'] is None else 'reset'}", flush=True)

    score_before = float(scene.score()[0])

    def press_at(tip_x_off: float, budget: int) -> bool:
        loc = knife_slot_loc(c.rod_top_z + 0.003)
        loc[:, 0] = (loc[:, 0] + tip_x_off).clamp(
            -(c.slot_x_half - c.blade_len / 2 - 0.002) - c.blade_len / 2 + 0.0,
            (c.slot_x_half - c.blade_len / 2 - 0.002) - c.blade_len / 2 + 0.0)
        pos_w = station_world(loc)
        _write_body(scene.knife, pos_w, q_use)
        pid_int.zero_()
        for _ in range(30):
            hold_wrench(fz=0.3, q_use=q_use, xy_target=pos_w[:, :2])
        fz = 8.0
        for i in range(budget):
            hold_wrench(fz=fz, q_use=q_use, xy_target=pos_w[:, :2])
            sev = scene.severed()[0]
            if bool(sev[0]) and bool(sev[1]):
                print(f"[smoke] chop: BOTH SEAMS SNAPPED at fz={fz:.0f}N", flush=True)
                return True
            if float((scene.knife.data.root_pos_w[0] - pos_w[0]).norm()) > 0.08:
                _write_body(scene.knife, pos_w, q_use)
                pid_int.zero_()
            if i % 60 == 59:
                fz = min(fz + 8.0, 40.0)
        return False

    chopped = press_at(0.0, 360)
    tries = 0
    while not chopped and tries < 3:
        tries += 1
        sev = scene.severed()[0]
        off = 0.0 if bool(sev[0]) == bool(sev[1]) else \
            (0.020 if bool(sev[1]) else -0.020)
        print(f"[smoke] chop retry #{tries} at tip_x_off={off:+.3f}", flush=True)
        chopped = press_at(off, 480)
    clear_wrench()
    # retract the knife (transport) and let everything settle
    _write_body(scene.knife, station_world(knife_slot_loc(c.cover_top + 0.10)), q_use)
    _step(5)
    park = scene.env_origins + torch.tensor([0.85, 0.35, 0.030], device=device)
    _write_body(scene.knife, park,
                torch.tensor([0.70711, 0.70711, 0.0, 0.0], device=device).expand(n, 4))
    for _ in range(12):
        _step(40)
        if bool(scene.success()[0]):
            break
    _step(60)
    _report("the-chop")
    _REC["on"] = False
    check("THE CHOP: knife press ramp severs BOTH seams; red piece drops and settles "
          "in the well; success, score 1.0 (and score never decreased)",
          chopped and bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 1.0) < 1e-6
          and float(scene.score()[0]) >= score_before)

    # ======== post-break constructions (NO resets from here: broken welds are permanent) ==========
    up_q = station_quat()

    def outer_home(body, sgn: float) -> None:
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = scene.rod_off + sgn * c.seg_len
        loc[:, 2] = c.rod_rest_z + 0.001
        _write_body(body, station_world(loc), up_q)

    # ---- 9. severed but parked OUTSIDE the station --------------------------------------------
    ground = scene.env_origins + torch.tensor([0.90, -0.40, c.rod_half + 0.001],
                                              device=device)
    _write_body(scene.mid, ground, up_q)
    _step(120)
    _report("severed-outside")
    check("negative (severed-elsewhere): seams severed but the red piece parked on "
          "the ground outside — sever credit only (score <= 0.65), no success",
          not bool(scene.mid_in_well()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.651)

    # ---- 10. severed but resting ON the cover plate -------------------------------------------
    cover_loc = torch.zeros(n, 3, device=device)
    cover_loc[:, 1] = c.slot_y_half + 0.030  # on a cover strip, clear of the slot
    cover_loc[:, 2] = c.cover_top + c.rod_half + 0.001
    _write_body(scene.mid, station_world(cover_loc), up_q)
    _step(120)
    _report("severed-on-cover")
    check("near-miss (severed-on-cover): red piece resting on the cover plate above "
          "the slot — not in the well, no success",
          not bool(scene.mid_in_well()[0]) and not bool(scene.success()[0]))

    # ---- 11. WRONG piece in the well ----------------------------------------------------------
    well_loc = torch.zeros(n, 3, device=device)
    well_loc[:, 2] = 0.010 + c.rod_half
    _write_body(scene.out_b, station_world(well_loc), up_q)  # beige end into the well
    # red piece parked outside (from check 10 it sits on the cover; move it out)
    _write_body(scene.mid, ground, up_q)
    _step(150)
    _report("wrong-piece")
    check("negative (wrong piece): a BEIGE end section dropped in the well, red piece "
          "elsewhere — outers not seated and red not in well, no success",
          not bool(scene.mid_in_well()[0]) and not bool(scene.outers_seated()[0])
          and not bool(scene.success()[0]))
    outer_home(scene.out_b, +1.0)  # put the beige end back on its anvil
    _step(90)

    # ---- 12. red in the well but an outer knocked OFF -----------------------------------------
    _write_body(scene.mid, station_world(well_loc), up_q)
    knock = scene.env_origins + torch.tensor([0.10, -0.45, c.rod_half + 0.001],
                                             device=device)
    _write_body(scene.out_a, knock, up_q)
    _step(150)
    _report("outer-knocked")
    check("negative (outer knocked off): red piece IS in the well but a beige end "
          "was knocked off its anvil — outers_seated refuses, no success",
          bool(scene.mid_in_well()[0]) and not bool(scene.outers_seated()[0])
          and not bool(scene.success()[0]))

    # ---- 13. restore the proper goal state -> success again -----------------------------------
    _REC["on"] = True
    outer_home(scene.out_a, -1.0)
    _step(200)
    _report("restore-goal")
    _REC["on"] = False
    check("restore goal: red piece in the well, both beige ends seated, station "
          "upright -> success, score 1.0",
          bool(scene.success()[0]) and abs(float(scene.score()[0]) - 1.0) < 1e-6)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.kerf_chop")
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
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(2)
