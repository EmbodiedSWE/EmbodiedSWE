#!/usr/bin/env python3
"""Fine-grained rubrics for all 13 tasks, computed from SAVED env states (pure torch,
no simulator). Input: the dict `env.get_states()` produced (checkpoint nodes and the
grading pipeline's states_v2 both store it). Output: float in [0, 1], num_envs=1.

Design (per user spec): every task gets gradual credit —
  * grasp/hold a part:            +0.10          (when the scene stores grasp welds)
  * approach its goal:            +0.15 * (1 - lateral_dist / 15 cm)
  * insertion/seating:            +0.65 * depth_fraction   (gated by coarse alignment)
  * official seat/success gates:  -> 1.00 for that part
  multi-part tasks average their parts; staged tasks (coffee, spatula, pen_holder,
  tool_packing, syringe) interpolate CONTINUOUSLY between their official stage values,
  so the rubric equals the official score at every latch point.

All thresholds/geometry are the scenes' own cfg constants (quoted in eval/grader docs);
drawer rest offsets were extracted from the vendored USDs. The official success()
definition is never altered — rubric 1.0 coincides with official seated/success gates.
"""
from __future__ import annotations

import math

import torch


# ------------------------------------------------------------------ quat helpers (wxyz)
def _q_rot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    uv = torch.stack([y * v[..., 2] - z * v[..., 1],
                      z * v[..., 0] - x * v[..., 2],
                      x * v[..., 1] - y * v[..., 0]], dim=-1)
    uuv = torch.stack([y * uv[..., 2] - z * uv[..., 1],
                       z * uv[..., 0] - x * uv[..., 2],
                       x * uv[..., 1] - y * uv[..., 0]], dim=-1)
    return v + 2 * (w.unsqueeze(-1) * uv + uuv)


def _q_inv_rot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qc = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    return _q_rot(qc, v)


def _axis(q: torch.Tensor, i: int) -> torch.Tensor:
    e = torch.zeros(q.shape[:-1] + (3,), dtype=q.dtype)
    e[..., i] = 1.0
    return _q_rot(q, e)


def _clamp01(x) -> float:
    return float(max(0.0, min(1.0, x)))


APPROACH_REF = 0.05  # m; "approaching" credit ramps in over the last 5 cm ONLY — parts
# spawn 10-20 cm from their goals, so a wider band would hand out credit at reset


def _part(held: bool, lat: float, axis_cos: float, depth_frac: float, seated: bool,
          lat_gate: float, axis_deg: float, has_grasp: bool = True) -> float:
    """One part's staged credit: hold 0.10 / approach 0.15 / depth 0.65 / seat -> 1.0."""
    if seated:
        return 1.0
    w_hold = 0.10 if has_grasp else 0.0
    w_appr = 0.15 if has_grasp else 0.25
    credit = (w_hold if held else 0.0) + w_appr * (1.0 - min(lat / APPROACH_REF, 1.0))
    if lat <= 2.0 * lat_gate and axis_cos >= math.cos(math.radians(2.0 * axis_deg)):
        credit += 0.65 * _clamp01(depth_frac)
    return min(credit, 0.99)


def _rel_in(frame_p, frame_q, p) -> torch.Tensor:
    return _q_inv_rot(frame_q, p - frame_p)


def _scene(state: dict) -> dict:
    return state.get("scene", state)


# =============================================================== assembly: allen_bolt
def rubric_allen_bolt(state: dict) -> float:
    """Bolts spawn PRE-STAGED in their holes (tip depth 4.9 mm); the task is to screw
    them home with the key. Credit: key held 0.10 + 0.90 * screwing progress from the
    staged depth to the seat depth; official seat gates -> 1.0."""
    s = _scene(state)
    plat, bolts = s["platforms"][0], s["bolts"][0]          # (B,13)
    held_any = bool(s["grasp_held"][0].any()) if "grasp_held" in s else False
    STAGE, SEAT = 0.004898, 0.022
    vals = []
    for b in range(bolts.shape[0]):
        offs = torch.stack([_rel_in(plat[p, :3], plat[p, 3:7], bolts[b, :3])
                            for p in range(plat.shape[0])])
        lat, near = offs[:, :2].norm(dim=-1).min(dim=0)
        z = offs[near, 2]
        depth = 0.038 - float(z)                            # plate_top - z
        cos = float((_axis(bolts[b, 3:7], 2) * _axis(plat[near, 3:7], 2)).sum())
        seated = (depth >= SEAT) and (float(lat) <= 0.004) and cos >= math.cos(math.radians(5))
        if seated:
            vals.append(1.0)
            continue
        progress = _clamp01((depth - STAGE) / (SEAT - STAGE))
        vals.append(min((0.10 if held_any else 0.0) + 0.90 * progress, 0.99))
    return sum(vals) / len(vals)


# =============================================================== assembly: bulb
def rubric_bulb(state: dict) -> float:
    s = _scene(state)
    socks, bulbs = s["sockets"][0], s["bulbs"][0]
    vals = []
    for b in range(bulbs.shape[0]):
        offs = torch.stack([_rel_in(socks[p, :3], socks[p, 3:7], bulbs[b, :3])
                            for p in range(socks.shape[0])])
        lat, near = offs[:, :2].norm(dim=-1).min(dim=0)
        z = float(offs[near, 2])
        cos = float((_axis(bulbs[b, 3:7], 2) * _axis(socks[near, 3:7], 2)).sum())
        seated = (z <= 0.027) and (float(lat) <= 0.015) and cos >= math.cos(math.radians(12))
        # depth ramp from the socket opening (0.0385) down to the seat (0.027)
        depth_frac = (0.0385 - z) / (0.0385 - 0.027)
        vals.append(_part(False, float(lat), cos, depth_frac, seated,
                          0.015, 12.0, has_grasp=False))
    return sum(vals) / len(vals)


# =============================================================== assembly: ikea_table
IKEA_SLOTS = torch.tensor([(0.25, 0.25), (-0.25, 0.25), (0.25, -0.25), (-0.25, -0.25)])


def rubric_ikea_table(state: dict) -> float:
    s = _scene(state)
    table, legs = s["table"][0], s["legs"][0]
    welded = s.get("welded")
    welded = welded[0] if welded is not None else torch.zeros(legs.shape[0], dtype=torch.bool)
    vals = []
    for k in range(legs.shape[0]):
        if bool(welded[k]):
            vals.append(1.0)
            continue
        off = _rel_in(table[:3], table[3:7], legs[k, :3])
        lat = float((off[:2] - IKEA_SLOTS).norm(dim=-1).amin())
        z = float(off[2])
        cos = float((_axis(legs[k, 3:7], 2) * _axis(table[3:7], 2)).sum())
        seated = (z <= 0.012) and lat <= 0.02 and cos >= math.cos(math.radians(10))
        depth_frac = (0.026 - z) / (0.026 - 0.012)          # on-stud spawn -> seat
        vals.append(_part(False, lat, cos, depth_frac, seated, 0.02, 10.0,
                          has_grasp=False))
    return sum(vals) / len(vals)


# =============================================================== assembly: nut_thread
def rubric_nut_thread(state: dict) -> float:
    s = _scene(state)
    bolts, nuts = s["bolts"][0], s["nuts"][0]
    vals = []
    for b in range(nuts.shape[0]):
        offs = torch.stack([_rel_in(bolts[p, :3], bolts[p, 3:7], nuts[b, :3])
                            for p in range(bolts.shape[0])])
        lat, near = offs[:, :2].norm(dim=-1).min(dim=0)
        z = float(offs[near, 2])
        cos = float((_axis(nuts[b, 3:7], 2) * _axis(bolts[near, 3:7], 2)).sum())
        seated = (z <= 0.012) and (float(lat) <= 0.015) and cos >= math.cos(math.radians(10))
        depth_frac = (0.025 - z) / (0.025 - 0.012)          # bolt-top rest -> seat
        vals.append(_part(False, float(lat), cos, depth_frac, seated,
                          0.015, 10.0, has_grasp=False))
    return sum(vals) / len(vals)


# =============================================================== assembly: pc family
def _slot_part(case, part, seat_pos, mouth_z, seat_depth, lat_gate, axis_deg, yaw_deg,
               yaw_axis, held) -> float:
    rel = _rel_in(case[:3], case[3:7], part[:3]) - torch.tensor(seat_pos)
    lat = float(rel[:2].norm())
    engaged = (mouth_z - seat_pos[2]) - float(rel[2])
    up_cos = float((_axis(part[3:7], 2) * _axis(case[3:7], 2)).sum())
    yaw_cos = float((_axis(part[3:7], yaw_axis) * _axis(case[3:7], yaw_axis)).sum())
    seated = (engaged >= seat_depth and lat <= lat_gate
              and up_cos >= math.cos(math.radians(axis_deg))
              and yaw_cos >= math.cos(math.radians(yaw_deg)))
    return _part(held, lat, min(up_cos, yaw_cos), engaged / seat_depth, seated,
                 lat_gate, axis_deg, has_grasp=True)


def rubric_pc_gpu(state: dict) -> float:
    s = _scene(state)
    held = bool(s["grasp_held"][0, 0]) if "grasp_held" in s else False
    return _slot_part(s["case"][0], s["card"][0], (-0.01595, 0.0293, 0.0035), 0.0085,
                      0.004, 0.003, 3.0, 3.0, 0, held)


RAM_SEATS = ((-0.1426893, -0.0678899, 0.0002058), (-0.1237320, -0.0678899, 0.0002058))


def rubric_pc_ram(state: dict) -> float:
    s = _scene(state)
    vals = []
    for k in range(2):
        held = bool(s["grasp_held"][0, k]) if "grasp_held" in s else False
        vals.append(_slot_part(s["case"][0], s[f"ram_{k}"][0], RAM_SEATS[k], 0.0046456,
                               0.0037, 0.003, 6.0, 3.0, 1, held))
    return sum(vals) / len(vals)


def rubric_pc_gpu_ram(state: dict) -> float:
    s = _scene(state)
    held = (s["grasp_held"][0] if "grasp_held" in s
            else torch.zeros(3, dtype=torch.bool))
    vals = [_slot_part(s["case"][0], s["card"][0], (-0.01595, 0.0293, 0.0035), 0.0085,
                       0.004, 0.003, 3.0, 3.0, 0, bool(held[0]))]
    for k in range(2):
        vals.append(_slot_part(s["case"][0], s[f"ram_{k}"][0], RAM_SEATS[k], 0.0046456,
                               0.0037, 0.003, 6.0, 3.0, 1, bool(held[k + 1])))
    return sum(vals) / len(vals)


MOBO_HOLES = torch.tensor([
    (+0.0376, -0.1448), (-0.1650, -0.1444), (-0.1651, +0.0109), (+0.0614, +0.0108),
    (+0.0619, +0.1345), (-0.0931, +0.1347), (-0.1648, +0.1343)])


def rubric_pc_motherboard(state: dict) -> float:
    """Screws spawn PRE-STAGED (tip 6 mm below the board, screw mechanic); the task is
    driving each the remaining 5 mm with the key. Credit per screw: key held 0.10 +
    0.90 * screwing progress from the staged depth; official seat gates -> 1.0."""
    s = _scene(state)
    case, bolts = s["case"][0], s["bolts"][0]
    key_held = bool(s["grasp_held"][0, 0]) if "grasp_held" in s else False
    STAGE, SEAT = 0.006, 0.011
    vals = []
    for b in range(bolts.shape[0]):
        rel = _rel_in(case[:3], case[3:7], bolts[b, :3])
        lat = float((rel[:2] - MOBO_HOLES).norm(dim=-1).amin())
        depth = 0.0 - float(rel[2])                          # board_top - z (tip depth)
        cos = float((_axis(bolts[b, 3:7], 2) * _axis(case[3:7], 2)).sum())
        seated = depth >= SEAT and lat <= 0.003 and cos >= math.cos(math.radians(5))
        if seated:
            vals.append(1.0)
            continue
        progress = _clamp01((depth - STAGE) / (SEAT - STAGE))
        vals.append(min((0.10 if key_held else 0.0) + 0.90 * progress, 0.99))
    return sum(vals) / len(vals)


# =============================================================== coffee (staged + frac)
COFFEE_CM = (0.0, 0.20)
COFFEE_SURFACE_Z = 0.994
COFFEE_POCKET = (-0.007, -0.100)
COFFEE_SPOUT = (-0.006, -0.193)
COFFEE_TRAY = (0.42, -0.10)


def _coffee_stage_fracs(s: dict) -> list:
    bodies = s["bodies"]
    cm = torch.tensor([COFFEE_CM[0], COFFEE_CM[1], COFFEE_SURFACE_Z])
    pod, cup, cover = bodies["pod"][0], bodies["cup"][0], bodies["cover"][0]
    tray = bodies["tray"][0]
    machine = s["machine"]
    # [0] pod -> pocket (xy inf-norm toward 0.020 tol, z into pocket band)
    pocket = cm + torch.tensor([COFFEE_POCKET[0], COFFEE_POCKET[1], 0.0])
    d_pod = float((pod[:2] - pocket[:2]).abs().max())
    z_rel = float(pod[2] - COFFEE_SURFACE_Z)
    z_in = 1.0 if 0.345 < z_rel < 0.386 else _clamp01(1 - abs(z_rel - 0.3655) / 0.30)
    f0 = _clamp01(1 - max(d_pod - 0.020, 0.0) / APPROACH_REF) * z_in
    # [1] cover closed: slide from open (-0.088) to closed (-0.010)
    cover_pos = float(cover[1]) - COFFEE_CM[1]
    f1 = _clamp01((cover_pos - (-0.088)) / ((-0.010) - (-0.088)))
    # [2] cup under spout
    spout = cm[:2] + torch.tensor(COFFEE_SPOUT)
    f2 = _clamp01(1 - max(float((cup[:2] - spout).norm()) - 0.050, 0.0) / APPROACH_REF)
    # [3] started / [4] brewed: brew timer progress
    running = bool(machine["_running"][0])
    timer = float(machine["_timer"][0])
    f3 = 1.0 if running or bool(machine["_filled"][0]) else 0.0
    f4 = _clamp01(1 - timer / 600.0) if running else (1.0 if bool(machine["_filled"][0]) else 0.0)
    # [5] cup out of the spout zone
    f5 = _clamp01((float((cup[:2] - spout).norm()) - 0.050) / 0.10)
    # [6] served: cup -> tray
    f6 = _clamp01(1 - max(float((cup[:2] - tray[:2]).norm()) - (0.205 - 0.046), 0.0)
                  / APPROACH_REF)
    return [f0, f1, f2, f3, f4, f5, f6]


def rubric_coffee(state: dict) -> float:
    s = _scene(state)
    flags = s["machine"]["_flags"][0]
    n = int(flags.long().sum())
    if n >= 7:
        return 1.0
    fr = _coffee_stage_fracs(s)
    # credit latched stages fully + continuous progress on the first unlatched one
    next_i = next((i for i in range(7) if not bool(flags[i])), None)
    extra = fr[next_i] if next_i is not None else 0.0
    return min((n + extra) / 7.0, 0.999)


# =============================================================== spatula (prefix stages)
SPAT_TABLE = {  # goal -> (latch keys, official stage values / 100)
    "serve": (("_lifted", "_wedged", "_loaded", "_served"), (0.10, 0.30, 0.60, 1.00)),
    "flip": (("_lifted", "_wedged", "_flipped"), (0.10, 0.30, 1.00)),
    "flip_serve": (("_lifted", "_wedged", "_flipped", "_loaded_pf", "_served"),
                   (0.10, 0.25, 0.50, 0.70, 1.00)),
}
SPAT_SURFACE_Z = 0.994


def rubric_spatula(state: dict, goal: str = "flip_serve") -> float:
    s = _scene(state)
    m = s["machine"]
    keys, vals = SPAT_TABLE[goal]
    flags = [bool(m[k][0]) for k in keys]
    k = 0
    while k < len(flags) and flags[k]:
        k += 1
    if k >= len(flags):
        return 1.0
    lo = vals[k - 1] if k > 0 else 0.0
    hi = vals[k]
    # continuous progress toward the next latch
    bodies = s["bodies"]
    present = m["_present"][0]
    bi = int(present.long().argmax())
    bread = bodies[["bread_s", "bread_m", "bread_l"][bi]][0]
    spat = bodies["spatula"][0]
    nxt = keys[k]
    if nxt == "_lifted":
        frac = _clamp01((float(spat[2]) - SPAT_SURFACE_Z) / 0.05)
    elif nxt == "_wedged":
        d = float((bread[:2] - spat[:2]).norm())
        frac = _clamp01(1 - d / APPROACH_REF)
    elif nxt == "_flipped":
        up_z = float(_axis(bread[3:7], 2)[2])
        frac = _clamp01((1.0 - up_z) / (1.0 + math.cos(math.radians(30))))
    elif nxt in ("_loaded", "_loaded_pf"):
        d = float((bread[:2] - spat[:2]).norm())
        h = _clamp01((float(bread[2]) - SPAT_SURFACE_Z) / 0.05)
        frac = 0.5 * _clamp01(1 - d / APPROACH_REF) + 0.5 * h
    else:  # _served
        plate = bodies["plate"][0]
        frac = _clamp01(1 - float((bread[:2] - plate[:2]).norm()) / APPROACH_REF)
    return min(lo + (hi - lo) * frac, 0.999)


# =============================================================== syringe (ledger-based)
def rubric_syringe(state: dict) -> float:
    s = _scene(state)
    dose = s["dose"]
    drawn = bool(dose["_drawn_ok"][0])
    liquid = float(dose["_liquid"][0])
    draw_frac = 1.0 if drawn else _clamp01(liquid / 0.95)
    # early-stage credit (a 4-hour run that picked the syringe and seated the tip on the
    # reservoir but never completed a draw must not read 0): barrel upright + tip near
    # the reservoir mouth. Both are geometry over the stored state; at spawn the syringe
    # lies flat at its home spot, so both gates read 0 at reset.
    sy_root = s["syringe"]["root"][0]
    axis = _axis(sy_root[3:7], 2)
    upright = _clamp01((float(axis[2]) - 0.35) / (0.85 - 0.35))
    res = s["bodies"]["reservoir"][0]
    tip = sy_root[:3] - axis * (0.2222 / 2 + 0.0288)
    d_res = float((tip[:2] - res[:2]).norm())
    seat_frac = _clamp01(1 - d_res / APPROACH_REF) * upright
    early = 0.08 * upright + 0.12 * seat_frac
    doses = dose["_doses"][0]
    tube_fracs = []
    for i in range(doses.shape[0]):
        d = float(doses[i])
        if 0.23 <= d <= 0.43:
            tube_fracs.append(1.0)
        elif d < 0.23:
            tube_fracs.append(_clamp01(d / 0.23))
        else:
            tube_fracs.append(_clamp01(1 - (d - 0.43) / 0.10))
    dose_frac = sum(tube_fracs) / len(tube_fracs)
    # park: barrel near home shelf (official radius 0.06), lying, settled
    sy = s["syringe"]
    p = sy["root"][0]
    d_home = float((p[:2] - torch.tensor((-0.06, -0.08))).norm())
    lying = abs(float(_axis(p[3:7], 2)[2])) < 0.35
    settled = float(p[7:10].norm()) < 0.05
    parked = d_home < 0.06 and lying and settled
    park_frac = 1.0 if parked else \
        _clamp01(1 - max(d_home - 0.06, 0.0) / APPROACH_REF) * (1.0 if lying else 0.5)
    doses_ok = all(0.23 <= float(doses[i]) <= 0.43 for i in range(doses.shape[0]))
    if drawn and doses_ok and parked:
        return 1.0
    # parking is the LAST step and the syringe SPAWNS at the park spot: park credit only
    # counts once the doses are delivered, else it is free at reset
    park_credit = 0.15 * park_frac if doses_ok else 0.0
    # early credit fades as real progress takes over (max, not sum, vs the draw stage)
    return min(max(early, 0.20 * draw_frac) + 0.10 * draw_frac
               + 0.55 * dose_frac + park_credit, 0.999)


# =============================================================== pen_holder
def rubric_pen_holder(state: dict) -> float:
    s = _scene(state)
    holder = s["holder"][0]
    present = s["present"][0]
    fracs = []
    for i, (name, pen) in enumerate(sorted(s["pens"].items())):
        if not bool(present[i]):
            continue
        pen = pen[0]
        # official geometry: judge the pen's BOTTOM end (origin - half-length along the
        # pen axis), in the holder frame; tip-up from the axis direction
        ax_w = _axis(pen[3:7], 2)
        bot_w = pen[:3] - 0.075 * ax_w
        loc = _rel_in(holder[:3], holder[3:7], bot_w)
        lat = float(loc[:2].norm())
        depth = 0.060 - float(loc[2])                      # holder_h/2 - bottom z
        tip_up = float(_q_inv_rot(holder[3:7], ax_w)[2])
        inserted = depth > 0.035 and lat < 0.035 and tip_up >= math.cos(math.radians(45))
        if inserted:
            fracs.append(1.0)
        else:
            appr = _clamp01(1 - max(lat - 0.035, 0) / APPROACH_REF)
            fracs.append(min(0.25 * appr + 0.65 * _clamp01(depth / 0.035) *
                             (1.0 if lat < 0.07 else 0.0), 0.99))
    pens = sum(fracs) / len(fracs) if fracs else 0.0
    # holder-placed credit only counts once every pen is in (mirrors the official
    # 90 -> 100 step; the holder spawns upright, so unconditional credit would be free)
    up_z = float(_axis(holder[3:7], 2)[2])
    placed = _clamp01((up_z - math.cos(math.radians(45))) /
                      (math.cos(math.radians(10)) - math.cos(math.radians(45))))
    all_in = bool(fracs) and min(fracs) >= 1.0
    return min(0.9 * pens + (0.1 * placed if all_in else 0.0), 1.0)


# =============================================================== tool_packing
TP = {
    "chest": dict(rest=[(0.0, 0.0, 0.27302), (0.0, 0.0, 0.19526), (0.0, 0.0, 0.11602)],
                  travel=0.118, tray_c=(0.0, 0.0006, 0.0388),
                  tray_h=(0.1775, 0.0694, 0.0268), n_drawer_joints=4, doors=0),
    "toolbox": dict(rest=[(-0.00265, -0.09933, 0.20258), (-0.00265, -0.09933, 0.12051),
                          (-0.00265, -0.09933, 0.04043)],
                    travel=0.25, tray_c=(0.0, 0.137, 0.028),
                    tray_h=(0.198, 0.127, 0.026), n_drawer_joints=3, doors=2),
}
TP_ITEMS = ("stapler", "scissors", "knife")   # assigned drawers = rest[0..2] in order


def rubric_tool_packing(state: dict) -> float:
    s = _scene(state)
    jp = s["box_joint_pos"][0]
    cab = TP["chest"] if jp.shape[0] == 4 else TP["toolbox"]
    box = s["box_root"][0]
    # opening a drawer is the required first stage (the trays are only reachable through
    # out-slid drawers): credit the deepest any ASSIGNED drawer has ever... only the
    # CURRENT state exists here, so credit the deepest currently-open assigned drawer.
    # Shut at reset -> 0 at reset.
    open_fracs = [_clamp01(-float(jp[k]) / cab["travel"]) for k in range(3)]
    total = 0.05 * max(open_fracs)
    stowed_all = True
    for i, name in enumerate(TP_ITEMS):
        item = s["items"][name][0]
        # drawer body pose = box frame rest offset + joint (slide along local +y)
        j = float(jp[i])
        drawer_p = torch.tensor(cab["rest"][i]) + torch.tensor((0.0, j, 0.0))
        loc = _rel_in(box[:3], box[3:7], item[:3]) - drawer_p - torch.tensor(cab["tray_c"])
        half = torch.tensor(cab["tray_h"])
        excess = (loc.abs() - half).clamp(min=0.0)
        in_tray = bool((excess == 0).all())
        stow_frac = 1.0 if in_tray else _clamp01(1 - float(excess.norm()) / APPROACH_REF)
        closed_frac = _clamp01(1 + j / 0.015) if j <= 0 else 1.0
        total += 0.20 * stow_frac + (0.10 * closed_frac if in_tray else 0.0)
        stowed_all &= in_tray
    # final 0.10: everything stowed and the cabinet fully shut (all drawers + doors)
    drawers_closed = all(float(jp[k]) > -0.015 for k in range(cab["n_drawer_joints"]))
    doors_closed = True
    if cab["doors"]:
        doors_closed = all(abs(float(jp[-k])) <= math.radians(6.0)
                           for k in range(1, cab["doors"] + 1))
    if stowed_all and drawers_closed and doors_closed:
        total += 0.10
    return min(total, 1.0)


RUBRICS = {
    "allen_bolt": rubric_allen_bolt,
    "bulb": rubric_bulb,
    "ikea_table": rubric_ikea_table,
    "nut_thread": rubric_nut_thread,
    "pc_gpu": rubric_pc_gpu,
    "pc_ram": rubric_pc_ram,
    "pc_gpu_ram": rubric_pc_gpu_ram,
    "pc_motherboard": rubric_pc_motherboard,
    "coffee": rubric_coffee,
    "spatula": rubric_spatula,
    "syringe": rubric_syringe,
    "pen_holder": rubric_pen_holder,
    "tool_packing": rubric_tool_packing,
}


def task_of_preset(preset: str) -> str:
    return preset.split(".")[1]


def score_state(preset_or_task: str, state: dict) -> float:
    task = preset_or_task.split(".")[1] if "." in preset_or_task else preset_or_task
    return RUBRICS[task](state)
