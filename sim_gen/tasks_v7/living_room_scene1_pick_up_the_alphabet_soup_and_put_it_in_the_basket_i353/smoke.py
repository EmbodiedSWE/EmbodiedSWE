"""Smoke battery for PantryRackOrderScene (sim_gen task
`living_room_scene1_pick_up_the_alphabet_soup_and_put_it_in_the_basket_i353`).

Every wrong outcome is CONSTRUCTED as a settled physical state (teleport constructs
or genuine physical pushes) and asserted REJECTED:

- reset sanity: settled, finite, score ~0
- randomization by READBACK across 8 seeds: the target permutation varies (>= 3
  distinct orders) and the roof tiles always sit at the rank slots matching it;
  staging-slot shuffle + continuous jitter (rack x, yaw, per-can xy)
- null policy: 2.5 s hands-off, no credit
- SEED strategy (lower the can into the container from above, as the seed's basket
  allows): the can lands ON THE ROOF — the rack has no top opening; no credit
- WRONG FIRST insertion (physical push): the can required at the MOUTH is pushed in
  first and seats fully inside -> stage 1 never latches, score ~0
- prefix-then-wrong (physical): correct deepest can (0.30 latches), then the
  mouth-required can pushed in second -> stage 2 never latches, frozen at 0.30
- completing that wrong run (physical): all three fully inside, upright, still —
  but the last two ranks swapped -> NEVER success, still 0.30
- near-miss: the deepest-required can left straddling the mouth plane -> not
  "fully inside", nothing latches
- tipped can: full correct depth order but the mouth can LYING DOWN -> upright
  gate rejects success
- wrong-order full arrangement constructed cold (deepest two swapped, all upright,
  settled) -> score ~0, never success
- success reproduction: the solve recipe (deepest-first physical pushes) -> 1.0
- latch regression: mouth can yanked out -> success collapses, latched 0.60 stays
- recovery: the yanked can physically pushed back in -> success returns to 1.0
  (the mouth rank is the one repairable rank; the deep ranks are not)
- audit: success() never True at any rejection-battery judged point
- final: no NaN anywhere

Records viewport rgb frames to frames.npz (cwd). Prints `SIM_GEN_SMOKE: ALL PASS n/n`.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
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

_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pantry_rack_order")().build(num_envs=args.num_envs,
                                                       device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    origin = scene.env_origins
    names = [s[0] for s in c.can_specs]
    masses = [s[3] for s in c.can_specs]

    from isaaclab.utils.math import quat_apply, quat_mul

    # --- recording (viewport rgb annotator) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.30, -0.95, 0.70)) + o),
                                tuple(np.array((0.40, 0.00, 0.08)) + o),
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

    checks: list[tuple[str, bool]] = []
    reject_violated = [False]

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def judge(tag: str, *, expect_reject: bool = True) -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        if expect_reject and ok:
            reject_violated[0] = True
        d = scene.depths()[0]
        ins = scene.inside()[0]
        print(f"[smoke] {tag:24s} | d=({float(d[0]):+.3f},{float(d[1]):+.3f},"
              f"{float(d[2]):+.3f}) in=({bool(ins[0])},{bool(ins[1])},{bool(ins[2])}) "
              f"s1_ever={bool(scene._s1_ever[0])} s2_ever={bool(scene._s2_ever[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)
        return s, ok

    def finite_all() -> bool:
        ok = True
        for b in [scene.rack, *scene.cans, *scene.tokens]:
            ok = ok and bool(torch.isfinite(b.data.root_state_w).all())
        return ok

    def place_world(body, pos_env, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor(pos_env, device=device).expand(n, 3) + origin
        st[:, 3:7] = torch.tensor(quat, device=device).expand(n, 4)
        body.write_root_state_to_sim(st, all_ids)

    def place_local(body, loc, q_local=None) -> None:
        """Pose write at rack-local `loc` in the rack's CURRENT (randomized) frame."""
        rq = scene.rack.data.root_quat_w
        rp = scene.rack.data.root_pos_w
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = rp + quat_apply(rq, torch.tensor(loc, device=device).expand(n, 3))
        if q_local is None:
            st[:, 3:7] = rq
        else:
            st[:, 3:7] = quat_mul(rq, torch.tensor(q_local, device=device).expand(n, 4))
        body.write_root_state_to_sim(st, all_ids)

    def settle(max_steps: int = 360, quiet_n: int = 30) -> bool:
        quiet = 0
        for _ in range(max_steps):
            step(1)
            quiet = quiet + 1 if bool(scene.still()[0].all()) else 0
            if quiet >= quiet_n:
                return True
        return False

    ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
    off_t = torch.tensor([0.0, 0.0, -0.017], device=device).expand(n, 3)
    mu_pair = (c.floor_mu + c.can_mu) / 2

    def push_in(ci: int, v_des: float = 0.12) -> bool:
        """The solve mechanism: stage on the apron (transport teleport), then a
        velocity-servo horizontal force slides the can through the mouth. Genuine
        physics — the probe asserts the can actually seated (never vacuous)."""
        body = scene.cans[ci]
        r = float(scene._radii[ci])
        chain = masses[ci] + sum(masses[j] for j in range(3)
                                 if j != ci and bool(scene.inside()[0, j]))
        place_local(body, (-(r + 0.045), 0.0,
                           c.floor_top + scene._height / 2 + 0.002))
        step(12)
        rq = scene.rack.data.root_quat_w
        u = quat_apply(rq, ex)
        ff = 1.1 * mu_pair * chain * 9.81
        cap, gain = 2.2, 4.0
        target = r + 0.022
        seated = False
        last_d = float(scene.depths()[0, ci])
        for i in range(1800):
            d = float(scene.depths()[0, ci])
            if d >= target:
                seated = True
                break
            v_along = (body.data.root_lin_vel_w * u).sum(dim=-1)
            f_mag = (ff + gain * (v_des - v_along)).clamp(min=0.0, max=cap)
            body.set_external_force_and_torque(
                (f_mag.unsqueeze(-1) * u).unsqueeze(1),
                torch.cross(off_t, f_mag.unsqueeze(-1) * u, dim=-1).unsqueeze(1),
                env_ids=all_ids, is_global=True)
            step(1)
            if (i + 1) % 120 == 0:
                if d - last_d < 0.005:
                    ff *= 1.4
                    cap = min(cap * 1.25, 4.0)
                last_d = d
        body.set_external_force_and_torque(torch.zeros(n, 1, 3, device=device),
                                           torch.zeros(n, 1, 3, device=device),
                                           env_ids=all_ids, is_global=True)
        settle(480)
        return seated

    # =========================== 1. reset sanity ============================================
    env.reset(seed=101)
    step(60)
    s, ok = judge("reset+settle")
    check("reset: settled, no NaN, success False, score <= 0.02, no can inside",
          finite_all() and not ok and s <= 0.02 and bool(scene.still()[0].all())
          and not bool(scene.inside()[0].any()))

    # =========================== 2-3. randomization by readback =============================
    perms: list[tuple] = []
    tok_ok = True
    rack_xs, yaws = [], []
    tom_xs, tom_ys, soup_ys = [], [], []
    tok_x = torch.tensor(c.token_x, device=device)
    for sd in range(8):
        env.reset(seed=sd)
        step(5)
        p = scene._perm[0]
        perms.append(tuple(p.tolist()))
        rank_of = p.argsort()
        for i in range(3):
            t_loc = scene._rack_local(scene.tokens[i].data.root_pos_w)[0]
            tok_ok = tok_ok and abs(float(t_loc[0] - tok_x[rank_of[i]])) < 0.005
        rq = scene.rack.data.root_quat_w[0]
        yaws.append(math.degrees(2.0 * math.atan2(float(rq[3]), float(rq[0]))))
        rack_xs.append(float((scene.rack.data.root_pos_w - origin)[0, 0]))
        t = (scene.cans[0].data.root_pos_w - origin)[0]
        sp = (scene.cans[2].data.root_pos_w - origin)[0]
        tom_xs.append(float(t[0]))
        tom_ys.append(float(t[1]))
        soup_ys.append(float(sp[1]))
    print(f"[smoke] readback 8 seeds: perms={perms}", flush=True)
    print(f"[smoke] rack_x={[f'{x:.3f}' for x in rack_xs]} "
          f"yaw={[f'{y:+.0f}' for y in yaws]}", flush=True)
    print(f"[smoke] tomato_x={[f'{x:.3f}' for x in tom_xs]} "
          f"tomato_y={[f'{y:+.2f}' for y in tom_ys]} "
          f"soup_y={[f'{y:+.2f}' for y in soup_ys]}", flush=True)
    check("randomization: target permutation varies (>= 3 distinct over 8 seeds) and "
          "the roof tiles always sit at the matching rank slots (readback)",
          len(set(perms)) >= 3 and tok_ok)
    n_tomx = len({round(x, 3) for x in tom_xs})
    check("randomization: staging slots shuffle (tomato and soup each span > 0.25 m "
          "in y) + continuous jitter (rack x > 1 cm, yaw > 3 deg, per-can xy)",
          max(tom_ys) - min(tom_ys) > 0.25 and max(soup_ys) - min(soup_ys) > 0.25
          and max(rack_xs) - min(rack_xs) > 0.01 and max(yaws) - min(yaws) > 3.0
          and n_tomx >= 5)

    # =========================== 4. null policy =============================================
    env.reset(seed=102)
    step(300)
    s, ok = judge("null policy 2.5s")
    check("null policy: hands-off leaves score <= 0.02, no success", s <= 0.02 and not ok)

    # =========================== 5. SEED strategy: no top opening ===========================
    # The seed's plan — carry the target can over the container and LOWER IT IN from
    # above — executed faithfully: the deepest-required can is released over the
    # rack's midpoint. It lands ON THE ROOF: the rack has no top opening.
    perm = scene._perm[0].tolist()
    t0 = perm[0]
    place_local(scene.cans[t0], (c.interior_len / 2, 0.0,
                                 c.roof_top + scene._height / 2 + 0.03))
    settle(480)
    zloc = float(scene._rack_local(scene.cans[t0].data.root_pos_w)[0, 2])
    print(f"[smoke] {names[t0]} after top drop: local_z={zloc:+.3f} "
          f"(roof plane at {c.roof_top:.3f})", flush=True)
    s, ok = judge("seed: lower from above")
    check("seed strategy: the can lowered from above lands ON the roof and never "
          "enters the rack — no can inside, score <= 0.02",
          not bool(scene.inside()[0].any()) and s <= 0.02 and not ok
          and zloc > c.roof_top - 0.005)

    # =========================== 6. WRONG FIRST insertion (physical) ========================
    env.reset(seed=103)
    step(30)
    perm = scene._perm[0].tolist()
    wrong = perm[2]  # the can required at the MOUTH, pushed in FIRST
    seated = push_in(wrong)
    s, ok = judge("wrong can first")
    check("wrong order: the MOUTH-required can pushed in first seats fully inside, "
          "but stage 1 never latches — score <= 0.02, no success",
          seated and bool(scene.inside()[0, wrong])
          and not bool(scene._s1_ever[0]) and s <= 0.02 and not ok)

    # =========================== 7. prefix-then-wrong (physical) ============================
    env.reset(seed=104)
    step(30)
    perm = scene._perm[0].tolist()
    ok1 = push_in(perm[0])  # correct deepest can -> 0.30 latches
    s_mid = float(scene.score()[0])
    ok2 = push_in(perm[2])  # WRONG second can (mouth-required)
    s, ok = judge("prefix then wrong")
    check("prefix-then-wrong: correct deepest can latches 0.30; the mouth-required "
          "can pushed second seats but stage 2 never latches — frozen at 0.30",
          ok1 and ok2 and s_mid >= 0.29 and bool(scene._s1_ever[0])
          and not bool(scene._s2_ever[0]) and bool(scene.inside()[0, perm[2]])
          and 0.29 <= s <= 0.301 and not ok)

    # =========================== 8. completed but mis-ordered (physical) ====================
    ok3 = push_in(perm[1])  # last can in: arrangement = t0, t2, t1 — ranks 2,3 swapped
    d = scene.depths()[0]
    s, ok = judge("full but swapped")
    check("completed-but-swapped: all three fully inside, upright, still — but the "
          "last two ranks are swapped -> NEVER success, score stays 0.30",
          ok3 and bool(scene.inside()[0].all())
          and float(d[perm[2]]) > float(d[perm[1]])
          and 0.29 <= s <= 0.301 and not ok)

    # =========================== 9. near-miss: straddling the mouth =========================
    env.reset(seed=105)
    step(30)
    perm = scene._perm[0].tolist()
    t0 = perm[0]
    r0 = float(scene._radii[t0])
    place_local(scene.cans[t0], (r0 - 0.010, 0.0,
                                 c.floor_top + scene._height / 2 + 0.002))
    settle(300)
    d0 = float(scene.depths()[0, t0])
    print(f"[smoke] straddle: {names[t0]} depth={d0:+.3f} "
          f"(fully-inside needs >= {r0 + c.in_margin:.3f})", flush=True)
    s, ok = judge("straddling the mouth")
    check("near-miss: the deepest-required can resting IN the mouth but not fully "
          "past it -> not inside, nothing latches, score <= 0.02",
          d0 > 0.0 and not bool(scene.inside()[0, t0])
          and not bool(scene._s1_ever[0]) and s <= 0.02 and not ok)

    # =========================== 10. tipped can: upright gate ===============================
    # Constructed full arrangement in the CORRECT depth order, but the mouth can is
    # LYING ON ITS SIDE (its centre is still inside the z band — only the upright
    # gate can reject it).
    env.reset(seed=106)
    step(30)
    perm = scene._perm[0].tolist()
    rr = [float(scene._radii[i]) for i in perm]
    d_0 = c.interior_len - rr[0] - 0.005
    d_1 = d_0 - rr[0] - rr[1] - 0.008
    d_2 = d_1 - rr[1] - rr[2] - 0.008
    z_up = c.floor_top + scene._height / 2 + 0.002
    place_local(scene.cans[perm[0]], (d_0, 0.0, z_up))
    place_local(scene.cans[perm[1]], (d_1, 0.0, z_up))
    hr = math.sqrt(0.5)
    place_local(scene.cans[perm[2]], (d_2, 0.0, c.floor_top + rr[2] + 0.002),
                q_local=(hr, hr, 0.0, 0.0))  # lying, axis along the channel width
    settle(360)
    ins = scene.inside()[0]
    s, ok = judge("mouth can tipped")
    check("tipped can: correct depth order but the mouth can lying on its side -> "
          "its centre passes the z band yet the upright gate rejects it, no success",
          bool(ins[perm[0]]) and bool(ins[perm[1]]) and not bool(ins[perm[2]])
          and not ok and s <= 0.601)

    # =========================== 11. wrong-order full arrangement (cold) ====================
    # All three upright, fully inside, settled — but the DEEPEST two are swapped.
    env.reset(seed=107)
    step(30)
    perm = scene._perm[0].tolist()
    rr = [float(scene._radii[i]) for i in perm]
    d_a = c.interior_len - rr[1] - 0.005          # t1 sits deepest (WRONG)
    d_b = d_a - rr[1] - rr[0] - 0.008             # t0 second
    d_c = d_b - rr[0] - rr[2] - 0.008             # t2 at the mouth rank (right)
    place_local(scene.cans[perm[1]], (d_a, 0.0, z_up))
    place_local(scene.cans[perm[0]], (d_b, 0.0, z_up))
    place_local(scene.cans[perm[2]], (d_c, 0.0, z_up))
    settle(360)
    s, ok = judge("deepest two swapped")
    check("wrong order: full arrangement, all upright and fully inside, deepest two "
          "swapped -> nothing latches, score <= 0.02, never success",
          bool(scene.inside()[0].all()) and s <= 0.02 and not ok
          and not bool(scene._s1_ever[0]))

    # =========================== 12. success reproduction ===================================
    env.reset(seed=108)
    step(30)
    perm = scene._perm[0].tolist()
    okp = all(push_in(ci) for ci in perm)
    quiet = 0
    for _ in range(400):
        step(1)
        quiet = quiet + 1 if bool(scene.success()[0]) else 0
        if quiet >= 30:
            break
    s, ok = judge("solve recipe", expect_reject=False)
    check("success reproduction: deepest-first physical pushes -> success True, "
          "score 1.0", okp and ok and s >= 0.999)

    # =========================== 13. latch regression =======================================
    place_world(scene.cans[perm[2]], (0.10, -0.45, scene._height / 2 + 0.002))
    settle(240)
    s, ok = judge("mouth can removed")
    check("latch regression: mouth can yanked out -> success collapses, latched "
          "credit stays (0.59 <= score <= 0.601)",
          not ok and 0.59 <= s <= 0.601 and not bool(scene.inside()[0, perm[2]]))

    # =========================== 14. recovery of the mouth rank =============================
    okr = push_in(perm[2])
    quiet = 0
    for _ in range(400):
        step(1)
        quiet = quiet + 1 if bool(scene.success()[0]) else 0
        if quiet >= 30:
            break
    s, ok = judge("mouth can re-inserted", expect_reject=False)
    check("recovery: the yanked mouth can pushed back in -> success returns, 1.0 "
          "(only the mouth rank is repairable; deep ranks are not)",
          okr and ok and s >= 0.999)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() never True at any rejection-battery judged point",
          not reject_violated[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.pantry_rack_order")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        for nm, okc in checks:
            if not okc:
                print(f"[smoke] FAILED CHECK: {nm}", flush=True)
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)

    code = 0 if all_ok else 1
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
    except Exception as e:  # noqa: BLE001 — fail fast, don't idle until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
