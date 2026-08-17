"""Smoke battery for LeakyBasketSealScene (sim_gen task
`living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket_i299`).

Every wrong outcome is CONSTRUCTED as a settled physical state and asserted REJECTED:

- reset sanity: settled, score ~0, finite state
- randomization by READBACK across 8 seeds: lid SLOT swap (the wide lid's side flips;
  lids always on opposite slots), bottle SLOT swap, continuous jitter (stand xy + yaw,
  per-object xy)
- null policy: 2.5 s hands-off, no credit
- SEED strategy (drop the ketchup into the basket as found): the bottle runs down the
  funnel, DISCHARGES through the open throat and lands on the floor under the stand ->
  no containment, ~0 credit — the seed's plan is physically void here
- WRONG lid: the narrow decoy dropped dead on the stand axis passes the throat in
  free fall and lands on the floor below -> no seal credit, ~0
- seal alone (wide lid seated): 0.30, not success
- WRONG bottle: brown bbq dropped onto the seated lid -> rests contained but earns no
  containment credit, not success
- BELOW-THE-APERTURE near-miss: ketchup standing on the floor UNDER the sealed basket,
  hugging the stand axis -> not contained (z gate), no latch
- BOTH bottles inside the sealed basket: exclusion clause holds -> capped 0.65, never
  success
- DECOY-IN-CAVITY exclusion: narrow lid resting on the seated plug + ketchup loaded ->
  success collapses despite seal + containment
- success reproduction (the solve recipe): success True, score 1.0
- latch regression x2: yank the ketchup out, then remove the plug -> success collapses
  but the latched 0.65 base credit persists (never 1.0)
- audit: success() was never True at any rejection-battery judged point
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
    env = ENVS.get("simgen.leaky_basket_seal")().build(num_envs=args.num_envs,
                                                       device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    origin = scene.env_origins

    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.45, -1.05, 0.85)) + o),
                                tuple(np.array((0.42, 0.00, 0.20)) + o),
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

    def loc_of(body) -> tuple[float, float, float]:
        p = scene._stand_local(body.data.root_pos_w)[0]
        return float(p[0]), float(p[1]), float(p[2])

    def judge(tag: str, *, expect_reject: bool = True) -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        if expect_reject and ok:
            reject_violated[0] = True
        pl = loc_of(scene.plug)
        kl = loc_of(scene.ketchup)
        print(f"[smoke] {tag:22s} | plug_z={pl[2]:+.3f} k_z={kl[2]:+.3f} "
              f"sealed={bool(scene.sealed()[0])} "
              f"k_in={bool(scene.contained(scene.ketchup)[0])} "
              f"bbq_in={bool(scene.contained(scene.bbq)[0])} "
              f"decoy_in={bool(scene.decoy_in_cavity()[0])} "
              f"sealed_ever={bool(scene._sealed_ever[0])} "
              f"in_ever={bool(scene._in_ever[0])} "
              f"ret_ever={bool(scene._ret_ever[0])} "
              f"still={bool(scene._still()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)
        return s, ok

    def finite_all() -> bool:
        ok = True
        for b in (scene.stand, scene.plug, scene.decoy, scene.ketchup, scene.bbq):
            ok = ok and bool(torch.isfinite(b.data.root_state_w).all())
        return ok

    def place(body, pos_env, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor(pos_env, device=device).expand(n, 3) + origin
        st[:, 3:7] = torch.tensor(quat, device=device).expand(n, 4)
        body.write_root_state_to_sim(st, all_ids)

    def stage_axis(body, local_z: float, xy=(0.0, 0.0), upright: bool = False) -> None:
        """Pose write at stand-local (xy, local_z) in the stand's CURRENT frame."""
        s_pos, s_quat = scene.stand.data.root_pos_w, scene.stand.data.root_quat_w
        loc = torch.tensor([xy[0], xy[1], local_z], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = s_pos + quat_apply(s_quat, loc)
        if upright:
            st[:, 3] = 1.0
        else:
            st[:, 3:7] = s_quat
        body.write_root_state_to_sim(st, all_ids)

    def settle_still(max_steps: int, need=None) -> None:
        quiet = 0
        for _ in range(max_steps):
            step(1)
            ok = bool(scene._still()[0]) if need is None else bool(need())
            quiet = quiet + 1 if ok else 0
            if quiet >= 30:
                break

    def seal_basket(max_steps: int = 720) -> bool:
        """The solve recipe: stage the WIDE lid on the axis ~8 cm above the seat."""
        stage_axis(scene.plug, c.z_throat + 0.105)
        quiet = 0
        for _ in range(max_steps):
            step(1)
            ok = bool(scene.sealed()[0]) and bool(scene._still_body(scene.plug)[0])
            quiet = quiet + 1 if ok else 0
            if quiet >= 30:
                return True
        return False

    def drop_bottle(body, xy=(0.014, 0.010), max_steps: int = 900) -> None:
        """Stage a bottle upright just above the rim over the cavity; hands off."""
        stage_axis(body, c.z_rim + 0.008 + c.body_h / 2, xy=xy, upright=True)
        settle_still(max_steps)

    # =========================== 1. reset sanity ============================================
    env.reset(seed=101)
    step(60)
    s, ok = judge("reset+settle")
    check("reset: settled, no NaN, success False, score <= 0.02",
          finite_all() and not ok and s <= 0.02 and bool(scene._still()[0]))

    # =========================== 2-4. randomization by readback =============================
    p_ys, k_ys, yaws, sx, p_xs, k_xs = [], [], [], [], [], []
    lids_opp, bots_opp = True, True
    for sd in range(8):
        env.reset(seed=sd)
        step(5)
        sp = (scene.stand.data.root_pos_w - origin)[0]
        sq = scene.stand.data.root_quat_w[0]
        yaws.append(math.degrees(2.0 * math.atan2(float(sq[3]), float(sq[0]))))
        sx.append(float(sp[0]))
        p = (scene.plug.data.root_pos_w - origin)[0]
        d = (scene.decoy.data.root_pos_w - origin)[0]
        k = (scene.ketchup.data.root_pos_w - origin)[0]
        q = (scene.bbq.data.root_pos_w - origin)[0]
        p_ys.append(float(p[1]))
        k_ys.append(float(k[1]))
        p_xs.append(float(p[0]))
        k_xs.append(float(k[0]))
        lids_opp = lids_opp and (float(p[1]) * float(d[1]) < 0)
        bots_opp = bots_opp and (float(k[1]) * float(q[1]) < 0)
    print(f"[smoke] readback 8 seeds: plug_y={[f'{y:+.2f}' for y in p_ys]} "
          f"k_y={[f'{y:+.2f}' for y in k_ys]}", flush=True)
    print(f"[smoke] stand_x={[f'{x:.3f}' for x in sx]} yaw={[f'{y:+.0f}' for y in yaws]} "
          f"plug_x={[f'{x:.3f}' for x in p_xs]} k_x={[f'{x:.3f}' for x in k_xs]}",
          flush=True)
    check("randomization: WIDE-lid slot side flips across seeds; lids always opposite",
          any(y > 0 for y in p_ys) and any(y < 0 for y in p_ys) and lids_opp)
    check("randomization: ketchup slot side flips across seeds; bottles always opposite",
          any(y > 0 for y in k_ys) and any(y < 0 for y in k_ys) and bots_opp)
    spread = (max(sx) - min(sx) > 0.01 and max(yaws) - min(yaws) > 3.0
              and max(p_xs) - min(p_xs) > 0.01 and max(k_xs) - min(k_xs) > 0.01)
    check("randomization: continuous jitter present (stand x, yaw, lid xy, bottle xy)",
          spread)

    # =========================== 5. null policy =============================================
    env.reset(seed=102)
    step(300)
    s, ok = judge("null policy 2.5s")
    check("null policy: hands-off leaves score <= 0.02, no success", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy: the basket LEAKS =========================
    # The seed's whole plan — drop the ketchup into the basket as found — executed
    # faithfully: bottle staged over the cavity, released. It runs down the funnel and
    # DISCHARGES through the open throat onto the floor under the stand.
    drop_bottle(scene.ketchup, xy=(0.010, 0.006), max_steps=600)
    kz = loc_of(scene.ketchup)[2]
    s, ok = judge("seed: drop in unsealed")
    check("seed strategy: ketchup dropped into the UNSEALED basket discharges through "
          "the throat to the floor (local z < 0.12), no containment, score <= 0.02",
          kz < 0.12 and not bool(scene.contained(scene.ketchup)[0])
          and not bool(scene._in_ever[0]) and s <= 0.02 and not ok)

    # =========================== 7. WRONG lid: the decoy cannot seal ========================
    # The narrow lid released dead on the axis — the strongest possible attempt to
    # seal with it — passes the 70 mm throat and lands on the floor below.
    # (Clear the discharged ketchup first: it settled standing right under the throat
    # and the falling decoy would land on its cap — probe debris, not the physics
    # under test.)
    place(scene.ketchup, (0.05, -0.45, c.body_h / 2 + 0.002))
    settle_still(60, need=lambda: scene._still_body(scene.ketchup)[0])
    stage_axis(scene.decoy, c.z_throat + 0.105)
    settle_still(600, need=lambda: scene._still_body(scene.decoy)[0])
    dz = loc_of(scene.decoy)[2]
    print(f"[smoke] decoy after axis drop: local_z={dz:+.3f}", flush=True)
    s, ok = judge("wrong lid: decoy drop")
    check("wrong lid: the NARROW lid dropped on the axis falls through the throat "
          "(local z < 0.12), no seal credit ever, score <= 0.02",
          dz < 0.12 and not bool(scene._sealed_ever[0])
          and not bool(scene.decoy_in_cavity()[0]) and s <= 0.02 and not ok)

    # =========================== 8. seal alone is 0.30, not success =========================
    env.reset(seed=103)
    step(30)
    sealed = seal_basket()
    s, ok = judge("seal alone")
    check("seal alone: wide lid funnel-centred and seated -> score ~0.30, not success",
          sealed and bool(scene.sealed()[0]) and 0.29 <= s <= 0.301 and not ok)

    # =========================== 9. WRONG bottle rejected ===================================
    drop_bottle(scene.bbq)
    s, ok = judge("bbq on the seal")
    check("wrong bottle: BROWN bbq resting contained on the seal -> no containment "
          "credit (score <= 0.301), not success",
          bool(scene.contained(scene.bbq)[0]) and bool(scene.sealed()[0])
          and s <= 0.301 and not ok)

    # =========================== 10. below-the-aperture near-miss ===========================
    # Ketchup standing on the floor UNDER the sealed basket, hugging the stand axis:
    # laterally dead-centre, but on the wrong side of the seal.
    stage_axis(scene.ketchup, c.body_h / 2 + 0.002, xy=(0.010, 0.006), upright=True)
    settle_still(240, need=lambda: scene._still_body(scene.ketchup)[0])
    kz = loc_of(scene.ketchup)[2]
    s, ok = judge("under the basket")
    check("near-miss: ketchup on the floor UNDER the sealed basket, on the axis -> "
          "not contained, no latch, not success",
          kz < 0.12 and not bool(scene.contained(scene.ketchup)[0])
          and not bool(scene._in_ever[0]) and not ok)

    # =========================== 11. BOTH bottles: exclusion ================================
    # Ketchup added to the sealed basket that already holds the bbq: containment +
    # retention latch (base 0.65) but the exclusion clause bars success.
    bq = loc_of(scene.bbq)
    off = (-0.016 if bq[0] > 0 else 0.016, -0.012 if bq[1] > 0 else 0.012)
    drop_bottle(scene.ketchup, xy=off, max_steps=1100)
    s, ok = judge("both bottles inside")
    check("exclusion: ketchup AND bbq inside the sealed basket -> capped 0.651, "
          "never success",
          bool(scene.contained(scene.ketchup)[0]) and bool(scene.contained(scene.bbq)[0])
          and bool(scene.sealed()[0]) and s <= 0.651 and not ok)

    # =========================== 12. decoy-in-cavity exclusion ==============================
    env.reset(seed=104)
    step(30)
    sealed = seal_basket()
    stage_axis(scene.decoy, c.z_throat + 0.105)  # drop the narrow lid onto the seal
    settle_still(400, need=lambda: scene._still_body(scene.decoy)[0])
    d_in = bool(scene.decoy_in_cavity()[0])
    dz = loc_of(scene.decoy)[2]
    print(f"[smoke] decoy on the seal: local_z={dz:+.3f} decoy_in={d_in}", flush=True)
    drop_bottle(scene.ketchup, max_steps=1100)
    s, ok = judge("decoy left in cavity")
    check("exclusion: narrow lid resting in the cavity on the seal + ketchup loaded -> "
          "success collapses (decoy must not be left inside)",
          sealed and d_in and bool(scene.decoy_in_cavity()[0])
          and bool(scene.contained(scene.ketchup)[0]) and not ok and s <= 0.651)

    # =========================== 13. success reproduction ===================================
    env.reset(seed=105)
    step(30)
    sealed = seal_basket()
    drop_bottle(scene.ketchup, max_steps=1100)
    quiet = 0
    for _ in range(400):
        step(1)
        quiet = quiet + 1 if bool(scene.success()[0]) else 0
        if quiet >= 30:
            break
    s, ok = judge("solve recipe", expect_reject=False)
    check("success reproduction: seal + gentle drop -> success True, score 1.0",
          sealed and ok and s >= 0.999)

    # =========================== 14-15. latch regression ====================================
    place(scene.ketchup, (0.05, -0.45, c.body_h / 2 + 0.002))
    settle_still(240)
    s, ok = judge("ketchup removed")
    check("latch regression: ketchup yanked out -> success collapses, latched credit "
          "stays (0.64 <= score <= 0.651)",
          not ok and 0.64 <= s <= 0.651 and not bool(scene.contained(scene.ketchup)[0]))

    place(scene.plug, (0.05, 0.45, c.plug_h / 2 + 0.002))
    settle_still(240)
    s, ok = judge("plug removed")
    check("latch regression: plug removed -> seal broken, not success, latched credit "
          "stays (0.64 <= score <= 0.651)",
          not ok and not bool(scene.sealed()[0]) and 0.64 <= s <= 0.651)

    # =========================== 16-17. audit + no-NaN ======================================
    check("rejection audit: success() never True at any rejection-battery judged point",
          not reject_violated[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.leaky_basket_seal")
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
