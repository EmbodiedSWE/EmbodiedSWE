"""Teleport solution for TrayPoiseScene (sim_gen task `draw_triangle_i416`) — the
task's legitimacy certificate.

The task's load-bearing interactions and how they are executed:

1. COMPUTE THE MOUNT POINT (perception + arithmetic, no privileged reads): the
   per-episode slug loadout is read back from the OBSERVED slug positions in the
   tray frame (which pocket each slug occupies), and the combined centre of mass
   x_com = (m_h*x_h + m_l*x_l) / (m_tray + m_h + m_l) is computed from the declared
   masses. The cap must go under THAT point — the scene asserts a centre mount
   always tips.
2. TRANSPORT THE LOADED TRAY (teleport): one pose write per body stages the tray —
   with both slugs at their seated pocket offsets, a rigid carry — HOVERING 10 mm
   above the cap, positioned so the cap axis sits under (x_com, 0) in the tray
   frame. Free-space transport only: the hover height is asserted NOT to read as
   mounted.
3. MOUNT (contact dynamics — never teleported into place): the ensemble is released
   and falls onto the cap. Whether it stands or capsizes is decided entirely by
   whether the computed point was right — the equilibrium, the rocking transient
   and the settle are all physics. The scene's streak-latched mount credit must
   confirm a SUSTAINED stand.
4. SERVE THE CUBES (teleport hover + contact drop): both golden cubes are staged
   12 mm above the deck at (x_com, +/-serve_y) in the LIVE tray frame —
   symmetric, so their drop impulses cancel in roll and their weight keeps the
   combined CoM on the cap axis — and released to land by contact. The balanced
   tray must carry the new load.
5. HANDS-OFF PERSISTENCE: >= 3.5 simulated seconds with no writes; success() must
   still hold at the end.

If a mount or serve attempt fails (tray off the cap), everything is honestly
re-collected to the floor and the attempt repeats (bounded retries) — latched
credit makes the printed SIM_GEN_SCORE trajectory non-decreasing regardless.

Run (forge): python -u -m simgen_tasks.<task>.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import math
import os
import threading

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tray_poise")().build(num_envs=args.num_envs,
                                               device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    from isaaclab.utils.math import quat_apply

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        tz = float(scene.tray.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
        cap = scene._tray_local(scene.pedestal.data.root_pos_w)[0]
        spd = {nm: float(b.data.root_lin_vel_w[0].norm())
               for nm, b in (("tray", scene.tray), ("H", scene.slug_h),
                             ("L", scene.slug_l), ("ca", scene.cube_a),
                             ("cb", scene.cube_b))}
        spd["trayW"] = float(scene.tray.data.root_ang_vel_w[0].norm())
        print("[solve]   speeds " + " ".join(f"{k}={v:.3f}" for k, v in spd.items()),
              flush=True)
        print(f"[solve] {tag:14s} | tray_z={tz:.3f} up_z={float(scene._tray_up_z()[0]):+.3f} "
              f"cap_in_tray=({float(cap[0]):+.3f},{float(cap[1]):+.3f}) "
              f"mounted={bool(scene._mounted()[0])} home={bool(scene._slugs_home()[0])} "
              f"served={[bool(v) for v in scene._served()[0]]} "
              f"still={bool(scene._still()[0])} "
              f"m_ever={bool(scene._mounted_ever[0])} "
              f"s_ever={[bool(v) for v in scene._served_ever[0]]} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def place(body, pos_w: torch.Tensor, quat_w: torch.Tensor) -> None:
        """Transport-only pose write: pose + ZERO velocities."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        body.write_root_state_to_sim(st, all_ids)

    def yaw_quat(yaw: float) -> torch.Tensor:
        q = torch.zeros(n, 4, device=device)
        q[:, 0], q[:, 3] = math.cos(yaw / 2), math.sin(yaw / 2)
        return q

    # ---------------- phase 0: reset, settle, read the loadout ------------------------------
    step(60)
    ph = scene._tray_local(scene.slug_h.data.root_pos_w)[0]
    pl = scene._tray_local(scene.slug_l.data.root_pos_w)[0]
    pockets = torch.tensor([-c.sock_pitch, 0.0, c.sock_pitch], device=device)
    hx_obs = float(pockets[int(torch.argmin((pockets - ph[0]).abs()))])
    lx_obs = float(pockets[int(torch.argmin((pockets - pl[0]).abs()))])
    # honest cross-check only — the answer is computed from the OBSERVED pockets
    assert abs(hx_obs - float(scene._hx[0])) < 1e-6, "loadout readback mismatch (H)"
    assert abs(lx_obs - float(scene._lx[0])) < 1e-6, "loadout readback mismatch (L)"
    m_tot = c.tray_mass + c.slug_h_mass + c.slug_l_mass
    x_com = (c.slug_h_mass * hx_obs + c.slug_l_mass * lx_obs) / m_tot
    pp = (scene.pedestal.data.root_pos_w - scene.env_origins)[0]
    pq = scene.pedestal.data.root_quat_w[0]
    p_yaw = 2.0 * math.atan2(float(pq[3]), float(pq[0]))
    print(f"[solve] layout readback (seed {args.seed}): "
          f"pedestal=({float(pp[0]):+.3f},{float(pp[1]):+.3f}) "
          f"yaw={math.degrees(p_yaw):+.1f}deg "
          f"slug_h_pocket={hx_obs:+.3f} slug_l_pocket={lx_obs:+.3f} "
          f"-> x_com={x_com:+.4f} (cap half-width {c.cap_w / 2:.3f})", flush=True)
    assert abs(x_com) > c.cap_w / 2 + 0.008, "loadout CoM should overhang the cap"
    report("reset")
    assert bool(scene._slugs_home()[0]), "slugs did not spawn seated in pockets"
    assert not bool(scene._mounted()[0]), "tray spawned mounted?!"
    s0 = print_score("P0 reset+settle, loadout read")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    def mount_attempt(hover: float) -> bool:
        """Teleport the loaded tray (rigid carry) to hover over the cap with the
        cap axis under (x_com, 0), release, settle. Returns sustained mount."""
        ped_xy = scene.pedestal.data.root_pos_w[:, 0:2]
        tq = yaw_quat(p_yaw)
        # tray position so that tray-frame (x_com, 0) lands on the cap axis
        off = quat_apply(tq, torch.tensor([x_com, 0.0, 0.0], device=device)
                         .expand(n, 3))
        tp = torch.zeros(n, 3, device=device)
        tp[:, 0:2] = ped_xy - off[:, 0:2]
        tp[:, 2] = scene.env_origins[:, 2] + c.mount_z + hover
        place(scene.tray, tp, tq)
        for sx, body, seat_z in ((hx_obs, scene.slug_h, c.seat_z_h),
                                 (lx_obs, scene.slug_l, c.seat_z_l)):
            loc = torch.tensor([sx, 0.0, seat_z], device=device).expand(n, 3)
            place(body, tp + quat_apply(tq, loc), tq)
        step(1)
        assert not bool(scene._mounted()[0]), \
            "hover pose already reads as mounted (teleport must not mount the tray)"
        # release: the drop, the rocking and the equilibrium are all physics
        for _ in range(600):
            env.step(no_action)
            if bool(scene._mounted_ever[0]):
                break
        # then require it to be a live, still stand right now
        step(60)
        return bool(scene._mounted_ever[0]) and bool(scene._mounted()[0]) \
            and bool(scene._slugs_home()[0])

    def recollect() -> None:
        """Honest failure recovery: everything back to the floor, re-seated."""
        tq = yaw_quat(0.0)
        tp = torch.zeros(n, 3, device=device)
        tp[:, 0] = c.tray_xy[0]
        tp[:, 1] = c.tray_xy[1]
        tp[:, 2] = c.ground_z + 0.002
        tp[:, 0:3] += scene.env_origins
        place(scene.tray, tp, tq)
        for sx, body, seat_z in ((hx_obs, scene.slug_h, c.seat_z_h),
                                 (lx_obs, scene.slug_l, c.seat_z_l)):
            loc = torch.tensor([sx, 0.0, seat_z + 0.004], device=device).expand(n, 3)
            place(body, tp + quat_apply(tq, loc), tq)
        for body, slot in ((scene.cube_a, c.cube_slots[0]),
                           (scene.cube_b, c.cube_slots[1])):
            cp = torch.zeros(n, 3, device=device)
            cp[:, 0], cp[:, 1] = slot[0], slot[1]
            cp[:, 2] = c.cube_s / 2 + 0.002
            cp[:, 0:3] += scene.env_origins
            place(body, cp, tq)
        step(60)

    def serve_cubes() -> bool:
        """Stage both cubes 12 mm above the LIVE deck at (x_com, +/-serve_y) —
        symmetric so the drop impulses cancel — release, settle."""
        tp = scene.tray.data.root_pos_w
        tq = scene.tray.data.root_quat_w
        zc = c.deck_top + c.cube_s / 2 + 0.012
        for body, sy in ((scene.cube_a, c.serve_y), (scene.cube_b, -c.serve_y)):
            loc = torch.tensor([x_com, sy, zc], device=device).expand(n, 3)
            place(body, tp + quat_apply(tq, loc), tq)
        for _ in range(480):
            env.step(no_action)
            if bool(scene.success()[0]):
                break
        step(60)
        return bool(scene.success()[0])

    # ---------------- phases 1-3: mount, then serve (bounded honest retries) ----------------
    done = False
    for attempt in range(3):
        if attempt > 0:
            print(f"[solve] attempt {attempt + 1}: re-collecting to the floor",
                  flush=True)
            recollect()
            report("recollected")
        hover = 0.010
        # -- P1/P2: transport to hover, release, physics decides the equilibrium --
        mounted = mount_attempt(hover)
        report("mount")
        if not mounted:
            print("[solve] mount attempt failed (tray off the cap)", flush=True)
            continue
        s2 = print_score("P2 loaded tray dropped onto the cap and stood (sustained)")
        assert s2 >= 0.35 - 1e-6, "mount credit missing"
        # -- P3: serve both cubes onto the live balanced tray --
        served = serve_cubes()
        report("serve")
        if not served:
            print("[solve] serve attempt failed (tray upset or cube off)", flush=True)
            continue
        done = True
        break
    if not done:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (could not reach success state)", flush=True)
        os._exit(1)
    s3 = print_score("P3 both cubes served onto the balanced tray")
    assert s3 >= 1.0 - 1e-6, "success should score exactly 1.0"

    # ---------------- phase 4: persistence (>= 3.5 simulated seconds, no writes) ------------
    hold = True
    for _ in range(10):  # 10 x 42 substeps = 3.5 s at 120 Hz
        step(42)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.5 s hands-off")
    ok = hold and bool(scene.success()[0]) and s4 >= s3 - 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
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
    except Exception as exc:  # noqa: BLE001 — die fast, Kit teardown hangs
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({exc})", flush=True)
        os._exit(1)
