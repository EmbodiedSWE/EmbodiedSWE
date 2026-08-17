"""Smoke / rubric-REJECTION battery for VaultLauncherScene — NullRobot, teleported probes.

solve.py is the acceptance proof (servo the turret onto the vault bearing by torque,
drop the ball into the chute, gravity launches it through the mouth into the vault).
This battery proves the rubric REJECTS wrong outcomes and that the strategic
differentiators from the seed (maniskill/pick_single_egad: grasp + lift 7.5 cm) are
physically load-bearing: LIFTING the ball earns nothing (the seed's entire plan),
the sealed vault cannot be entered by gentle hand-placement (a ball released inside
the snout rolls back OUT the rising floor), and the AIM is load-bearing (a launch
from a misaimed turret misses). Every probe is CONSTRUCTED as a settled state
(teleport, real physics steps, judge); constructed partial states may earn latched
partial credit but none may reach success() unless the ball genuinely ends up
inside the vault interior.

Checks:
   1. settle/no-NaN      — seeded reset settles finite; ball resting in its tray,
                           turret misaimed >= 16 deg; score ~0, no success;
   2. randomization      — three seeded resets (max-pairwise): vault bearing, vault
                           xy, turret initial yaw, tray xy and ball xy all differ;
   3. readback consistency — across seeds the STORED vault bearing matches both the
                           vault yaw readback and the base->vault position bearing;
                           the turret yaw readback matches the stored initial yaw;
                           the misaim is always toward the center and in the
                           declared 18..55 deg band; both bearing signs occur;
   4. null-policy        — 300 idle steps: ball stays in the tray, turret holds its
                           misaim, score ~0, no success;
   5. SEED STRATEGY      — the seed's whole plan is grasp+LIFT: a force-servo lifts
                           the ball 10 cm and sets it back down — and earns NOTHING
                           (no latch, score ~0, no success);
   6. roof near-miss     — ball settled ON the vault roof: on the box but not IN
                           it — entered never latches, no success;
   7. wall-hug near-miss — ball settled on the ground hugging the OUTSIDE of the
                           snout tunnel wall — entered never latches, no success;
   8. hand-placement     — ball RELEASED at rest inside the snout tunnel (the best
                           any hand could do): the rising one-way floor rolls it
                           back OUT the mouth — no success, only latched partial
                           credit (entered ~0.30);
   9. misaim launch      — turret teleported 18 deg off the bearing, ball dropped
                           into the chute: a real launch that MISSES the mouth —
                           loaded latches, entered never does, no success. The aim
                           is load-bearing;
  10. slow entry         — ball constructed AT the mouth moving inward at 0.25 m/s
                           (a gentle roll-in): it cannot carry the 12 mm ridge and
                           rolls back out — no success;
  11. fast entry         — full latched pipeline: turret aimed (latch), ball parked
                           in the chute (latch), then constructed at the mouth at
                           1.2 m/s: it carries the ridge, drops 72 mm, settles —
                           genuine success. With check 10 this is the one-way
                           velocity gate the task is built on;
  12. revocation         — from success, the ball parked far away on the ground:
                           success reads False and the score falls to the latched
                           0.70 cap — success is judged live;
  13. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.pick_single_egad_i328.smoke --headless
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

_qz = task_scene._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

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


def _write_body(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None,
                vel_w: torch.Tensor | None = None) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    if vel_w is not None:
        st[:, 7:10] = vel_w
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    err = float(scene.aim_err_deg()[0])
    bv = scene._vault_local(scene.ball.data.root_pos_w)[0]
    print(f"[smoke] {tag:16s} | aim_err={err:+7.2f}deg "
          f"ball_vault=({float(bv[0]):+.3f},{float(bv[1]):+.3f},{float(bv[2]):+.3f}) "
          f"in_vault={bool(scene.ball_in_vault()[0])} "
          f"aimed={bool(scene._aimed[0])} loaded={bool(scene._loaded[0])} "
          f"entered={bool(scene._entered[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.vault_launcher")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.55, -1.25, 1.05)) + o),
                                tuple(np.array((0.00, 0.05, 0.12)) + o),
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
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    zero = torch.zeros(n, 1, 3, device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)

    def ball_wrench_w(f_w: torch.Tensor) -> None:
        q = scene.ball.data.root_quat_w
        scene.ball.set_external_force_and_torque(
            quat_apply_inverse(q, f_w).reshape(n, 1, 3), zero)

    def set_turret_yaw(yaw: torch.Tensor) -> None:
        """Pure-quat yaw write ON the axle (the turret body origin sits on it) —
        joint-consistent, so the base is untouched."""
        _refresh()
        _write_body(scene.turret, scene.base.data.root_pos_w, _qz(yaw))

    def ball_to_vault_local(local_xyz, v_local_x: float = 0.0) -> None:
        """Construct the ball at a vault-local point, optionally moving inward
        (+x, toward the interior) at v_local_x m/s."""
        _refresh()
        qv = scene.vault.data.root_quat_w
        loc = torch.tensor(local_xyz, device=device).expand(n, 3)
        pos = scene.vault.data.root_pos_w + quat_apply(qv, loc)
        vel = quat_apply(qv, torch.tensor([v_local_x, 0.0, 0.0],
                                          device=device).expand(n, 3))
        _write_body(scene.ball, pos, None, vel)

    def tun_floor_at(x_local: float) -> float:
        """Tunnel floor-top height at vault-local x (mouth..ridge)."""
        t = (x_local - c.mouth_x) / (c.ridge_x - c.mouth_x)
        return c.tun_floor_z0 + t * (c.tun_floor_z1 - c.tun_floor_z0)

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def score0() -> float:
        _refresh()
        return float(scene.score()[0])

    def succ0() -> bool:
        _refresh()
        return bool(scene.success()[0])

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(180)
    _report("settle")
    _REC["on"] = False
    _refresh()
    ball_z = float(scene.ball.data.root_pos_w[0, 2] - env.iscene.env_origins[0, 2])
    tray_d = float((scene.ball.data.root_pos_w[0, :2]
                    - scene.tray.data.root_pos_w[0, :2]).norm())
    check("settle/no-NaN: layout settles finite; ball resting in its tray, turret "
          "misaimed >= 16 deg, score ~0, no success",
          bool(scene._finite()[0]) and 0.015 < ball_z < 0.060 and tray_d < 0.06
          and abs(float(scene.aim_err_deg()[0])) >= 16.0
          and score0() <= 0.03 and not succ0())

    # ================= 2. randomization is real (3-seed max-pairwise readback) ===================
    def readback():
        _refresh()
        return {
            "bear": math.degrees(float(scene.vault_bear[0])),
            "vxy": scene.vault.data.root_pos_w[0, :2].clone(),
            "tyaw": float(scene.turret_yaw_deg()[0]),
            "trayxy": scene.tray.data.root_pos_w[0, :2].clone(),
            "ballxy": scene.ball.data.root_pos_w[0, :2].clone(),
        }

    obs = []
    for sd in (101, 202, 303):
        torch.manual_seed(sd)
        env.reset()
        _step(10)
        obs.append(readback())
    pairs = [(0, 1), (0, 2), (1, 2)]
    d_bear = max(dyaw(obs[i]["bear"], obs[j]["bear"]) for i, j in pairs)
    d_vxy = max(float((obs[i]["vxy"] - obs[j]["vxy"]).norm()) for i, j in pairs)
    d_tyaw = max(dyaw(obs[i]["tyaw"], obs[j]["tyaw"]) for i, j in pairs)
    d_txy = max(float((obs[i]["trayxy"] - obs[j]["trayxy"]).norm()) for i, j in pairs)
    d_bxy = max(float((obs[i]["ballxy"] - obs[j]["ballxy"]).norm()) for i, j in pairs)
    print(f"[smoke] randomization max-pairwise deltas: bear={d_bear:.1f}deg "
          f"vault_xy={d_vxy * 1000:.0f}mm turret_yaw={d_tyaw:.1f}deg "
          f"tray_xy={d_txy * 1000:.0f}mm ball_xy={d_bxy * 1000:.0f}mm", flush=True)
    check("randomization-is-real: vault bearing, vault xy, turret initial yaw, "
          "tray xy and ball xy readback all differ across seeds",
          d_bear > 5.0 and d_vxy > 0.04 and d_tyaw > 5.0 and d_txy > 0.04
          and d_bxy > 0.04)

    # ================= 3. stored randomization matches pose readback ==============================
    consistent = True
    seen_sign = {True: 0, False: 0}
    for sd in range(300, 310):
        torch.manual_seed(sd)
        env.reset()
        _step(5)
        _refresh()
        bear = math.degrees(float(scene.vault_bear[0]))
        vy = yaw_of(scene.vault.data.root_quat_w[0])
        d = scene.vault.data.root_pos_w[0] - scene.base.data.root_pos_w[0]
        pos_bear = math.degrees(math.atan2(float(d[1]), float(d[0])))
        tyaw = float(scene.turret_yaw_deg()[0])
        iyaw = math.degrees(float(scene.init_yaw[0]))
        err0 = tyaw - bear
        consistent &= dyaw(bear, vy) < 1.0                    # vault faces its bearing
        consistent &= dyaw(bear, pos_bear) < 1.0              # vault stands on its bearing
        consistent &= dyaw(tyaw, iyaw) < 1.5                  # turret at its stored yaw
        consistent &= 16.0 <= abs(err0) <= 57.0               # declared misaim band
        consistent &= (err0 > 0) == (bear < 0)                # misaim toward the center
        seen_sign[bear >= 0] += 1
        if all(seen_sign.values()) and sd >= 304:
            break
    print(f"[smoke] bearing signs over seeds: +={seen_sign[True]} -={seen_sign[False]} "
          f"consistent={consistent}", flush=True)
    check("readback consistency: stored bearing matches vault yaw AND base->vault "
          "position bearing; turret yaw matches stored init yaw; misaim in band, "
          "toward the center; both bearing signs occur",
          consistent and all(v > 0 for v in seen_sign.values()))

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _refresh()
    ball0 = scene.ball.data.root_pos_w[0].clone()
    err0 = float(scene.aim_err_deg()[0])
    _step(300)
    _report("null-policy")
    drift = float((scene.ball.data.root_pos_w[0] - ball0).norm())
    check("null-policy-fails: 300 idle steps, ball stays in the tray (drift < 2 cm), "
          "turret holds its misaim, score ~0, no success",
          drift < 0.02 and abs(float(scene.aim_err_deg()[0]) - err0) < 2.0
          and score0() <= 0.03 and not succ0())

    # ================= 5. negative: the SEED'S OWN STRATEGY (grasp + lift) ========================
    # maniskill/pick_single_egad succeeds by lifting the object 7.5 cm. Here a
    # force-servo lifts the ball 10 cm, holds, and sets it back down — real
    # contact-free flight, real set-down — and earns nothing.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _refresh()
    ball_m = float(scene.ball.root_physx_view.get_masses()[0].sum())
    z0 = float(scene.ball.data.root_pos_w[0, 2])
    z_peak = z0
    _REC["on"] = True
    for phase, z_des_off, steps in (("lift", 0.10, 240), ("lower", 0.005, 240)):
        for _ in range(steps):
            _refresh()
            z = scene.ball.data.root_pos_w[:, 2]
            vz = scene.ball.data.root_lin_vel_w[:, 2]
            f = ball_m * (9.81 + 30.0 * (z0 + z_des_off - z) - 10.0 * vz)
            ball_wrench_w(f.clamp(0.0, 4.0 * ball_m * 9.81).unsqueeze(-1) * ez)
            _step(1)
            z_peak = max(z_peak, float(scene.ball.data.root_pos_w[0, 2]))
    scene.ball.set_external_force_and_torque(zero, zero)
    _step(120)
    _report("seed-lift")
    _REC["on"] = False
    z_end = float(scene.ball.data.root_pos_w[0, 2])
    print(f"[smoke] seed lift: z {z0:.3f} -> peak {z_peak:.3f} -> end {z_end:.3f}",
          flush=True)
    check("negative (SEED strategy): the ball force-lifted 10 cm and set back down "
          "— the seed's entire success criterion — earns NOTHING (no latch, "
          "score ~0, no success)",
          z_peak > z0 + 0.08 and abs(z_end - z0) < 0.02
          and not bool(scene._loaded[0]) and not bool(scene._entered[0])
          and score0() <= 0.03 and not succ0())

    # ================= 6. near-miss: ON the vault is not IN the vault =============================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    ball_to_vault_local((0.0, 0.0, c.wall_z + 0.012 + c.ball_r + 0.003))
    _step(180)
    _report("roof-rest")
    bv = scene._vault_local(scene.ball.data.root_pos_w)[0]
    check("near-miss (roof): ball settled ON the vault roof — over the interior "
          "in xy but never inside — entered never latches, no success",
          bool(scene._finite()[0]) and float(bv[2]) > c.wall_z - 0.005
          and not bool(scene._entered[0]) and not succ0() and score0() <= 0.03)

    # ================= 7. near-miss: hugging the tunnel's OUTSIDE wall ============================
    tmx = (c.mouth_x + c.ridge_x) / 2
    torch.manual_seed(100)
    env.reset()
    _step(60)
    ball_to_vault_local((tmx, c.tun_half_w + c.wall_t + c.ball_r + 0.003,
                         c.ball_r + 0.002))
    _step(180)
    _report("wall-hug")
    bv = scene._vault_local(scene.ball.data.root_pos_w)[0]
    check("near-miss (wall hug): ball settled on the ground against the snout's "
          "OUTSIDE wall — beside the tunnel, never in it — entered never "
          "latches, no success",
          bool(scene._finite()[0]) and float(bv[2]) < 0.05
          and not bool(scene._entered[0]) and not succ0() and score0() <= 0.03)

    # ================= 8. hand-placement: released in the snout rolls back OUT ====================
    # The best a hand could ever do: the ball AT REST on the tunnel floor, ~90 mm
    # in (as deep as fingers + ball reach). The rising one-way floor must roll it
    # back out the mouth. (Partial credit for the visit is fine; success is not.)
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    x_in = -0.100
    ball_to_vault_local((x_in, 0.0, tun_floor_at(x_in) + c.ball_r + 0.001))
    _step(360)
    _report("hand-placed")
    _REC["on"] = False
    bv = scene._vault_local(scene.ball.data.root_pos_w)[0]
    check("hand-placement rejected: a ball RELEASED at rest inside the snout rolls "
          "back out the mouth (one-way floor) — outside again, no success, "
          "entered-only credit",
          bool(scene._finite()[0]) and float(bv[0]) < c.mouth_x - 0.01
          and bool(scene._entered[0]) and not bool(scene.ball_in_vault()[0])
          and not succ0() and 0.29 <= score0() <= 0.3000001)

    # ================= 9. misaim launch: the aim is load-bearing ==================================
    # Turret teleported 18 deg off the bearing (aimed never latches: 18 > 4), the
    # ball dropped into the chute exactly like the solve: a REAL launch that must
    # miss the flared mouth (the import-time audit puts 15 deg well outside it).
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _refresh()
    bear = scene.vault_bear.clone()
    sgn_c = torch.where(bear >= 0, -1.0, 1.0)
    set_turret_yaw(bear + sgn_c * math.radians(18.0))
    _step(30)
    aimed_pre = bool(scene._aimed[0])
    _REC["on"] = True
    _refresh()
    slope = (c.chan_z0 - c.chan_z1) / (c.chan_x1 - c.chan_x0)
    floor_top = c.chan_z0 - slope * (c.drop_x - c.chan_x0)
    loc = torch.tensor([c.drop_x, 0.0, floor_top + c.ball_r + c.drop_hover],
                       device=device).expand(n, 3)
    pos = scene.turret.data.root_pos_w + quat_apply(scene.turret.data.root_quat_w, loc)
    _write_body(scene.ball, pos)
    drop_p = scene.ball.data.root_pos_w[0].clone()
    _step(600)
    _report("misaim-launch")
    _REC["on"] = False
    flew = float((scene.ball.data.root_pos_w[0, :2] - drop_p[:2]).norm())
    ball_z = float(scene.ball.data.root_pos_w[0, 2] - env.iscene.env_origins[0, 2])
    print(f"[smoke] misaim launch: ball flew {flew:.3f}m from the drop, final "
          f"z={ball_z:.3f}", flush=True)
    check("misaim launch rejected: a real launch from a turret 18 deg off the "
          "bearing flies and MISSES — loaded latches, entered never does, no "
          "success (the aim is load-bearing)",
          (not aimed_pre) and not bool(scene._aimed[0]) and flew > 0.30
          and ball_z < 0.06 and bool(scene._loaded[0])
          and not bool(scene._entered[0]) and not succ0()
          and score0() <= 0.2000001)

    # ================= 10. slow entry rolls back: the one-way velocity gate =======================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    x_m = c.mouth_x + 0.010
    ball_to_vault_local((x_m, 0.0, tun_floor_at(x_m) + c.ball_r + 0.001),
                        v_local_x=0.25)
    _step(360)
    _report("slow-entry")
    _REC["on"] = False
    bv = scene._vault_local(scene.ball.data.root_pos_w)[0]
    check("slow entry rejected: a 0.25 m/s roll-in cannot carry the 12 mm ridge "
          "and rolls back out the mouth — no success",
          bool(scene._finite()[0]) and float(bv[0]) < c.mouth_x - 0.01
          and not bool(scene.ball_in_vault()[0]) and not succ0())

    # ================= 11. fast entry commits: genuine success ====================================
    # Full latched pipeline first (aimed + loaded), then the mouth construction at
    # launch speed: past the ridge the 72 mm drop keeps the ball forever.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _refresh()
    _REC["on"] = True
    set_turret_yaw(scene.vault_bear.clone())
    _step(10)                                    # aimed latches (3-step persistence)
    loc = torch.tensor([0.180, 0.0, c.chan_z1 + c.ball_r + 0.001],
                       device=device).expand(n, 3)
    pos = scene.turret.data.root_pos_w + quat_apply(scene.turret.data.root_quat_w, loc)
    _write_body(scene.ball, pos)
    _step(8)                                     # loaded latches (ball resting in the flat run)
    aimed_l, loaded_l = bool(scene._aimed[0]), bool(scene._loaded[0])
    ball_to_vault_local((x_m, 0.0, tun_floor_at(x_m) + c.ball_r + 0.001),
                        v_local_x=1.20)
    for _ in range(40):
        _step(10)
        _refresh()
        if bool(scene.ball_in_vault()[0]) \
                and float(scene.ball.data.root_lin_vel_w[0].norm()) < 0.5 * c.settle_speed:
            break
    _step(120)
    _report("fast-entry")
    _REC["on"] = False
    commit_ok = aimed_l and loaded_l and bool(scene.ball_in_vault()[0]) \
        and succ0() and score0() >= 0.999
    check("fast entry commits: at 1.2 m/s the ball carries the ridge, drops into "
          "the interior and settles — genuine success (with check 10: the "
          "one-way velocity gate)",
          commit_ok)

    # ================= 12. revocation: success is judged live =====================================
    _refresh()
    park = scene.base.data.root_pos_w[0:1].clone().expand(n, 3).clone()
    park[:, 0] -= 1.2
    park[:, 1] -= 1.2
    park[:, 2] = c.ball_r + 0.002
    _write_body(scene.ball, park)
    _step(90)
    _report("revoked")
    s12 = score0()
    check("revocation: the ball parked far away on the ground — success reads "
          "False and the score falls to the latched 0.70 cap",
          commit_ok and not succ0() and 0.6999 <= s12 <= 0.7000005)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.vault_launcher")
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
    except Exception as exc:  # noqa: BLE001 - Kit teardown hangs; die loudly NOW
        print(f"SIM_GEN_SMOKE: FAIL (exception: {exc!r})", flush=True)
        import traceback

        traceback.print_exc()
        os._exit(1)
