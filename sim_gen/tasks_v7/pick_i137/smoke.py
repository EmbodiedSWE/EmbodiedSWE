"""Smoke / rubric-REJECTION battery for GravityVaultScene — NullRobot, teleported probes.

solve.py is the acceptance proof (force-driven gate extraction + gravity feed scores
1.0). This battery proves the rubric REJECTS wrong outcomes and that the physical
claims the task rests on are load-bearing: the chamber really is sealed (a ball pressed
onto the roof stays out), the spanning gate really blocks the drop (the seed's
carry-and-release parks on it), the interlock really is one-way (an inward shove jams
on the knob), the blue decoy pays nothing, and success() refuses a moving ball. Every
probe is CONSTRUCTED (teleport, real physics steps, judge) — instrumentation, never a
solution: success() is monitored at EVERY step and must never turn True anywhere in
this battery (the audit is itself a check).

Checks:
   1. settle/no-NaN     — seeded reset settles finite; gate SPANS by readback
                          (pin_clear False), balls on the floor outside; score ~0;
   2. randomization     — seeded resets: vault xy + yaw, gate insertion depth and the
                          red-ball spawn all differ pairwise (READBACK); the gate's
                          knob side takes BOTH values across seeds;
   3. null policy       — 240 idle steps: nothing latches, score ~0, no success;
   4. SEALED CHAMBER    — the seed's direct placement: the red ball set on the chamber
                          roof and PRESSED down at 2x weight for half a second. The
                          roof takes the load (the probe is real: z readback), the
                          ball never enters, the chamber latch never fires;
   5. GATE BLOCKS       — the exact solve drop (release above the funnel mouth) with
                          the gate still spanning: the ball falls the tower and PARKS
                          ON the gate (z readback), the chamber never latches (partial
                          `entered` credit <= 0.25 is by design), the impact does not
                          shove the gate axially (wedge-drop guard), no success;
   6. latched credit    — teleporting the parked ball back to the floor drops every
                          live predicate but the latched `entered` credit survives
                          (score >= 0.25), still no success;
   7. ONE-WAY INTERLOCK — the gate shoved INWARD with escalating force: it slides
                          (probe moved) then JAMS when the knob hits the tower wall;
                          the tongue still spans, pin_clear never True, no gate credit;
   8. WRONG OBJECT      — gate parked away (constructed), the BLUE decoy dropped down
                          the tower into the chamber: no chamber/entered credit, no
                          success, score stays at the constructed gate credit 0.30;
   9. settle gate + cap — the red ball placed inside the chamber SLIDING at 0.5 m/s:
                          all three latches fire, the non-success score is CAPPED at
                          exactly 0.75, and success() refuses while anything moves
                          (the probe is dismantled before it can settle);
  10. rejection audit   — success() was never True at any step of this battery;
  11. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.pick_i137.smoke --headless
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

_qmul, _qinv, _qapply, _qz = (task_scene._qmul, task_scene._qinv,
                              task_scene._qapply, task_scene._qz)

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() -----------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
_AUD = {"on": False, "hits": 0}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if _AUD["on"] and bool(env.scene.success()[0]):
            _AUD["hits"] += 1
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


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    rl = scene._vault_local(scene.red.data.root_pos_w)[0]
    gl = scene._vault_local(scene.gate.data.root_pos_w)[0]
    print(f"[smoke] {tag:18s} | red_l=({float(rl[0]):+.3f},{float(rl[1]):+.3f},"
          f"{float(rl[2]):+.3f}) gate_l=({float(gl[0]):+.3f},{float(gl[1]):+.3f},"
          f"{float(gl[2]):+.3f}) pin_clear={bool(scene.pin_clear()[0])} "
          f"l_gate={bool(scene._l_gate[0])} l_ent={bool(scene._l_entered[0])} "
          f"l_ch={bool(scene._l_chamber[0])} in_ch={bool(scene.in_chamber(scene.red)[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main --------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.gravity_vault")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    zero = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.35, -0.95, 0.85)) + o),
                                tuple(np.array((0.45, 0.0, 0.20)) + o),
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

    def vq() -> torch.Tensor:
        _refresh()
        return scene.vault.data.root_quat_w

    def v_yaw() -> float:
        q = vq()[0]
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    def red_local() -> torch.Tensor:
        _refresh()
        return scene._vault_local(scene.red.data.root_pos_w)[0]

    def gate_y_out() -> float:
        _refresh()
        gl = scene._vault_local(scene.gate.data.root_pos_w)[0]
        return float(scene.gate_side[0]) * float(gl[1])

    def write_local(body, local_xyz, *, quat=None, lin_vel_w=None) -> None:
        """Teleport a body to a vault-frame position. Pure transport — the judge and
        the physics do the rest."""
        _refresh()
        st = torch.zeros(n, 13, device=device)
        loc = torch.tensor([list(local_xyz)], device=device, dtype=torch.float)
        st[:, 0:3] = scene.vault.data.root_pos_w + _qapply(vq(), loc.expand(n, 3))
        st[:, 3:7] = quat if quat is not None else torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], device=device).expand(n, 4)
        if lin_vel_w is not None:
            st[:, 7:10] = lin_vel_w
        body.write_root_state_to_sim(st, _all_ids())
        _refresh()

    # ================= 1. settle / no-NaN / baseline zero =======================================
    env.reset(seed=0)
    _AUD["on"] = True
    _step(240)
    _report("reset")
    rl = red_local()
    check("settle/no-NaN: seeded reset settles finite; gate SPANS by readback "
          "(pin_clear False); balls on the floor outside; score ~0, no success",
          bool(scene._finite()[0]) and not bool(scene.pin_clear()[0])
          and c.gate_in_y - 0.01 < gate_y_out() < c.gate_in_y + c.gate_retract_jitter + 0.01
          and float(rl[2]) < 0.06 and not bool(scene.in_chamber(scene.red)[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization readback ================================================
    vaults, yaws, depths, reds, sides = [], [], [], [], []
    for s in (0, 1, 2):
        env.reset(seed=s)
        _step(30)
        _refresh()
        vp = (scene.vault.data.root_pos_w - scene.env_origins)[0]
        vaults.append((float(vp[0]), float(vp[1])))
        yaws.append(v_yaw())
        depths.append(gate_y_out() - c.gate_in_y)
        r = red_local()
        reds.append((float(r[0]), float(r[1])))
    for s in range(8):
        env.reset(seed=s)
        _refresh()
        sides.append(float(scene.gate_side[0]))
    print(f"[smoke] rand readback: vaults={vaults} yaws={[f'{y:+.1f}' for y in yaws]} "
          f"depths={[f'{d:.4f}' for d in depths]} reds={reds} sides={sides}", flush=True)

    def pairwise(vals, tol) -> bool:
        return all(abs(a - b) > tol for i, a in enumerate(vals) for b in vals[i + 1:])

    check("randomization: vault xy+yaw, gate depth and red spawn differ pairwise "
          "across seeds 0/1/2; gate side takes both values across seeds 0..7",
          pairwise([v[0] for v in vaults], 3e-4) and pairwise([v[1] for v in vaults], 3e-4)
          and pairwise(yaws, 0.1) and pairwise(depths, 5e-5)
          and pairwise([r[0] for r in reds], 3e-4) and pairwise([r[1] for r in reds], 3e-4)
          and (1.0 in sides) and (-1.0 in sides))

    # ================= 3. null policy ===========================================================
    env.reset(seed=0)
    _step(240)
    _report("null-policy")
    check("null policy: 240 idle steps latch nothing — score ~0, no success",
          float(scene.score()[0]) <= 0.01 and not bool(scene._l_gate[0])
          and not bool(scene._l_entered[0]) and not bool(scene._l_chamber[0])
          and not bool(scene.success()[0]))

    # ================= 4. SEALED CHAMBER: pressed onto the roof =================================
    # The seed's own strategy — carry the ball to the target and place it — aimed at
    # the chamber directly: set the ball on the roof shelf and press DOWN at 2x weight.
    _REC["on"] = True
    # roof shelf between the tower wall (|x| 0.065) and the roof edge (|x| 0.11):
    # centre at 0.098 clears the wall by 8 mm and keeps the CoM over the roof
    write_local(scene.red, (0.098, 0.0, c.z_roof_top + c.ball_r + 0.004))
    _step(60)
    press = torch.zeros(n, 1, 3, device=device)
    press[0, 0, 2] = -2.0 * c.ball_mass * 9.81
    on_roof = True
    for _ in range(120):
        scene.red.set_external_force_and_torque(press, zero, env_ids=_all_ids(),
                                                is_global=True)
        _step(1)
        z = float(red_local()[2])
        on_roof = on_roof and (c.z_roof_top - 0.005 < z < c.z_roof_top + 0.05)
    scene.red.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    _step(60)
    _report("roof-press")
    check("sealed chamber: red ball pressed onto the roof at 2x weight stays ON the "
          "roof (probe real, z readback) — never enters, chamber latch never fires",
          on_roof and not bool(scene.in_chamber(scene.red)[0])
          and not bool(scene._l_chamber[0]) and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= 5. GATE BLOCKS: the solve drop against a spanning gate ==================
    y_out_before = gate_y_out()
    write_local(scene.red, (0.0, 0.0, c.z_mouth + 0.03))
    _step(300)
    _report("drop-on-gate")
    rl = red_local()
    z_on_gate = c.gate_rest_z + c.tongue_t / 2 + c.ball_r  # ~0.283
    check("gate blocks: the exact solve release with the gate spanning parks the ball "
          "ON the gate (z readback), chamber never latches, the impact does not shove "
          "the gate axially, no success",
          abs(float(rl[2]) - z_on_gate) < 0.02
          and abs(float(rl[0])) < 0.05 and abs(float(rl[1])) < 0.05
          and bool(scene._l_entered[0]) and not bool(scene._l_chamber[0])
          and abs(gate_y_out() - y_out_before) < 0.02
          and float(scene.score()[0]) <= c.w_entered + 1e-4
          and not bool(scene.success()[0]))

    # ================= 6. latched credit ========================================================
    write_local(scene.red, (0.32, -0.25, c.ball_r + 0.003))
    _step(60)
    _report("latched")
    check("latched credit: ball teleported back to the floor — live predicates drop "
          "but the `entered` credit survives (score >= 0.25), still no success",
          not bool(scene.in_shaft(scene.red)[0]) and not bool(scene.in_chamber(scene.red)[0])
          and bool(scene._l_entered[0])
          and float(scene.score()[0]) >= c.w_entered - 1e-4
          and float(scene.score()[0]) <= c.w_entered + 1e-4
          and not bool(scene.success()[0]))

    # ================= 7. ONE-WAY INTERLOCK: inward shove jams on the knob ======================
    side = float(scene.gate_side[0])
    u_in = torch.tensor([[0.0, -side, 0.0]], device=device)
    y0 = gate_y_out()
    y_prev = y0
    jammed = False
    moved = False
    for f_mag in (1.5, 3.0, 5.0):
        f = torch.zeros(n, 1, 3, device=device)
        f[0, 0, :] = _qapply(vq(), u_in)[0] * f_mag
        for _ in range(300):
            scene.gate.set_external_force_and_torque(f, zero, env_ids=_all_ids(),
                                                     is_global=True)
            _step(1)
        y_now = gate_y_out()
        if f_mag >= 3.0 and abs(y_now - y_prev) < 0.004:
            jammed = True
        y_prev = y_now
    scene.gate.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    _step(60)
    moved = (y0 - gate_y_out()) > 0.02
    _report("inward-jam")
    print(f"[smoke] inward shove: y_out {y0:+.4f} -> {gate_y_out():+.4f} "
          f"(moved={moved} jammed={jammed})", flush=True)
    check("one-way interlock: an inward shove moves the gate (probe real) then JAMS "
          "on the knob before the tongue can clear — pin never clears, no gate credit",
          moved and jammed and gate_y_out() > -0.035
          and not bool(scene.pin_clear()[0]) and not bool(scene._l_gate[0])
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 8. WRONG OBJECT: the blue decoy pays nothing =============================
    env.reset(seed=3)
    _step(60)
    _REC["on"] = True
    # constructed open state: park the gate away (this legitimately latches the 0.30
    # gate credit — the point of the check is that BLUE adds nothing on top)
    write_local(scene.gate, (-0.35, 0.0, c.tongue_t / 2 + 0.002))
    _step(60)
    write_local(scene.blue, (0.0, 0.0, c.z_mouth + 0.03))
    _step(360)
    _report("blue-decoy")
    _refresh()
    check("wrong object: blue decoy delivered into the chamber — chamber/entered "
          "latches never fire, no success, score stays at the gate credit 0.30",
          bool(scene.in_chamber(scene.blue)[0]) and bool(scene._l_gate[0])
          and not bool(scene._l_chamber[0]) and not bool(scene._l_entered[0])
          and abs(float(scene.score()[0]) - c.w_gate) < 1e-4
          and not bool(scene.success()[0]))

    # ================= 9. settle gate + score cap ===============================================
    # The red ball INSIDE the chamber but sliding at 0.5 m/s: every latch fires, the
    # non-success score hits the 0.75 cap, and success() must refuse while anything
    # moves. Dismantled (transport) long before friction could stop it (~27 steps).
    vel = _qapply(vq(), torch.tensor([[0.5 * 1.0, 0.0, 0.0]], device=device))
    vel[:, 2] = 0.0
    write_local(scene.red, (-0.03, 0.0, c.floor_t + c.ball_r + 0.002), lin_vel_w=vel)
    moving_ok = True
    for _ in range(4):
        _step(1)
        v = float(scene.red.data.root_lin_vel_w[0].norm())
        moving_ok = moving_ok and v > 0.1 and bool(scene.in_chamber(scene.red)[0]) \
            and not bool(scene.settled_red()[0]) and not bool(scene.success()[0])
    s_cap = float(scene.score()[0])
    _report("settle-gate")
    # dismantle before it can settle into a real success
    write_local(scene.red, (0.32, -0.25, c.ball_r + 0.003))
    _step(30)
    _REC["on"] = False
    check("settle gate + cap: a sliding in-chamber ball latches everything, the "
          "non-success score is capped at exactly 0.75, success refuses while moving",
          moving_ok and bool(scene._l_chamber[0]) and bool(scene._l_entered[0])
          and s_cap >= 0.75 - 1e-4 and s_cap <= 0.75 + 1e-5
          and not bool(scene.success()[0]))

    # ================= 10. rejection audit ======================================================
    check("rejection audit: success() was never True at any step of this battery",
          _AUD["hits"] == 0)

    # ================= save + verdict ===========================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.gravity_vault")
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
    main()
