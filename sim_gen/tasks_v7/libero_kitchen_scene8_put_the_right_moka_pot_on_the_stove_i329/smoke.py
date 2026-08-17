"""Smoke / rubric-REJECTION battery for CubbyRakeStoveScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a real insert->hook->drag->
place run and the latched credit is monotone along it). This battery proves the
rubric REJECTS wrong outcomes and that the claims the task rests on are load-
bearing. Note on the seed: the END STATE deliberately overlaps the seed's
(copper pot on the burner) — the strategic difference lives in REACHABILITY
(check 5 demonstrates the roofed cubby is a hard gate: the goal object cannot
leave upward even under a sustained over-weight lift) and in the extra success
clauses (steel pot must REMAIN in its lane, the rake must end clear of the
burner) that checks 9 and 10 show are live.

Checks:
  1.  settle/no-NaN — seeded reset settles finite; both pots upright deep in
                      their lanes, burner empty, score ~0, no success;
  2.  randomization — three seeded resets: READBACK copper-pot depth + yaw,
                      steel-pot yaw, rake xy; max-pairwise deltas all real;
  3.  lane coin-flip— over 10 resets the COPPER pot spawns in BOTH lanes
                      (identity is appearance, not position);
  4.  null-policy   — 240 idle steps: pots stay put, score ~0, no success;
  5.  roof-jam      — a 1.3x-weight lift wrench presses the copper pot up: it
                      rises (verified non-vacuous) then JAMS under the roof and
                      never leaves the cubby; released it lands back upright in
                      its lane — no extraction credit, no success;
  6.  wrong pot     — the STEEL pot seated on the burner (verified): pot_on_pad
                      watches the copper pot only — score ~0, no success;
  7.  pot tipped    — the copper pot lying on its SIDE on the burner: upright
                      clause refuses the pad latch — no success;
  8.  apron-only    — copper pot dragged-path (reach shift, then apron rest):
                      reach + out latches fire, run CAPS at 0.50, no success;
  9.  steel out     — steel pot displaced onto the open counter FIRST, then the
                      copper pot fully delivered to the burner: all latches
                      fire (cap 0.75) but in_cubby(steel) refuses success;
  10. rake at stove — the rake parked against the burner FIRST (rake_clear
                      verified False), then the copper pot fully delivered:
                      cap 0.75, rake_clear refuses success;
  11. frames.npz    — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_kitchen_scene8_put_the_right_moka_pot_on_the_stove_i329.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=24)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version on some GPUs and silently rejects
# RTX -> the annotator returns EMPTY frames. Disable the check.
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
    from . import scene as task_scene  # noqa: F401 - registers the scene
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

G = 9.81

# ----- module state wired up in main() ----------------------------------------------------------
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
    d = scene.pot_local(scene.pot_t)[0]
    print(f"[smoke] {tag:14s} | potT_loc=({float(d[0]):+.3f},{float(d[1]):+.3f},"
          f"{float(d[2]):+.3f}) inT={bool(scene.in_cubby(scene.pot_t)[0])} "
          f"inW={bool(scene.in_cubby(scene.pot_w)[0])} "
          f"out={bool(scene.pot_out()[0])} pad={bool(scene.pot_on_pad()[0])} "
          f"clear={bool(scene.rake_clear()[0])} "
          f"latches=({int(scene._l_reach[0])},{int(scene._l_out[0])},{int(scene._l_pad[0])}) "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main -------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cubby_rake_stove")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply_inverse

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.35, -1.00, 0.90)) + o),
                                tuple(np.array((0.42, 0.00, 0.10)) + o),
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

    # ----- shared machinery ----------------------------------------------------------------------
    origins = env.iscene.env_origins
    zero3 = torch.zeros(n, 1, 3, device=device)

    def cub_spot(x: float, y: float, dz: float) -> torch.Tensor:
        """World point at cubby-local xy, `dz` above the cubby floor top."""
        p = scene.cubby.data.root_pos_w.clone()
        p[:, 0] += x
        p[:, 1] += y
        p[:, 2] += c.floor_top + dz
        return p

    def pad_spot(dz: float) -> torch.Tensor:
        """World point over the burner centre, `dz` above the pad top."""
        p = scene.stove.data.root_pos_w.clone()
        p[:, 2] += c.plate_t + c.pad_h + dz
        return p

    def counter_spot(x: float, y: float, dz: float) -> torch.Tensor:
        """World point at deck-nominal xy, `dz` above the deck top."""
        p = origins.clone()
        p[:, 0] += x
        p[:, 1] += y
        p[:, 2] += c.deck_top + dz
        return p

    def _qy(ang: float) -> torch.Tensor:
        q = torch.zeros(n, 4, device=device)
        q[:, 0], q[:, 2] = math.cos(ang / 2), math.sin(ang / 2)
        return q

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def deliver_copper() -> None:
        """Teleport-path the copper pot along the demonstrated route so every
        latch fires: mouth-ward shift inside its lane (reach), rest on the
        apron (out), then seated on the burner pad (pad). Callers arrange the
        blocking condition FIRST so no prefix of this path reaches success()."""
        s = 1.0 if float(scene.pot_local(scene.pot_t)[0, 1]) > 0 else -1.0
        x0 = float(scene._x0[0])
        _write_body(scene.pot_t, cub_spot(x0 + 0.06, s * c.lane_y, 0.001))
        _step(30)
        _write_body(scene.pot_t,
                    cub_spot(c.mouth_x + c.out_margin + 0.025, s * c.lane_y, 0.001))
        _step(120)
        _write_body(scene.pot_t, pad_spot(0.008))
        _step(240)

    # ================= 1. settle / no-NaN =======================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(120)
    _report("settle")
    _REC["on"] = False
    fin = bool(torch.isfinite(scene.rake.data.root_pos_w).all()
               and torch.isfinite(scene.pot_t.data.root_state_w).all()
               and torch.isfinite(scene.pot_w.data.root_state_w).all())
    check("settle/no-NaN: both pots upright deep in their lanes, burner empty, "
          "score ~0, no success",
          fin and bool(scene.in_cubby(scene.pot_t)[0]) and bool(scene.in_cubby(scene.pot_w)[0])
          and bool(scene.pot_up(scene.pot_t)[0]) and bool(scene.pot_up(scene.pot_w)[0])
          and not bool(scene.pot_on_pad()[0]) and not bool(scene.pot_out()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback, 3-seed max-pairwise) =================
    def readback():
        _refresh()
        return (float(scene.pot_local(scene.pot_t)[0, 0]),
                yaw_of(scene.pot_t.data.root_quat_w[0]),
                yaw_of(scene.pot_w.data.root_quat_w[0]),
                (scene.rake.data.root_pos_w[0, :2] - origins[0, :2]).clone())

    obs = []
    for s in (101, 202, 303):
        torch.manual_seed(s)
        env.reset()
        _step(10)
        obs.append(readback())
    pairs = [(0, 1), (0, 2), (1, 2)]
    d_x = max(abs(obs[i][0] - obs[j][0]) for i, j in pairs)
    d_t = max(dyaw(obs[i][1], obs[j][1]) for i, j in pairs)
    d_w = max(dyaw(obs[i][2], obs[j][2]) for i, j in pairs)
    d_r = max(float((obs[i][3] - obs[j][3]).norm()) for i, j in pairs)
    print(f"[smoke] randomization max-pairwise deltas: potT_x={d_x * 1000:.1f}mm "
          f"potT_yaw={d_t:.1f}deg potW_yaw={d_w:.1f}deg rake_xy={d_r * 1000:.1f}mm",
          flush=True)
    check("randomization-is-real: copper depth + yaw, steel yaw and rake xy "
          "readback all differ across seeds",
          d_x > 0.002 and d_t > 5.0 and d_w > 5.0 and d_r > 0.003)

    # ================= 3. copper-lane coin flip =================================================
    lanes = []
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        lanes.append(1 if float(scene.pot_local(scene.pot_t)[0, 1]) > 0 else -1)
    print(f"[smoke] copper-pot lanes over 10 resets: {lanes}", flush=True)
    check("lane coin-flip: the copper pot spawns in BOTH lanes over 10 resets "
          "(identity by appearance, not position)",
          1 in lanes and -1 in lanes)

    # ================= 4. null policy fails =====================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, pots stay deep in their lanes, "
          "score ~0, no success",
          bool(scene.in_cubby(scene.pot_t)[0]) and bool(scene.in_cubby(scene.pot_w)[0])
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. roof-jam: no top exit from the cubby ==================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    z0 = float(scene.pot_local(scene.pot_t)[0, 2])
    f_w = torch.zeros(n, 3, device=device)
    f_w[:, 2] = 1.3 * c.pot_mass * G
    max_z = z0
    for _ in range(150):
        q = scene.pot_t.data.root_quat_w
        scene.pot_t.set_external_force_and_torque(
            quat_apply_inverse(q, f_w).unsqueeze(1), zero3)
        _step(1)
        max_z = max(max_z, float(scene.pot_local(scene.pot_t)[0, 2]))
    scene.pot_t.set_external_force_and_torque(zero3, zero3)
    _step(300)
    _report("roof-jam")
    _REC["on"] = False
    rise = max_z - z0
    print(f"[smoke] roof-jam: rise={rise * 1000:.1f}mm max_local_z={max_z * 1000:.1f}mm",
          flush=True)
    check("roof-jam: a 1.3x-weight lift presses the pot up (verified risen) but "
          "it JAMS under the roof and never leaves the cubby; released it lands "
          "back upright in its lane — no extraction credit, no success",
          rise > 0.030 and max_z < 0.075
          and bool(scene.in_cubby(scene.pot_t)[0]) and bool(scene.pot_up(scene.pot_t)[0])
          and not bool(scene._l_out[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.205)

    # ================= 6. WRONG pot on the burner ===============================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_body(scene.pot_w, pad_spot(0.008))
    _step(300)
    _report("wrong-pot")
    _REC["on"] = False
    dw = float((scene.pot_w.data.root_pos_w[0, :2]
                - scene.stove.data.root_pos_w[0, :2]).norm())
    check("wrong-pot rejected: STEEL pot seated on the burner (verified) — "
          "pot_on_pad watches the copper pot only; score ~0, no success",
          dw <= c.pad_xy_tol and bool(scene.pot_up(scene.pot_w)[0])
          and not bool(scene.pot_on_pad()[0]) and bool(scene.in_cubby(scene.pot_t)[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 7. copper pot TIPPED on the burner =======================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_body(scene.pot_t, pad_spot(c.pot_across / 2 + 0.004), _qy(math.pi / 2))
    _step(300)
    _report("pot-tipped")
    _REC["on"] = False
    check("tipped rejected: copper pot lying on its SIDE on the burner — upright "
          "clause refuses the pad latch, no success",
          not bool(scene.pot_up(scene.pot_t)[0]) and not bool(scene.pot_on_pad()[0])
          and not bool(scene._l_pad[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.205)

    # ================= 8. extraction only: pot left on the apron caps at 0.50 ===================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    s8 = 1.0 if float(scene.pot_local(scene.pot_t)[0, 1]) > 0 else -1.0
    x08 = float(scene._x0[0])
    _write_body(scene.pot_t, cub_spot(x08 + 0.06, s8 * c.lane_y, 0.001))
    _step(30)
    _write_body(scene.pot_t,
                cub_spot(c.mouth_x + c.out_margin + 0.025, s8 * c.lane_y, 0.001))
    _step(240)
    _report("apron-only")
    _REC["on"] = False
    check("apron-only capped: copper pot dragged-path to rest on the apron — "
          "reach + out latches fire, score caps at 0.50, no success",
          bool(scene._l_reach[0]) and bool(scene._l_out[0]) and not bool(scene._l_pad[0])
          and bool(scene.pot_out()[0]) and not bool(scene.success()[0])
          and 0.495 <= float(scene.score()[0]) <= 0.505)

    # ================= 9. steel pot disturbed out of the cubby ==================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    # blocking condition FIRST: the steel pot displaced onto the open counter
    _write_body(scene.pot_w, counter_spot(0.55, 0.10, 0.001))
    _step(120)
    steel_out = not bool(scene.in_cubby(scene.pot_w)[0])
    deliver_copper()
    _report("steel-out")
    _REC["on"] = False
    check("steel-out rejected: steel pot on the open counter, copper pot fully "
          "delivered to the burner — all latches fire (cap 0.75) but "
          "in_cubby(steel) refuses success",
          steel_out and bool(scene.pot_on_pad()[0])
          and bool(scene._l_reach[0]) and bool(scene._l_out[0]) and bool(scene._l_pad[0])
          and not bool(scene.success()[0])
          and 0.745 <= float(scene.score()[0]) <= 0.7501)

    # ================= 10. rake left against the burner =========================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    # blocking condition FIRST: park the rake with its bar end just off the
    # stove plate (no contact) but well inside the clear radius
    _write_body(scene.rake,
                counter_spot(0.38, c.stove_pos[1], c.bar_t / 2 + c.flange_drop + 0.002))
    _step(240)
    rake_blocking = not bool(scene.rake_clear()[0])
    deliver_copper()
    _report("rake-at-stove")
    _REC["on"] = False
    check("rake-at-stove rejected: rake bar parked inside the clear radius "
          "(verified), copper pot fully delivered — cap 0.75, rake_clear "
          "refuses success",
          rake_blocking and bool(scene.pot_on_pad()[0]) and bool(scene.in_cubby(scene.pot_w)[0])
          and not bool(scene.rake_clear()[0]) and not bool(scene.success()[0])
          and 0.745 <= float(scene.score()[0]) <= 0.7501)

    # ================= save + verdict ===========================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.cubby_rake_stove")
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
    except BaseException:  # noqa: BLE001 - die fast; Kit teardown would hang until the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)
