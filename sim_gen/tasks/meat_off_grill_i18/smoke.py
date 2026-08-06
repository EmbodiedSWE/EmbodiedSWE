"""Smoke / rubric-REJECTION battery for SearServeScene — NullRobot, teleported probes.

solve.py (the real Franka solution) is the acceptance proof: it demonstrates the rubric
accepts correct outcomes and that latched credit is monotone along a real trajectory.
This battery proves the rubric REJECTS wrong outcomes. Every probe is CONSTRUCTED as a
settled state (teleport, real physics steps, judge) — instrumentation, never a solution:
no probe here reaches success().

Checks:
  1. settle/no-NaN   — reset layout (raw patties on the board) settles finite, score 0;
  2. randomization   — READBACK across 3 seeds: grill/plate move, yaw changes;
  3. subset sampling — patty count varies across resets (1 and 2 both occur);
  4. null policy     — 240 idle steps: score ~0, cooks nothing, never succeeds;
  5. SEED STRATEGY   — the seed's whole plan is transport: every present patty carried
                       straight to the plate, settled flat (sanity-asserted via the
                       geometric on_plate) -> score exactly 0, success rejected;
  6. near-miss early — the seed's "grab it off the grill promptly": pulled off at
                       0.6*cook_min, then plated perfectly -> not cooked, no credit;
  7. overcook        — left on the pad past burn_time, then plated perfectly: the spoil
                       latch holds (served False, ~0 score) ...
  8. terminality     — ... and is TERMINAL (more steps never un-spoil it);
  9. near-miss pad   — patty resting 25 mm off the pad center (just outside the
                       fully-inside slack): the cook clock never starts;
 10. wrong surface   — patty resting on the grill BODY beside the pad: no accrual;
 11. wrong pose      — a COOKED patty lying on its SIDE on the plate: not served;
 12. wrong place     — a COOKED patty delivered to the prep board: not served;
 13. one-slot        — two patties can never BOTH be on the pad: dry config arithmetic
                       + a physical stacking attempt;
 14-15. calibration  — cook clock rate ~1 s/s of real pad time; the raw / cooked /
                       burned window boundaries land where configured.

ALWAYS records video via the viewport rgb annotator (RTX driver-version override, 3-render
ghost flush) and saves frames.npz in the CURRENT WORKING DIRECTORY. Bodies are driven
straight through scene handles; the NullRobot applies nothing.

Run (forge): python -m simgen_tasks.meat_off_grill_i18.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=10)
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
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

try:
    from .scene import SearServeSceneCfg  # noqa: E402  (import registers the scene + env)
except ImportError:  # standalone execution fallback
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import SearServeSceneCfg  # noqa: E402


# ----- driver: stepping + recording ---------------------------------------------------------------
class Driver:
    """Steps the env with the empty NullRobot action, records viewport frames, and provides
    predicate-polled settling (settling time is physics, not what the checks are about)."""

    def __init__(self, env) -> None:
        self.env = env
        self.scene = env.scene
        self.no_action = torch.empty(0, device=env.device)
        self.all_ids = torch.arange(env.num_envs, device=env.device)
        self.step_i = 0
        self.frames: list[np.ndarray] = []
        self.annot = None

    def make_state(self, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> torch.Tensor:
        st = torch.zeros(self.env.num_envs, 13, device=self.env.device)
        st[:, 0:3] = self.env.iscene.env_origins + torch.tensor(
            [float(v) for v in pos], device=self.env.device)
        st[:, 3:7] = torch.tensor([float(v) for v in quat], device=self.env.device)
        return st

    def step(self, k: int) -> None:
        for _ in range(k):
            self.env.step(self.no_action, render=True)
            if self.annot is not None and self.step_i % args.record_every == 0:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    self.env.sim.render()
                arr = np.asarray(self.annot.get_data())
                if arr.size:
                    self.frames.append(arr[..., :3].astype(np.uint8).copy())
            self.step_i += 1

    def settle_until(self, pred, max_steps: int = 300, poll: int = 12) -> bool:
        self.step(poll)  # ALWAYS force real steps first (zero-step teleport trap)
        if pred():
            return True
        waited = poll
        while waited < max_steps:
            self.step(poll)
            waited += poll
            if pred():
                return True
        return False

    def put_patty(self, name: str, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        """Teleport a patty (zero velocity) and refresh buffers so the authored pose is
        what the very next predicates / physics step see."""
        self.scene.patties[name].write_root_state_to_sim(
            self.make_state(pos, quat), self.all_ids)
        self.env.iscene.update(0.0)


# ----- main battery ---------------------------------------------------------------------------------
def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.sear_and_serve")().build(
        num_envs=args.num_envs, device=device, scene_cfg=SearServeSceneCfg())
    scene = env.scene
    c = scene.cfg
    drv = Driver(env)
    origin0 = env.iscene.env_origins[0]
    dt = env.dt

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin0.detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.00, -1.00, 0.85)) + o),
                                tuple(np.array((-0.10, 0.00, 0.05)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        drv.annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        drv.annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(drv.annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)
        drv.annot = None

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def report(tag: str) -> None:
        print(f"[smoke] {tag:12s} | present={scene.present[0].int().tolist()} "
              f"cook_t={[f'{v:.2f}' for v in scene.cook_t[0].tolist()]} "
              f"cooked={scene.cooked()[0].int().tolist()} "
              f"burned={scene.burned[0].int().tolist()} "
              f"on_pad={scene.on_pad()[0].int().tolist()} "
              f"on_plate={scene.on_plate()[0].int().tolist()} "
              f"served={scene.served()[0].int().tolist()} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(drv.frames)}", flush=True)

    def first_present() -> int:
        return int(torch.nonzero(scene.present[0])[0])

    def pad_target(i: int, off_xy=(0.0, 0.0)) -> tuple:
        """Rest pose on the sear pad, offset in the GRILL body frame (yaw-aware)."""
        from isaaclab.utils.math import quat_apply

        gp = scene.grill.data.root_pos_w[0] - origin0
        gq = scene.grill.data.root_quat_w[0]
        h = c.patties[i][2]
        loc = torch.tensor([off_xy[0], off_xy[1], c.pad_top_local + h / 2 + 0.001],
                           device=env.device)
        w = gp + quat_apply(gq.unsqueeze(0), loc.unsqueeze(0))[0]
        return (float(w[0]), float(w[1]), float(gp[2]) + c.pad_top_local + h / 2 + 0.001)

    def body_top_target(i: int, off_x: float) -> tuple:
        """Rest pose on the grill BODY top beside the pad (grill frame offset)."""
        from isaaclab.utils.math import quat_apply

        gp = scene.grill.data.root_pos_w[0] - origin0
        gq = scene.grill.data.root_quat_w[0]
        h = c.patties[i][2]
        loc = torch.tensor([off_x, 0.0, 0.0], device=env.device)
        w = gp + quat_apply(gq.unsqueeze(0), loc.unsqueeze(0))[0]
        return (float(w[0]), float(w[1]),
                float(gp[2]) + c.grill_size[2] / 2 + h / 2 + 0.001)

    def plate_target(i: int, slot: int = 0) -> tuple:
        plate_p = (scene.plate.data.root_pos_w[0] - origin0).tolist()
        sx, sy = c.plate_slots[slot]
        h = c.patties[i][2]
        return (plate_p[0] + sx, plate_p[1] + sy, plate_p[2] + c.plate_t / 2 + h / 2 + 0.002)

    def board_target(i: int) -> tuple:
        bp = (scene.board.data.root_pos_w[0] - origin0).tolist()
        h = c.patties[i][2]
        return (bp[0], bp[1], bp[2] + c.board_size[2] / 2 + h / 2 + 0.002)

    def cook_to(i: int, dwell_s: float) -> None:
        """Probe constructor: put patty i on the pad and step real dwell time."""
        drv.put_patty(list(scene.patties)[i], pad_target(i))
        drv.step(int(round(dwell_s / dt)))

    # =========================== 1. show =======================================================
    torch.manual_seed(0)
    env.reset()
    report("reset")
    drv.step(60)
    report("show")
    finite = all(torch.isfinite(b.data.root_state_w).all()
                 for b in [scene.board, scene.grill, scene.plate, *scene.patties.values()])
    check("reset settles finite with score 0",
          finite and float(scene.score()[0]) == 0.0 and not bool(scene.success()[0]))

    # =========================== 2. randomization by readback ==================================
    obs = []
    for s in (201, 202, 203):
        torch.manual_seed(s)
        env.reset()
        drv.step(5)
        obs.append((scene.grill.data.root_pos_w[0, :2] - origin0[:2],
                    scene.plate.data.root_pos_w[0, :2] - origin0[:2],
                    scene.grill.data.root_quat_w[0].clone()))
    grill_moves = max(float((a[0] - b[0]).norm()) for a in obs for b in obs)
    plate_moves = max(float((a[1] - b[1]).norm()) for a in obs for b in obs)
    yaw_moves = min(float((a[2] - b[2]).norm()) for i, a in enumerate(obs)
                    for j, b in enumerate(obs) if i < j)
    print(f"[smoke] randomization: grill dxy={grill_moves * 1000:.1f}mm "
          f"plate dxy={plate_moves * 1000:.1f}mm grill dq={yaw_moves:.3f}", flush=True)
    check("randomization is real (grill+plate move on readback, yaw changes)",
          grill_moves > 0.005 and plate_moves > 0.005 and yaw_moves > 0.01)
    counts = set()
    for s in range(210, 222):
        torch.manual_seed(s)
        env.reset()
        counts.add(int(scene.present[0].sum()))
    print(f"[smoke] subset-sampled patty counts over 12 resets: {sorted(counts)}", flush=True)
    check("patty-count subset sampling varies", counts == {1, 2})

    # =========================== 3. null policy ================================================
    torch.manual_seed(42)
    env.reset()
    drv.step(240)
    report("null")
    check("null policy scores ~0, cooks nothing, no success",
          float(scene.score()[0]) <= 0.05 and float(scene.cook_t.max()) == 0.0
          and not bool(scene.success()[0]))

    # =========================== 4. negative A: the seed's own strategy =========================
    # meat_off_grill's plan is pure transport: pick the meat up and put it at the target.
    # Execute it perfectly — every present patty straight from the board onto the plate,
    # settled flat (sanity-asserted via the geometric on_plate) — and it must earn NOTHING.
    torch.manual_seed(11)
    env.reset()
    drv.step(50)
    slot = 0
    for i, name in enumerate(scene.patties):
        if not bool(scene.present[0, i]):
            continue
        drv.put_patty(name, plate_target(i, slot))
        slot += 1
    drv.step(90)
    report("seed-strat")
    plated = bool((scene.on_plate()[0] | ~scene.present[0]).all())
    check("negative A: seed strategy (raw meat straight to the plate) scores 0",
          plated and float(scene.score()[0]) <= 1e-6 and not bool(scene.success()[0]))

    # =========================== 5. near-miss: pulled off too early =============================
    torch.manual_seed(13)
    env.reset()
    drv.step(50)
    i0 = first_present()
    name0 = list(scene.patties)[i0]
    cook_to(i0, 0.6 * c.cook_min)  # the seed's prompt grab-off: 0.9 s < cook_min
    early_ct = float(scene.cook_t[0, i0])
    drv.put_patty(name0, plate_target(i0, 0))
    drv.settle_until(lambda: bool(scene.on_plate()[0, i0]) and bool(scene.settled()[0, i0]),
                     max_steps=240)
    report("near-miss")
    print(f"[smoke]   near-miss cook_t at removal: {early_ct:.2f}s "
          f"(cook_min {c.cook_min:.2f}s)", flush=True)
    check("near-miss: patty pulled off the grill too early is not cooked, no serve credit",
          not bool(scene.cooked()[0, i0]) and not bool(scene.served()[0, i0])
          and float(scene.score()[0]) <= 0.21 and not bool(scene.success()[0]))

    # =========================== 6-7. negative B: overcook + terminality ========================
    torch.manual_seed(12)
    env.reset()
    drv.step(50)
    i0 = first_present()
    name0 = list(scene.patties)[i0]
    cook_to(i0, c.burn_time + 0.5)
    burned_now = bool(scene.burned[0, i0])
    cooked_too = bool(scene.cooked()[0, i0])
    drv.put_patty(name0, plate_target(i0, 0))
    drv.settle_until(lambda: bool(scene.on_plate()[0, i0]) and bool(scene.settled()[0, i0]),
                     max_steps=240)
    report("overcook")
    check("negative B: overcooked patty is spoiled — plating it earns ~nothing",
          burned_now and cooked_too and not bool(scene.served()[0, i0])
          and float(scene.score()[0]) <= 0.15 and not bool(scene.success()[0]))
    drv.step(120)
    check("negative B: the burn latch is terminal",
          bool(scene.burned[0, i0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.15)

    # =========================== 8. near-miss: just outside the pad slack =======================
    # Steak fully-inside slack is pad_half - r = 15.5 mm; rest it 25 mm off center (still
    # physically ON the pad surface) — the cook clock must never start.
    torch.manual_seed(15)
    env.reset()
    drv.step(50)
    i0 = first_present()
    drv.put_patty(list(scene.patties)[i0], pad_target(i0, off_xy=(0.025, 0.0)))
    drv.step(int(round(c.cook_min / dt)))
    report("off-center")
    check("near-miss: patty resting 25 mm off the pad center (outside fully-inside "
          "slack) never starts the cook clock",
          float(scene.cook_t[0, i0]) == 0.0 and not bool(scene.ever_on_pad[0, i0])
          and float(scene.score()[0]) <= 1e-6)

    # =========================== 9. wrong surface: grill body beside the pad ====================
    torch.manual_seed(16)
    env.reset()
    drv.step(50)
    i0 = first_present()
    drv.put_patty(list(scene.patties)[i0], body_top_target(i0, off_x=0.085))
    drv.step(int(round(c.cook_min / dt)))
    report("body-top")
    check("wrong surface: patty resting on the grill body beside the pad accrues nothing",
          float(scene.cook_t[0, i0]) == 0.0 and float(scene.score()[0]) <= 1e-6)

    # =========================== 10. wrong pose: cooked patty lying on its side =================
    torch.manual_seed(17)
    env.reset()
    drv.step(50)
    i0 = first_present()
    name0 = list(scene.patties)[i0]
    cook_to(i0, 1.2 * c.cook_min)  # properly cooked, not burned
    tgt = plate_target(i0, 0)
    r0 = c.patties[i0][1]
    c45 = math.cos(math.pi / 4)
    drv.put_patty(name0, (tgt[0], tgt[1],
                          float(scene.plate.data.root_pos_w[0][2] - origin0[2])
                          + c.plate_t / 2 + r0 + 0.002),
                  quat=(c45, 0.0, c45, 0.0))  # pitched 90 deg: on its rolling edge
    drv.step(150)
    report("lying")
    check("wrong pose: a COOKED patty lying on its side is never served",
          bool(scene.cooked()[0, i0]) and not bool(scene.burned[0, i0])
          and not bool(scene.served()[0, i0]) and not bool(scene.success()[0]))

    # =========================== 11. wrong place: cooked patty back on the board ================
    torch.manual_seed(18)
    env.reset()
    drv.step(50)
    i0 = first_present()
    name0 = list(scene.patties)[i0]
    cook_to(i0, 1.2 * c.cook_min)
    drv.put_patty(name0, board_target(i0))
    drv.settle_until(lambda: bool(scene.settled()[0, i0]), max_steps=180)
    report("wrong-place")
    check("wrong place: a COOKED patty delivered to the prep board is not served, "
          "no success",
          bool(scene.cooked()[0, i0]) and not bool(scene.served()[0, i0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.50)

    # =========================== 12. one-slot pad ================================================
    r0, r1 = c.patties[0][1], c.patties[1][1]
    slack_diag = math.hypot((c.pad_half - r0) + (c.pad_half - r1),
                            (c.pad_half - r0) + (c.pad_half - r1))
    print(f"[smoke] one-slot dry math: max both-inside separation {slack_diag * 1000:.1f}mm "
          f"< touching distance {(r0 + r1) * 1000:.1f}mm", flush=True)
    seed_both = None
    for s in range(80, 130):
        torch.manual_seed(s)
        env.reset()
        if int(scene.present[0].sum()) == 2:
            seed_both = s
            break
    drv.step(40)
    names = list(scene.patties)
    drv.put_patty(names[0], pad_target(0))
    drv.step(30)
    gp = pad_target(0)
    drv.put_patty(names[1], (gp[0], gp[1],
                             gp[2] + c.patties[0][2] / 2 + c.patties[1][2] / 2 + 0.001))
    drv.step(60)
    report("stacked")
    both_on = bool(scene.on_pad()[0, 0]) and bool(scene.on_pad()[0, 1])
    check("one-slot pad: two patties can never both be on the pad (dry math + stacking probe)",
          slack_diag < r0 + r1 - 0.008 and not both_on and seed_both is not None)

    # =========================== 13-14. calibration: the cook clock =============================
    outcomes = []
    for dwell in (0.5 * c.cook_min, 1.2 * c.cook_min, c.burn_time + 0.5):
        torch.manual_seed(14)
        env.reset()
        drv.step(40)
        i0 = first_present()
        cook_to(i0, dwell)
        outcomes.append((dwell, float(scene.cook_t[0, i0]),
                         bool(scene.cooked()[0, i0]), bool(scene.burned[0, i0])))
    for dwell, ct, ck, bn in outcomes:
        print(f"[smoke] CALIBRATION dwell={dwell:.2f}s -> cook_t={ct:.2f}s "
              f"cooked={ck} burned={bn}", flush=True)
    rate_ok = all(0.80 <= ct / dwell <= 1.05 for dwell, ct, _ck, _bn in outcomes)
    check("calibration: cook clock tracks real pad time (rate ~1 s/s)", rate_ok)
    check("calibration: window boundaries behave (raw / cooked / burned)",
          outcomes[0][2] is False and outcomes[0][3] is False
          and outcomes[1][2] is True and outcomes[1][3] is False
          and outcomes[2][3] is True)

    # =========================== save + verdict ================================================
    arr = (np.stack(drv.frames, axis=0) if drv.frames
           else np.zeros((0, 600, 960, 3), np.uint8))
    np.savez_compressed(args.out, frames=arr, env="simgen.sear_and_serve")
    print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    check("video: frames captured and saved to frames.npz", len(drv.frames) > 10)

    n_ok = sum(ok for _n, ok in checks)
    if n_ok == len(checks):
        print(f"SIM_GEN_SMOKE: ALL PASS {n_ok}/{len(checks)}", flush=True)
        code = 0
    else:
        for name, ok in checks:
            if not ok:
                print(f"[smoke] FAILED CHECK: {name}", flush=True)
        print(f"SIM_GEN_SMOKE: FAIL {n_ok}/{len(checks)}", flush=True)
        code = 1

    # Kit teardown hangs are routine — hard-exit behind a watchdog.
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()
