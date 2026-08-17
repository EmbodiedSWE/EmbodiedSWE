"""Smoke / rubric-REJECTION battery for SoundingWellsScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome, the sensed
answer matches the hidden state, and the latched credit is monotone along a real
trajectory on seeds hiding the deep well in every position). This battery proves the
rubric REJECTS wrong outcomes, and that the claims the task rests on — a filled well
physically refuses the rod far proud of the flush tolerance, the dummy always
protrudes retrievably — are load-bearing physics. Every probe is CONSTRUCTED as a
settled state (teleport, real physics steps, judge) — instrumentation, never a
solution: no probe here reaches success().

Checks:
  1. settle/no-NaN     — seeded reset settles finite; rod and dummy flat on the
                         ground, both slugs seated INSIDE the two decoy wells
                         (readback); score 0, no success;
  2. randomization     — three seeded resets (max-pairwise deltas): well_0 xy,
                         terrace yaw, and rod xy readback all differ;
  3. hidden-state      — over 9 resets the TRUE-well index takes at least 2 (of 3)
     coverage            distinct values (histogram printed);
  4. null-policy       — 240 idle steps: score ~0, no success (nothing moves);
  5. SEED-strategy     — the seed's whole plan is "deposit the object into the
     analog / decoy      receptacle": the rod is dropped upright into a DECOY well
     stall               exactly as into the true one. The hidden slug STOPS it: the
                         cap settles 25..50 mm PROUD of the rim (resting ON the
                         filler — not jammed at the mouth, asserted by the depth
                         band), sound credit latches (score = 0.50 cap) but success
                         REFUSES — depositing without sensing fails;
  6. near-miss         — the rod parked CAP-DOWN on the TRUE well's rim (inverted:
     (delivered, not     the wide cap cannot enter the bore, so it rests stably on
     planted)            the rim annulus, shaft pointing up). Delivered to the
                         right well and touching the rim — but not upright, tip in
                         no bore, cap 12 mm proud -> no sound credit, no success;
  7. wrong object      — the DUMMY BAR dropped upright into the TRUE well: it
                         bottoms out but ALWAYS protrudes >= 15 mm above the rim
                         (the retrievability claim is physics), and success refuses
                         (the capped probe rod is the goal object, not the bar);
  8. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.living_room_scene2_pick_up_the_butter_and_put_it_in_the_basket_i401.smoke --headless
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

_qmul, _qy, _qz = task_scene._qmul, task_scene._qy, task_scene._qz

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
    print(f"[smoke] {tag:16s} | true={int(scene.true_idx[0])} "
          f"rod_z={float(scene.rod.data.root_pos_w[0, 2]):+.3f} "
          f"up_z={float(scene._rod_up_z()[0]):+.2f} "
          f"gap={float(scene.cap_gap()[0]) * 1000:+.1f}mm "
          f"sounded={bool(scene.sounded_now()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.sounding_wells")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.15, -0.85, 0.70)) + o),
                                tuple(np.array((0.38, 0.00, 0.05)) + o),
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

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def drop_rod_into(i: int, settle: int = 150) -> None:
        """CONSTRUCT: rod upright over well i's mouth, tip 10 mm above the rim,
        yaw-aligned — then hands-off settle (what stops it is contact)."""
        _refresh()
        wp = scene.wells[i].data.root_pos_w
        wq = scene.wells[i].data.root_quat_w
        pos = wp.clone()
        pos[:, 2] += c.base_t + c.wall_h + 0.010 + c.shaft_len
        _write_body(scene.rod, pos, wq)
        _step(settle)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    t0 = int(scene.true_idx[0])
    slug_ok = True
    for k, slug in enumerate(scene.slugs):
        decoy = (t0 + 1 + k) % 3
        d = float((slug.data.root_pos_w[0, :2]
                   - scene.wells[decoy].data.root_pos_w[0, :2]).norm())
        sz = float(slug.data.root_pos_w[0, 2] - scene.wells[decoy].data.root_pos_w[0, 2])
        slug_ok &= d < 0.01 and c.base_t - 0.002 < sz < c.base_t + c.slug_h
    check("settle/no-NaN: layout settles finite; rod and dummy flat on the ground, "
          "slugs seated inside the two decoy wells; score 0, no success",
          bool(torch.isfinite(scene.rod.data.root_state_w).all())
          and bool(torch.isfinite(scene.dummy.data.root_state_w).all())
          and float(scene.rod.data.root_pos_w[0, 2]) < 0.06
          and float(scene.dummy.data.root_pos_w[0, 2]) < 0.05
          and slug_ok
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback, 3-seed max-pairwise) ===================
    def readback():
        _refresh()
        return (scene.wells[0].data.root_pos_w[0, :2].clone(),
                yaw_of(scene.wells[0].data.root_quat_w[0]),
                scene.rod.data.root_pos_w[0, :2].clone())

    obs = []
    for sd in (101, 202, 303):
        torch.manual_seed(sd)
        env.reset()
        _step(10)
        obs.append(readback())
    d_w = max(float((obs[a][0] - obs[b][0]).norm())
              for a, b in ((0, 1), (0, 2), (1, 2)))
    d_y = max(dyaw(obs[a][1], obs[b][1]) for a, b in ((0, 1), (0, 2), (1, 2)))
    d_r = max(float((obs[a][2] - obs[b][2]).norm())
              for a, b in ((0, 1), (0, 2), (1, 2)))
    print(f"[smoke] randomization max-pairwise deltas: well0_xy={d_w * 1000:.1f}mm "
          f"terrace_yaw={d_y:.1f}deg rod_xy={d_r * 1000:.1f}mm", flush=True)
    check("randomization-is-real: well_0 xy, terrace yaw, rod xy readback differ "
          "across seeds",
          d_w > 0.005 and d_y > 2.0 and d_r > 0.003)

    # ================= 3. hidden-state coverage ===================================================
    seen = {}
    for s in range(9):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        t = int(scene.true_idx[0])
        seen[t] = seen.get(t, 0) + 1
    print(f"[smoke] true-well histogram over 9 resets: {dict(sorted(seen.items()))}",
          flush=True)
    check("hidden-state coverage: the TRUE well index takes >= 2 distinct values "
          "over 9 resets", len(seen) >= 2)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. SEED-strategy analog: deposit into a DECOY well =========================
    # The seed's whole skill is "put the object in the receptacle". Do exactly that
    # without sensing: drop the rod into a decoy well with the very move that wins
    # in the deep one. The hidden slug must STOP it — cap 25..50 mm proud (resting
    # ON the filler at the asserted depth, so the stall is real slug contact, not a
    # mouth jam), sound credit latches (score caps at 0.50), success refuses.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    t = int(scene.true_idx[0])
    decoy = (t + 1) % 3
    drop_rod_into(decoy)
    _report("decoy-stall")
    gap_d = float(scene.rod.data.root_pos_w[0, 2] - scene._rim_z()[0, decoy])
    up_d = float(scene._rod_up_z()[0])
    print(f"[smoke] rod deposited into decoy well {decoy}: cap {gap_d * 1000:+.1f} mm "
          f"proud (design stall {(c.shaft_len - (c.wall_h - c.slug_h)) * 1000:.0f} mm), "
          f"up_z={up_d:+.2f}", flush=True)
    check("SEED-analog (deposit without sensing): rod dropped into a decoy well "
          "stalls on the hidden slug 25..50 mm proud of the rim — sound credit "
          "only (score = 0.50), no success",
          0.025 < gap_d < 0.050 and up_d > 0.95
          and bool(scene._sound[0]) and not bool(scene.success()[0])
          and 0.49 <= float(scene.score()[0]) <= 0.501)
    _REC["on"] = False

    # ================= 6. near-miss: delivered to the RIGHT well but not planted ==================
    # The rod parked CAP-DOWN on the true well's rim (inverted). The wide cap
    # cannot enter the bore, so it rests stably on the rim annulus with the shaft
    # pointing up. It must STAY up on the rim (readback — the probe is not
    # vacuous), yet earn nothing: not upright, tip in no bore, cap ~12 mm proud.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _refresh()
    wp = scene.wells[t].data.root_pos_w
    wq = scene.wells[t].data.root_quat_w
    pos = wp.clone()
    pos[:, 2] += c.base_t + c.wall_h + c.cap_t + 0.005
    quat = _qmul(wq, _qy(torch.full((n,), math.pi, device=device)))
    _write_body(scene.rod, pos, quat)
    _step(150)
    _report("cap-down")
    rim_t = float(scene._rim_z()[0, t])
    rod_z = float(scene.rod.data.root_pos_w[0, 2])
    din6 = float((scene.rod.data.root_pos_w[0, :2] - wp[0, :2]).norm())
    print(f"[smoke] rod parked cap-down on the true well {t}: origin z={rod_z:.3f} "
          f"(rim {rim_t:.3f}), xy off-centre {din6 * 1000:.1f} mm, "
          f"up_z={float(scene._rod_up_z()[0]):+.2f}", flush=True)
    check("near-miss (delivered, not planted): rod parked CAP-DOWN on the TRUE "
          "well's rim stays up on the rim — not upright, tip in no bore, no sound "
          "credit, no success, score <= 0.20",
          rod_z > rim_t + 0.005 and din6 < 0.02
          and float(scene._rod_up_z()[0]) < -0.9
          and not bool(scene.sounded_now()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.20)
    _REC["on"] = False

    # ================= 7. wrong object: the dummy bar in the TRUE well ============================
    # The uncapped dummy dropped upright into the deep well bottoms out but ALWAYS
    # protrudes (dummy_len > bore depth by >= 20 mm): the retrievability claim is
    # physics. And success refuses — the goal object is the capped probe rod.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _refresh()
    t = int(scene.true_idx[0])
    wp = scene.wells[t].data.root_pos_w
    wq = scene.wells[t].data.root_quat_w
    pos = wp.clone()
    pos[:, 2] += c.base_t + c.wall_h + 0.010 + c.dummy_len / 2
    _write_body(scene.dummy, pos, wq)
    _step(150)
    _report("dummy-in-well")
    dz_top = float(scene.dummy.data.root_pos_w[0, 2] + c.dummy_len / 2
                   - scene._rim_z()[0, t])
    din = float((scene.dummy.data.root_pos_w[0, :2] - wp[0, :2]).norm())
    print(f"[smoke] dummy planted in the TRUE well {t}: top {dz_top * 1000:+.1f} mm "
          f"above the rim, xy off-centre {din * 1000:.1f} mm", flush=True)
    check("wrong object: the uncapped dummy bar planted in the TRUE well protrudes "
          ">= 15 mm above the rim (retrievable) and earns no success",
          din < 0.02 and dz_top >= 0.015 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.20)
    _REC["on"] = False

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.sounding_wells")
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
