"""Smoke / rubric-REJECTION battery for RidgePoiseScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts the constructed equilibrium and the
latched credit is monotone along a real trajectory). This battery proves the rubric
REJECTS wrong outcomes, and that every success() clause is load-bearing — including the
cheats that make bars PHYSICALLY STAND by the wrong means. Every probe is CONSTRUCTED
as a state (teleport, real physics steps, judge) — instrumentation, never a solution.

Checks:
   1. settle/no-NaN     — seeded reset settles finite; both bars flat on the floor,
                          slugs seated in their pockets; score 0, no success;
   2. randomization     — READBACK across seeded resets: ridge yaw / ridge xy /
                          red-bar spawn all differ, and the SLUG-POCKET pair (the
                          episode's hidden balance points) takes >= 3 distinct values;
   3. null-policy       — 300 idle steps: bars stay on the floor, score ~0;
   4. GEOMETRIC CENTER  — the money rejection: the red bar set down on the ridge by
                          its MIDDLE (slug aboard) — the off-center load tips it and
                          it falls to the floor; nothing latches, score ~0;
   5. SLUGLESS cheat    — both slugs teleported to the floor, both EMPTY bars stacked
                          centered on the ridge: the stack PHYSICALLY STANDS level in
                          the exact judged height bands — and is rejected by the
                          slug-in-own-bar clauses; score ~0 (the inference the task
                          exists for cannot be deleted);
   6. FLOOR STACK       — the correct cross-stack (balance points and all) built ON
                          THE FLOOR next to the ridge: blue genuinely rests poised on
                          red's rails — rejected because red is not on the ridge;
                          score ~0 (transport without poising is nothing);
   7. BLUE-BESIDE-RED   — red poised correctly, then blue poised DIRECTLY ON THE FIN
                          beside it (a genuine second balance, ends airborne) — a
                          real skill, and still rejected: blue must ride red's rails
                          (30 mm band), not the ridge; score stays at the red partial;
   8. INVERTED stack    — blue poised on the fin, red crosswise on blue: the full
                          two-tier stack stands — rejected by bar identity (red
                          carries, blue rides); score ~0;
   9. PARALLEL stack    — red poised, blue balanced on red's deck ALIGNED with it
                          (standing, correct heights) — rejected by the crosswise
                          clause; score stays at the red partial;
  10. REMOVE-SLUG       — full success CONSTRUCTED live (score 1.0), then the blue
                          slug teleported to the floor: blue's balance point jumps to
                          its middle, GRAVITY tips the blue bar off the stack,
                          success COLLAPSES while the latched score survives at the
                          0.60 partial cap — the equilibrium is physics, not
                          bookkeeping;
  11. rejection audit   — success() never fired at any judged step during the
                          negative probes (checks 4-9);
  12. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.approach_grasp_i29.smoke --headless
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
    from . import scene as task_scene  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
_AUDIT = {"on": False, "fired": False}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if _AUDIT["on"]:
            _AUDIT["fired"] |= bool(env.scene.success()[0])
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


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ridge_poise")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    RED, BLUE = 0, 1
    ratio = c.slug_mass / (c.slug_mass + c.bar_mass)
    hover = 0.008
    from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.30, -0.95, 0.80)) + o),
                                tuple(np.array((0.35, 0.00, 0.14)) + o),
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

    def bar(i):
        return scene.bars[c.bar_names[i]]

    def slug(i):
        return scene.slugs[c.slug_names[i]]

    def status():
        _refresh()
        return scene._status()

    def slug_loc(i) -> torch.Tensor:
        _refresh()
        return scene._local(bar(i), slug(i).data.root_pos_w)

    def x_bal(i) -> float:
        return ratio * float(slug_loc(i)[0, 0])

    def ridge_loc(body) -> torch.Tensor:
        _refresh()
        return scene._local(scene.ridge, body.data.root_pos_w)

    def report(tag: str) -> None:
        s = status()
        rl_r, rl_b = ridge_loc(bar(RED))[0], ridge_loc(bar(BLUE))[0]
        print(f"[smoke] {tag:16s} | red@ridge=({float(rl_r[0]):+.3f},{float(rl_r[1]):+.3f},"
              f"{float(rl_r[2]):+.3f}) blue@ridge=({float(rl_b[0]):+.3f},"
              f"{float(rl_b[1]):+.3f},{float(rl_b[2]):+.3f}) "
              f"red_poised={bool(s['red_poised'][0])} blue_on_red={bool(s['blue_on_red'][0])} "
              f"slugs=({bool(s['slug_red'][0])},{bool(s['slug_blue'][0])}) "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(_REC['frames'])}", flush=True)

    def score() -> float:
        _refresh()
        return float(scene.score()[0])

    def success() -> bool:
        _refresh()
        return bool(scene.success()[0])

    def co_place(i: int, loc_ridge_xyz, yaw_off: float, carry_slug: bool = True) -> None:
        """TRANSPORT bar i (and, if aboard, its slug — pose preserved RELATIVE to the
        bar) to a hover pose given in the RIDGE frame; zero velocity."""
        _refresh()
        q_ridge = scene.ridge.data.root_quat_w
        p_ridge = scene.ridge.data.root_pos_w
        rel_p = scene._local(bar(i), slug(i).data.root_pos_w).clone()
        rel_q = quat_mul(quat_inv(bar(i).data.root_quat_w), slug(i).data.root_quat_w)
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0], loc[:, 1], loc[:, 2] = loc_ridge_xyz
        yaw = torch.full((n,), yaw_off, device=device)
        qy = torch.zeros(n, 4, device=device)
        qy[:, 0], qy[:, 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        q_new = quat_mul(q_ridge, qy)
        p_new = p_ridge + quat_apply(q_ridge, loc)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = p_new
        st[:, 3:7] = q_new
        bar(i).write_root_state_to_sim(st, _all_ids())
        if carry_slug:
            ss = torch.zeros(n, 13, device=device)
            ss[:, 0:3] = p_new + quat_apply(q_new, rel_p)
            ss[:, 3:7] = quat_mul(q_new, rel_q)
            slug(i).write_root_state_to_sim(ss, _all_ids())

    def on_red_place(loc_red_xyz, yaw_off_world_of_red: float) -> None:
        """TRANSPORT the blue bar (slug co-carried) to a hover given in the RED bar's
        frame, long axis rotated `yaw_off` from RED's long axis; zero velocity."""
        _refresh()
        q_red = bar(RED).data.root_quat_w
        p_red = bar(RED).data.root_pos_w
        rel_p = scene._local(bar(BLUE), slug(BLUE).data.root_pos_w).clone()
        rel_q = quat_mul(quat_inv(bar(BLUE).data.root_quat_w),
                         slug(BLUE).data.root_quat_w)
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0], loc[:, 1], loc[:, 2] = loc_red_xyz
        yaw = torch.full((n,), yaw_off_world_of_red, device=device)
        qy = torch.zeros(n, 4, device=device)
        qy[:, 0], qy[:, 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        q_new = quat_mul(q_red, qy)
        p_new = p_red + quat_apply(q_red, loc)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = p_new
        st[:, 3:7] = q_new
        bar(BLUE).write_root_state_to_sim(st, _all_ids())
        ss = torch.zeros(n, 13, device=device)
        ss[:, 0:3] = p_new + quat_apply(q_new, rel_p)
        ss[:, 3:7] = quat_mul(q_new, rel_q)
        slug(BLUE).write_root_state_to_sim(ss, _all_ids())

    def to_floor(body, x: float, y: float, half_h: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.env_origins
        st[:, 0] += x
        st[:, 1] += y
        st[:, 2] += half_h + 0.003
        st[:, 3] = 1.0
        body.write_root_state_to_sim(st, _all_ids())

    def poise_red(center_x: float | None = None) -> None:
        """Constructed red placement: hover across the fin. center_x=None means the
        MEASURED balance point (correct); a number forces that ridge-x center."""
        xb = x_bal(RED)
        cx = -xb if center_x is None else center_x
        co_place(RED, (cx, 0.0, c.ridge_top_z + c.slab_t / 2 + hover), 0.0)
        _step(360)

    def poise_blue_on_fin(y_off: float) -> None:
        """Constructed BLUE placement directly on the fin (its own balance point)."""
        xb = x_bal(BLUE)
        co_place(BLUE, (-xb, y_off, c.ridge_top_z + c.slab_t / 2 + hover), 0.0)
        _step(360)

    def stack_blue_crosswise() -> None:
        """Constructed blue placement crosswise over red's deck (the solution move)."""
        xb = x_bal(BLUE)
        red_y = float(ridge_loc(bar(RED))[0, 1])
        co_place(BLUE, (0.0, red_y - xb,
                        c.ridge_top_z + c.slab_t + c.rail_h + c.slab_t / 2 + hover),
                 math.pi / 2)
        _step(420)

    def fresh(seed: int, settle: int = 150) -> None:
        torch.manual_seed(seed)
        env.reset()
        _step(settle)

    # ================= 1. settle / no-NaN =========================================================
    fresh(100)
    _REC["on"] = True
    _step(60)
    _REC["on"] = False
    report("settle")
    s = status()
    bodies = [*scene.bars.values(), *scene.slugs.values()]
    finite = all(bool(torch.isfinite(b.data.root_pos_w).all()) for b in bodies)
    floor_z = all(float(ridge_loc(bar(i))[0, 2]) < 0.05 for i in (RED, BLUE))
    check("settle/no-NaN: seeded reset settles finite; both bars flat on the floor "
          "with slugs seated in their pockets; score 0, no success",
          finite and floor_z and bool(s["slug_red"][0]) and bool(s["slug_blue"][0])
          and score() <= 0.01 and not success())

    # ================= 2. randomization is real (readback) ========================================
    def cell_of(i: int) -> int:
        x = float(slug_loc(i)[0, 0])
        return int(np.argmin([abs(x - cc) for cc in c.cell_centers]))

    obs = []
    for k in range(6):
        torch.manual_seed(300 + k)
        env.reset()
        _step(15)
        q = scene.ridge.data.root_quat_w
        yaw = math.degrees(2.0 * math.atan2(float(q[0, 3]), float(q[0, 0])))
        obs.append((yaw, scene.ridge.data.root_pos_w[0, :2].clone(),
                    bar(RED).data.root_pos_w[0, :2].clone(),
                    (cell_of(RED), cell_of(BLUE))))
    d_yaw = max(o[0] for o in obs) - min(o[0] for o in obs)
    d_rp = max(float((a[1] - b[1]).norm()) for a in obs for b in obs)
    d_bar = max(float((a[2] - b[2]).norm()) for a in obs for b in obs)
    pockets = {o[3] for o in obs}
    print(f"[smoke] randomization: ridge yaw spread {d_yaw:.1f}deg, ridge xy spread "
          f"{d_rp * 1000:.0f}mm, red-bar spawn spread {d_bar * 1000:.0f}mm, "
          f"pocket pairs {sorted(pockets)}", flush=True)
    check("randomization-is-real: ridge yaw / ridge xy / red-bar spawn READBACK all "
          "vary and the slug-pocket pair (the hidden balance points, read from the "
          "slugs' settled bar-frame positions) takes >= 3 values across 6 resets",
          d_yaw > 5.0 and d_rp > 0.005 and d_bar > 0.05 and len(pockets) >= 3)

    # ================= 3. null policy fails =======================================================
    fresh(100)
    _step(300)
    report("null-policy")
    check("null-policy-fails: 300 idle steps, both bars still on the floor, "
          "score ~0, no success",
          float(ridge_loc(bar(RED))[0, 2]) < 0.05 and score() <= 0.01
          and not success())

    _AUDIT["on"] = True  # ---- negative probes below: success must never fire ----

    # ================= 4. GEOMETRIC CENTER: the money rejection ===================================
    fresh(100)
    _REC["on"] = True
    print(f"[smoke] geometric-center probe: red slug_x={float(slug_loc(RED)[0, 0]):+.4f}"
          f" (balance point {x_bal(RED) * 1000:+.1f} mm) — placing by the MIDDLE",
          flush=True)
    poise_red(center_x=0.0)  # middle over the fin: CoM 39-62 mm off the flat
    _step(240)
    report("geom-center")
    _REC["on"] = False
    s = status()
    red_tipped = (not bool(scene._level(bar(RED))[0])) \
        and float(ridge_loc(bar(RED))[0, 2]) < (c.ridge_top_z + c.slab_t / 2 - 0.005)
    check("negative (GEOMETRIC CENTER): the red bar set down on the ridge by its "
          "middle, slug aboard — the off-center load TIPS it off the 24 mm flat "
          "(ends wrecked: off-level, fallen out of the poised band, typically "
          "leaning against the ridge); not poised, nothing latches, score ~0",
          red_tipped and not bool(s["red_poised"][0])
          and score() <= 0.01 and not success())

    # ================= 5. SLUGLESS cheat: the stack stands, the slugs are gone ===================
    fresh(100)
    to_floor(slug(RED), -0.70, 0.50, c.slug_size[2] / 2)
    to_floor(slug(BLUE), -0.70, -0.50, c.slug_size[2] / 2)
    _step(120)
    _REC["on"] = True
    co_place(RED, (0.0, 0.0, c.ridge_top_z + c.slab_t / 2 + hover), 0.0,
             carry_slug=False)
    _step(300)
    on_red_place((0.0, 0.0, c.slab_t + c.rail_h + hover), math.pi / 2)
    _step(420)
    report("slugless")
    _REC["on"] = False
    s = status()
    stands = bool(s["red_poised"][0]) and bool(s["blue_on_red"][0])
    check("near-miss (SLUGLESS cheat): both slugs dumped on the floor, both EMPTY "
          "bars stacked centered on the ridge — the cross-stack physically STANDS "
          "level in the exact judged bands, and is rejected by the slug-in-own-bar "
          "clauses; score ~0 (deleting the load deletes the task's inference)",
          stands and not bool(s["slug_red"][0]) and not bool(s["slug_blue"][0])
          and not success() and score() <= 0.01)

    # ================= 6. FLOOR STACK: correct pair, wrong place ==================================
    fresh(100)
    _REC["on"] = True
    co_place(RED, (0.32, 0.0, c.slab_t / 2 + hover), 0.0)  # flat ground beside the ridge
    _step(240)
    on_red_place((0.0, -x_bal(BLUE), c.slab_t + c.rail_h + hover), math.pi / 2)
    _step(420)
    report("floor-stack")
    _REC["on"] = False
    s = status()
    check("negative (FLOOR STACK / seed strategy): the correct cross-stack — blue "
          "genuinely poised crosswise on red's rails at its balance point — built ON "
          "THE FLOOR beside the ridge: transported but never poised on the fin; "
          "blue_on_red holds, red_poised does not, no success, score ~0",
          bool(s["blue_on_red"][0]) and bool(s["slug_red"][0])
          and bool(s["slug_blue"][0]) and not bool(s["red_poised"][0])
          and not success() and score() <= 0.01)

    # ================= 7. BLUE-BESIDE-RED: two balances, no stack =================================
    fresh(100)
    _REC["on"] = True
    poise_red()
    st7 = status()
    ok_red = bool(st7["red_poised"][0])
    poise_blue_on_fin(y_off=0.10)
    _step(240)
    report("blue-beside-red")
    _REC["on"] = False
    s = status()
    blue_rl = ridge_loc(bar(BLUE))[0]
    blue_on_fin = bool(scene._level(bar(BLUE))[0]) \
        and bool(scene._settled(bar(BLUE))[0]) \
        and abs(float(blue_rl[2]) - (c.ridge_top_z + c.slab_t / 2)) < c.z_tol
    check("near-miss (BLUE BESIDE RED): red poised correctly AND blue poised "
          "directly on the fin beside it — a genuine second balance, ends airborne — "
          "still rejected: blue must ride the red bar's rails (30 mm higher band); "
          "score stays at the red partial (0.30)",
          ok_red and blue_on_fin and not bool(s["blue_on_red"][0])
          and not success() and abs(score() - c.w_poised) < 0.011)

    # ================= 8. INVERTED stack: identity is load-bearing ================================
    # The drop of the 0.28 kg red assembly onto a bar poised on the 24 mm fin can rock
    # it off on impact (solve.py's stacking phase retries for the same reason), so this
    # probe retries the CONSTRUCTION until the inverted stack genuinely stands; the
    # rejection clauses are then evaluated on the standing stack.
    _REC["on"] = True
    z_top = c.ridge_top_z + c.slab_t + c.rail_h + c.slab_t / 2
    inverted_stands = False
    for attempt in range(6):
        fresh(100 + attempt)
        xb = x_bal(BLUE)
        co_place(BLUE, (-xb, 0.0, c.ridge_top_z + c.slab_t / 2 + hover), 0.0)
        _step(420)
        _refresh()
        blue_rl8 = ridge_loc(bar(BLUE))[0]
        if not (bool(scene._level(bar(BLUE))[0])
                and bool(scene._settled(bar(BLUE))[0])
                and abs(float(blue_rl8[2]) - (c.ridge_top_z + c.slab_t / 2)) < c.z_tol):
            continue  # blue itself failed to poise this attempt
        # red crosswise on BLUE's deck (mirror of the solution move, roles swapped):
        # red-local x maps to blue-frame y after the 90 deg yaw, so red's slug-shifted
        # CoM lands centered between blue's rails when the red bar sits at blue-frame
        # (x=0, y=-x_bal(RED)); the whole stack's CoM stays over the fin
        q_blue = bar(BLUE).data.root_quat_w.clone()
        p_blue = bar(BLUE).data.root_pos_w.clone()
        rel_p = scene._local(bar(RED), slug(RED).data.root_pos_w).clone()
        rel_q = quat_mul(quat_inv(bar(RED).data.root_quat_w),
                         slug(RED).data.root_quat_w)
        yaw = torch.full((n,), math.pi / 2, device=device)
        qy = torch.zeros(n, 4, device=device)
        qy[:, 0], qy[:, 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        q_new = quat_mul(q_blue, qy)
        off = torch.zeros(n, 3, device=device)
        # blue is poised with the FIN under its CoM, i.e. at blue-frame x = x_bal(BLUE)
        # (not under its center) — red must be set down over the fin, else the combined
        # CoM leaves the 24 mm flat and the whole stack tips
        off[:, 0] = xb
        off[:, 1] = -x_bal(RED)
        off[:, 2] = c.slab_t + c.rail_h + 0.004  # gentle 4 mm drop onto the rails
        p_new = p_blue + quat_apply(q_blue, off)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = p_new
        st[:, 3:7] = q_new
        bar(RED).write_root_state_to_sim(st, _all_ids())
        ss = torch.zeros(n, 13, device=device)
        ss[:, 0:3] = p_new + quat_apply(q_new, rel_p)
        ss[:, 3:7] = quat_mul(q_new, rel_q)
        slug(RED).write_root_state_to_sim(ss, _all_ids())
        _step(420)
        _refresh()
        red_rl = ridge_loc(bar(RED))[0]
        blue_rl8 = ridge_loc(bar(BLUE))[0]
        inverted_stands = bool(scene._level(bar(RED))[0]) \
            and bool(scene._settled(bar(RED))[0]) \
            and abs(float(red_rl[2]) - z_top) < 2 * c.z_tol \
            and bool(scene._level(bar(BLUE))[0]) \
            and abs(float(blue_rl8[2]) - (c.ridge_top_z + c.slab_t / 2)) < c.z_tol
        if inverted_stands:
            break
        print(f"[smoke] inverted attempt {attempt} did not stand, retrying", flush=True)
    report("inverted")
    _REC["on"] = False
    s = status()
    check("near-miss (INVERTED stack): blue poised on the fin with red crosswise on "
          "top — the two-tier stack STANDS and is rejected by bar identity (red "
          "carries, blue rides): no clause holds, score ~0",
          inverted_stands and not bool(s["red_poised"][0])
          and not bool(s["blue_on_red"][0]) and not success() and score() <= 0.01)

    # ================= 9. PARALLEL stack: crosswise is load-bearing ===============================
    fresh(100)
    _REC["on"] = True
    poise_red()
    ok_red9 = bool(status()["red_poised"][0])
    # blue ALIGNED with red on the rail deck, its balance point over the fin plane
    xb = x_bal(BLUE)
    co_place(BLUE, (-xb, 0.0,
                    c.ridge_top_z + c.slab_t + c.rail_h + c.slab_t / 2 + hover), 0.0)
    _step(420)
    report("parallel")
    _REC["on"] = False
    s = status()
    blue_loc_red = scene._local(bar(RED), bar(BLUE).data.root_pos_w)[0]
    parallel_stands = bool(scene._level(bar(BLUE))[0]) \
        and bool(scene._settled(bar(BLUE))[0]) \
        and abs(float(blue_loc_red[2]) - (c.slab_t + c.rail_h)) < 2 * c.z_tol
    check("near-miss (PARALLEL stack): blue balanced on red's deck ALIGNED with red "
          "— standing at the correct height, both bars level — rejected by the "
          "crosswise clause; score stays at the red partial (0.30)",
          ok_red9 and parallel_stands and not bool(s["blue_on_red"][0])
          and not success() and abs(score() - c.w_poised) < 0.011)

    _AUDIT["on"] = False  # ---- end of negative probes ----

    # ================= 10. REMOVE-SLUG: success collapses, latches survive ========================
    fresh(100)
    _REC["on"] = True
    poise_red()
    stack_blue_crosswise()
    for _ in range(8):
        if success():
            break
        _step(60)
    report("full-solution")
    built = success()
    score_before = score()
    to_floor(slug(BLUE), -0.70, -0.50, c.slug_size[2] / 2)
    _step(500)
    report("slug-removed")
    _REC["on"] = False
    s = status()
    blue_rl10 = ridge_loc(bar(BLUE))[0]
    z_ride = c.ridge_top_z + c.slab_t + c.rail_h + c.slab_t / 2
    blue_dislodged = (not bool(scene._level(bar(BLUE))[0])) \
        or abs(float(blue_rl10[2]) - z_ride) > 1.5 * c.z_tol
    score_after = score()
    check("REMOVE-SLUG: constructed success is live (score 1.0), then the blue slug "
          "teleported to the floor — the blue bar's balance point jumps to its "
          "middle and GRAVITY dislodges it (tipped off-level / out of the ride "
          "band); success COLLAPSES while the latched score survives at the 0.60 "
          "partial cap",
          built and score_before >= 0.999 and blue_dislodged
          and not bool(s["blue_on_red"][0])
          and not success() and abs(score_after - 0.60) < 0.011)

    # ================= 11. rejection audit ========================================================
    check("rejection audit: success() never fired at any step of the negative "
          "probes (checks 4-9)", not _AUDIT["fired"])

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ridge_poise")
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
    except BaseException as e:  # noqa: BLE001 - die fast, never idle until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL (exception: {e})", flush=True)
        os._exit(2)
