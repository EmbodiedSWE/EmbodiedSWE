"""Smoke / rubric-REJECTION battery for BurnerSnuffScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the
latched credit is monotone along a real trajectory). This battery proves the
rubric REJECTS wrong outcomes, and that the claims the task rests on — the color
identity of the target pad, the mouth-down flip, the enclosure tolerance, and
the physically inherent clear-before-cap order — are load-bearing. Every probe
is CONSTRUCTED as a settled state (teleport, real physics steps, judge) —
instrumentation, never a solution: no probe here reaches success().

Checks:
   1. settle/no-NaN     — seeded reset settles finite; kettle standing on the lit
                          burner post, cup mouth-UP on the deck; score 0, no success;
   2. randomization     — two seeded resets: READBACK burner xy, cup xy, cup yaw and
                          kettle yaw all differ;
   3. side permutation  — over 10 resets the green trivet appears on BOTH sides;
   4. null-policy       — 240 idle steps: score ~0, no success (everything stays put);
   5. kettle-only       — kettle correctly on the trivet, cup untouched (mouth-up at
                          its spawn): the stove is still lit -> no success, score <= 0.45;
   6. WRONG PAD         — kettle parked on the WHITE PLATE + cup correctly seated over
                          the post: everything done except the color binding -> the
                          trivet clause refuses, no success (color identity is load-
                          bearing, position is not);
   7. bare deck         — kettle on the bare deck (well clear of the burner) + cup
                          correctly seated -> no success;
   8. unflipped cover   — kettle on the trivet + cup placed mouth-UP resting on the
                          post top ("cover" without the 180-degree flip): rim reads far
                          above the band and the mouth points up -> not capped, no
                          success;
   9. cap near-miss xy  — kettle on the trivet + cup mouth-down GENUINELY seated on
                          the hob plate (rim in the band, post enclosed) but its axis
                          19 mm off the post axis (physically possible: annular
                          clearance is 22 mm; tolerance is 15 mm) -> capped refuses,
                          no success;
  10. beside-burner     — kettle on the trivet + cup mouth-down seated flat on the
                          DECK beside the burner (rim at deck level reads inside the
                          rim band!) -> the axis clause alone refuses, no success;
  11. BLOCKED CAP       — fresh reset (kettle still ON the post): the cup is dropped
                          mouth-down from a hover over the OCCUPIED burner — it lands
                          high on the kettle (98 mm body vs 94 mm interior) and can
                          never reach the rim band -> not capped, no success: the
                          clear-before-cap order is physics, not fiat;
  12. kettle-perch      — cup correctly seated over the post, kettle then rested on
                          TOP of the seated cup (both bodies "at the burner", flame
                          covered) -> kettle is not on the trivet, no success;
  13. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_kitchen_scene8_turn_off_the_stove_i13.smoke --headless
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

_qmul, _qz, _qx = task_scene._qmul, task_scene._qz, task_scene._qx

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
    print(f"[smoke] {tag:18s} | on_burner={bool(scene.kettle_on_burner()[0])} "
          f"on_trivet={bool(scene.kettle_on_trivet()[0])} "
          f"capped={bool(scene.capped()[0])} "
          f"cup_upz={float(scene._up_w(scene.cup)[0, 2]):+.2f} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.burner_snuff")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.10, -0.75, 0.70)) + o),
                                tuple(np.array((0.28, 0.00, 0.07)) + o),
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

    def down_quat(yaw: float = 0.0) -> torch.Tensor:
        """Mouth-down cup orientation (180-degree flip about x, then yaw)."""
        return _qmul(_qz(torch.full((n,), yaw, device=device)),
                     _qx(torch.full((n,), math.pi, device=device)))

    def put(body, xy_w: torch.Tensor, z_center: float | torch.Tensor,
            quat: torch.Tensor | None = None, drop: float = 0.004) -> None:
        """CONSTRUCT: write the body just above its rest pose (probe
        instrumentation; the caller settles when the arrangement is complete)."""
        pos = torch.zeros(n, 3, device=device)
        pos[:, 0:2] = xy_w
        pos[:, 2] = z_center
        pos[:, 2] += drop
        _write_body(body, pos, quat)

    def burner_xy() -> torch.Tensor:
        _refresh()
        return scene.burner.data.root_pos_w[:, :2].clone()

    def deck_rel(x: float, y: float) -> torch.Tensor:
        """World xy for a deck-anchored point (env origins included via readback)."""
        _refresh()
        base = scene.deck.data.root_pos_w[:, :2].clone()
        base[:, 0] += x - c.deck_pos[0]
        base[:, 1] += y - c.deck_pos[1]
        return base

    # frequently used rest heights (world z, single-env origins at z=0)
    oz = float(env.iscene.env_origins[0, 2])
    z_kettle_trivet = oz + c.deck_top + c.trivet_t + c.kettle_h / 2
    z_kettle_plate = oz + c.deck_top + c.plate_t + c.kettle_h / 2
    z_kettle_deck = oz + c.deck_top + c.kettle_h / 2
    z_cup_seated = oz + c.deck_top + c.ring_t + c.cup_h / 2
    z_cup_on_deck = oz + c.deck_top + c.cup_h / 2

    def kettle_to_trivet() -> None:
        _refresh()
        put(scene.kettle, scene.trivet.data.root_pos_w[:, :2].clone(),
            z_kettle_trivet, _qz(torch.full((n,), math.pi, device=device)), drop=0.010)
        _step(90)

    def seat_cup(offset_xy=(0.0, 0.0), drop: float = 0.004) -> None:
        bxy = burner_xy()
        bxy[:, 0] += offset_xy[0]
        bxy[:, 1] += offset_xy[1]
        put(scene.cup, bxy, z_cup_seated, down_quat(), drop=drop)
        _step(120)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    fin = bool(torch.isfinite(scene.kettle.data.root_pos_w).all()
               and torch.isfinite(scene.cup.data.root_pos_w).all())
    check("settle/no-NaN: kettle standing on the lit post, cup mouth-up on the deck; "
          "score 0, no success",
          fin and bool(scene.kettle_on_burner()[0])
          and float(scene._up_w(scene.cup)[0, 2]) > 0.9
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (scene.burner.data.root_pos_w[0, :2].clone(),
                scene.cup.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.cup.data.root_quat_w[0]),
                yaw_of(scene.kettle.data.root_quat_w[0]))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_b, a_c, a_cy, a_ky = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_b, b_c, b_cy, b_ky = readback()
    d_b, d_c = float((a_b - b_b).norm()), float((a_c - b_c).norm())
    d_cy, d_ky = dyaw(a_cy, b_cy), dyaw(a_ky, b_ky)
    print(f"[smoke] randomization deltas: burner_xy={d_b * 1000:.1f}mm "
          f"cup_xy={d_c * 1000:.1f}mm cup_yaw={d_cy:.1f}deg kettle_yaw={d_ky:.1f}deg",
          flush=True)
    check("randomization-is-real: burner xy, cup xy, cup yaw, kettle yaw readback differ",
          d_b > 0.003 and d_c > 0.005 and d_cy > 5.0 and d_ky > 5.0)

    # ================= 3. trivet/plate side permutation ===========================================
    sides = []
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        # verify the color pads really moved: side readback vs trivet y sign
        t_y = float(scene.trivet.data.root_pos_w[0, 1] - env.iscene.env_origins[0, 1])
        assert (t_y > 0) == (int(scene.side[0]) > 0), "side flag vs trivet pose mismatch"
        sides.append(int(scene.side[0]))
    print(f"[smoke] trivet sides over 10 resets: {sides}", flush=True)
    check("side permutation: the green trivet appears on BOTH sides over 10 resets",
          (1 in sides) and (-1 in sides))

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. kettle-only partial =====================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    kettle_to_trivet()
    _step(60)
    _report("kettle-only")
    check("partial (kettle only): kettle correctly on the trivet, cup untouched — "
          "the stove is still lit: no success, score <= 0.45",
          bool(scene.kettle_on_trivet()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.45)

    # ================= 6. WRONG PAD: color identity is load-bearing ===============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _refresh()
    put(scene.kettle, scene.plate.data.root_pos_w[:, :2].clone(), z_kettle_plate,
        _qz(torch.full((n,), math.pi, device=device)), drop=0.010)
    _step(90)
    seat_cup()
    _report("wrong-pad")
    check("negative (WRONG PAD): kettle parked on the WHITE PLATE + burner correctly "
          "capped — the color clause refuses: no success, score <= 0.51",
          bool(scene.capped()[0]) and not bool(scene.kettle_on_trivet()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.51)
    _REC["on"] = False

    # ================= 7. bare deck ===============================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    put(scene.kettle, deck_rel(0.16, -0.08), z_kettle_deck,
        _qz(torch.full((n,), math.pi, device=device)), drop=0.010)
    _step(90)
    seat_cup()
    _report("bare-deck")
    check("negative (bare deck): kettle set down on the bare deck + burner correctly "
          "capped — no trivet, no success, score <= 0.51",
          bool(scene.capped()[0]) and not bool(scene.kettle_on_trivet()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.51)

    # ================= 8. unflipped cover =========================================================
    # Cup rested MOUTH-UP on the post top: "covering" without the flip. Its floor
    # sits on the post; the mouth points up and the rim reads far above the band.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    kettle_to_trivet()
    z_cup_on_post = oz + c.deck_top + c.post_top_dz + c.cup_h / 2
    put(scene.cup, burner_xy(), z_cup_on_post, None, drop=0.004)
    _step(120)
    _report("unflipped")
    check("negative (unflipped): cup rested mouth-UP on the post top — no flip, rim "
          "far above the band: not capped, no success, score <= 0.45",
          not bool(scene.capped()[0]) and not bool(scene._flip[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.45)

    # ================= 9. cap near-miss: axis off by 19 mm ========================================
    # GENUINELY seated (mouth-down, rim resting in the band, post inside the
    # interior — physically possible at 19 mm: the annular clearance is 22 mm) but
    # outside the 15 mm tolerance. The axis clause must refuse.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    kettle_to_trivet()
    seat_cup(offset_xy=(0.019, 0.0))
    _report("near-miss-xy")
    _refresh()
    off = float((scene.cup.data.root_pos_w[0, :2] - scene.burner.data.root_pos_w[0, :2]).norm())
    up = scene._up_w(scene.cup)
    rim_dz = float(scene.cup.data.root_pos_w[0, 2] + up[0, 2] * (c.cup_h / 2)
                   - scene._ring_top_z()[0])
    print(f"[smoke] near-miss: axis offset={off * 1000:.1f}mm "
          f"(tol {c.cap_xy_tol * 1000:.0f}mm, clearance "
          f"{(c.int_half - c.post_s / 2) * 1000:.0f}mm) rim_dz={rim_dz * 1000:+.1f}mm",
          flush=True)
    check("near-miss (axis): cup genuinely seated over the post but 19 mm off its "
          "axis — rim in the band, post enclosed, capped still refuses, no success",
          off > c.cap_xy_tol and rim_dz < c.rim_hi
          and not bool(scene.capped()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.60)

    # ================= 10. beside the burner ======================================================
    # Cup mouth-down seated flat on the DECK: its rim height (deck level) is inside
    # the rim band relative to the hob plate — ONLY the axis clause refuses.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    kettle_to_trivet()
    put(scene.cup, deck_rel(0.16, 0.09), z_cup_on_deck, down_quat(), drop=0.004)
    _step(120)
    _report("beside-burner")
    check("negative (beside): cup mouth-down seated flat on the deck beside the "
          "burner — the axis clause alone refuses: not capped, no success",
          float(scene._up_w(scene.cup)[0, 2]) < -0.9
          and not bool(scene.capped()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.60)

    # ================= 11. BLOCKED CAP: the order is physics ======================================
    # Fresh reset — the kettle still stands ON the post. The cup is dropped
    # mouth-down from a hover over the occupied burner (the cap move, made too
    # early). The kettle body (98 mm) is wider than the interior (94 mm): the cup
    # lands high on the kettle and can never reach the rim band.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _refresh()
    k_top = float(scene.kettle.data.root_pos_w[0, 2]) + c.kettle_h / 2
    put(scene.cup, burner_xy(), k_top + 0.012 + c.cup_h / 2, down_quat(), drop=0.0)
    _step(180)
    _report("blocked-cap")
    _refresh()
    up = scene._up_w(scene.cup)
    rim_dz = float(scene.cup.data.root_pos_w[0, 2] + up[0, 2] * (c.cup_h / 2)
                   - scene._ring_top_z()[0])
    print(f"[smoke] blocked cap: cup rim settled {rim_dz * 1000:+.1f}mm above the hob "
          f"(band {c.rim_lo * 1000:+.0f}..{c.rim_hi * 1000:+.0f}mm)", flush=True)
    check("BLOCKED CAP: cup dropped over the OCCUPIED burner lands high on the "
          "kettle and never reaches the rim band — clear-before-cap is physics",
          not bool(scene.capped()[0]) and not bool(scene._cap[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.35)
    _REC["on"] = False

    # ================= 12. kettle perched on the seated cup =======================================
    # Correct cap, then the kettle rested on TOP of the seated cup: flame covered
    # and both bodies "at the burner" — but the kettle is not on the trivet.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    # park the kettle on the deck first (frees the post), then cap, then perch
    put(scene.kettle, deck_rel(0.14, -0.15), z_kettle_deck,
        _qz(torch.full((n,), math.pi, device=device)), drop=0.010)
    _step(90)
    seat_cup()
    z_kettle_on_cup = z_cup_seated + c.cup_h / 2 + c.kettle_h / 2
    put(scene.kettle, burner_xy(), z_kettle_on_cup,
        _qz(torch.full((n,), math.pi, device=device)), drop=0.006)
    _step(150)
    _report("kettle-perch")
    check("negative (perch): kettle rested on TOP of the correctly seated cup — "
          "flame covered but the kettle is not on the trivet: no success, "
          "score <= 0.55",
          bool(scene.capped()[0]) and not bool(scene.kettle_on_trivet()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.55)
    _REC["on"] = False

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.burner_snuff")
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
