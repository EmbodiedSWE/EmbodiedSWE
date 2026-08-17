"""Smoke battery for StowflatScene (sim_gen task
`libero_kitchen_scene5_close_the_top_drawer_of_the_cabinet_i418`) — REJECTION-ONLY:
every check either verifies basic health/randomization or CONSTRUCTS/EXECUTES a
wrong outcome and asserts the rubric refuses it. success() must never fire
anywhere in the battery. (The drawer is JOINTED to the kinematic station —
prismatic limits are the hard stops — so "stolen drawer" fakes are structurally
impossible and are guarded, not constructed.)

 1. settle/no-NaN     — drawer at its sampled opening q0, carton upright in the
                        drawer's front band, companion inside, authored masses
                        readback (custom-spawner trap), score ~0, no success.
 2. randomization A   — the drawer opening q0 varies across resets and the
                        SETTLED drawer tracks the sample every time (readback).
 3. randomization B   — the carton's floor pose varies (x span) and its side
                        split (left vs right) is seen BOTH ways; the companion's
                        pose varies; the cargo settles inside the drawer.
 4. null policy       — 240 idle steps -> score ~0, no success.
 5. SEED strategy     — the seed's whole skill (push the drawer shut) executed
                        for REAL: an 8 N PD push on the drawer. The upright
                        carton rams the slab and wedges against the front
                        panel: the drawer HARD-JAMS far from seated (q >= 20 mm
                        at all times), score ~0 (closing credit is gated on the
                        stowed cargo), never success.
 6. upright-in-closed — drawer CONSTRUCTED seated with the carton upright
                        inside (a physically unreachable state): the flat
                        clause refuses instantly — no step is taken on the
                        interpenetrating construct.
 7. eject cheat       — the carton removed to the ground (the i58-style
                        strategy: clear the obstruction OUT), then a real push
                        seats the drawer. The drawer closes — but the carton is
                        not inside: no flat latch, no gated closing credit,
                        score ~0, never success.
 8. companion eject   — carton constructed FLAT inside (the flat latch fires,
                        0.25) but the companion bar removed to the ground; a
                        real push seats the drawer: closing credit stays gated
                        off, score ~0.25, never success.
 9. near-miss + latch — carton flat inside, real push stops the drawer FLUSH
                        but SHORT (q ~ 35 mm): no success, score < 0.70; then
                        the drawer is pulled back OPEN with a real force — the
                        latched credit survives, the live clause reads open,
                        still no success.
10. rejection audit   — success() observed False at every step of the battery.
11. final no-NaN.
12. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene5_close_the_top_drawer_of_the_cabinet_i418.smoke --headless
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
_REC = {"on": True, "annot": None, "frames": [], "i": 0}
_AUDIT = {"saw_success": False}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        _AUDIT["saw_success"] |= bool(env.scene.success()[0])
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
    print(f"[smoke] {tag:16s} | q={float(scene.drawer_open()[0]):+.4f} "
          f"axz={float(scene.carton_axis_z()[0]):.2f} "
          f"flat={bool(scene.carton_flat()[0])} in={bool(scene.carton_in()[0])} "
          f"comp={bool(scene.comp_in()[0])} "
          f"latch(f/c)=({int(scene._fflat[0])},{float(scene._fclose[0]):.2f}) "
          f"settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.stowflat_cabinet")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.15, -0.95, 0.95)) + o),
                                tuple(np.array((-0.05, 0.00, 0.35)) + o),
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
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def succ() -> bool:
        s = bool(scene.success()[0])
        _AUDIT["saw_success"] |= s
        return s

    def origin(dx: float, dy: float, dz: float) -> torch.Tensor:
        return (scene.env_origins
                + torch.tensor([dx, dy, dz], device=device)).expand(n, 3)

    def q_now() -> float:
        return float(scene.drawer_open()[0])

    def qx(theta_rad: float) -> torch.Tensor:
        """wxyz quat for a rotation about +X (lays the carton's long axis flat)."""
        q = torch.zeros(n, 4, device=device)
        q[:, 0] = math.cos(theta_rad / 2)
        q[:, 1] = math.sin(theta_rad / 2)
        return q

    def pd_drawer(q_target: float, steps: int, clamp: float = 8.0) -> float:
        """REAL actuation: a force-limited PD push/pull on the drawer along its
        slide axis (the seed's whole skill), what a hand would apply. Returns
        the MINIMUM opening observed during the drive (the peak of the probe —
        gravity/stops can restore state before a post-settle readback)."""
        zero = torch.zeros(n, 1, 3, device=device)
        q_min = q_now()
        for _ in range(steps):
            v = scene.drawer.data.root_lin_vel_w[:, 0]
            f = (200.0 * (q_target - scene.drawer_open()) - 30.0 * v).clamp(-clamp, clamp)
            fw = torch.zeros(n, 3, device=device)
            fw[:, 0] = f
            scene.drawer.set_external_force_and_torque(fw.reshape(n, 1, 3), zero)
            _step(1)
            q_min = min(q_min, q_now())
        scene.drawer.set_external_force_and_torque(zero, zero)
        _step(90)
        q_min = min(q_min, q_now())
        return q_min

    def construct_carton_flat() -> None:
        """Teleport-construct the carton LYING FLAT inside the drawer (what the
        honest tip achieves), long axis along y, resting on the drawer floor."""
        q0 = q_now()
        _write_body(scene.carton,
                    origin(q0 - 0.0725, -0.050, c.floor_z1 + c.carton_w / 2 + 0.002),
                    qx(math.pi / 2))
        _step(60)

    # ================= 1. settle / no-NaN / masses ================================================
    torch.manual_seed(11)
    env.reset()
    _step(150)
    _report("reset")
    m_drawer = float(scene.drawer.root_physx_view.get_masses()[0].sum())
    m_carton = float(scene.carton.root_physx_view.get_masses()[0].sum())
    m_comp = float(scene.comp.root_physx_view.get_masses()[0].sum())
    print(f"[smoke] masses: drawer={m_drawer:.3f} carton={m_carton:.3f} "
          f"comp={m_comp:.3f}", flush=True)
    check("settle/no-NaN: drawer at its sampled q0, carton upright inside the "
          "drawer front band, companion inside, authored masses in effect, "
          "score ~0, no success",
          bool(scene._finite()[0]) and bool(scene.drawer_in_channel()[0])
          and abs(q_now() - float(scene.q0[0])) < 0.006
          and float(scene.carton_axis_z()[0]) > 0.95
          and bool(scene.carton_in()[0]) and bool(scene.comp_in()[0])
          and abs(m_drawer - c.drawer_mass) < 0.1 * c.drawer_mass
          and abs(m_carton - c.carton_mass) < 0.1 * c.carton_mass
          and abs(m_comp - c.comp_mass) < 0.1 * c.comp_mass
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2+3. randomization readback ================================================
    q0s, cxs, csides, bys, tracks, inside = [], [], [], [], True, True
    for k in range(10):
        torch.manual_seed(20 + k)
        env.reset()
        _step(90)
        _refresh()
        q0s.append(float(scene.q0[0]))
        cl = scene._drawer_frame(scene.carton)[0]
        bl = scene._drawer_frame(scene.comp)[0]
        cxs.append(float(cl[0]))
        csides.append(1.0 if float(cl[1]) > 0 else -1.0)
        bys.append(float(bl[1]))
        tracks = tracks and abs(q_now() - float(scene.q0[0])) < 0.006
        inside = inside and bool(scene.carton_in()[0]) and bool(scene.comp_in()[0]) \
            and float(scene.carton_axis_z()[0]) > 0.95
    qspan = max(q0s) - min(q0s)
    cxspan = max(cxs) - min(cxs)
    byspan = max(bys) - min(bys)
    print(f"[smoke] readback: q0={[f'{q * 1000:.0f}' for q in q0s]}mm span={qspan * 1000:.0f}mm "
          f"cx_span={cxspan * 1000:.1f}mm csides={csides} by_span={byspan * 1000:.0f}mm "
          f"tracks={tracks} inside={inside}", flush=True)
    check("randomization A: the drawer opening q0 varies across resets and the "
          "settled drawer tracks the sample every time",
          qspan > 0.012 and tracks)
    check("randomization B: the carton's floor pose varies with the side split "
          "seen both ways, the companion's pose varies, and the cargo settles "
          "upright inside the drawer",
          cxspan > 0.010 and (1.0 in csides) and (-1.0 in csides)
          and byspan > 0.050 and inside)

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 5. SEED strategy (real push on the drawer — HARD JAM) =====================
    torch.manual_seed(41)
    env.reset()
    _step(150)
    q_min5 = pd_drawer(0.0, 480)     # the seed's whole skill: hand-push the drawer shut
    _report("seed-skill")
    check("SEED strategy: an 8 N push on the drawer rams the upright carton into "
          "the slab and wedges it against the front panel — the drawer "
          "hard-jams far from seated (q >= 20 mm throughout), the carton stays "
          "inside, score ~0 (closing gated on the stowed cargo), never success",
          bool(scene._finite()[0]) and q_min5 >= 0.020 and q_now() >= 0.020
          and bool(scene.carton_in()[0]) and bool(scene.drawer_in_channel()[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 6. upright-in-closed construct (instant refusal) ==========================
    torch.manual_seed(51)
    env.reset()
    _step(120)
    qc = 0.004
    _write_body(scene.drawer, origin(qc, 0.0, 0.0))
    _write_body(scene.carton,
                origin(qc - 0.0725, 0.030, c.floor_z1 + c.carton_hz / 2 + 0.002))
    _write_body(scene.comp, origin(qc - 0.155, -0.05, c.floor_z1 + c.comp_lz / 2 + 0.002))
    _refresh()
    inst = (not succ()) and (not bool(scene.carton_flat()[0])) \
        and float(scene.score()[0]) <= 0.05
    check("upright-in-closed construct: drawer written seated with the carton "
          "still upright inside (physically unreachable through the mouth) — "
          "the flat clause refuses instantly, score ~0",
          inst)
    # do NOT step the interpenetrating construct — clear it with a fresh reset

    # ================= 7. eject cheat (the i58 strategy: remove the obstruction) =================
    torch.manual_seed(61)
    env.reset()
    _step(150)
    _write_body(scene.carton, origin(0.45, 0.30, c.carton_w / 2 + 0.002), qx(math.pi / 2))
    _step(90)
    q_min7 = pd_drawer(0.0, 420)
    _report("eject-cheat")
    check("eject cheat: the carton removed to the ground and the drawer pushed "
          "shut for real — the drawer seats, but the carton is not inside: no "
          "flat latch, no gated closing credit, score ~0, never success",
          q_min7 <= c.q_goal and q_now() <= c.q_goal + 0.004
          and not bool(scene.carton_in()[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 8. companion eject =========================================================
    torch.manual_seed(71)
    env.reset()
    _step(150)
    construct_carton_flat()
    _write_body(scene.comp, origin(0.45, -0.30, c.comp_lz / 2 + 0.002))
    _step(60)
    s8_flat = float(scene.score()[0])
    pd_drawer(0.0, 420)
    _report("comp-eject")
    check("companion eject: carton flat inside (flat latch 0.25) but the white "
          "bar removed to the ground, drawer pushed seated for real — closing "
          "credit stays gated off: score ~0.25, never success",
          q_now() <= c.q_goal + 0.004
          and abs(s8_flat - c.w_flat) < 0.03
          and not bool(scene.comp_in()[0])
          and float(scene.score()[0]) <= c.w_flat + 0.05 and not succ())

    # ================= 9. near-miss flush + latch regression ======================================
    torch.manual_seed(81)
    env.reset()
    _step(150)
    q0_9 = float(scene.q0[0])
    construct_carton_flat()
    pd_drawer(0.035, 360)            # real push, stopped FLUSH but short of seated
    _report("near-miss")
    s9 = float(scene.score()[0])
    near_ok = (0.015 <= q_now() <= 0.060) and (not succ()) \
        and bool(scene.carton_flat()[0]) and bool(scene.comp_in()[0]) \
        and s9 <= 0.70 + 1e-3
    fclose_latched = float(scene._fclose[0])
    pd_drawer(q0_9, 420)             # real pull back toward the sampled opening
    _report("latch-regress")
    check("near-miss: the drawer pushed flush but short (q ~ 35 mm) with the "
          "cargo stowed — no success, score < 0.70",
          near_ok and fclose_latched >= 0.5)
    check("latch regression: the drawer pulled back open with a real force — "
          "the latched closing credit survives, the live clause reads open "
          "again, no success",
          float(scene._fclose[0]) >= fclose_latched - 1e-4
          and q_now() >= 0.5 * q0_9
          and float(scene.score()[0]) >= c.w_flat + c.w_close * fclose_latched - 1e-3
          and not succ())

    # ================= 10-12. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.stowflat_cabinet")
        print(f"[smoke] wrote {args.out}: {arr.shape}", flush=True)
    check("video: >10 frames recorded", len(_REC["frames"]) > 10)

    n_pass = sum(1 for _, ok in checks if ok)
    if n_pass == len(checks):
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
        code = 0
    else:
        for nm, ok in checks:
            if not ok:
                print(f"[smoke] FAILED: {nm}", flush=True)
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        code = 1
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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
