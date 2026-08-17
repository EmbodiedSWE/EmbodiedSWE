"""Smoke battery for BallastVaultScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT a solution (the solution is solve.py — load the iron block into the lever
cup, drop the carton through the held-open mouth, unload so the lid seals itself; the
Franka strategy is TASK.md's embodiment argument). Teleported states here are rubric
INSTRUMENTATION: construct an outcome as a settled state under the scene's live hinge
plant, then assert the rubric's verdict on it. The `lid_drive` buffer is used ONLY as
a stand-in for "an arm holds the lid" in two checks; solve.py never touches it.

One linear run, 14 named checks:
  1. settle    — clean reset: finite state, lid resting CLOSED (readback ~0 deg),
                 all three items on their staging slots, score ~0;
  2. random    — vault xy/yaw and the slot permutation differ across seeds, and each
                 item PHYSICALLY sits on its assigned slot (readback, not trust);
  3. null      — 2 s of nothing: score < 0.05, no success;
  4. decoy     — the WHITE FOAM block, given exactly the iron's transport (hover +
                 drop into the cup), lands in the cup and the lid STAYS CLOSED: the
                 wrong tool fails through the plant, not through a rubric clause;
  5. seed plan — the seed's whole strategy (carry the juice to the receptacle and
                 release it above): the carton dropped over the closed vault bounces
                 off the lid and never enters — rejected at ~0;
  6. accept    — the iron block's identical transport DOES swing the lid to its
                 65-deg stop and HOLD it (the accept side of the decoy boundary);
  7. dilemma   — end state "never unloaded": carton inside but iron still in the cup
                 holding the lid open — rejected, capped at the latched 0.65;
  8. perched   — full goal except the iron parked ON TOP of the closed lid (not set
                 aside) — rejected by the resting-clear clause;
  9. transient — mid-flight carton inside the vault volume does NOT arm the insert
                 latch or success (settled outcomes only);
 10. wrong obj — the FOAM sealed inside instead of the carton (proper open + close):
                 rejected, score stays at the open/approach credit;
 11. wrong pl. — the IRON sealed inside the vault (lid held open by the drive probe
                 while both are dropped in, then released): rejected — a block inside
                 is not "set aside";
 12. exactness — constructed full goal state -> success() and score == 1.0;
 13. latch     — the drive probe re-opens the sealed lid and HOLDS it: success is
                 revoked while held; the latched 0.65 remains (earned credit does not
                 evaporate; the live 0.35 does);
 14. frames    — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.libero_pick_orange_juice_i129.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=10)
parser.add_argument("--max_frames", type=int, default=280)
parser.add_argument("--out", type=str, default="frames.npz")  # CWD — the pipeline fetches it
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe (proven on this render stack): kit mis-decodes the driver version and
# silently rejects RTX -> the annotator returns EMPTY frames.
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

robobench.discover()
try:
    from simgen_tasks.libero_pick_orange_juice_i129 import scene as scene_mod
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod

# Watchdog: never leave a GPU zombie.
threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                 os._exit(3))).start()

HOVER_CUP = 0.060   # m above the cup floor along its tilted normal (same as solve.py)
DROP_Y = -0.045     # vault-local carton drop line (same as solve.py)
DROP_Z = 0.33
HOLD_TAU = -1.5     # N*m drive probe = "an arm holds the lid open" (opening is negative)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ballast_vault")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Mass readback: the lid's balance IS the mechanism, and custom compound spawners
    # silently IGNORE root mass_props — so verify what PhysX actually derived from the
    # authored densities and FAIL LOUDLY on regression (folded into check 1).
    mass_ok = False
    try:
        got = {}
        for nm, b in (("vault", scene.vault), ("lid", scene.lid),
                      ("carton", scene.items["carton"]), ("iron", scene.items["iron"]),
                      ("foam", scene.items["foam"])):
            m = b.root_physx_view.get_masses()
            com = b.root_physx_view.get_coms()
            got[nm] = float(m.flatten()[0])
            print(f"[smoke][mass] {nm}: m={m.flatten().tolist()} "
                  f"com={com.flatten().tolist()}", flush=True)
        mass_ok = abs(got["iron"] - scene.cfg.iron_mass) < 0.05 * scene.cfg.iron_mass \
            and abs(got["foam"] - scene.cfg.foam_mass) < 0.5 * scene.cfg.foam_mass \
            and abs(got["vault"] - scene.cfg.vault_mass) < 0.05 * scene.cfg.vault_mass
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke][mass] readback failed: {exc!r}", flush=True)

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.45, -0.80, 0.90)) + o),
                                tuple(np.array((0.40, 0.00, 0.20)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (800, 500))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames — check the RTX recipe",
                  flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action, render=annot is not None)
            if annot is not None and step_i % args.record_every == 0 \
                    and len(frames) < args.max_frames:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def deg() -> float:
        return math.degrees(float(scene.lid_angle()[0]))

    def report(tag: str) -> None:
        print(f"[smoke] {tag:12s} | lid={deg():6.1f}deg "
              f"iron_in_cup={bool(scene.in_cup('iron')[0])} "
              f"foam_in_cup={bool(scene.in_cup('foam')[0])} "
              f"carton_inside={bool(scene.carton_inside()[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def settle_until(pred, max_steps: int = 600, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    from isaaclab.utils.math import quat_apply

    def teleport(body, pos_w: torch.Tensor, quat_w: torch.Tensor) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        body.write_root_state_to_sim(st, all_ids)

    def vault_local_w(local_xyz: tuple) -> torch.Tensor:
        loc = torch.tensor([list(local_xyz)], device=device)
        return scene.vault.data.root_pos_w + quat_apply(scene.vault.data.root_quat_w, loc)

    def drop_into_cup(name: str) -> None:
        """Transport-only: hover the block above the cup along its tilted normal, release."""
        tilt = math.radians(c.cup_tilt_deg)
        loc = torch.tensor([[0.0, c.cup_y - HOVER_CUP * math.sin(tilt),
                             c.cup_z + HOVER_CUP * math.cos(tilt)]], device=device)
        w = scene.lid.data.root_pos_w + quat_apply(scene.lid.data.root_quat_w, loc)
        qx = torch.zeros(1, 4, device=device)
        qx[0, 0], qx[0, 1] = math.cos(tilt / 2), math.sin(tilt / 2)
        teleport(scene.items[name], w, scene_mod._qmul(scene.lid.data.root_quat_w, qx))

    def drop_carton_in() -> bool:
        teleport(scene.items["carton"], vault_local_w((0.0, DROP_Y, DROP_Z)),
                 scene.vault.data.root_quat_w)
        return settle_until(
            lambda: bool(scene.carton_inside()[0])
            and float(scene.items["carton"].data.root_lin_vel_w.norm()) < 0.05)

    def park_block(name: str, xy=( -0.10, 0.30)) -> None:
        """Set a block down on the open floor, well away from the vault (env-local)."""
        p = torch.zeros(1, 3, device=device)
        p[0, 0], p[0, 1] = xy
        p[0, 2] = c.puck_h / 2 + 0.002
        p += scene.env_origins[0]
        teleport(scene.items[name], p, torch.tensor([[1.0, 0, 0, 0]], device=device))

    def load_iron(max_steps: int = 720) -> bool:
        drop_into_cup("iron")
        return settle_until(lambda: bool(scene.in_cup("iron")[0]) and deg() > 58.0
                            and abs(float(scene.lid_rate()[0])) < 0.2, max_steps=max_steps)

    def unload_iron(max_steps: int = 900) -> bool:
        park_block("iron")
        return settle_until(lambda: deg() < c.closed_tol_deg
                            and abs(float(scene.lid_rate()[0])) < 0.05, max_steps=max_steps)

    # ========================= 1. settle / clean-slate ========================================
    env.reset(seed=3)
    step(60)
    report("settled")
    finite = all(bool(torch.isfinite(b.data.root_state_w).all())
                 for b in (scene.vault, scene.lid, *scene.items.values())) \
        and bool(torch.isfinite(scene.score()).all())
    slots = scene.slot_of[0].tolist()
    on_slot = True
    for i, nm in enumerate(scene.ITEM_NAMES):
        p = (scene.items[nm].data.root_pos_w[0] - scene.env_origins[0]).tolist()
        want_x = c.table_pos[0] + c.slot_x[slots[i]]
        on_slot &= abs(p[0] - want_x) < c.slot_jitter + 0.02 \
            and abs(p[1] - c.table_pos[1]) < c.slot_jitter + 0.02 \
            and abs(p[2] - c.table_size[2]) < 0.09
    check("settle: clean reset (finite, lid resting closed, items staged, masses as "
          "authored, score ~0)",
          finite and deg() < 2.0 and on_slot and mass_ok
          and float(scene.score()[0]) < 0.05)

    # ========================= 2. randomization (readback) ====================================
    draws = []
    perms = set()
    phys_ok = True
    for seed in (11, 12, 13, 14):
        env.reset(seed=seed)
        step(30)
        vp = (scene.vault.data.root_pos_w[0] - scene.env_origins[0]).tolist()
        q = scene.vault.data.root_quat_w[0]
        yaw = math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))
        perm = tuple(scene.slot_of[0].tolist())
        perms.add(perm)
        draws.append((perm, round(vp[0], 3), round(vp[1], 3), round(yaw, 1)))
        for i, nm in enumerate(scene.ITEM_NAMES):  # slot_of must match PHYSICAL positions
            p = (scene.items[nm].data.root_pos_w[0] - scene.env_origins[0]).tolist()
            phys_ok &= abs(p[0] - (c.table_pos[0] + c.slot_x[perm[i]])) < c.slot_jitter + 0.02
    print(f"[smoke] draws (perm, vault xy, yaw): {draws}", flush=True)
    check("randomization is real (vault pose + slot permutation differ; physical readback)",
          len({str(d) for d in draws}) >= 3 and len(perms) >= 2 and phys_ok
          and all(abs(d[3]) <= c.vault_yaw_deg + 1.0 for d in draws))

    # ========================= 3. null policy =================================================
    env.reset(seed=3)
    step(240)  # 2 s of nothing
    report("null")
    check("null policy: score < 0.05 and no success",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 4. decoy: the foam block cannot open the lid ===================
    env.reset(seed=21)
    step(30)
    drop_into_cup("foam")
    step(360)  # 3 s — plenty for any opening to show
    report("foam-cup")
    foam_in = bool(scene.in_cup("foam")[0])  # the probe must actually land (not vacuous)
    check("decoy: foam block lands in the cup yet the lid stays closed (plant-enforced)",
          foam_in and deg() < 10.0 and not bool(scene.open_latch[0] > 0.5))

    # ========================= 5. the seed's plan: drop on the closed vault ===================
    env.reset(seed=31)
    step(30)
    teleport(scene.items["carton"], vault_local_w((0.0, DROP_Y, DROP_Z)),
             scene.vault.data.root_quat_w)
    settle_until(lambda: float(scene.items["carton"].data.root_lin_vel_w.norm()) < 0.05,
                 max_steps=600)
    report("seed-plan")
    check("seed strategy: carton dropped over the CLOSED vault never enters — rejected ~0",
          not bool(scene.carton_inside()[0]) and deg() < 5.0
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.05)

    # ========================= 6. accept: the iron block opens + holds ========================
    env.reset(seed=41)
    step(30)
    ok_open = load_iron()
    a0 = deg()
    step(240)  # 2 s more: HELD open, not a bounce
    report("iron-cup")
    check("accept: iron block swings the lid to its stop and HOLDS it open",
          ok_open and a0 > 58.0 and deg() > 58.0 and bool(scene.in_cup("iron")[0]))

    # ========================= 7. dilemma: never unloaded =====================================
    ok_ins = drop_carton_in()
    step(120)
    report("not-unloaded")
    check("negative (never unloaded): carton inside but iron still holds the lid — rejected",
          ok_ins and deg() > 58.0 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.66)

    # ========================= 8. perched: iron left on top of the closed lid =================
    # Continue the episode properly (unload -> sealed), then put the iron ON the lid.
    ok_seal = unload_iron()
    report("sealed")
    top_w = vault_local_w((0.0, -0.02, c.hinge_z + c.puck_h / 2 + 0.004))
    teleport(scene.items["iron"], top_w, scene.vault.data.root_quat_w)
    step(360)
    report("perched")
    check("negative (perched): iron resting ON the closed lid is not 'set aside' — rejected",
          ok_seal and bool(scene.carton_inside()[0]) and deg() < c.closed_tol_deg
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.66)

    # ========================= 9+12. transient guard + exactness ==============================
    env.reset(seed=51)
    step(30)
    ok_open = load_iron()
    teleport(scene.items["carton"], vault_local_w((0.0, DROP_Y, DROP_Z)),
             scene.vault.data.root_quat_w)
    step(6)  # mid-flight: inside the vault footprint, still falling fast
    falling = float(scene.items["carton"].data.root_lin_vel_w.norm()) > 0.2
    check("transient: a falling carton arms nothing (insert latch and success stay off)",
          ok_open and falling and float(scene.insert_latch[0]) < 0.5
          and not bool(scene.success()[0]))
    ok_ins = settle_until(
        lambda: bool(scene.carton_inside()[0])
        and float(scene.items["carton"].data.root_lin_vel_w.norm()) < 0.05)
    ok_seal = unload_iron()
    ok = settle_until(lambda: bool(scene.success()[0]), max_steps=600)
    report("goal-state")
    check("exactness: full goal state -> success() and score == 1.0",
          ok_ins and ok_seal and ok and abs(float(scene.score()[0]) - 1.0) < 1e-3)
    goal_state = scene.get_state(all_ids)  # reused by the latch check (13)

    # ========================= 10. wrong object: foam sealed inside ===========================
    env.reset(seed=61)
    step(30)
    ok_open = load_iron()
    teleport(scene.items["foam"], vault_local_w((0.0, DROP_Y, DROP_Z)),
             scene.vault.data.root_quat_w)
    step(30)  # a just-teleported body has zero velocity — let it actually fall first
    ok_foam = settle_until(
        lambda: bool((scene._vault_frame(scene.items["foam"].data.root_pos_w)[0, 2]
                      < c.height).item())
        and float(scene.items["foam"].data.root_lin_vel_w.norm()) < 0.05)
    foam_in_vault = bool((scene._vault_frame(scene.items["foam"].data.root_pos_w)[0, 2]
                          < c.height).item())
    ok_seal = unload_iron()
    report("foam-sealed")
    print(f"[smoke]   wrong-object clauses: ok_open={ok_open} ok_foam={ok_foam} "
          f"foam_in_vault={foam_in_vault} ok_seal={ok_seal} "
          f"foam_local={scene._vault_frame(scene.items['foam'].data.root_pos_w)[0].tolist()}",
          flush=True)
    check("negative (wrong object): FOAM sealed inside instead of the carton — rejected",
          ok_open and ok_foam and foam_in_vault and ok_seal
          and not bool(scene.carton_inside()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.36)

    # ========================= 11. wrong place: iron sealed inside ============================
    # The drive probe stands in for "an arm holds the lid open" while both items are
    # dropped inside; releasing it lets the lid seal. A block inside is not set aside.
    env.reset(seed=71)
    step(30)
    scene.lid_drive[0] = HOLD_TAU
    held = settle_until(lambda: deg() > 58.0, max_steps=480)
    teleport(scene.items["iron"], vault_local_w((0.0, DROP_Y, DROP_Z + 0.02)),
             scene.vault.data.root_quat_w)
    step(90)
    ok_carton = drop_carton_in()
    scene.lid_drive[0] = 0.0  # release: the lid seals on its own
    sealed = settle_until(lambda: deg() < c.closed_tol_deg
                          and abs(float(scene.lid_rate()[0])) < 0.05, max_steps=900)
    iron_in_vault = bool((scene._vault_frame(scene.items["iron"].data.root_pos_w)[0, 2]
                          < c.height).item())
    report("iron-sealed")
    check("negative (wrong place): IRON sealed inside with the carton — rejected",
          held and ok_carton and sealed and iron_in_vault
          and bool(scene.carton_inside()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.66)

    # ========================= 13. achievement latch ==========================================
    scene.set_state(goal_state, all_ids)
    step(30)
    ok_goal = bool(scene.success()[0])
    scene.lid_drive[0] = HOLD_TAU  # an arm pries the sealed lid open and HOLDS it
    reopened = settle_until(lambda: deg() > 45.0, max_steps=480)
    step(60)
    revoked = not bool(scene.success()[0])
    latched = float(scene.score()[0])
    scene.lid_drive[0] = 0.0
    report("reopened")
    check("achievement latch: prying the lid open revokes success; latched 0.65 remains",
          ok_goal and reopened and revoked and abs(latched - 0.65) < 0.02)

    # ========================= 14. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ballast_vault")
        print(f"[smoke] saved {arr.shape} -> {os.path.abspath(args.out)}", flush=True)
    check("video frames recorded", len(frames) >= 20)

    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        bad = [nm for nm, ok in checks if not ok]
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)} — failing: {bad}", flush=True)
    # Kit teardown hangs are routine: watchdog then hard exit.
    threading.Timer(10.0, lambda: os._exit(0 if all_ok else 1)).start()
    try:
        env.close()
        app.close()
    finally:
        os._exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
