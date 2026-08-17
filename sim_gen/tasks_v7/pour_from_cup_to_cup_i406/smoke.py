"""Smoke / rubric-REJECTION battery for RationTicketScene — NullRobot, physical probes.

solve.py is the acceptance proof (the rubric accepts the right-place-count-stop
outcome and the printed credit is monotone along a real trajectory). This battery
proves the rubric REJECTS wrong outcomes, and that the claims the task rests on — the
ticket count is exact, overfill earns nothing, an upside-down cup covering balls
contains nothing, abandoned leftovers block success — hold on SETTLED constructed
states, not fiat. Every constructed state is built the honest way (bodies dropped
above their targets with zero velocity and settled by real contact), then judged.
No probe here is asserted to reach success().

Checks:
   1. settle/no-NaN     — seeded reset settles finite; present balls rest inside the
                          bin, absent balls parked in the depot, both cups mouth-down,
                          SHOWN dot count on each side == quota readback; score ~0;
   2. randomization     — three seeded resets: max-pairwise READBACK deltas of pad_a,
                          cup_a and ball_0 positions all nonzero (GPU RNG two-seed
                          collisions masked — use 3-seed max-pairwise);
   3. coverage          — over 10 resets: >= 2 distinct quota pairs, an ASYMMETRIC
                          pair (q_a != q_b) appears, >= 2 distinct present counts;
   4. null-policy       — 240 idle steps: cups still mouth-down, balls still in the
                          bin, score ~0, no success;
   5. BULK-DUMP         — the seed strategy's end state ("move everything into the
                          target"): both cups placed, then EVERY present ball dropped
                          into cup A — VERIFIED overfilled (count > quota); success
                          refuses and the score caps at the quota's own fill credit
                          (overfill earns nothing);
   6. OFF-BY-ONE        — right total delivered, wrong allocation (one ball of pad
                          A's ticket delivered to pad B instead): leftovers fine,
                          both cups placed, but exactness fails both pads -> no
                          success, fill credit exactly one ball short;
   7. OFF-PAD near-miss — cup A upright, correctly filled, settled — but centred
                          pad_tol + 2 cm AWAY from its pad: latch never arms, its
                          balls count for nothing, no success;
   8. MOUTH-DOWN COVER  — cup A left upside-down ON its pad, covering exactly
                          quota_a balls (geometrically inside the interior — only
                          the upright gate rejects); count reads 0, latch never
                          arms, no success;
   9. LEFTOVER LOOSE    — both pads exactly right but ONE surplus ball abandoned on
                          the open floor: base credit is FULL (0.75) yet success
                          refuses — the leave-the-rest-in-the-bin clause is
                          load-bearing;
  10. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.pour_from_cup_to_cup_i406.smoke --headless
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

SCATTER = ((0.0, 0.0), (0.009, 0.0), (-0.005, 0.008))


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


def _drop_body(body, pos_w: torch.Tensor, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
    """TRANSPORT write: pose ABOVE the target, zero velocity — gravity finishes."""
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    st[:, 3:7] = torch.tensor(quat, device=_ENV.device)
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    pres = scene.present[0]
    print(f"[smoke] {tag:16s} | counts={scene.pad_counts()[0].tolist()} "
          f"quota={scene.quota[0].tolist()} "
          f"in_bin={int((scene.in_bin()[0] & pres).sum())}/{int(pres.sum())} "
          f"latch={scene._placed_latch[0].tolist()} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ration_ticket_station")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    origin = env.iscene.env_origins  # (n, 3)

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.95, -0.70, 0.85)) + o),
                                tuple(np.array((0.0, 0.0, 0.08)) + o),
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

    # --- construction helpers (drop + settle: contact does the seating) ---
    def settle_until(pred, max_steps: int, poll: int = 10) -> bool:
        waited = 0
        while waited < max_steps:
            _step(poll)
            waited += poll
            if pred():
                return True
        return False

    def pad_xy(j: int) -> tuple[float, float]:
        _refresh()
        p = scene.pads[j].data.root_pos_w[0] - origin[0]
        return float(p[0]), float(p[1])

    def place_cup(j: int, off=(0.0, 0.0), mouth_down: bool = False,
                  expect_on_pad: bool = True) -> bool:
        px, py = pad_xy(j)
        quat = (0.0, 1.0, 0.0, 0.0) if mouth_down else (1.0, 0.0, 0.0, 0.0)
        _drop_body(scene.cups[j],
                   origin + torch.tensor([px + off[0], py + off[1],
                                          c.cup_h / 2 + 0.025], device=device), quat)
        if expect_on_pad:
            return settle_until(lambda: bool(scene.on_pad()[0, j].any()),
                                max_steps=int(3.0 / env.dt))
        _step(int(1.2 / env.dt))
        return True

    def bin_balls():
        _refresh()
        inb = scene.in_bin()[0] & scene.present[0]
        return [i for i in range(c.n_balls) if bool(inb[i])]

    def fill_cup(j: int, n: int, require_contained: bool = True) -> bool:
        for k in range(n):
            cand = bin_balls()
            if not cand:
                return False
            i = cand[0]
            _refresh()
            cup_p = scene.cups[j].data.root_pos_w[0] - origin[0]
            dx, dy = SCATTER[k % len(SCATTER)]
            _drop_body(scene.balls[i],
                       origin + torch.tensor(
                           [float(cup_p[0]) + dx, float(cup_p[1]) + dy,
                            float(cup_p[2]) + c.cup_h / 2 + c.ball_r + 0.025],
                           device=device))
            if require_contained:
                if not settle_until(lambda i=i, j=j: bool(scene.in_cup()[0, j, i]),
                                    max_steps=int(3.0 / env.dt)):
                    return False
            else:
                _step(int(0.8 / env.dt))
        return True

    def cup_upz(j: int) -> float:
        _refresh()
        _p, _q, upz, _v = scene._cup_tensors()
        return float(upz[0, j])

    def score() -> float:
        _refresh()
        return float(scene.score()[0])

    def succ() -> bool:
        _refresh()
        return bool(scene.success()[0])

    def counts():
        _refresh()
        return scene.pad_counts()[0].tolist()

    # ================= 1. settle / no-NaN + ticket readback =======================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(150)
    _report("settle")
    _REC["on"] = False
    quota = scene.quota[0].tolist()
    pres = scene.present[0]
    n_pres = int(pres.sum())
    ball_pos = torch.stack([b.data.root_pos_w[0] for b in scene.balls])
    depot_ok = True
    for i in range(c.n_balls):
        if not bool(pres[i]):
            d = ball_pos[i, :2] - origin[0, :2] - torch.tensor(
                [c.depot[0], c.depot[1] + 0.07 * i], device=device)
            depot_ok = depot_ok and float(d.norm()) < 0.05
    dots_ok = True
    for j in range(2):
        shown = sum(1 for i in range(c.quota_max)
                    if float(scene.dots[j][i].data.root_pos_w[0, 2] - origin[0, 2]) > -0.02)
        dots_ok = dots_ok and shown == quota[j]
    check("settle/no-NaN: present balls rest in the bin, absent balls parked in the "
          "depot, both cups mouth-down, SHOWN dots per side == quota, score ~0",
          bool(torch.isfinite(ball_pos).all())
          and int((scene.in_bin()[0] & pres).sum()) == n_pres and depot_ok
          and cup_upz(0) < -0.9 and cup_upz(1) < -0.9 and dots_ok
          and score() <= 0.02 and not succ())

    # ================= 2. randomization is real (3-seed max-pairwise readback) ===================
    def readback():
        _refresh()
        return torch.cat([scene.pads[0].data.root_pos_w[0, :2] - origin[0, :2],
                          scene.cups[0].data.root_pos_w[0, :2] - origin[0, :2],
                          scene.balls[0].data.root_pos_w[0, :2] - origin[0, :2]]).clone()

    obs = []
    for sd in (101, 202, 303):
        torch.manual_seed(sd)
        env.reset()
        _step(10)
        obs.append(readback())
    dmax = torch.zeros(3)
    for a in range(3):
        for b in range(a + 1, 3):
            d = (obs[a] - obs[b]).reshape(3, 2).norm(dim=-1)
            dmax = torch.maximum(dmax, d.cpu())
    print(f"[smoke] randomization max-pairwise deltas: pad_a={float(dmax[0]) * 1000:.1f}mm "
          f"cup_a={float(dmax[1]) * 1000:.1f}mm ball0={float(dmax[2]) * 1000:.1f}mm", flush=True)
    check("randomization-is-real: pad, cup and ball positions READBACK differ across "
          "seeded resets (3-seed max-pairwise)",
          float(dmax[0]) > 0.003 and float(dmax[1]) > 0.005 and float(dmax[2]) > 0.003)

    # ================= 3. coverage: quotas vary, asymmetric pair, ball counts vary ================
    pairs, presents = [], []
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        pairs.append(tuple(scene.quota[0].tolist()))
        presents.append(int(scene.present[0].sum()))
    print(f"[smoke] over 10 resets: quota pairs={pairs} present={presents}", flush=True)
    check("coverage: >= 2 distinct quota pairs, an asymmetric pair appears, >= 2 "
          "distinct present-ball counts over 10 resets",
          len(set(pairs)) >= 2 and any(a != b for a, b in pairs)
          and len(set(presents)) >= 2)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps — cups still mouth-down, every present "
          "ball still in the bin, score ~0, no success",
          cup_upz(0) < -0.9 and cup_upz(1) < -0.9
          and int((scene.in_bin()[0] & scene.present[0]).sum()) == int(scene.present[0].sum())
          and score() <= 0.02 and not succ())

    # ================= 5. BULK-DUMP (the seed strategy) is rejected ===============================
    torch.manual_seed(100)
    env.reset()
    _step(120)
    quota = scene.quota[0].tolist()
    total = quota[0] + quota[1]
    _REC["on"] = True
    ok_place = place_cup(0) and place_cup(1)
    ok_dump = fill_cup(0, len(bin_balls()), require_contained=False)  # move EVERYTHING
    _step(300)
    _report("bulk-dump")
    _REC["on"] = False
    cnt = counts()
    cap = 2 * c.w_place + c.w_fill * quota[0] / total
    check("BULK-DUMP: the seed's move-everything strategy — cup A VERIFIED "
          "overfilled beyond its ticket — success refuses, overfill earns nothing "
          "(score capped at the quota's own fill credit)",
          ok_place and ok_dump and cnt[0] >= quota[0] + 1
          and score() <= cap + 0.02 and not succ())

    # ================= 6. OFF-BY-ONE allocation is rejected =======================================
    torch.manual_seed(100)
    env.reset()
    _step(120)
    quota = scene.quota[0].tolist()
    total = quota[0] + quota[1]
    ok_place = place_cup(0) and place_cup(1)
    ok_fill = fill_cup(0, quota[0] - 1) and fill_cup(1, quota[1] + 1)
    _step(240)
    _report("off-by-one")
    cnt = counts()
    expect = 2 * c.w_place + c.w_fill * (total - 1) / total
    s_now = score()
    print(f"[smoke] off-by-one: counts={cnt} quota={quota} score={s_now:.3f} "
          f"expected~{expect:.3f}", flush=True)
    check("OFF-BY-ONE: right total delivered but one ball allocated to the wrong "
          "pad — leftovers fine, cups placed, yet no success and fill credit exactly "
          "one ball short",
          ok_place and ok_fill and cnt[0] == quota[0] - 1 and cnt[1] == quota[1] + 1
          and abs(s_now - expect) <= 0.02 and not succ())

    # ================= 7. OFF-PAD near-miss =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(120)
    quota = scene.quota[0].tolist()
    total = quota[0] + quota[1]
    off = c.pad_tol + 0.020
    ok_a = place_cup(0, off=(0.0, off), expect_on_pad=False)  # outboard, AWAY from the pad
    ok_b = place_cup(1)
    ok_fill = fill_cup(0, quota[0]) and fill_cup(1, quota[1])
    _step(240)
    _report("off-pad")
    _refresh()
    d_a = float((scene.cups[0].data.root_pos_w[0, :2]
                 - scene.pads[0].data.root_pos_w[0, :2]).norm())
    held = int((scene.in_cup()[0, 0] & scene.present[0]).sum())
    cap = c.w_place + c.w_fill * quota[1] / total
    check("OFF-PAD near-miss: cup A upright, correctly filled (VERIFIED contains its "
          "quota) but centred past pad_tol — latch never arms, its balls count for "
          "nothing, no success",
          ok_a and ok_b and ok_fill and d_a > c.pad_tol
          and cup_upz(0) > 0.9 and held == quota[0]
          and not bool(scene._placed_latch[0, 0]) and counts()[0] == 0
          and score() <= cap + 0.02 and not succ())

    # ================= 8. MOUTH-DOWN COVER cheat ==================================================
    torch.manual_seed(100)
    env.reset()
    _step(120)
    quota = scene.quota[0].tolist()
    total = quota[0] + quota[1]
    px, py = pad_xy(0)
    # lay quota_a balls on the pad floor (spread inside the cup's interior footprint)
    layouts = {1: ((0.0, 0.0),),
               2: ((-0.0142, 0.0), (0.0142, 0.0)),
               3: tuple((0.0165 * math.cos(a), 0.0165 * math.sin(a))
                        for a in (math.pi / 2, math.pi / 2 + 2 * math.pi / 3,
                                  math.pi / 2 + 4 * math.pi / 3))}
    cover_ids = []
    for k, (dx, dy) in enumerate(layouts[quota[0]]):
        i = bin_balls()[0]
        cover_ids.append(i)
        _drop_body(scene.balls[i],
                   origin + torch.tensor([px + dx, py + dy, c.ball_r + 0.006], device=device))
        _step(int(0.5 / env.dt))
    # drop the cup MOUTH-DOWN over them, centred on the pad
    ok_cover = place_cup(0, mouth_down=True, expect_on_pad=False)
    ok_b = place_cup(1) and fill_cup(1, quota[1])
    _step(240)
    _report("cover-cheat")
    _refresh()
    # the balls really are inside the inverted interior (only the upright gate rejects)
    cp = scene.cups[0].data.root_pos_w[0]
    inside_geom = all(
        float((scene.balls[i].data.root_pos_w[0, :2] - cp[:2]).norm()) < c.cup_inner_r
        for i in cover_ids)
    cap = c.w_place + c.w_fill * quota[1] / total
    check("MOUTH-DOWN COVER: cup A upside-down ON its pad covering exactly its "
          "quota (balls VERIFIED inside the inverted interior) — count reads 0, "
          "latch never arms, no success",
          ok_cover and ok_b and cup_upz(0) < -0.9 and inside_geom
          and counts()[0] == 0 and not bool(scene._placed_latch[0, 0])
          and score() <= cap + 0.02 and not succ())

    # ================= 9. LEFTOVER LOOSE on the floor blocks success ==============================
    torch.manual_seed(100)
    env.reset()
    _step(120)
    quota = scene.quota[0].tolist()
    _REC["on"] = True
    ok_place = place_cup(0) and place_cup(1)
    ok_fill = fill_cup(0, quota[0]) and fill_cup(1, quota[1])
    stray = bin_balls()[0]
    _drop_body(scene.balls[stray],
               origin + torch.tensor([0.45, 0.0, c.ball_r + 0.010], device=device))
    _step(300)
    _report("leftover-loose")
    _REC["on"] = False
    _refresh()
    loose = (not bool(scene.in_bin()[0, stray])
             and not bool(scene.in_cup()[0, :, stray].any()))
    cnt = counts()
    s_now = score()
    check("LEFTOVER LOOSE: both pads exactly right but one surplus ball abandoned "
          "on the open floor — base credit FULL (0.75) yet success refuses (the "
          "leave-the-rest-in-the-bin clause is load-bearing)",
          ok_place and ok_fill and loose and cnt == quota
          and abs(s_now - (2 * c.w_place + c.w_fill)) <= 2e-3 and not succ())

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ration_ticket_station")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    check("video: frames captured and saved to frames.npz", len(_REC["frames"]) > 10)

    n_pass = sum(ok for _nm, ok in checks)
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
