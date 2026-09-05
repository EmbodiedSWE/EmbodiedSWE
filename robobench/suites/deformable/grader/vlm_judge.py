"""VLM plausibility gate for "final" grader stages — the judge and the final-frame capture.

Deformable state is high-dimensional and every programmatic metric is a proxy (a pitcher dunked
into the mug passes "milk in mug"; a crumpled ball passes "small footprint"). So a grader's LAST
rung can be a plausibility gate: the final scene is rendered once per env, the PNG is shown to
`gpt-5.6-terra` with a task-specific description of the intended final scene and the degenerate
states to reject, and the model answers strict JSON `{"plausible": bool, "reason": str}`. The gate
can only take credit away (`BaseGrader` ANDs it into success), never add it.

Two halves:
  * `judge_plausibility(png_bytes, task_prompt)` — the API call (deterministic: temperature 0,
    fixed system prompt, strict JSON parse; bounded retries on 5xx / timeouts, then raise).
  * `FrameCapture` — the viewport RGB capture, the same recipe as `eval/tools/scene_view.py`
    (our own camera prim + replicator rgb annotator, Kit app pumping), one frame per
    env from a fixed per-task camera pose, with the OTHER envs hidden while an env is filmed.
    `gate_final_frames(...)` ties both halves together for a final stage: it never raises —
    every exception is printed in full and recorded in that env's details with value 0.

Endpoint, key and model are HARDCODED here on purpose (project rule): this module is privileged
grading code, never shipped to agents. Nothing in this module imports Isaac at import time.
"""

from __future__ import annotations

import base64
import json
import tempfile
import time
import traceback
import urllib.error
import urllib.request
import uuid
from pathlib import Path

# seed-code.bytedance.com: the PUBLIC ByteDance platform host (reachable from Modal grading
# containers; verified 2026-09-05 with image input), OpenAI Responses API. The previous AIDP
# gateway is corp-only and was unreachable from every grading container, so the gate had never
# once been evaluated.
JUDGE_URL = "https://seed-code.bytedance.com/v1/responses"
JUDGE_KEY = "plat_F-l52t0Z2RC_9aKlDk_DndtbARtUXFLhzzh6UWBSkwY"
MODEL = "model_hub/robo_g56_terra"   # gpt-5.6-terra
READ_TIMEOUT = 150  # s per attempt (the laptop bridge's proven value)
MAX_ATTEMPTS = 4  # bounded: transient 5xx / 429 / timeouts retry, then raise
RETRY_STATUSES = (429, 500, 502, 503, 504)
OUT_DIR = Path("/out")  # the grading container's verdict mount (eval/grader/grade.py)

SYSTEM_PROMPT = (
    "You are a strict visual inspector for a robotics benchmark. You are shown ONE rendered image "
    "of the FINAL state of a simulated manipulation task, together with a description of what a "
    "plausibly completed final scene looks like and a list of degenerate outcomes that must be "
    "rejected. Decide whether the image shows a plausible completed final scene. Judge ONLY what is "
    "visible in the image: do not assume anything happened off-screen, do not reward effort or "
    "partial progress, and reject if any listed degenerate state is visible or if the described "
    "final arrangement is not clearly visible. Answer with a single JSON object and NOTHING else — "
    "no prose, no markdown, no code fences — exactly of the form "
    '{"plausible": true, "reason": "<one sentence>"} or '
    '{"plausible": false, "reason": "<one sentence>"}.'
)


# ----- the judge ---------------------------------------------------------------------------------
def _post(payload: dict) -> tuple[int, dict, str]:
    """One POST to the judge endpoint. Returns (http status, parsed JSON body, log id). Raises on
    transport failure (URLError / timeout) so the caller can decide whether to retry."""
    logid = f"cosigen-{uuid.uuid4().hex}"
    req = urllib.request.Request(
        JUDGE_URL, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {JUDGE_KEY}",
                 "X-TT-LOGID": logid})
    try:
        with urllib.request.urlopen(req, timeout=READ_TIMEOUT) as resp:
            return resp.status, json.loads(resp.read()), logid
    except urllib.error.HTTPError as exc:  # 4xx/5xx: hand the status + body back
        raw = exc.read().decode(errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"error": {"message": raw}}
        return exc.code, body, logid


def _message_text(body: dict) -> str:
    """The assistant text of an OpenAI Responses object (all output_text parts of its messages)."""
    parts = [c.get("text", "") for o in body.get("output", []) if o.get("type") == "message"
             for c in o.get("content", []) if isinstance(c, dict) and c.get("type") == "output_text"]
    if not parts:
        raise ValueError(f"no output_text in the judge's response: {json.dumps(body, ensure_ascii=False)[:1000]}")
    return "".join(parts)


def parse_verdict(text: str) -> dict:
    """STRICT parse of the model's answer: the whole reply must be one JSON object with a
    boolean `plausible` and a string `reason`. No regex rescue of prose — anything else is an
    error (a non-JSON answer must surface, not be guessed at)."""
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge answer is not JSON ({exc}): {text!r}") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("plausible"), bool) \
            or not isinstance(parsed.get("reason"), str):
        raise ValueError(f"judge answer is not {{plausible: bool, reason: str}}: {text!r}")
    return parsed


def judge_plausibility(image_png_bytes: bytes, task_prompt: str) -> tuple[bool, dict]:
    """Ask gpt-5.6-terra whether `image_png_bytes` shows a plausible completed final scene for
    `task_prompt`. Returns (plausible, details) where details carries the model, log id, HTTP
    status, attempt count, the prompt, the RAW response body and the parsed answer — everything
    needed to audit the decision from verdict.json. Raises after MAX_ATTEMPTS transient
    failures, on any other HTTP error, and on a non-JSON answer; nothing is swallowed."""
    b64 = base64.b64encode(image_png_bytes).decode()
    payload = {
        "model": MODEL,
        "instructions": SYSTEM_PROMPT,
        "reasoning": {"effort": "medium"},
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": task_prompt},
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64}", "detail": "high"},
        ]}],
    }
    last = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            status, body, logid = _post(payload)
        except Exception as exc:  # noqa: BLE001 -- transport failure (timeout, DNS, TLS): retry, bounded
            last = f"{type(exc).__name__}: {exc}"
            print(f"[vlm_judge] attempt {attempt}/{MAX_ATTEMPTS}: transport failure {last}", flush=True)
            if attempt < MAX_ATTEMPTS:
                time.sleep(min(2 ** attempt, 16))
            continue
        if status in RETRY_STATUSES:
            last = f"HTTP {status}: {json.dumps(body, ensure_ascii=False)}"
            print(f"[vlm_judge] attempt {attempt}/{MAX_ATTEMPTS} (logid {logid}): {last}", flush=True)
            if attempt < MAX_ATTEMPTS:
                time.sleep(min(2 ** attempt, 16))
            continue
        if status != 200:
            raise RuntimeError(f"judge HTTP {status} (logid {logid}): {json.dumps(body, ensure_ascii=False)}")
        text = _message_text(body)
        parsed = parse_verdict(text)
        details = {"model": MODEL, "logid": logid, "http_status": status, "attempts": attempt,
                   "prompt": task_prompt, "raw_response": body, "answer_text": text,
                   "parsed": parsed, "decision": parsed["plausible"]}
        print(f"[vlm_judge] {MODEL} (logid {logid}): plausible={parsed['plausible']} — "
              f"{parsed['reason']}", flush=True)
        return parsed["plausible"], details
    raise RuntimeError(f"judge call failed after {MAX_ATTEMPTS} attempts; last: {last}")


# ----- the final frame ---------------------------------------------------------------------------
def png_bytes(frame) -> bytes:
    """Encode an (H, W, 3) uint8 array as PNG bytes."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return buf.getvalue()


def final_frame_paths(num_envs: int) -> list[Path]:
    """Where each env's final frame goes: next to its verdict (`/out/traj_<e>/final.png`, the
    layout eval/grader/grade.py writes) when `/out` exists, else a fresh temp dir."""
    if OUT_DIR.is_dir():
        return [OUT_DIR / f"traj_{e:03d}" / "final.png" for e in range(num_envs)]
    tmp = Path(tempfile.mkdtemp(prefix="cosigen_final_"))
    return [tmp / f"traj_{e:03d}" / "final.png" for e in range(num_envs)]


class FrameCapture:
    """Final-frame RGB capture from the grader's OWN camera prim.

    Isaac Lab 2.3.2's recipe (drive `/OmniverseKit_Persp` through `sim.set_camera_view` and
    `sim.render()`) does nothing on isaaclab develop / the Newton stack: there both calls only
    forward to registered *visualizers*, and a headless grading app has none — measured
    2026-09-05: `sim.render()` produced 0 pixels, `set_camera_view` left the frame identical.
    What works on every version: a `UsdGeom.Camera` we define ourselves, posed by writing its
    look-at transform, a replicator render product on that prim, and pumping the Kit app loop
    (`omni.kit.app.get_app().update()`), which is what drives RTX. Requires the app launched
    with `enable_cameras=True` (grade.py --cameras); anything missing raises — no fake images.
    """

    CAMERA_PATH = "/World/cosigen_final_cam"

    def __init__(self, env, size: tuple[int, int] = (1280, 720)) -> None:
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom

        self.env = env
        self.size = (int(size[0]), int(size[1]))
        stage = omni.usd.get_context().get_stage()
        cam = UsdGeom.Camera.Define(stage, self.CAMERA_PATH)
        cam.CreateFocalLengthAttr(18.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        self._xform = UsdGeom.Xformable(cam.GetPrim())
        try:
            import carb.settings

            carb.settings.get_settings().set("/rtx/post/aa/op", 2)  # 2 = FXAA: single-frame AA, no ghosting
        except Exception as exc:  # noqa: BLE001 -- a quality tweak must not block capture
            print(f"[vlm_judge] could not force FXAA ({exc!r}); the capture may ghost", flush=True)
        product = rep.create.render_product(self.CAMERA_PATH, self.size)
        self._annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._annot.attach([product])
        self._tick(8)  # warm up so the first real frame is populated
        probe = self._annot.get_data()
        shape = getattr(probe, "shape", None)
        if not shape or len(shape) < 3 or shape[0] == 0:
            raise RuntimeError(
                "camera capture returned no pixels — the grading app must be launched with "
                "AppLauncher(headless=True, enable_cameras=True) (grade.py --cameras)")
        print(f"[vlm_judge] capture ready, rgb shape {tuple(shape)} from {self.CAMERA_PATH}", flush=True)

    def _tick(self, n: int) -> None:
        """Sync physics state into the render scene, then advance the Kit app loop n frames.

        `sim.render()` renders nothing here (no visualizer) but is what flushes Newton state
        into Fabric — body transforms AND deformable mesh points (NewtonManager.pre_render ->
        sync_particles_to_usd, for meshes tagged newton:particleOffset/Count). Without it the
        frame shows the cloth at its spawn pose while the rubric says "folded" (measured
        2026-09-05 on tshirt_fable_5_1). `kit.update()` is the only thing that drives RTX."""
        import omni.kit.app

        self.env.sim.render()
        kit = omni.kit.app.get_app()
        for _ in range(n):
            kit.update()

    def _look_at(self, eye_w, target_w) -> None:
        from pxr import Gf

        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye_w)), Gf.Vec3d(*map(float, target_w)),
                                       Gf.Vec3d(0.0, 0.0, 1.0))  # Z-up world
        self._xform.ClearXformOpOrder()
        self._xform.AddTransformOp().Set(view.GetInverse())  # camera-to-world

    def _env_prims(self):
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        prims = []
        for i in range(int(self.env.num_envs)):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{i}")
            if prim and prim.IsValid():
                prims.append((i, UsdGeom.Imageable(prim)))
        return prims

    def frame(self, env_idx: int, eye, target):
        """Render env `env_idx` from `eye` -> `target` (env-local meters, added to that env's
        origin) with every other env hidden (visibility only — physics untouched; restored
        afterwards). Returns an (H, W, 3) uint8 array; raises if the renderer gives no pixels."""
        import numpy as np

        origin = self.env.iscene.env_origins[env_idx].detach().cpu().numpy().astype(float)
        eye_w = tuple(np.asarray(eye, dtype=float) + origin)
        target_w = tuple(np.asarray(target, dtype=float) + origin)
        prims = self._env_prims()
        try:
            for i, im in prims:
                if i != env_idx:
                    im.MakeInvisible()
            self._look_at(eye_w, target_w)
            self._tick(30)  # settle the moved camera + visibility edits; RTX accumulates samples
            arr = np.asarray(self._annot.get_data())
        finally:
            for _, im in prims:
                im.MakeVisible()
        if arr.ndim != 3 or arr.shape[0] == 0:
            raise RuntimeError(f"camera capture returned no pixels for env {env_idx} (shape {arr.shape})")
        return arr[:, :, :3].astype(np.uint8).copy()


def open_capture(env, tag: str) -> FrameCapture | None:
    """Attach the final-frame capture at grader setup. A failure is printed with its full
    traceback and yields None — `gate_final_frames` then re-attempts the attach at verdict
    time and, if that fails too, records the error in the details with the gate at 0."""
    try:
        return FrameCapture(env)
    except Exception:  # noqa: BLE001 -- printed in full here, surfaces again (recorded) in the gate
        print(f"[{tag}] final-frame capture not available at setup:\n{traceback.format_exc()}",
              flush=True)
        return None


def gate_final_frames(env, capture: FrameCapture | None, task_prompt: str, eye, target,
                      prepare=None) -> tuple[list[float], list[dict]]:
    """The body of a grader's VLM final stage: for every env, render the final frame (after the
    optional `prepare()` — e.g. pushing MPM particle positions into the renderer), save it as
    PNG next to the verdict, ask the judge, and return (values, details) with value 1.0 for a
    plausible frame and 0.0 otherwise. `capture` is the FrameCapture opened at setup; if that
    failed (None) one attach is attempted here. Nothing is swallowed: every exception is printed
    with its traceback and recorded verbatim in that env's details (value 0 — the gate fails)."""
    n = int(env.num_envs)
    paths = final_frame_paths(n)
    values = [0.0] * n
    details: list[dict] = [{"prompt": task_prompt, "image": str(paths[e])} for e in range(n)]
    if capture is None:
        try:
            capture = FrameCapture(env)
        except Exception:  # noqa: BLE001 -- recorded + printed below; the gate fails for every env
            err = traceback.format_exc()
            print(f"[vlm_judge] final-frame capture unavailable:\n{err}", flush=True)
            for d in details:
                d["error"] = err
            return values, details
    if prepare is not None:
        try:
            prepare()
        except Exception:  # noqa: BLE001
            err = traceback.format_exc()
            print(f"[vlm_judge] render preparation failed:\n{err}", flush=True)
            for d in details:
                d["error"] = err
            return values, details
    for e in range(n):
        try:
            frame = capture.frame(e, eye, target)
            png = png_bytes(frame)
            paths[e].parent.mkdir(parents=True, exist_ok=True)
            paths[e].write_bytes(png)
            print(f"[vlm_judge] env {e}: final frame {frame.shape[1]}x{frame.shape[0]} -> {paths[e]}",
                  flush=True)
            ok, info = judge_plausibility(png, task_prompt)
            details[e].update(info)
            values[e] = 1.0 if ok else 0.0
        except Exception:  # noqa: BLE001 -- printed in full, recorded, value stays 0
            err = traceback.format_exc()
            print(f"[vlm_judge] env {e}: gate FAILED with an error (value 0):\n{err}", flush=True)
            details[e]["error"] = err
    return values, details
