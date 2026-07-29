#!/usr/bin/env python3
"""Build an interactive viewer for one grade folder: frames on one side,
score curve + rubric bars on the other, synced on sim time.

    python3 eval/scripts/view_grade.py experiments/<exp>/runs/<run>/grades/<name> [--serve 8110]

Reads verdict.json + progress.jsonl (+ frames.jsonl if the grade was rendered)
and writes <grade>/viewer.html — a single static page; frames are referenced
relatively, so serve the grade folder itself (--serve does it).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

MAX_CURVE_POINTS = 10_000  # decimate per-step curves; plenty for lookup + plot


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grade", help="a grade output folder (verdict.json + progress.jsonl [+ frames.jsonl])")
    ap.add_argument("--serve", type=int, default=None, metavar="PORT",
                    help="after building, serve the folder on this port")
    args = ap.parse_args()

    gdir = Path(args.grade).resolve()
    verdict_file = gdir / "verdict.json"
    # a budget-killed grade has no verdict — still viewable (frames + partial curve)
    verdict = json.loads(verdict_file.read_text()) if verdict_file.exists() else {}
    grade_meta = json.loads((gdir / "grade.json").read_text()) if (gdir / "grade.json").exists() else {}

    progress = load_jsonl(gdir / "progress.jsonl")
    if not progress:
        sys.exit(f"no progress.jsonl in {gdir} — nothing to plot")
    if len(progress) > MAX_CURVE_POINTS:
        keep = max(1, len(progress) // MAX_CURVE_POINTS)
        progress = progress[::keep] + [progress[-1]]
    frames = load_jsonl(gdir / "frames.jsonl")

    stages = list(progress[0].get("stages", {}).keys())
    # weights out of the criteria string ("picked ×0.2 (once) · engaged ×0.2 · ...")
    weights = dict(re.findall(r"(\w+) ×([\d.]+)", verdict.get("criteria", "")))

    data = {
        "title": "/".join(p for p in (grade_meta.get("exp", gdir.parent.parent.parent.name).split("/")[-1],
                                      gdir.parent.parent.name, gdir.name) if p),
        "verdict": {k: verdict.get(k) for k in ("success", "score", "criteria", "solve_wall_s", "solve_sim_steps")},
        "stages": stages,
        "weights": weights,
        "progress": [{"t": r["sim_time_s"], "p": r["progress"],
                      "s": [r["stages"].get(n) for n in stages]} for r in progress],
        "frames": [{"t": r["sim_time_s"], "f": r["file"]} for r in frames],
    }

    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = gdir / "viewer.html"
    out.write_text(html)
    print(f"built: {out}  ({len(data['progress'])} curve points, {len(data['frames'])} frames)")

    if args.serve:
        import http.server
        import functools
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(gdir))
        print(f"serving: http://localhost:{args.serve}/viewer.html   (Ctrl-C to stop)")
        http.server.ThreadingHTTPServer(("", args.serve), handler).serve_forever()
    else:
        print(f"serve:  python3 -m http.server 8110 --directory {gdir}"
              f"\nthen:   http://localhost:8110/viewer.html")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>grade viewer</title>
<style>
  :root { --bg:#0D1418; --panel:#151D22; --ink:#E6ECEF; --mut:#AAB8C0; --line:#28343B;
          --acc:#4EC3C9; --good:#5FC084; --bad:#E06C60; }
  @media (prefers-color-scheme: light) {
    :root { --bg:#F5F8F8; --panel:#FFFFFF; --ink:#17252C; --mut:#4C5E68; --line:#DDE6E7;
            --acc:#0B6B70; --good:#2E7C4B; --bad:#B3402F; } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font-family:system-ui,"Segoe UI",Roboto,Arial,sans-serif; font-size:15px; }
  header { display:flex; align-items:baseline; gap:14px; padding:14px 20px; border-bottom:1px solid var(--line); flex-wrap:wrap; }
  header h1 { margin:0; font-size:17px; font-weight:650; font-family:ui-monospace,Consolas,monospace; }
  .badge { font-weight:700; padding:2px 10px; border-radius:999px; font-size:13px; }
  .ok  { background:color-mix(in srgb, var(--good) 18%, transparent); color:var(--good); }
  .bad { background:color-mix(in srgb, var(--bad) 18%, transparent);  color:var(--bad); }
  .na  { background:color-mix(in srgb, var(--line) 55%, transparent); color:var(--mut); }
  .crit { color:var(--mut); font-size:12.5px; }
  main { display:flex; gap:16px; padding:16px 20px; flex-wrap:wrap; }
  #left { flex:3 1 560px; min-width:340px; }
  #right { flex:2 1 380px; min-width:320px; display:flex; flex-direction:column; gap:14px; }
  #shot { width:100%; aspect-ratio:16/9; background:#000; border:1px solid var(--line); border-radius:10px; }
  #noframes { padding:40px; text-align:center; color:var(--mut); border:1px dashed var(--line); border-radius:10px; }
  #controls { display:flex; align-items:center; gap:10px; margin-top:10px; }
  button { background:var(--panel); color:var(--ink); border:1px solid var(--line);
           border-radius:8px; padding:6px 14px; font-size:15px; cursor:pointer; }
  button:hover { border-color:var(--acc); }
  select { background:var(--panel); color:var(--ink); border:1px solid var(--line); border-radius:8px; padding:5px; }
  #scrub { flex:1; accent-color:var(--acc); }
  #clock { font-family:ui-monospace,Consolas,monospace; font-size:13px; color:var(--mut); white-space:nowrap; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
  .card h2 { margin:0 0 8px; font-size:12px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--mut); }
  canvas { width:100%; }
  .stage { display:grid; grid-template-columns: 92px 1fr 56px; align-items:center; gap:10px; margin:7px 0; }
  .stage .name { font-family:ui-monospace,Consolas,monospace; font-size:13px; }
  .stage .w { color:var(--mut); font-size:11px; }
  .bar { height:14px; background:color-mix(in srgb, var(--line) 55%, transparent); border-radius:7px; overflow:hidden; }
  .bar i { display:block; height:100%; background:var(--acc); border-radius:7px; width:0%; transition:width .06s linear; }
  .stage .val { text-align:right; font-family:ui-monospace,Consolas,monospace; font-size:13px; }
  .stage.total .bar i { background:var(--good); }
  kbd { color:var(--mut); font-size:11.5px; }
</style>
<body>
<header>
  <h1 id="title"></h1>
  <span id="badge" class="badge"></span>
  <span id="score"></span>
  <span class="crit" id="crit"></span>
</header>
<main>
  <div id="left">
    <canvas id="shot" hidden></canvas>
    <div id="noframes" hidden>no frames in this grade — run it with --render for video</div>
    <div id="controls">
      <button id="play">&#9654;</button>
      <select id="speed"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select>
      <input id="scrub" type="range" min="0" max="10000" value="0">
      <span id="clock"></span>
    </div>
    <kbd>space play/pause &nbsp;·&nbsp; &#8592;/&#8594; step 1 s</kbd>
  </div>
  <div id="right">
    <div class="card"><h2>progress</h2><canvas id="plot" height="240"></canvas></div>
    <div class="card"><h2>rubric</h2><div id="stages"></div></div>
  </div>
</main>
<script>
const D = __DATA__;
const P = D.progress, F = D.frames, T = P[P.length-1].t;
const colors = ["#E0A93E","#7FB4E0","#B07FE0","#6FCF8F","#E07F9E"];

document.getElementById("title").textContent = D.title;
const ok = D.verdict.success;
const badge = document.getElementById("badge");
badge.textContent = ok === true ? "success" : ok === false ? "failed" : "no verdict";
badge.className = "badge " + (ok === true ? "ok" : ok === false ? "bad" : "na");
document.getElementById("score").textContent = D.verdict.score == null ? "" : "score " + D.verdict.score;
document.getElementById("crit").textContent = D.verdict.criteria || "";

const stagesDiv = document.getElementById("stages");
const bars = [];
function addBar(name, w, cls) {
  const row = document.createElement("div");
  row.className = "stage" + (cls ? " " + cls : "");
  row.innerHTML = `<span class="name">${name} <span class="w">${w ? "×"+w : ""}</span></span>
                   <span class="bar"><i></i></span><span class="val"></span>`;
  stagesDiv.appendChild(row);
  bars.push({fill: row.querySelector("i"), val: row.querySelector(".val")});
}
addBar("progress", "", "total");
D.stages.forEach((n) => addBar(n, D.weights[n] || ""));

const shot = document.getElementById("shot"), sctx = shot.getContext("2d");
const warmed = new Set();
function prefetch(k) {  // rolling look-ahead — never flood the connection
  for (let j = k + 1; j <= Math.min(F.length - 1, k + 15); j++)
    if (!warmed.has(j)) { warmed.add(j); const i = new Image(); i.src = F[j].f; }
}
let pendingF = -1, shownF = -1;
function show(k) {  // double-buffered: the old frame stays until the new one is decoded
  if (k === shownF || k === pendingF) return;
  pendingF = k;
  const img = new Image();
  img.src = F[k].f;
  img.decode().then(() => {
    if (pendingF !== k) return;  // superseded by a later seek
    if (shot.width !== img.naturalWidth) { shot.width = img.naturalWidth; shot.height = img.naturalHeight; }
    sctx.drawImage(img, 0, 0);
    shownF = k;
  }).catch(() => { pendingF = -1; });
  prefetch(k);
}
if (F.length) { shot.hidden = false; show(0); }
else document.getElementById("noframes").hidden = false;

function bisect(arr, t) {  // last index with .t <= t
  let lo = 0, hi = arr.length - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (arr[mid].t <= t) lo = mid; else hi = mid - 1; }
  return lo;
}

const plot = document.getElementById("plot"), ctx = plot.getContext("2d");
let plotW = 0, plotH = 240;
const css = getComputedStyle(document.documentElement);
function drawPlot(t) {
  if (plot.clientWidth !== plotW) { plotW = plot.clientWidth; plot.width = plotW * devicePixelRatio; plot.height = plotH * devicePixelRatio; ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0); }
  const W = plotW, H = plotH, x = tt => 34 + (W-40)*(tt/T), y = p => (H-22) - (H-34)*p;
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle = css.getPropertyValue("--line"); ctx.fillStyle = css.getPropertyValue("--mut");
  ctx.font = "10px ui-monospace,monospace"; ctx.lineWidth = 1;
  for (const g of [0,0.5,1]) { ctx.beginPath(); ctx.moveTo(x(0),y(g)); ctx.lineTo(x(T),y(g)); ctx.stroke(); ctx.fillText(g.toFixed(1), 8, y(g)+3); }
  D.stages.forEach((n,i) => {  // faint per-stage lines
    ctx.strokeStyle = colors[i % colors.length] + "88"; ctx.beginPath();
    P.forEach((r,k) => { const v = r.s[i] ?? 0; k ? ctx.lineTo(x(r.t), y(v)) : ctx.moveTo(x(r.t), y(v)); });
    ctx.stroke();
  });
  ctx.strokeStyle = css.getPropertyValue("--acc"); ctx.lineWidth = 2; ctx.beginPath();
  P.forEach((r,k) => k ? ctx.lineTo(x(r.t), y(r.p)) : ctx.moveTo(x(r.t), y(r.p)));
  ctx.stroke();
  ctx.strokeStyle = css.getPropertyValue("--bad"); ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(x(t), y(1.05)); ctx.lineTo(x(t), y(-0.02)); ctx.stroke();
  ctx.fillStyle = css.getPropertyValue("--mut");
  ctx.fillText("0 s", x(0)-4, H-6); ctx.fillText(T.toFixed(0)+" s sim", x(T)-30, H-6);
}

let t = 0, playing = false, lastTs = null;
const playBtn = document.getElementById("play"), scrub = document.getElementById("scrub"),
      clock = document.getElementById("clock"), speedSel = document.getElementById("speed");

function render() {
  const r = P[bisect(P, t)];
  bars[0].fill.style.width = (100*r.p)+"%";
  bars[0].val.textContent = r.p.toFixed(3);
  D.stages.forEach((n,i) => { const v = r.s[i] ?? 0;
    bars[i+1].fill.style.width = (100*v)+"%"; bars[i+1].val.textContent = v.toFixed(2); });
  if (F.length) show(bisect(F, t));
  if (!scrubbing) scrub.value = Math.round(10000 * t / T);
  clock.textContent = t.toFixed(1) + " / " + T.toFixed(1) + " s sim";
  drawPlot(t);
}
function tick(ts) {
  if (playing) {
    if (lastTs !== null) t = Math.min(T, t + (ts - lastTs)/1000 * parseFloat(speedSel.value));
    lastTs = ts;
    if (t >= T) toggle(false);
    render();
  }
  requestAnimationFrame(tick);
}
function toggle(on) {
  playing = on === undefined ? !playing : on;
  if (playing && t >= T - 1e-6) t = 0;  // play again from the start
  lastTs = null;
  playBtn.innerHTML = playing ? "&#10074;&#10074;" : "&#9654;";
}
playBtn.onclick = () => toggle();
let scrubbing = false, wasPlaying = false;
scrub.addEventListener("pointerdown", () => { scrubbing = true; wasPlaying = playing; toggle(false); });
scrub.addEventListener("pointerup", () => { scrubbing = false; if (wasPlaying) toggle(true); });
scrub.oninput = () => { t = T * scrub.value / 10000; render(); };
addEventListener("keydown", e => {
  if (e.code === "Space") { e.preventDefault(); toggle(); }
  if (e.code === "ArrowRight") { t = Math.min(T, t+1); render(); }
  if (e.code === "ArrowLeft")  { t = Math.max(0, t-1); render(); }
});
addEventListener("resize", () => render());
render(); requestAnimationFrame(tick);
</script>
"""


if __name__ == "__main__":
    main()
