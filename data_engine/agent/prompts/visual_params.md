# Visual-params session — give every scene its VISUAL_PARAMS look axis

You are inside a **data_gen campaign** that multiplies verified episodes into a
demonstration dataset. The MULTIPLY stage replays each verified episode under
K different looks — but a look is only more than a camera angle if the scene
declares `VISUAL_PARAMS` bands (materials, colors, lighting) for the renderer
to draw from. These scenes currently declare none, so their visual passes vary
NOTHING but camera pose:

{scenes}

Scene files (edit IN PLACE):

{scene_paths}

## What to add

A module-level `VISUAL_PARAMS` dict in each scene file, using the same band
grammar as `PHYSICAL_PARAMS` (see other suite scenes and
`data_engine/engine/sampler.py` in the framework for the exact spec; campaign
root is your cwd: `{gen}`). Give bands to the visual knobs the scene actually
has — table/background materials or colors, light intensity/temperature/
direction, object albedo where it does not encode task-relevant identity.
Every band's nominal value must reproduce today's look.

Hard constraints:

1. RENDER-ONLY knobs. `VISUAL_PARAMS` must not touch geometry, poses, masses,
   friction, or anything physics reads — physics provenance of already-verified
   episodes must stay exactly true. If a knob could plausibly alter contact or
   dynamics, it does not belong here.
2. THE SCENE MUST STILL BUILD. If your edit makes any code path read a cfg
   attribute (e.g. `c.light_intensity` in `assets()`), DECLARE that attribute
   on the scene's cfg dataclass with today's value as its default. The
   orchestrator verifies your edit by actually building and running each edited
   scene — an AttributeError at build time rejects the whole session. If a
   scene listed above already declares `VISUAL_PARAMS`, that is why: a previous
   edit broke its build — read the scene, find the breakage, fix it.
3. Do not change anything else in the scene files: no new objects, no camera
   edits, no refactors. This session adds the missing declaration, nothing more.
4. Task-critical appearance stays recognizable: never randomize a color the
   grader or the task semantics depend on (e.g. a color-matched target).

The scene's `apply_visual_params(env, values)` hook applies live knobs at
render time; build-consumed knobs work with no extra code. If the scene class
lacks the hook and your knobs need one, add the minimal hook.

Verify your bands parse: a 1-env smoke render of any verified episode with
`--visual_draw 1` (see `render.py --help`; a GPU with Isaac is available)
must complete and show a visibly different look. The orchestrator re-checks
the scene FILES for `VISUAL_PARAMS` after your session — an empty stub
declaration passes nothing downstream, so make the bands real.

You are running autonomously: no one answers questions; your final message
ends the session.
