# This level: scene

The engine's whole idea is to take one solve that already works and multiply
it into many verified episodes. At the scene level you multiply the **world**:
each cell you make is one new version of the world the task happens in — with
its judge kept truthful — and the same delivered solve is run in it, batch
after batch, every episode graded. Different worlds are what make the
dataset's episodes observably different: different clutter, different tables,
different starting arrangements.

One rule above all: **the delivered solve must still succeed in your world.**
A world nobody can solve produces no data. When in doubt, change less.

## Gentle modifications only

Make simple, believable edits that preserve the task's contact-critical
mechanics:

- **Add non-task clutter** when it can rest stably outside the task and robot
  motion volumes. Visual-only assets are set dressing, not graspable objects.
- **Add task instances** when the task naturally extends to multiple targets.
  This changes what "done" means, so the judge must change with it.
- **Change the work surface**: a taller or lower table, or a different table
  model. Moving the surface moves the task in the robot's workspace, so do
  the small calculation for the new poses (heights, approach points) and
  change the necessary parameters in this cell's own `solve.py` so the solve
  still reaches them — and check the new height is actually reachable before
  building on it.

**Leave the hard parts exactly as the suite ships them**: the contact
mechanics (threads, insertion channels), the sim/PhysX settings, and the
welding/fastening mechanisms — those needed dedicated expert checks the
session cannot redo.

**Where new objects come from**: the suite's asset folders are linked under
`scenes/<scene>/assets/`; read how the start scene spawns its own objects and
spawn new ones the same way (`rigid` -> a free object, `static`/`articulation`
-> a fixture, purely visual geometry -> set dressing only, never a graspable
"distractor"). Place things clear of the task. Objects not in the assets can be
built from simple shapes (a box, a cylinder, with a material); do not import
asset files from outside. The robot base and its table are fixed — do not
propose moving them or changing the table height.

## The scene and its judge move as a pair

Where a change shifts what success looks like, rewrite the scene's own check
functions AND `grader/grader.py` so they describe the new world truthfully.
Two ways this happens:

- **Directly** — you extended the task to more target instances, so success
  now requires all of them.
- **By accident** — you added something "irrelevant" that the judge happens
  to measure. A distractor can alter a nearest-object check or a loose-object
  count. Before calling an object irrelevant, read what the grader measures
  and make sure the new object is invisible to it — or update the judge.

Where the meaning of success didn't move, leave the grader alone. An episode
graded by an outdated judge is worse than no episode: it is wrong data that
looks right.

## The four edits people forget

Adding an object is more than spawning it. Every new object needs all four:

1. **Spawn** it in `assets()`, at a spot that cannot collide with the task
   (on the surface, clear of the parts and the robot's path).
2. **Reset** it in `reset(env_ids)`, so every episode puts it back — with its
   own small placement variation if you want one (copy the scene's existing
   pattern).
3. **State**: add it to `get_state` / `set_state`. Skip this and saved
   episodes silently lose the object when restored later — the mistake that
   hurts most and shows up last.
4. **Describe** it in `describe()`. That text is all a solving agent knows
   about the world; an object missing from the description does not exist
   for it.

## Physical parameters — usually nothing to do

The scene already carries its own physics dials with sensible ranges
(`PHYSICAL_PARAMS` on the scene class): friction, mass and similar physical
properties. `generate` samples them on its own — each env in a batch gets its
own world, and env 0 always keeps the untouched original. You only act here
if:

- **your new object should vary too** (say, its friction): add one entry to
  `PHYSICAL_PARAMS` and one matching block in `apply_physical_params` (copy
  an existing one). Unknown names are rejected loudly, so a half-done
  extension cannot slip through.
- **your testing shows a shipped range is wrong**: tighten a range that kills
  the yield, and say so in `SUMMARY.md`.

Only physical properties of objects belong there — never grading thresholds,
controller gains, sim settings, or start-pose jitter (starting poses belong
to the scene's reset). To stop a dial from being sampled, set its entry to
`None`; don't delete the line — the nominal values are applied through the
same list, so deleting changes the normal world too.

`PHYSICAL_PARAMS`, `VISUAL_PARAMS` and `CAMERAS` are REQUIRED declarations on
every scene you ship — including `scene_0`: adding these declarations is the
one edit you may make to the start scene (nothing else in it may change).
`PHYSICAL_PARAMS` bands the world physics the task actually has (friction,
masses, small pose offsets; nominal = today's values; ranges the delivered solve
still succeeds under — measure, do not guess); without it every env of a wide
batch is the same world and the dynamics stage has nothing to vary.
`VISUAL_PARAMS` is the look axis of the visual stage: band the render-only knobs
the scene has (materials, colors, lighting; nominal = today's look; never a
color the task's semantics or grader depend on, never anything physics reads).
`CAMERAS` is the scene's own declared viewpoints (there is no pipeline default):
frame them so the WHOLE workspace is visible — robot base, target objects and
everything manipulated — and check a rendered frame; the render probe only
proves the camera builds, not that it frames anything. Camera heights are
relative to the scene's `surface_z` as the preset configures it.

## Verification

    generate --headless . --scene scene_N --num_envs <N> --seed 0

This generates one batch of data on your scene, under `data/<batch>/`:
one `ep_NNNN/` folder per episode, success/fail in each episode's `meta.json`,
and the batch summary (yield) in `data/<batch>/meta.json`. Physical parameters
are sampled automatically (env 0 always keeps the plain, unsampled world); add
`--nominal` to turn sampling off when you want a pure baseline. Judge by
success: does the delivered solve still succeed in your world? Env 0 failing
points at your scene edit, not at the physics ranges. A cell whose tests never
produce a successful episode must not ship: fix it or delete it.

Either way, write `SUMMARY.md` at the cell root: what you tried, what you
changed, the yields you measured, and what you learned — failed ideas
included; they are what the next session learns from.
