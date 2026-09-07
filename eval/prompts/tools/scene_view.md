== Seeing the scene (the scene_view tool) ==
Numbers lie by omission: a distance can shrink while the part tips over. Look at the scene.

    from scene_view import Viewer

    viewer = Viewer(env)                      # once, after env.reset()
    viewer.snapshot("before")                 # PNG of the current state
    for t in range(steps):
        env.step(action)
        viewer.grab()                         # buffer footage of the run (cheap)
    viewer.save_frames(6, "insert attempt")   # 6 stills sampled across the run
    viewer.save_video("insert attempt")       # or the same footage as one mp4

Files land under `/workspace/.footage/`, numbered in order. Saving a path or printing its name
does not show you the scene: transport the actual PNG pixels through your CLI's image-capable
read/view/attachment mechanism and inspect them. If your interface cannot transport images,
state that visual evidence is unavailable instead of claiming the image confirms a diagnosis.

  * Boot with cameras on, or capture cannot exist: `AppLauncher(headless=True,
    enable_cameras=True)`. `Viewer` raises at construction if pixels are unavailable.
  * The first rendering boot in a container compiles shaders — noticeably slower than the
    normal boot. Run long captures in the background and poll their log rather than
    sitting on a foreground timeout.
  * Snapshot before/after every maneuver you care about, and transport and inspect the LAST
    still of a failed run before deciding why it failed. Record the failed attempt even when the
    image disproves your initial diagnosis.
  * `viewer.clear()` between attempts keeps each video covering exactly one run.
  * The default view frames a tabletop workspace. For anything else — or when the work is
    occluded — pass your own angle: `Viewer(env, eye=(1.0, -1.2, 1.6),
    target=(0.4, 0.0, 1.0))` (positions relative to the env origin).

Images diagnose development runs; they do not replace the final integrated fresh-reset
verification.
