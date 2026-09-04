# real_to_sim

TODO.

(For now: see `background/README.md` for the background stage — real scene →
splat backdrop — and `objects/README.md` for the objects stage — real object
photos → sim assets.)

Runtime splat rendering inside robobench envs (robot link gaussians re-posed from the sim every
step, GSWorld style) lives in the importable package `gsworld/` — see
`gsworld/README.md`. The stages here produce the splats; `gsworld` consumes them.
