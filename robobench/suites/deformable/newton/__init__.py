"""Newton substrates for the deformable suite.

One `SimCfg` subclass per material class — `cloth_sim.NewtonSimCfg` (VBD cloth, tshirt),
`mpm_sim.MpmSimCfg` (implicit-MPM liquids, latte), `dough_sim.DoughSimCfg` (elastoplastic MPM,
dumpling), `rod_sim.RodSimCfg` (VBD/AVBD rods, knot) — plus the managers the scenes build on:
`coupled_manager.NewtonCoupledMJWarpMPMManager` (latte and dumpling share it verbatim) and
`lace_manager.NewtonLaceVBDManager` (per-substep kinematic handles + the rod contact recipe).

Import-light: everything that touches newton / isaaclab is deferred into functions, so the
registry side of the suite works in any venv.
"""
