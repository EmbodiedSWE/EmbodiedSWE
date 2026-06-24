"""Task-family suites.

Each sub-package (e.g. `assembly`) bundles its own scenes, assets, and configs, and registers them
into `robobench.core.registries` on import. There is no separate task layer — each scene *is* one
task (it carries its own goal; success criteria live in an optional Verifier). Shared machinery
(`robobench.core`) and reusable embodiments (`robobench.robots`) live one level up and are shared
across all suites — so a new suite adds a folder here without touching the others.
"""
