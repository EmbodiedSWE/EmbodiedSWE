"""CoSiGen / robobench code-as-policy sim helpers (run ON the GPU render server).

AGGREGATOR (2026-07-26 reorg): the former monolith now lives in purpose modules —
  cosigen_config   feature flags, gating sets, sibling reload, leakage guard
  cosigen_prompts  every agent-facing prompt section (editable text blocks)
  cosigen_view     per-env oracle view + restorable-state tensor utils
  cosigen_apis     the control-API classes, one per embodiment
  cosigen_session  suite registry, env build, make_prompt, namespace, run_policy
  cosigen_opt      the optimize tool's parallel parameter-search engine
This module re-exports the whole surface so `import cosigen_loop` keeps working for
the render server, cosigen_opt, tests, and agent programs alike.

NOTE: cosigen_apis/cosigen_session import torch/robobench at top — import this only
AFTER AppLauncher has booted (the server does that), matching robobench's app-first rule.
"""
from __future__ import annotations

import numpy as np  # noqa: F401 -- part of the historical public surface
import torch  # noqa: F401

from cosigen_config import *  # noqa: F401,F403
from cosigen_config import (_CKPT_FNS, _FEATURES_OFF, _OPT_FNS,  # noqa: F401
                            _RELOAD_HDFS_DIR, _import_sibling,
                            _refresh_sibling_on_reload)
from cosigen_prompts import *  # noqa: F401,F403
from cosigen_prompts import _compose_doc, _emb_screw, _strip_doc_section  # noqa: F401
from cosigen_view import *  # noqa: F401,F403
from cosigen_view import _clip_words, _np, _overwrite_replica_rows, _q_wxyz  # noqa: F401
from cosigen_apis import *  # noqa: F401,F403
from cosigen_session import *  # noqa: F401,F403
from cosigen_session import _suite_of  # noqa: F401
