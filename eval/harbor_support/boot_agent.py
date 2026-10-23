"""Harbor infrastructure fixture: boot/reset the exported world without grading."""

import shlex

from harbor.agents.base import BaseAgent


BOOT_PROGRAM = r'''
import json
import os
from pathlib import Path

protocol = json.loads(Path('/task/protocol.json').read_text())
print('GPU_BOOT_START ' + protocol['preset'], flush=True)
print('uid={} home={} cwd={}'.format(os.getuid(), Path.home(), Path.cwd()), flush=True)
try:
    import pinocchio
except ImportError:
    pass
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
print('GPU_BOOT_APP_READY', flush=True)
import robobench
assert Path(robobench.__file__).is_relative_to('/bench')
robobench.discover()
from robobench.core.registries import ENVS
env = ENVS.get(protocol['preset'])().build(num_envs=protocol['num_envs'], seed=protocol['seeds'][0])
env.reset(seed=protocol['seeds'][0])
assert env.num_envs == protocol['num_envs']
print('HARBOR_GPU_BOOT_OK ' + protocol['preset'], flush=True)
# Kit teardown can outlive the test budget; the process/container owns all resources.
os._exit(0)
'''


class ContainerBootAgent(BaseAgent):
    @staticmethod
    def name():
        return "robobench-container-boot"

    def version(self):
        return "1"

    async def setup(self, environment):
        result = await environment.exec(
            'test "$(id -un)" = agent && test "$(id -u)" != 0 '
            '&& test ! -e /tests/graders && test ! -w /task/protocol.json '
            '&& test ! -w /bench/robobench/__init__.py '
            '&& test -z "${VLM_JUDGE_API_KEY:-}"', timeout_sec=30)
        if result.return_code:
            raise RuntimeError("Agent user, world permissions, or grader separation check failed")

    async def run(self, instruction, environment, context):
        result = await environment.exec(
            "/opt/venv/bin/python -u -c " + shlex.quote(BOOT_PROGRAM), timeout_sec=1800)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "gpu-boot.log").write_text(
            (result.stdout or "") + "\n" + (result.stderr or ""))
        if result.return_code or "HARBOR_GPU_BOOT_OK " not in (result.stdout or ""):
            raise RuntimeError("GPU build/reset failed; inspect agent/gpu-boot.log")
        # An empty final delivery exercises the separate verifier without invoking a grader.
        context.metadata = {"gpu_build_reset": "passed", "delivery": "empty"}
