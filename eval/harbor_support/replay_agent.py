"""Harbor acceptance agent: deliver trusted references using the public submit CLI.

This is a transport fixture, not a benchmark solver. Reference files stay outside
the exported task so replay cannot invalidate the export's content receipt.
"""

import json
import shlex
import shutil
import tarfile
import tempfile
from pathlib import Path
from uuid import uuid4

from harbor.agents.base import BaseAgent

from .catalog import TASKS
from .files import content_hash, inventory, write_json
from .reference import load_references


class ReferenceReplayAgent(BaseAgent):
    def __init__(self, *args, manifest: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.references = load_references(Path(manifest))

    @staticmethod
    def name():
        return "robobench-reference-replay"

    def version(self):
        return "1"

    async def setup(self, environment):
        # These checks run as Harbor's declared agent user, not as root.
        result = await environment.exec(
            'test "$(id -un)" = agent && test "$(id -u)" != 0 '
            '&& test -z "${VLM_JUDGE_API_KEY:-}" && test ! -e /tests/graders '
            '&& test ! -w /bench/robobench/__init__.py '
            '&& test ! -w /task/protocol.json '
            '&& for d in /opt/venv/lib/python*/site-packages/isaacsim/kit/cache '
            '/opt/venv/lib/python*/site-packages/isaacsim/kit/data '
            '/opt/venv/lib/python*/site-packages/isaacsim/kit/logs; '
            'do test -d "$d" && test -w "$d" || exit 1; done', timeout_sec=30)
        if result.return_code != 0:
            raise RuntimeError("Agent user, grader separation, credential isolation, world/cache permissions check failed")

    async def run(self, instruction, environment, context):
        result = await environment.exec("cat /task/protocol.json", timeout_sec=30)
        if result.return_code != 0:
            raise RuntimeError("Cannot read the task protocol")
        protocol = json.loads(result.stdout)
        task = next(t for t in TASKS if t.preset == protocol["preset"])
        case = self.references["tasks"][task.id]
        if len(case["checkpoints"]) > protocol["max_submissions"]:
            raise ValueError("Reference checkpoints exceed the exported submission limit")
        deliveries = []
        for index, record in enumerate([*case["checkpoints"], case["final"]]):
            source = Path(record["path"])
            remote = f"/tmp/robobench-reference-{uuid4().hex}.tar"
            with tempfile.TemporaryDirectory() as temp:
                frozen = Path(temp) / "solution"
                shutil.copytree(source, frozen)
                if (content_hash(inventory(frozen)) != record["sha256"]
                        or content_hash(inventory(source)) != record["sha256"]):
                    raise ValueError("Reference solution changed before upload")
                archive = Path(temp) / "solution.tar"
                with tarfile.open(archive, "w") as tar:
                    tar.add(frozen, arcname=".")
                archive.chmod(0o644)  # Docker copies as root; the agent must be able to read it.
                await environment.upload_file(archive, remote)
            try:
                # find includes dotfiles and leaves the agent-owned destination in place.
                command = ('find /workspace/solution -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + '
                           '&& tar --no-same-owner -xf ' + shlex.quote(remote) + ' -C /workspace/solution')
                if index < len(case["checkpoints"]):
                    command += " && submit " + shlex.quote(f"reference checkpoint {index + 1}")
                result = await environment.exec(command, timeout_sec=120)
                if result.return_code != 0:
                    raise RuntimeError(f"Reference delivery {index + 1} failed: {result.stderr}")
            finally:
                await environment.exec("rm -f -- " + shlex.quote(remote), timeout_sec=30)
            deliveries.append({"submission": f"{index + 1:06d}" if index < len(case["checkpoints"]) else "final",
                               "sha256": record["sha256"]})
        context.metadata = {"task": task.id, "deliveries": deliveries}
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.logs_dir / "reference-deliveries.json", context.metadata)
