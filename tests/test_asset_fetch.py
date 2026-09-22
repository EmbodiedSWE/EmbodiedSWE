"""Download integrity, offline use and concurrent first loads; no simulator/network required."""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from robobench.scripts import fetch_assets as assets


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.a = "robobench/suites/example/assets/a.usd"
        self.b = "robobench/backdrops/assets/room/scene_visual.usd"
        self.remote = self.root / "remote"
        self.remote.write_bytes(b"good asset")
        self.meta = {"bytes": 10, "sha256": hashlib.sha256(b"good asset").hexdigest()}
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"revision": "a"*40, "files": {self.a: self.meta, self.b: self.meta}, "bundles": {}}))
        for attr, value in (("ROOT", self.root), ("BUNDLE_DIR", self.root/"cache"), ("MANIFEST", self.manifest)):
            p = patch.object(assets, attr, value)
            p.start()
            self.addCleanup(p.stop)
        assets._VERIFIED.clear()

    def test_selective_fetch_pins_revision_and_reuses_verified_local_files(self):
        with patch("huggingface_hub.hf_hub_download", return_value=str(self.remote)) as download:
            self.assertEqual(assets.fetch(prefixes=[self.a]), 0)
            self.assertEqual(download.call_args.kwargs["revision"], "a"*40)
            self.assertFalse((self.root/self.b).exists())
            with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
                self.assertEqual(assets.fetch(prefixes=[self.a]), 0)
            self.assertEqual(download.call_count, 1)

    def test_same_size_corruption_is_repaired(self):
        p = self.root/self.a
        p.parent.mkdir(parents=True)
        p.write_bytes(b"bad! asset")
        with patch("huggingface_hub.hf_hub_download", return_value=str(self.remote)):
            assets.fetch(prefixes=[self.a])
        self.assertEqual(p.read_bytes(), b"good asset")

    def test_invalid_download_does_not_replace_existing_file(self):
        p = self.root/self.a
        p.parent.mkdir(parents=True)
        p.write_bytes(b"previous version")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            assets._atomic_copy(io.BytesIO(b"truncated"), self.a, self.meta)
        self.assertEqual(p.read_bytes(), b"previous version")
        self.assertEqual(list(p.parent.glob("*.partial")), [])

    def test_offline_missing_group_fails_without_network(self):
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}), patch("huggingface_hub.hf_hub_download") as download:
            with self.assertRaisesRegex(FileNotFoundError, "Offline"):
                assets.fetch(prefixes=[self.a])
            download.assert_not_called()

    def test_concurrent_loads_only_download_once(self):
        def download(*args, **kwargs):
            time.sleep(.05)
            return str(self.remote)
        with patch("huggingface_hub.hf_hub_download", side_effect=download) as mocked:
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: assets.fetch(prefixes=[self.a]), range(2)))
            self.assertEqual(results, [0, 0])
            self.assertEqual(mocked.call_count, 1)

    def test_manifest_cannot_escape_destination(self):
        self.manifest.write_text(json.dumps({"files": {"../escape": self.meta}}))
        with self.assertRaisesRegex(ValueError, "Unsafe asset path"):
            assets.load_manifest()

    def test_manifest_update_preserves_groups_not_materialized_locally(self):
        old_bundle = {"path": "bundles/old.tar", "sha256": "0"*64, "bytes": 10, "files": 1}
        self.manifest.write_text(json.dumps({"revision": "a"*40, "files": {self.a: self.meta},
                                             "bundles": {assets._asset_root(self.a): old_bundle}}))
        p = self.root/self.b
        p.parent.mkdir(parents=True)
        p.write_bytes(b"good asset")
        assets.update_manifest()
        manifest = assets.load_manifest()
        self.assertEqual(manifest["files"][self.a], self.meta)
        self.assertEqual(manifest["bundles"][assets._asset_root(self.a)], old_bundle)
        self.assertEqual(manifest["revision"], "a"*40)
        self.assertIn(self.b, manifest["files"])

    def test_upload_preserves_matching_remote_bundle_without_original_local_tar(self):
        bundle = {"path": "bundles/legacy.tar", "sha256": "0"*64, "bytes": 10240, "files": 1}
        self.manifest.write_text(json.dumps({"files": {self.a: self.meta},
                                            "bundles": {assets._asset_root(self.a): bundle}}))
        remote_files = [SimpleNamespace(path=path, size=meta["bytes"],
                                       lfs=SimpleNamespace(sha256=meta["sha256"]))
                        for path, meta in ((self.a, self.meta), (bundle["path"], bundle))]
        with patch("huggingface_hub.HfApi") as api_cls, patch.object(assets, "_build_bundle") as rebuild:
            api = api_cls.return_value
            api.list_repo_tree.return_value = remote_files
            api.repo_info.return_value.sha = "b"*40
            for staged in (False, True):
                with self.subTest(staged=staged):
                    if staged:
                        assets.BUNDLE_DIR.mkdir(parents=True)
                        (assets.BUNDLE_DIR/"legacy.tar").write_bytes(b"different tar metadata")
                    self.assertEqual(assets.upload(100), 0)
                    rebuild.assert_not_called()
                    api.create_commit.assert_not_called()
                    self.assertEqual(assets.load_manifest()["revision"], "b"*40)


if __name__ == "__main__":
    unittest.main()
