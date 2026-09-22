"""Room defaults belong to suite configs and registry discovery stays offline."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from robobench.core import ENVS, EnvCfg
from robobench.core.rooms import prepare_room


class RoomConfigTests(unittest.TestCase):
    def test_every_builtin_registered_config_declares_its_room_in_its_suite(self):
        import robobench
        with patch("robobench.scripts.fetch_assets.fetch", side_effect=AssertionError("registry must stay offline")):
            robobench.discover()
            configs = [ENVS.get(name)() for name in ENVS.list()]
            self.assertTrue(configs)
            for cfg in configs:
                self.assertTrue(callable(cfg.room), cfg.describe())
                self.assertTrue(cfg.room.__module__.endswith(".configs.envs"), cfg.describe())

    def test_explicit_disable_needs_no_assets_or_simulator(self):
        with patch("robobench.core.rooms.ensure_assets") as fetch:
            self.assertEqual(prepare_room(EnvCfg(scene="pc_gpu", room=None, env_spacing=3), None, None), (None, 3))
            fetch.assert_not_called()

    def test_factory_copies_settings_and_uses_scene_overrides(self):
        from robobench.suites.assembly.configs.envs import _pc_room
        cfg = EnvCfg(scene="pc_gpu", room=_pc_room)
        a = _pc_room(cfg, SimpleNamespace(cfg=SimpleNamespace(surface_z=0.0)), None)
        b = _pc_room(cfg, SimpleNamespace(cfg=SimpleNamespace(surface_z=0.5)), None)
        self.assertAlmostEqual(b["room"]["floor_world_z"] - a["room"]["floor_world_z"], 0.5)
        b["room"]["anchor"][0] = 999
        self.assertNotEqual(a["room"]["anchor"][0], 999)

    def test_room_id_cannot_escape_asset_directory(self):
        cfg = EnvCfg(scene="example", room={"room": {"id": "../escape"}})
        with patch("robobench.core.rooms.ensure_assets") as fetch:
            with self.assertRaisesRegex(ValueError, "Invalid room asset id"):
                prepare_room(cfg, None, None)
            fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
