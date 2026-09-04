"""CPU-only checks for gsworld (no Isaac, no GPU): PLY round trip, similarity/rigid transforms,
model loading and per-link re-posing, scene-config schema.

    pytest tests/test_gsworld.py        # or, without pytest:
    python tests/test_gsworld.py
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gsworld.splat import GaussianSet, decompose_similarity, quat_from_matrix, quat_mul, quat_rotate


def _rand_set(n: int = 50, seed: int = 0) -> GaussianSet:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return GaussianSet(
        means=rng.normal(size=(n, 3)).astype(np.float32), quats=q.astype(np.float32),
        log_scales=rng.normal(size=(n, 3)).astype(np.float32), logit_opacities=rng.normal(size=n).astype(np.float32),
        sh0=rng.normal(size=(n, 3)).astype(np.float32), shN=rng.normal(size=(n, 15, 3)).astype(np.float32))


def _write_ply(path, g: GaussianSet, two_scales: bool = False) -> None:
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(45)] \
        + ["opacity", "scale_0", "scale_1"] + ([] if two_scales else ["scale_2"]) + ["rot_0", "rot_1", "rot_2", "rot_3"]
    rest = g.shN.transpose(0, 2, 1).reshape(g.n, 45)  # inria layout: (N, 3, 15) flattened
    cols = [g.means, np.zeros((g.n, 3), np.float32), g.sh0, rest, g.logit_opacities[:, None],
            g.log_scales[:, :2] if two_scales else g.log_scales, g.quats]
    arr = np.concatenate(cols, axis=1).astype("<f4")
    hdr = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % g.n + "".join(f"property float {p}\n" for p in props) + "end_header\n"
    with open(path, "wb") as f:
        f.write(hdr.encode())
        f.write(arr.tobytes())


def test_ply_round_trip(tmp_path):
    g = _rand_set()
    _write_ply(tmp_path / "a.ply", g)
    r = GaussianSet.from_ply(tmp_path / "a.ply")
    for k in ("means", "quats", "log_scales", "logit_opacities", "sh0", "shN"):
        np.testing.assert_allclose(getattr(r, k), getattr(g, k), atol=1e-6)
    assert r.sh_degree == 3


def test_to_ply_round_trip_with_semantics(tmp_path):
    g = _rand_set()
    g.to_ply(tmp_path / "w.ply", semantics=np.arange(g.n))
    r = GaussianSet.from_ply(tmp_path / "w.ply")
    for k in ("means", "quats", "log_scales", "logit_opacities", "sh0", "shN"):
        np.testing.assert_allclose(getattr(r, k), getattr(g, k), atol=1e-6)
    raw = (tmp_path / "w.ply").read_bytes()
    assert b"property float semantics" in raw.split(b"end_header")[0]


def test_2dgs_ply_gets_thin_third_axis(tmp_path):
    g = _rand_set()
    _write_ply(tmp_path / "s.ply", g, two_scales=True)
    r = GaussianSet.from_ply(tmp_path / "s.ply")
    assert r.log_scales.shape == (g.n, 3)
    assert np.all(r.log_scales[:, 2] < r.log_scales[:, :2].min(axis=1))


def test_quat_helpers_match_scipy():
    rng = np.random.default_rng(1)
    R1, R2 = Rotation.random(random_state=rng), Rotation.random(random_state=rng)
    q1, q2 = quat_from_matrix(R1.as_matrix()), quat_from_matrix(R2.as_matrix())
    np.testing.assert_allclose(quat_from_matrix((R1 * R2).as_matrix()) * np.sign(quat_mul(q1, q2)[0]),
                               quat_mul(q1, q2) * np.sign(quat_mul(q1, q2)[0]), atol=1e-9)
    v = rng.normal(size=(7, 3))
    np.testing.assert_allclose(quat_rotate(q1, v), R1.apply(v), atol=1e-9)


def test_similarity_transform_moves_means_scales_and_orientations():
    g = _rand_set()
    s, R, t = 1.7, Rotation.from_euler("xyz", [10, -20, 35], degrees=True).as_matrix(), np.array([0.3, -0.2, 0.9])
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    out = g.transform(T)
    np.testing.assert_allclose(out.means, (s * R @ g.means.astype(np.float64).T).T + t, atol=1e-5)
    np.testing.assert_allclose(out.log_scales, g.log_scales + np.log(s), atol=1e-6)
    # orientation: rotating the gaussian's own frame by R
    qR = quat_from_matrix(R)
    np.testing.assert_allclose(out.quats, quat_mul(np.broadcast_to(qR, g.quats.shape), g.quats), atol=1e-5)
    s2, R2, t2 = decompose_similarity(T)
    assert abs(s2 - s) < 1e-9 and np.allclose(R2, R) and np.allclose(t2, t)


def test_posed_equals_rigid_transform():
    g = _rand_set()
    R = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    t = np.array([1.0, 2.0, 3.0])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    a, b = g.posed(t, quat_from_matrix(R)), g.transform(T)
    np.testing.assert_allclose(a.means, b.means, atol=1e-5)
    np.testing.assert_allclose(a.quats, b.quats, atol=1e-5)
    np.testing.assert_allclose(a.log_scales, b.log_scales, atol=1e-6)


def test_robot_splat_from_labeled_ply(tmp_path):
    """A labeled model posed at a scan pose loads to link-local frames and re-poses correctly."""
    from gsworld.model import SplatModel, save_scan_poses

    g = _rand_set(20)
    labels = np.array([0] * 10 + [1] * 10)
    # scan pose: link l0 at (0,1,0.5) identity, link l1 at (1,0,0) rotated 90deg about z; base unused
    Rz = Rotation.from_euler("z", 90, degrees=True)
    scan_pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.5]], dtype=np.float32)   # bodies: base, l1, l0
    scan_quat = np.array([[1, 0, 0, 0], quat_from_matrix(Rz.as_matrix()), [1, 0, 0, 0]], dtype=np.float32)
    world = GaussianSet.concat([g.select(labels == 0).posed(scan_pos[2], scan_quat[2]),
                                g.select(labels == 1).posed(scan_pos[1], scan_quat[1])])
    world.to_ply(tmp_path / "robot.ply", semantics=labels)
    save_scan_poses(tmp_path / "robot_poses.json", ["base", "l1", "l0"], scan_pos, scan_quat, ["l0", "l1"])
    rs = SplatModel(tmp_path / "robot.ply", tmp_path / "robot_poses.json", device="cpu")
    assert rs.link_names == ["l0", "l1"] and rs.local.n == 20
    np.testing.assert_allclose(rs.local.means, g.means, atol=1e-5)  # back in link-local frames
    # re-pose at a new configuration: l0 moved to (2,2,2)
    out = rs.posed_numpy(np.array([[0, 0, 0], [1, 0, 0], [2, 2, 2]], np.float32), scan_quat, ["base", "l1", "l0"])
    np.testing.assert_allclose(out.means[:10], g.means[:10] + 2.0, atol=1e-5)
    np.testing.assert_allclose(out.means[10:], Rz.apply(g.means[10:]) + [1, 0, 0], atol=1e-5)
    with pytest.raises(KeyError):
        rs.bind_bodies(["base", "l1"])  # missing l0 (strict)
    assert rs.bind_bodies(["base", "l1"], strict=False) == ["l1"] and rs.local.n == 10

def test_scene_config_schema(tmp_path):
    """A scene JSON declares the env, the models and the camera; paths resolve relative to it."""
    import json

    from gsworld.config import SplatSceneCfg
    from gsworld.model import save_scan_poses

    g = _rand_set(10)
    g.to_ply(tmp_path / "robot.ply", semantics=np.zeros(10))
    save_scan_poses(tmp_path / "robot_poses.json", ["l0"], np.zeros((1, 3)), [[1, 0, 0, 0]], ["l0"])
    _rand_set(7).to_ply(tmp_path / "table.ply")
    _rand_set(5).to_ply(tmp_path / "cup.ply")  # an unlabeled object: one body
    save_scan_poses(tmp_path / "cup_poses.json", ["cup"], [[0.5, 0.0, 0.1]], [[1, 0, 0, 0]], ["cup"])
    (tmp_path / "scene.json").write_text(json.dumps({
        "env": {"scene": "table", "robot": "franka_robotiq", "control_mode": "joint",
                "scene_cfg": {"layout": "gsworld_table", "pedestal": False}},
        "robot": {"ply": "robot.ply", "poses": "robot_poses.json"},
        "static": [{"path": "table.ply", "crop": [[-9, -9, -9], [9, 9, 9]]}],
        "objects": [{"name": "cup", "ply": "cup.ply", "poses": "cup_poses.json"}],
        "camera": {"eye": [1, 2, 3], "target": [0, 0, 0], "size": [640, 480]},
    }))
    c = SplatSceneCfg.load(tmp_path / "scene.json")
    assert c.env.scene == "table" and c.env.robot == "franka_robotiq" and c.env.scene_cfg["pedestal"] is False
    assert c.camera.size == (640, 480) and c.camera.eye == (1, 2, 3)
    assert [s.n for s in c.load_static()] == [7]
    cup = c.load_objects(device="cpu")["cup"]
    assert cup.link_names == ["cup"] and cup.local.n == 5  # unlabeled PLY -> single body
    assert c.robot_splat("cpu").link_names == ["l0"]
    # the env block instantiates registered cfg dataclasses from plain dicts
    import robobench
    from robobench.core import SCENES
    from gsworld.config import _make_cfg

    robobench.discover()
    scfg = _make_cfg(SCENES.get("table"), c.env.scene_cfg)
    assert scfg.pedestal is False and scfg.layout == "gsworld_table"


if __name__ == "__main__":  # `python tests/test_gsworld.py` — no pytest needed
    import sys
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        for name, fn in list(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn(Path(d)) if fn.__code__.co_argcount else fn()
                print(f"PASS {name}")
    sys.exit(0)
