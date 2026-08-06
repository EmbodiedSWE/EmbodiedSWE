"""Offline sanity test for agent-chosen checkpointing in cosigen_loop's tree.

No Isaac needed: builds an AssemblyApi shell via object.__new__ and stubs the
state-capture/persistence methods, then drives begin_turn/checkpoint/end_turn/
drain_turn_payload/list_checkpoints/goto exactly as run_policy + the server do.

Run: /home/tiger/cap-x/.venv/bin/python scripts/tests/ckpt_tree_offline_test.py
"""
import sys, types

sys.path.insert(0, "/home/tiger/cap-x/CoSiGen/eval")
sys.path.insert(0, "/home/tiger/cap-x/CoSiGen")

# cosigen_loop imports isaac-lab bits at module level? It imports torch, numpy; check lazily.
import cosigen_loop as CL


class DummyEnv:
    def get_states(self, ids):
        return {"dummy": True}

    def set_states(self, states):
        pass


def make_api():
    api = object.__new__(CL.AssemblyApi)
    api._env = DummyEnv()
    api.device = "cpu"
    api._ckpt_nodes = {}
    api._ckpt_seq = 0
    api._ckpt_current = None
    api._turn_no = 0
    api._steps_used = 0
    api._rec_total = 0
    api._turn_frame_start = 0
    api._turn_images = []
    # stub captures/persistence
    api._api_state_env0 = lambda: {}
    api.success = lambda: False
    api.scene_summary = lambda: "scene"
    api.obs_snapshot = lambda: {}
    api._node_thumb = lambda: ""
    api._persist_tree = lambda **kw: None
    api._restore_to = getattr(api, "_restore_to", None)
    return api


def main():
    api = make_api()

    # ---- turn 1: agent does NOT checkpoint ----
    api.begin_turn("print('probe')")
    nid = api.end_turn("print('probe')", 0, stdout="probe output")
    assert nid is None, f"no-checkpoint turn must return None, got {nid}"
    payload = api.drain_turn_payload()
    tree = payload["tree"]
    assert tree["new_node"] is None, tree
    assert "diff_context" not in payload, "diff_context must be absent without a new node"
    # only the lazy root exists
    assert len(api._ckpt_nodes) == 1 and api._ckpt_current == "n1"
    print("PASS turn without checkpoint: only root n1, new_node=None, no diff_context")

    # ---- turn 2: agent checkpoints explicitly ----
    api.begin_turn("move_to('left', [0,0,0])\ncheckpoint('grasped leg_0 firmly')")
    cid = api.checkpoint("grasped leg_0 firmly")
    out = api.end_turn("...", 0, stdout="full turn stdout here, kept in full " * 5)
    assert out == cid == "n2"
    node = api._ckpt_nodes[cid]
    assert node["parent"] == "n1", node["parent"]
    assert node["label"] == "grasped leg_0 firmly"
    assert node["code"].startswith("move_to"), "checkpoint must capture the turn's code"
    assert node["stdout_tail"].startswith("full turn stdout"), "end_turn back-fills stdout"
    payload = api.drain_turn_payload()
    assert payload["tree"]["new_node"] == "n2"
    assert "diff_context" in payload, "diff_context needed for annotator on checkpoint turns"
    print("PASS explicit checkpoint: n2 parent=n1, code+stdout captured, annotator ctx present")

    # ---- checkpoint log: captured up to the moment of checkpoint(), readable on demand ----
    import io
    api.begin_turn("...")
    buf = io.StringIO()
    api._turn_stdout_buf = buf          # as run_policy does
    buf.write("leg_0 at (0.3, 0.1); grasp verified\n")
    cid_log = api.checkpoint("log test")
    buf.write("LATER OUTPUT that must NOT appear in the checkpoint log\n")
    api._turn_stdout_buf = None
    api.end_turn("...", 0, stdout=buf.getvalue())
    log = api.get_checkpoint_log(cid_log)
    assert "grasp verified" in log and "LATER OUTPUT" not in log, log
    assert "(no log recorded" in api.get_checkpoint_log("n1")
    print("PASS get_checkpoint_log: log snapshot at checkpoint time, later output excluded")

    # ---- turn 4: two checkpoints in one turn -> payload points at the last ----
    api.begin_turn("...")
    c1 = api.checkpoint("mid-turn state")
    c2 = api.checkpoint("end-turn state")
    api.end_turn("...", 0, stdout="x")
    assert api._ckpt_nodes[c2]["parent"] == c1
    assert api.drain_turn_payload()["tree"]["new_node"] == c2
    print(f"PASS multi-checkpoint turn: {c1}<-{c2} chained, payload targets {c2}")

    # ---- rendering: parent labels + render_checkpoint footer hint ----
    listing = api.list_checkpoints()
    print("\n--- list_checkpoints ---\n" + listing + "\n---")
    assert "(root)" in listing and f"(parent {c1})" in listing and "(parent n1)" in listing
    assert "render_checkpoint" in listing, "footer hint must nudge visual inspection"
    print("PASS listing shows parent of every node + render hint")

    # ---- goto returns parent info; requires stubbing restore ----
    api.n = 1
    api._restore_api_state = lambda st: None
    api._rl_log = []
    hdr = api.goto("n2")
    print("\n--- goto('n2') ---\n" + hdr.splitlines()[0])
    assert "parent n1" in hdr.splitlines()[0]
    print("PASS goto header shows parent")

    print("\nALL PASS")


if __name__ == "__main__":
    main()
