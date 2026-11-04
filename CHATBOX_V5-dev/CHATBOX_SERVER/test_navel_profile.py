import json

from modules.graph_relationship.viz.server import seed_robot_graph
from modules.navel_profile import ACTIONS, CONNECTION


def _nodes(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["nodes"]


def test_navel_graph_has_the_robot_its_gestures_and_how_to_connect(tmp_path):
    p = str(tmp_path / "kg_navel.json")
    seed_robot_graph(p, "navel")
    nodes = _nodes(p)
    ids = {n["id"] for n in nodes}
    assert "navel" in ids and "navel:capability" in ids
    assert len(ACTIONS) == 11 and not any(i.startswith("navel:action:") for i in ids)   # gestures are not nodes
    assert {f"navel:connection:{k}" for k in CONNECTION} <= ids
    assert sum(n["node_type"] == "topic" for n in nodes) == 7            # its declared topics
    assert not any(n["node_type"] == "person" for n in nodes)            # nothing of the ChatBox's Jay graph
    assert "8765" in next(n for n in nodes if n["id"] == "navel:connection:websocket")["items"][0]


def test_seeding_twice_changes_nothing_and_tops_up_an_old_file(tmp_path):
    p = str(tmp_path / "kg_navel.json")
    seed_robot_graph(p, "navel"); n = len(_nodes(p))
    seed_robot_graph(p, "navel")
    assert len(_nodes(p)) == n
    from modules.graph_relationship.schema import Embodiment, RobotNode
    from modules.graph_relationship.store import InMemoryGraphStore
    old = InMemoryGraphStore(); old.upsert_node(RobotNode(id="navel", name="Navel", embodiment=Embodiment.CAT))
    q = str(tmp_path / "old.json"); old.save(q)                           # a Navel file from before the profile existed
    seed_robot_graph(q, "navel")
    assert "navel:capability" in {x["id"] for x in _nodes(q)}
