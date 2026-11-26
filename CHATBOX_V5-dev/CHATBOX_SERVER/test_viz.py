"""test_viz.py — the standalone KG viz serves the graph file, culture override and persona, and stubs the chat panel."""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from modules.graph_relationship.schema import Embodiment, PersonNode, RobotNode
from modules.graph_relationship.store import InMemoryGraphStore
from modules.graph_relationship.viz.server import GraphState, make_handler
from modules.role_seed import seed_role_library


def test_viz_serves_graph_culture_persona_and_a_disabled_chat(tmp_path):
    kg = str(tmp_path / "kg.json")
    s = InMemoryGraphStore()
    s.upsert_node(RobotNode(id="chatbox", name="ChatBox", embodiment=Embodiment.CAT))
    s.upsert_node(PersonNode(id="jay", display_name="Jay"))
    seed_role_library(s)
    s.save(kg)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(GraphState(kg)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    get = lambda p: json.load(urllib.request.urlopen(base + p))
    post = lambda p, b: json.load(urllib.request.urlopen(
        urllib.request.Request(base + p, json.dumps(b).encode(), {"Content-Type": "application/json"})))
    try:
        assert {"Person", "Robot"} <= {n["type"] for n in get("/graph.json")["nodes"]}
        assert post("/culture", {"active_culture": "maori"})["active"] == "maori" and get("/culture")["active"] == "maori"
        assert post("/persona", {"role": "role:tutor", "ocean": dict(O=0, C=.8, E=-.9, A=.6, N=.1)})["ok"]
        assert (tmp_path / "persona_assignment.json").exists()
        assert get("/chat")["available"] is False
    finally:
        srv.shutdown()
