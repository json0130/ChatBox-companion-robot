"""What the Navel graph starts with: the robot, what it can do, its gesture list and how to reach it.

Facts come from the robot-robot-interaction repo (branch feat/navel-demo-client, commit dad21e4):
navel-client/client_config.json (tags, ports, modules), navel-client/client.py (how it connects) and
lab-demo-server/data/snapshot (navel_001's declared topics). The ChatBox graph shares none of it.

    robot --has_capability--> capability  (what Navel can do)  --about--> topics it declares
    robot --has_capability--> one node per connection fact

Like the ChatBox, the gestures are NOT nodes: the list below is what a robot client / the viz
reads, and the capability node only mentions that Navel gestures.
"""
from __future__ import annotations

import os
from typing import Dict

from modules.graph_relationship.schema import (
    CapabilityNode, HasCapabilityEdge, Provenance)
from modules.graph_relationship.seed import seed_from_spec

SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_relationship", "specs", "navel_spec.json")
SOURCE = "spec:navel_profile"

# Navel's own gesture tags (client_config.json allowed_tags / tag_descriptions): it performs these, not the ChatBox's
ACTIONS: Dict[str, str] = {
    "DEFAULT": "neutral pose, when no expressive action fits",
    "POINT": "point towards a person, object, display or direction",
    "EMPHASISE": "hand or arm gesture to reinforce an important statement",
    "AGREE": "affirmative gesture such as a nod, to confirm or agree",
    "LISTEN": "attentive pose that invites the person to keep talking",
    "UNCERTAIN": "hesitant gesture when unsure or unable to give a definite answer",
    "DENY": "negative gesture to refuse, disagree or correct",
    "GREET": "raise one hand in a friendly greeting: welcome, hello, goodbye",
    "HAPPY": "cheerful or celebratory gesture for praise, success or good news",
    "SURPRISE": "surprised reaction to unexpected information",
    "SAD": "subdued, sympathetic gesture for sadness or difficulty",
}

# How to reach it. NOTE the direction: Navel does NOT dial out; it runs a server and waits to be dialled.
CONNECTION: Dict[str, str] = {
    "websocket": "Navel runs a WebSocket server on port 8765 and waits; the server dials out to it (navel-client/client.py)",
    "register": "on start it POSTs {server_url}/robots/register (its IP and ws_port) as client_id navel_001, robot_name Navel",
    "server": "server_url in navel-client/client_config.json: the lab demo / central server, port 5000 (run one of them)",
    "planner": "Method B planner service, POST /method_b/plan on port 8001 (navel-server/tools/method_b_plan_service.py)",
    "messages": "JSON messages with an event field: persona_update, demo_step, chat_response, speech_response, tts_stop; it sends ack, speech, tts_state",
}


def _capability(store, robot_id: str, cap_id: str, text: str) -> None:
    store.upsert_node(CapabilityNode(id=cap_id, items=[text]))
    store.upsert_edge(HasCapabilityEdge(source_id=robot_id, target_id=cap_id,
                                        provenance=Provenance(source=SOURCE, confidence=1.0)))


def seed_navel(store) -> None:
    """Idempotent: ids are fixed, so running it again just refreshes the same nodes."""
    seed_from_spec(store, SPEC)                      # robot, abilities, declared topics
    for key, text in CONNECTION.items():
        _capability(store, "navel", f"navel:connection:{key}", f"connect: {text}")
