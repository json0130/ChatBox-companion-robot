from dataclasses import dataclass, field


@dataclass
class OceanTraits:
    o: float  # Openness          [0, 1]
    c: float  # Conscientiousness [0, 1]
    e: float  # Extraversion      [0, 1]
    agreeableness: float           # [0, 1]
    n: float  # Neuroticism       [0, 1]


@dataclass
class PersonaConfig:
    robot_id: str
    ocean: OceanTraits
    relationship_tier: str = "unknown"


@dataclass
class PADWeights:
    w_user: float = 0.4       # weight of live affect-stream offset
    w_rel: float = 0.3        # weight of relationship-tier offset
    alpha_decay: float = 0.15  # per-turn decay back toward baseline


# ChatBox: warm, outgoing, highly agreeable companion (from client_config.json)
CHATBOX_PERSONA = PersonaConfig(
    robot_id="chatbox",
    ocean=OceanTraits(o=0.6, c=0.7, e=0.8, agreeableness=0.9, n=0.3),
)

# ElleBot: calm, conscientious, moderately extraverted companion
ELLEBOT_PERSONA = PersonaConfig(
    robot_id="ellebot",
    ocean=OceanTraits(o=0.75, c=0.85, e=0.55, agreeableness=0.75, n=0.15),
)
