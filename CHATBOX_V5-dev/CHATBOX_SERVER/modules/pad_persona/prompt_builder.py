def build_system_prompt(
    persona_name: str,
    descriptors: dict[str, str],
    relationship_tier: str,
    memory_context: str = "",
) -> str:
    """Assemble the LLM system prompt from PAD descriptors and relationship context.

    Args:
        persona_name:      Display name of the robot (e.g. "ChatBox", "ElleBot").
        descriptors:       Output of PADEngine.to_language_descriptors():
                           {"pleasure": str, "arousal": str, "dominance": str}
        relationship_tier: One of "close", "family", "known", "visitor", "unknown".
        memory_context:    Retrieved RAG snippets about this user; empty string if none.

    Returns:
        Formatted system prompt string ready to pass to the LLM.
    """
    p_word = descriptors.get("pleasure",  "neutral")
    a_word = descriptors.get("arousal",   "moderate")
    d_word = descriptors.get("dominance", "neutral")

    tier_notes = {
        "close":   "You know this person well and feel very comfortable with them.",
        "family":  "This person is like family to you; speak with warmth and ease.",
        "known":   "You recognise this person from previous interactions.",
        "visitor": "This person is a new face; be welcoming but a little more formal.",
        "unknown": "You are meeting this person for the first time; be friendly and open.",
    }
    relationship_note = tier_notes.get(relationship_tier, tier_notes["unknown"])

    # --- memory injection point ---
    # Insert retrieved RAG context here so the model can reference past interactions.
    # Replace the empty-string guard with actual retrieved snippets from RagModule.search().
    memory_section = ""
    if memory_context.strip():
        memory_section = f"\n\nRelevant things you remember about this person:\n{memory_context.strip()}"

    prompt = (
        f"You are {persona_name}, a gentle, playful, and caring emotional support robot "
        f"for children. You always respond concisely (1-2 sentences), in plain casual language, "
        f"and you MUST begin every reply with an action tag in square brackets, e.g. [GREETING].\n\n"
        f"Right now respond in a {p_word}, {a_word}, {d_word} manner.\n\n"
        f"{relationship_note}"
        f"{memory_section}"
    )

    return prompt
