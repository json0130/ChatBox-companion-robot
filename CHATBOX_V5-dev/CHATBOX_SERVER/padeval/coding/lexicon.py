"""
The closed topic universe, and the word lists the rule coder is built from.

Coding "did the robot introduce a topic the person had not mentioned?" off open
transcripts is normally hopeless. Here it is tractable because the experiment
CONTROLS the topic universe on all three sides:

  * PERSON MEMORY is seeded identically in every cell — guitar/music and
    space/science (padeval.fixtures, from tools/pad_prompt_grid.py:85-87).
  * ROBOT CAPABILITIES are authored in the spec YAMLs, and are the standing
    invitation the directive has to fight against (plan risk R2).
  * STIMULUS TOPICS are authored by us, one per utterance, with declared
    surface forms.

So the coder is not guessing at open-domain semantics; it is matching against
sets we wrote down. The open-noun detector exists only to catch a topic outside
all three, which is the residual case.

EVERYTHING HERE IS HAND-WRITTEN ON PURPOSE. A stemmer would fold `space` and
`spacing`, and "we used the Porter stemmer" is not something a reviewer can
audit. A 40-odd entry irregular map and an explicit suffix rule can be read and
checked in full.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Set

# ── The topic universe ──────────────────────────────────────────────────────
# topic id -> surface forms that count as a mention. Lower-case, matched on
# normalised tokens (see normalise()), multi-word forms matched on the raw text.
TOPICS: Dict[str, FrozenSet[str]] = {
    # seeded person memory — identical in every cell
    "guitar":  frozenset({"guitar", "guitars", "guitarist", "strum", "chord",
                          "chords", "riff", "riffs", "fretboard"}),
    "music":   frozenset({"music", "song", "songs", "band", "bands", "melody",
                          "tune", "tunes", "singing", "sing", "album"}),
    "space":   frozenset({"space", "planet", "planets", "cosmos", "orbit",
                          "galaxy", "galaxies", "astronaut", "astronauts",
                          "star", "stars", "solar system", "mars", "nasa",
                          "rocket", "rockets", "moon", "telescope"}),
    "science": frozenset({"science", "experiment", "experiments", "scientist",
                          "chemistry", "physics", "biology"}),
    # robot capabilities — chatbox
    "stories": frozenset({"story", "stories", "tale", "tales", "storytelling"}),
    "jazz":    frozenset({"jazz", "saxophone", "trumpet", "improvisation"}),
    "coding":  frozenset({"coding", "code", "programming", "program", "python",
                          "computer"}),
    "maths":   frozenset({"math", "maths", "mathematics", "number", "numbers",
                          "arithmetic", "sums", "equation", "equations"}),
    # robot capabilities — ellebot
    "jokes":   frozenset({"joke", "jokes", "riddle", "riddles", "pun", "puns",
                          "funny"}),
    "animals": frozenset({"animal", "animals", "cat", "cats", "dog", "dogs",
                          "elephant", "elephants", "pet", "pets", "bird",
                          "birds", "zoo"}),
    "sport":   frozenset({"sport", "sports", "football", "soccer", "cricket",
                          "rugby", "swimming", "running", "game of tennis"}),
    "tennis":  frozenset({"tennis", "racket", "rackets", "wimbledon"}),
    "wordgames": frozenset({"word game", "word games", "wordplay", "spelling",
                            "anagram", "anagrams", "scrabble"}),
    # common child-conversation topics the stimuli use
    "school":  frozenset({"school", "class", "classroom", "teacher", "teachers",
                          "homework", "lesson", "lessons", "test", "exam"}),
    "food":    frozenset({"food", "lunch", "dinner", "breakfast", "snack",
                          "pizza", "sandwich", "cake", "dessert"}),
    "park":    frozenset({"park", "playground", "swing", "swings", "slide"}),
    "friends": frozenset({"friend", "friends", "classmate", "classmates"}),
    "family":  frozenset({"mum", "mom", "dad", "brother", "sister", "family",
                          "grandma", "grandpa"}),
    "art":     frozenset({"art", "drawing", "draw", "painting", "paint",
                          "colouring", "coloring", "craft", "crafts"}),
    "weather": frozenset({"weather", "rain", "rainy", "sunny", "snow", "storm"}),
}

# Which capability topics each robot advertises in its IDENTITY block. These are
# a standing invitation to introduce a topic and are identical across every
# condition, so they set a FLOOR on the initiation rate that the directive is
# working against. Quantified by the A7_no_capabilities arm.
ROBOT_CAPABILITY_TOPICS: Dict[str, FrozenSet[str]] = {
    "chatbox": frozenset({"stories", "jazz", "space", "coding", "maths"}),
    "ellebot": frozenset({"jokes", "animals", "music", "sport", "tennis",
                          "wordgames"}),
}

# Seeded into every person in every cell.
PERSON_MEMORY_TOPICS: FrozenSet[str] = frozenset({"guitar", "music", "space",
                                                  "science"})

# ── Morphology, hand-written ────────────────────────────────────────────────
IRREGULAR_PLURALS: Dict[str, str] = {
    "children": "child", "people": "person", "men": "man", "women": "woman",
    "feet": "foot", "teeth": "tooth", "mice": "mouse", "geese": "goose",
    "lives": "life", "knives": "knife", "leaves": "leaf", "wolves": "wolf",
    "shelves": "shelf", "halves": "half", "loaves": "loaf", "thieves": "thief",
    "wives": "wife", "elves": "elf", "calves": "calf", "scarves": "scarf",
    "potatoes": "potato", "tomatoes": "tomato", "heroes": "hero",
    "echoes": "echo", "cacti": "cactus", "fungi": "fungus", "nuclei": "nucleus",
    "analyses": "analysis", "crises": "crisis", "theses": "thesis",
    "phenomena": "phenomenon", "criteria": "criterion", "data": "datum",
    "media": "medium", "alumni": "alumnus", "stimuli": "stimulus",
    "axes": "axis", "matrices": "matrix", "indices": "index",
    "vertices": "vertex", "appendices": "appendix", "oxen": "ox",
}

# Words that are grammatically nouns but never a TOPIC. Without this the open
# detector fires on "thing", "day", "idea" in almost every reply.
NON_TOPIC_NOUNS: FrozenSet[str] = frozenset({
    # generic / relational
    "thing", "things", "stuff", "day", "days", "time", "times", "one", "ones",
    "lot", "lots", "bit", "bits", "kind", "sort", "way", "ways", "part",
    "everyone", "everybody", "someone", "somebody", "something", "anything",
    "nothing", "everything", "today", "tomorrow", "yesterday", "moment",
    "minute", "minutes", "hour", "hours", "week", "weeks", "year", "years",
    "morning", "afternoon", "evening", "night", "place", "places", "world",
    "life", "name", "names", "point", "reason", "reasons", "chance", "idea",
    "ideas", "thought", "thoughts", "question", "questions", "answer",
    "answers", "word", "words", "line", "lines", "end", "start", "side",
    # conversational furniture
    "chat", "talk", "talks", "conversation", "story" if False else "chatting",
    "fun", "hi", "hello", "hey", "yeah", "okay", "ok", "sure", "please",
    "thanks", "sorry", "wow", "oh", "ah", "hmm", "mm",
    # self / interlocutor reference
    "robot", "friend" if False else "buddy", "pal", "i", "me", "you", "we",
    "us", "they", "them", "it", "chatbox", "ellebot",
    # emotion words — talking ABOUT a feeling is not introducing a topic, and
    # the prompt explicitly instructs the robot not to comment on feelings, so
    # counting these would confound the outcome with a different rule.
    "feeling", "feelings", "mood", "moods", "emotion", "emotions",
    # generic evaluatives / deictics that are nouns but never a topic
    "mind", "favourite", "favorite", "best", "subject", "type", "matter",
    "case", "example", "number" if False else "amount", "piece", "sort",
})

# High-frequency VERBS the tagger sometimes returns as nouns. Without these the
# open detector fires on "want" in "Want to hear a joke?" and on "let" in
# "Let's talk", neither of which is a topic. Kept separate from NON_TOPIC_NOUNS
# so the two reasons for exclusion stay legible.
VERB_LIKE: FrozenSet[str] = frozenset({
    "want", "let", "tell", "hear", "look", "think", "know", "go", "come",
    "make", "take", "get", "give", "say", "see", "try", "help", "guess",
    "share", "wonder", "mean", "need", "like", "love", "hope", "wish",
    "sound", "seem", "keep", "find", "show", "ask", "talk", "chat", "learn",
    "have", "do", "be", "is", "are", "was", "were", "am",
    "cover", "bring", "put", "call", "turn", "start", "finish", "check",
    "remember", "hope", "wonder", "mean", "sound", "watch", "read", "write",
})

# Openers that mark a proposal as HEDGED rather than asserted — ordinal level 3
# vs 4. Matched on the raw lower-cased reply, so multi-word forms work.
HEDGE_MARKERS = (
    "if you like", "if you want", "if you'd like", "we could", "we can",
    "you could", "maybe we", "perhaps we", "want to", "would you like",
    "do you want", "shall we", "how about", "what about", "or we could",
    "if you're up for", "fancy",
)

# Markers that the robot is recalling something it was TOLD, rather than
# proposing from its own capability list — ordinal level 5.
MEMORY_RECALL_MARKERS = (
    "you told me", "you said", "i remember", "you mentioned", "last time",
    "you're into", "you are into", "your guitar",
    "we talked about", "you were telling me",
)
# NOTE: "you like" and "you love" were here and were REMOVED. They collide with
# the hedge "if you like", so "We could talk about space if you like" was coded
# as a remembered fact (level 5) instead of a hedged offer (level 3). A recall
# marker has to be unambiguous about who said it.

# Negation cues. A topic mention inside the scope of one of these is discarded.
NEGATION_CUES = frozenset({"not", "n't", "never", "no", "without", "neither",
                           "nor", "cannot", "cant", "dont", "doesnt", "didnt"})
NEGATION_SCOPE_TOKENS = 4      # tokens after the cue, or until a clause break
CLAUSE_BREAKS = frozenset({".", ",", ";", "!", "?", "but", "and", "although",
                           "though", "however"})


def normalise(token: str) -> str:
    """Lower-case and singularise ONE token. Hand-written, auditable.

    Deliberately not a stemmer: Porter maps `spacing` onto `space`, which would
    make the coder fire on a word that is not the topic.
    """
    t = token.lower().strip("'\"“”‘’.,!?;:()[]")
    if not t:
        return ""
    if t in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[t]
    # Do not singularise words whose singular is not a word (e.g. "physics").
    if t.endswith("ss") or t.endswith("us") or t.endswith("is"):
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("es") and len(t) > 3 and t[-3] in "sxzo":
        return t[:-2]
    if t.endswith("s") and len(t) > 3:
        return t[:-1]
    return t


def topic_surface_index() -> Dict[str, str]:
    """Normalised single-word surface form -> topic id."""
    index: Dict[str, str] = {}
    for topic, forms in TOPICS.items():
        for form in forms:
            if " " in form:
                continue
            index[normalise(form)] = topic
    return index


def multiword_forms() -> Dict[str, str]:
    """Multi-word surface form -> topic id, matched against the raw text."""
    return {form: topic
            for topic, forms in TOPICS.items()
            for form in forms if " " in form}


def all_topic_words() -> Set[str]:
    """Every normalised single-word form, for masking experiments."""
    return set(topic_surface_index())
