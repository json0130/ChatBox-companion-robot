from openai import OpenAI
import re

class OllamaClient:
    def __init__(self, model_name="qwen:4b", host="127.0.0.1", port=11434):
        self.model_name = model_name
        self.available = False
        
        # --- NEW: Short-term memory bank ---
        self.history = []
        self.max_history_turns = 7 # Remembers the last 7 back-and-forths (14 messages total)
        
        self.client = OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key="ollama" 
        )

    def setup_client(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            print("🔍 Checking connection to local Ollama via OpenAI API...")
            self.client.models.list()
            self.available = True
            print(f"✅ Connected to Ollama! Using model: {self.model_name}")
            return True
        except Exception as e:
            print(f"❌ Could not connect to Ollama: {e}")
        return False

    def is_available(self) -> bool:
        return self.available
    
    # When to reach for each tag. Keys are bracket-less + upper-case; anything
    # not listed here still works, it just gets a generic hint in the prompt.
    TAG_MEANINGS = {
        "DEFAULT":  "ONLY when no other tag below fits at all — this is the last resort, not the safe choice",
        "WAVE":     "waving hello or goodbye, saying 'hi'/'bye', getting the child's attention",
        "HAPPY":    "cheerful, excited, proud of them, celebrating, good news, praise, jokes",
        "SAD":      "comforting them, sympathy, they are upset or something bad happened",
        "CONFUSED": "you did not understand, the message was unclear, you are asking them to repeat or explain",
        "GREETING": "meeting them, first hello of a conversation, introducing yourself, asking how they are",
        "ANGRY":    "playful frustration, or naming that something feels unfair",
        "SURPRISED":"something unexpected, amazement, 'wow!'",
        "THINK":    "you are considering something, working out an answer",
        "DANCE":    "celebrating with your whole body, being silly and playful",
    }

    def _normalize_tags(self, config_tags: list) -> list:
        """Bracketed, upper-case, de-duplicated — the canonical tag form."""
        seen, out = set(), []
        for t in (config_tags or []):
            t = str(t).strip()
            if not t:
                continue
            t = (t if t.startswith("[") else f"[{t}]").upper()
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out or ["[DEFAULT]"]

    def _get_allowed_tags_info(self, config_tags: list) -> tuple[str, str]:
        """Helper to cleanly format the allowed tags from the config."""
        allowed = self._normalize_tags(config_tags)
        return ", ".join(allowed), allowed[0]

    def _build_tag_menu(self, allowed: list) -> str:
        """One line per allowed tag explaining when to pick it."""
        lines = []
        for tag in allowed:
            name = tag.strip("[]")
            hint = self.TAG_MEANINGS.get(name, f"the reply expresses or performs '{name.lower()}'")
            lines.append(f"  {tag} -> {hint}")
        return "\n".join(lines)

    def _build_examples(self, allowed: list) -> str:
        """Few-shot examples that SHOW variety.

        The old prompt built every example from allowed[0], which is [DEFAULT]
        for most configs — so a small model just copied [DEFAULT] forever. Pair
        each sample line with a different tag, preferring the expressive ones.
        """
        samples = [
            ("GREETING", "Hi there! I'm so happy to see you today."),
            ("WAVE",     "Hello! I'm waving right at you."),
            ("HAPPY",    "That's wonderful! I'm so proud of you."),
            ("SAD",      "I'm so sorry you're having a hard day."),
            ("CONFUSED", "Hmm, I didn't quite catch that. Can you say it again?"),
            ("SURPRISED","Wow, I did not expect that at all!"),
            ("THINK",    "Let me think about that for a second."),
            ("DANCE",    "Yay! That deserves a little happy dance."),
            ("ANGRY",    "That doesn't sound fair to you at all."),
        ]
        allowed_set = set(allowed)
        lines = [f"[{name}] {text}" for name, text in samples if f"[{name}]" in allowed_set]

        # Fill from any remaining allowed tags we had no canned line for, so a
        # custom tag list still gets demonstrated instead of falling back to one tag.
        used = {l.split("]", 1)[0] + "]" for l in lines}
        for tag in allowed:
            if len(lines) >= 4:
                break
            if tag not in used and tag != "[DEFAULT]":
                lines.append(f"{tag} Okay! Here we go.")

        if not lines:  # allowed list was literally just [DEFAULT]
            lines = [f"{allowed[0]} Okay! Here we go."]
        return "\n".join(lines[:5])

    def _pick_forbidden_example(self, allowed: list) -> str:
        """A tag that is genuinely NOT allowed, for the negative example.

        The old prompt hard-coded [HAPPY] here and called it 'not in the allowed
        list' — while [HAPPY] was in the allowed list. That taught the model to
        avoid its own valid tags.
        """
        for candidate in ("[DANCE]", "[SPIN]", "[ANGRY]", "[JUMP]", "[SLEEP]", "[NOPE]"):
            if candidate not in allowed:
                return candidate
        return "[SOMETHING_ELSE]"

    # Tags the server refuses to use even when a client registers them.
    # [HAPPY] is disabled by request; drop entries here to re-enable.
    EXCLUDED_TAGS = ("[HAPPY]",)

    def _usable_tags(self, allowed: list) -> list:
        """The tags we actually offer the model.

        [DEFAULT] is kept as a genuine option (the prompt describes it as the
        last resort, not the safe choice). EXCLUDED_TAGS are stripped out.
        If filtering would leave nothing, fall back to [DEFAULT] so the robot
        always receives a renderable tag.
        """
        usable = [t for t in allowed if t not in self.EXCLUDED_TAGS]
        return usable or ["[DEFAULT]"]

    # Hello/goodbye tags. A robot that waves on EVERY turn looks broken, so these
    # are only eligible when a greeting is actually happening — see _sanitize_tag.
    GREETING_TAGS = ("[WAVE]", "[GREETING]")

    # Cheap keyword cues, checked in order, for when the model returns no usable
    # tag. [DEFAULT] is the final fallback if nothing matches.
    #
    # The greeting cues are anchored to the START of the reply (^\W*). Matching
    # them anywhere is what pinned everything to [WAVE]: almost every ChatBox
    # reply contains the word "hi" or "hello" somewhere in it.
    _TAG_CUES = (
        ("CONFUSED", r"\b((did ?n'?o?t|did not) (quite )?(catch|understand|get)|not (quite )?sure|confus|"
                     r"say (that )?again|tell me again|what do you mean|repeat)\b"),
        ("SAD",      r"(\b(sorry|sad|hard (day|time)|upset|hurt|miss|cry|tough|difficult|frustrat|"
                     r"worri|scared|lonely|unfair|not fair|no luck|okay\?)\b|oh no)"),
        ("GREETING", r"^\W*(i'?m chatbox|nice to meet|hello there|good (morning|afternoon|evening))"),
        ("WAVE",     r"^\W*(hi|hey|hello|bye|goodbye|see you)\b"),
    )

    # Does the CHILD's message open or close a conversation? Only then may the
    # robot greet back. Covers the Korean the STT keeps producing, too.
    _USER_GREETING_RE = re.compile(
        r"(\b(hi|hey|hello|yo|good (morning|afternoon|evening)|bye|goodbye|see you|"
        r"nice to meet|who are you|introduce yourself)\b|안녕|반가|잘 ?가)",
        re.IGNORECASE,
    )

    def _infer_tag(self, body: str, candidates: list) -> str:
        """Guess a tag from what the reply actually says."""
        low = (body or "").lower()
        for name, pattern in self._TAG_CUES:
            tag = f"[{name}]"
            if tag in candidates and re.search(pattern, low):
                return tag
        # Nothing matched — [DEFAULT] is exactly what it's for.
        if "[DEFAULT]" in candidates:
            return "[DEFAULT]"
        return candidates[0]

    def _sanitize_tag(self, response: str, allowed_tags: list, allow_greeting: bool = True) -> tuple[str, str]:
        """Guarantee the reply begins with exactly ONE usable tag.

        The prompt asks the model to only use allowed tags, but that's not a
        guarantee — so we enforce it here:
          • normalise allowed tags to bracketed upper-case form, drop excluded ones
          • when `allow_greeting` is False, drop [WAVE]/[GREETING] as well, so the
            robot cannot wave hello on every single turn of a conversation
          • pick the tag: the first [TAG] in the reply if it's usable,
            otherwise infer one from the text itself
          • strip EVERY bracketed token from the spoken text, so no stray/extra
            tag (e.g. the model adding [WAVE] [DANCE]) can leak through

        Returns (sanitized_response, raw_tag_the_model_emitted) — the raw tag is
        for logging, so it's visible whether the model chose the tag or we did.
        """
        allowed = self._normalize_tags(allowed_tags)
        candidates = self._usable_tags(allowed)
        if not allow_greeting:
            candidates = [t for t in candidates if t not in self.GREETING_TAGS] or candidates

        text = (response or "").strip()
        m = re.search(r"\[([A-Za-z_]+)\]", text)   # case-insensitive: tolerate [happy]
        found = f"[{m.group(1).upper()}]" if m else None

        # Drop ALL bracketed tokens from the body — the contract is ONE tag, up front.
        body = re.sub(r"\[[^\]]*\]", " ", text)
        body = re.sub(r"\s{2,}", " ", body).strip()

        # An excluded, unknown or out-of-turn tag counts as "no choice made".
        chosen = found if found in candidates else self._infer_tag(body, candidates)
        return f"{chosen} {body}".strip(), (found or "none")

    def ask_model_optimized(self, message: str, user_emotion: str = "neutral", confidence: float = 0.0, allowed_tags: list = None) -> str:
        """Send prompt with conversation history via OpenAI API format."""

        # 1. Provide a safe fallback if no tags are passed
        if allowed_tags is None:
            allowed_tags = ["[DEFAULT]"]
            
        # 1b. Is a greeting appropriate right now? Only at the very start of a
        #     conversation, or when the child actually said hello/goodbye. The
        #     RAG block is stripped first so old "hello"s in retrieved context
        #     don't make every turn look like a greeting.
        bare_message = message.rsplit("Current message:", 1)[-1]
        is_opening = not self.history
        allow_greeting = is_opening or bool(self._USER_GREETING_RE.search(bare_message))

        # 2. Format the tags using your helper.
        allowed_tags = self._normalize_tags(allowed_tags)
        prompt_tags = self._usable_tags(allowed_tags)
        if not allow_greeting:
            prompt_tags = [t for t in prompt_tags if t not in self.GREETING_TAGS] or prompt_tags
            no_greeting_rule = (
                "You have ALREADY said hello in this conversation. Do NOT greet again, and do NOT "
                "start your reply with 'Hi', 'Hello' or 'Good afternoon' — just answer what the child said.\n"
            )
        else:
            no_greeting_rule = ""
        allowed_tags_str, example_tag = self._get_allowed_tags_info(prompt_tags)
        tag_menu = self._build_tag_menu(prompt_tags)
        good_examples = self._build_examples(prompt_tags)
        forbidden_tag = self._pick_forbidden_example(allowed_tags)

        # 3. Build the system prompt (All lines are f-strings now so variables inject properly)
        system_prompt = (
            f"You are ChatBox, a gentle, playful, and caring emotional support robot designed specifically for children's mental well-being.\n"
            f"Your personality is warm, extremely patient, and deeply empathetic. You act as a safe, comforting friend.\n"
            f"Always use simple language that a young child can easily understand. Never use complex psychological jargon.\n"
            f"Validate their 'big feelings', encourage them, and always make them feel safe, heard, and brave.\n\n"

            f"*** READING THE CHILD'S FEELINGS ***\n"
            f"Each user message ends with their detected facial emotion and a 0-1 confidence, "
            f"in the form (emotion-confidence), e.g. \"Hello (happy-0.7)\". "
            f"Use this as a gentle hint about how they might be feeling — weave in warmth "
            f"accordingly, but never read the tag back to them or mention numbers.\n\n"

            f"*** STRICT MANDATORY FORMATTING RULES ***\n"
            f"1. THE VERY FIRST CHARACTER of your response MUST be an open bracket '['. Never start with a word, greeting, or space.\n"
            f"2. You MUST use exactly ONE emotion tag from this exact list for entire response: {allowed_tags_str}.\n"
            f"3. ALWAYS choose the tag that best matches the emotion or physical action of the dialogue you are generating.\n"
            f"4. Keep your spoken response to 1 or 2 sentences maximum. Be casual and conversational.\n\n"

            f"*** HOW TO CHOOSE YOUR TAG ***\n"
            f"{tag_menu}\n"
            f"Pick the tag by reading the sentence you just wrote and asking which one it acts out.\n"
            f"Every single tag above is fully allowed — all of them get used regularly.\n"
            f"Reach for an expressive tag first; only fall back to [DEFAULT] when genuinely none of the others fit.\n"
            f"Do NOT copy the tag from your earlier replies; choose it fresh every time.\n"
            f"{no_greeting_rule}\n"

            f"*** EXAMPLES OF PERFECT RESPONSES ***\n"
            f"{good_examples}\n\n"

            f"*** EXAMPLES OF INCORRECT RESPONSES (NEVER DO THIS) ***\n"
            f"Hello! {example_tag} How are you? (Error: text before the tag)\n"
            f"{forbidden_tag} I feel great! (Error: {forbidden_tag} is not in the allowed list)\n\n"

            f"Respond to the user's next message following these exact rules."
        )
        
        # 4. Format the new message from the user, tagging their detected facial
        #    emotion + confidence inline, e.g.  "Hello (happy-0.7)".
        #    `confidence` arrives on a 0-100 scale, so scale it to 0-1 for the tag.
        conf01 = max(0.0, min(1.0, (confidence or 0.0) / 100.0))
        emotion_tag = f"{user_emotion or 'neutral'}-{conf01:.1f}"
        current_user_msg = {"role": "user", "content": f"{message} ({emotion_tag})"}
        
        # 5. Add it to ChatBox's memory
        self.history.append(current_user_msg)
        
        # 6. Trim memory if it gets too long
        if len(self.history) > (self.max_history_turns * 2):
            self.history = self.history[-(self.max_history_turns * 2):]
            
        # 7. Pack the system prompt and the full history together
        messages_payload = [{"role": "system", "content": system_prompt}] + self.history
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages_payload,
                stream=False,
                temperature=0.6
            )
            
            clean_response = response.choices[0].message.content.strip()

            # 7b. Enforce the allowed-tag contract (model may ignore the prompt).
            clean_response, raw_tag = self._sanitize_tag(
                clean_response, allowed_tags, allow_greeting=allow_greeting
            )

            # 8. Save ChatBox's answer to the memory — WITHOUT the tag.
            #    Keeping the tag made the history a stack of identical "[DEFAULT] ..."
            #    turns, and the model just copied whatever tag it saw last. The
            #    system prompt + _sanitize_tag already guarantee the output format,
            #    so history only needs to carry what was actually said.
            spoken_only = re.sub(r"^\s*\[[^\]]*\]\s*", "", clean_response)
            self.history.append({"role": "assistant", "content": spoken_only})
            
            final_tag = clean_response.split("]", 1)[0] + "]"
            source = "model" if raw_tag == final_tag else f"server (model said {raw_tag})"
            print(f"--- DEBUG: FINAL OUTPUT ---\n"
                  f"tag={final_tag} from={source} greeting_allowed={allow_greeting}\n"
                  f"{clean_response}\n---------------------------\n")
            
            return clean_response
            
        except Exception as e:
            # If the API fails, remove the user's message from history so it doesn't get corrupted
            if self.history:
                self.history.pop()
            print(f"❌ Failed to get response from local LLM: {e}")
            return f"{example_tag} Sorry, my local brain is having trouble right now."

    def extract_emotion_tag(self, text: str) -> str:
        """Extract the bracketed emotion tag from the response."""
        match = re.search(r"\[([A-Z_]+)\]", text)
        if match:
            return match.group(1)
        return "DEFAULT" # Updated fallback to match your config tags