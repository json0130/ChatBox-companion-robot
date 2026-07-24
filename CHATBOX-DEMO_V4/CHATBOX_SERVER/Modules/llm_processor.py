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
    
    def _get_allowed_tags_info(self, config_tags: list) -> tuple[str, str]:
        """Helper to cleanly format the allowed tags from the config."""
        if config_tags and isinstance(config_tags, list) and len(config_tags) > 0:
            allowed_list = ", ".join(config_tags)
            safe_example = config_tags[0]
            return allowed_list, safe_example
        
        return "[DEFAULT]", "[DEFAULT]"

    def _sanitize_tag(self, response: str, allowed_tags: list) -> str:
        """Guarantee the reply begins with exactly ONE tag from `allowed_tags`.

        The prompt asks the model to only use allowed tags, but that's not a
        guarantee — so we enforce it here:
          • normalise allowed tags to bracketed upper-case form
          • pick the tag: the first [TAG] in the reply if it's allowed,
            otherwise the first allowed tag (the safe default)
          • strip EVERY bracketed token from the spoken text, so no stray/extra
            tag (e.g. the model adding [WAVE] [DANCE]) can leak through
        Result is always exactly one allowed tag, followed by the spoken text.
        """
        allowed = [(t if t.startswith("[") else f"[{t}]").upper()
                   for t in (allowed_tags or []) if str(t).strip()]
        if not allowed:
            allowed = ["[DEFAULT]"]
        default = allowed[0]

        text = (response or "").strip()
        m = re.search(r"\[([A-Za-z_]+)\]", text)   # case-insensitive: tolerate [happy]
        found = f"[{m.group(1).upper()}]" if m else None
        chosen = found if found in allowed else default

        # Drop ALL bracketed tokens from the body — the contract is ONE tag, up front.
        body = re.sub(r"\[[^\]]*\]", " ", text)
        body = re.sub(r"\s{2,}", " ", body).strip()
        return f"{chosen} {body}".strip()

    def ask_model_optimized(self, message: str, user_emotion: str = "neutral", confidence: float = 0.0, allowed_tags: list = None) -> str:
        """Send prompt with conversation history via OpenAI API format."""

        # 1. Provide a safe fallback if no tags are passed
        if allowed_tags is None:
            allowed_tags = ["[DEFAULT]"]
            
        # 2. Format the tags using your helper
        allowed_tags_str, example_tag = self._get_allowed_tags_info(allowed_tags)
        
        # 3. Build the system prompt (All lines are f-strings now so variables inject properly)
        system_prompt = (
            f"You are CHATBOX, a gentle, playful, and caring emotional support robot designed specifically for children's mental well-being.\n"
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
            
            f"*** EXAMPLES OF PERFECT RESPONSES ***\n"
            f"{example_tag} Hello there! It's so nice to meet you.\n"
            f"{example_tag} I'm not sure I understand what you mean.\n"
            f"{example_tag} I'm so sorry you are having a hard day.\n\n"
            
            f"*** EXAMPLES OF INCORRECT RESPONSES (NEVER DO THIS) ***\n"
            f"Hello! [WAVE] How are you? (Error: Text before the tag)\n"
            f"[HAPPY] I feel great! (Error: Tag is not in the allowed list)\n\n"
            
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
            clean_response = self._sanitize_tag(clean_response, allowed_tags)

            # 8. Save ChatBox's answer to the memory
            self.history.append({"role": "assistant", "content": clean_response})
            
            print(f"--- DEBUG: FINAL OUTPUT ---\n{clean_response}\n---------------------------\n")
            
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