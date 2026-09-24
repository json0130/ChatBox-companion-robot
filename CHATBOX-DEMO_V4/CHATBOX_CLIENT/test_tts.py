#!/usr/bin/env python3
"""
test_tts.py — Standalone TTS tester.
Run from inside CHATBOX_CLIENT folder:

    python3 test_tts.py                # plays the pepeha as plain text (gTTS / Piper)
    python3 test_tts.py --ssml         # plays the pepeha via SSML+phonemes (edge-tts, NZ Neural voice)
    python3 test_tts.py hello world    # plays any plain-text args
    python3 test_tts.py --ssml-file x.ssml   # plays a custom SSML file
"""
import json
import logging
import sys
import os

logging.basicConfig(level=logging.WARNING)  # suppress DEBUG noise

sys.path.insert(0, os.path.dirname(__file__))
from OutputModules.edge_tts_output import EdgeTTSOutputModule

# ── Plain-text fallback ────────────────────────────────────────────────────────
PEPEHA_LINES = [
    "Ko Rangitoto te maunga.",
    "Ko Waitematā te moana.",
    "Ko Tāmaki Makaurau tōku kāinga.",
    "Ko te Whare Wānanga o Tāmaki Makaurau tōku whare wānanga.",
    "Ko ChatBox tōku ingoa.",
    "Tēnā koutou, tēnā koutou, tēnā koutou katoa.",
]

# ── SSML pepeha — uses IPA phonemes for accurate Māori pronunciation ──────────
# Namespace fixed to the W3C SSML spec: www.w3.org/2001/10/synthesis
PEPEHA_SSML = """\
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-NZ">
  <voice name="en-NZ-MitchellNeural">

    <phoneme alphabet="ipa" ph="tɛːnaː kou.tou kaː.tɔa">Tēnā koutou katoa</phoneme>.<break time="400ms"/>

    Ko <phoneme alphabet="ipa" ph="ɾa.ŋi.to.to">Rangitoto</phoneme> te maunga.<break time="300ms"/>

    Ko <phoneme alphabet="ipa" ph="wai.tɛ.ma.taː">Waitematā</phoneme> te moana.<break time="300ms"/>

    Ko <phoneme alphabet="ipa" ph="taː.ma.ki ma.kou.ɾoe">Tāmaki Makaurau</phoneme> tōku kāinga.<break time="300ms"/>

    Ko <phoneme alphabet="ipa" ph="tɛ ɸa.ɾɛ waː.na.ŋa o taː.ma.ki ma.kou.ɾaw">te Whare Wānanga o Tāmaki Makaurau</phoneme> tōku whare wānanga.<break time="300ms"/>

    Ko ChatBox tōku ingoa.<break time="400ms"/>

    <phoneme alphabet="ipa" ph="tɛːnaː kou.tu tɛːnaː kou.too tɛːnaː kou.toe kaː.tɔa">Tēnā koutou, tēnā koutou, tēnā koutou katoa</phoneme>.

  </voice>
</speak>"""


def main():
    args = sys.argv[1:]

    with open("client_config.json") as f:
        config = json.load(f)

    tts_cfg = config.get("edge_tts_config", {})
    tts = EdgeTTSOutputModule("edge_tts_output", tts_cfg)
    tts.start()

    if '--ssml-file' in args:
        idx = args.index('--ssml-file')
        ssml_path = args[idx + 1]
        with open(ssml_path) as f:
            ssml = f.read()
        print(f"Playing SSML from: {ssml_path}")
        tts.process_ssml(ssml)

    elif '--ssml' in args:
        print("Playing SSML pepeha (edge-tts, NZ Neural voice + IPA phonemes):")
        print(PEPEHA_SSML[:120] + "...\n")
        tts.process_ssml(PEPEHA_SSML)

    else:
        lines = args if args else PEPEHA_LINES
        print("Playing (plain text):")
        for line in lines:
            print(f"  {line}")
            tts.process_output(line)

    tts.tts_queue.join()
    tts.stop()


if __name__ == "__main__":
    main()
