# voice-to-voice

Local TTS demo with Kokoro-82M. Paste an LLM response and hear it spoken.
English and Spanish are detected automatically; you can also pick a voice
yourself. Runs on the GPU if one is available, falls back to CPU.

## Quickstart

    ./run.sh

The first run creates a Python 3.12 venv and installs requirements with uv,
then serves the UI at http://127.0.0.1:7860.

## What it does

- Light preprocessing so LLM output sounds natural: strips headings, bold,
  backticks, links, bare URLs, and citation markers, turns bullet lists into
  sentences, and replaces fenced code blocks with a short spoken placeholder
  ("There is a code block here." / "Hay un bloque de código aquí.").
  Technical terms, commands, and filenames survive intact; prose is never
  summarized or rewritten.
- Language auto-detection with manual override.
- Speed slider, voice picker, audio player.

## Voices

US English: af_heart, af_alloy, af_aoede, af_bella, af_jessica, af_kore,
af_nicole, af_nova, af_river, af_sarah, af_sky, am_adam, am_echo, am_eric,
am_fenrir, am_liam, am_michael, am_onyx, am_puck, am_santa

UK English: bf_alice, bf_emma, bf_isabella, bf_lily, bm_daniel, bm_fable,
bm_george, bm_lewis

Spanish: ef_dora, em_alex, em_santa

Conversational picks: af_heart / af_bella / am_michael (English),
ef_dora / em_alex (Spanish).

## Tests

    .venv/bin/python test_preprocess.py

Checks the markdown cleaner and the EN/ES language detector (21 checks).

## Run as a service (systemd user unit)

    mkdir -p ~/.config/systemd/user
    cp deploy/kokoro-tts.service ~/.config/systemd/user/
    systemctl --user daemon-reload
    systemctl --user enable --now kokoro-tts

    systemctl --user status kokoro-tts     # state
    journalctl --user -u kokoro-tts -f     # logs
    systemctl --user restart kokoro-tts    # after pulling changes
