"""Local TTS quality test app for Kokoro-82M.

Paste an LLM response (English or Spanish) into the Gradio textbox. The app
cleans the markdown lightly (code bodies become a short spoken placeholder),
detects the language, picks a Kokoro voice, and writes the speech to a WAV
file under ./out.

Start it with:

    .venv/bin/python app.py
"""

import re
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 24000
OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPELINE_KWARGS = {"device": DEVICE}

CODE_PLACEHOLDER = {
    "en": "There is a code block here.",
    "es": "Hay un bloque de código aquí.",
}

DEFAULT_VOICE = {"a": "af_heart", "b": "bf_emma", "e": "ef_dora"}

US_ENGLISH_VOICES = [
    "af_heart",
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
]
UK_ENGLISH_VOICES = [
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
]
SPANISH_VOICES = [
    "ef_dora",
    "em_alex",
    "em_santa",
]
ALL_VOICES = US_ENGLISH_VOICES + UK_ENGLISH_VOICES + SPANISH_VOICES

# Words shared by both languages (a, no, en, la, ...) stay in neither set, so
# they never tip the vote for either side.
ES_SPANISH_ONLY = {
    "de", "que", "el", "los", "las", "del", "una", "por", "con", "para",
    "es", "al", "lo", "como", "más", "pero", "sus", "ya", "este", "esta",
    "esto", "entre", "cuando", "muy", "sin", "sobre", "también", "hasta",
    "hay", "donde", "desde", "todo", "todos", "uno", "unos", "otra",
    "otras", "otro", "otros", "ellos", "ellas", "nos", "yo", "él", "ella",
    "si", "sí", "año", "porque", "ese", "esa", "esos", "esas", "ser",
    "está", "son", "se", "su", "mí", "tú", "usted", "aquí", "allí",
}
EN_ENGLISH_ONLY = {
    "the", "of", "to", "and", "is", "it", "you", "that", "for", "on",
    "with", "as", "was", "are", "this", "be", "have", "from", "or", "an",
    "at", "not", "which", "by", "so", "we", "they", "his", "her", "its",
    "can", "will", "if", "one", "all", "would", "there", "their", "what",
    "about", "out", "who", "get", "has", "him", "like", "when", "now",
    "your", "my", "these", "some", "very", "do", "into", "up", "than",
    "them", "over", "also", "just", "only", "most", "other", "such",
    "because", "between", "after", "under", "our", "where", "while",
    "both", "however", "more", "any", "each", "few", "own", "same", "too",
    "but", "been", "were",
}


def detect_language(text: str) -> str:
    """Return "en" or "es". Ambiguous or empty text defaults to "en"."""
    words = re.findall(r"\w+", text.lower())
    es_hits = sum(1 for word in words if word in ES_SPANISH_ONLY)
    en_hits = sum(1 for word in words if word in EN_ENGLISH_ONLY)
    accents = sum(1 for char in text if char in "¿¡ñáéíóúüÁÉÍÓÚÑ")
    return "es" if es_hits + 2 * accents > en_hits else "en"


def preprocess(md: str, lang: str) -> str:
    """Lightly clean markdown so TTS sounds natural.

    Prose is never summarized, reordered, translated, or dropped: only
    syntax, URLs, citations, and code bodies are removed. ``lang`` ("en" or
    "es") only picks the sentence that replaces fenced code blocks.
    """
    placeholder = CODE_PLACEHOLDER[lang]

    # Fenced code. Odd-index parts are code bodies (first line may be a
    # language tag). An unterminated fence leaves a trailing odd part, which
    # this loop also covers.
    parts = md.split("```")
    for i in range(1, len(parts), 2):
        parts[i] = placeholder + "\n"
    md = "".join(parts)

    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)  # images
    md = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", md)  # links: keep the text
    md = re.sub(r"https?://\S+", "", md)  # bare URLs
    md = re.sub(r"`([^`]*)`", r"\1", md)  # inline code: keep technical terms

    md = re.sub(r"\*\*\*([^*]+)\*\*\*", r"\1", md)
    md = re.sub(r"\*\*([^*]+)\*\*", r"\1", md)
    md = re.sub(r"\*([^*]+)\*", r"\1", md)
    md = re.sub(r"~~([^~]+)~~", r"\1", md)
    # Underscore emphasis only away from word chars, so snake_case survives.
    md = re.sub(r"(?<![\w])__?([^_]+)__?(?![\w])", r"\1", md)

    md = re.sub(r"^#{1,6}\s+", "", md, flags=re.MULTILINE)  # headings
    md = re.sub(r"^\s*>\s?", "", md, flags=re.MULTILINE)  # blockquotes
    md = re.sub(
        r"^[ \t]*([-*_])[ \t]*(?:[ \t]*\1){2,}[ \t]*$",
        "",
        md,
        flags=re.MULTILINE,
    )  # horizontal rules

    md = re.sub(r"\[\s*\d+(?:\s*[,;-]\s*\d+)*\s*\]", "", md)  # [1] [2, 3] [3-5]
    md = re.sub(r"\[[A-Z][A-Za-z]*(?:\s+et al\.?)?(?:\s*,\s*\d{4})?\]", "", md)
    md = re.sub(r"\[[A-Z][A-Za-z]*\d{4}\]", "", md)  # [Smith2020]

    md = re.sub(r"^\s*[-*+]\s+", "", md, flags=re.MULTILINE)  # bullets
    md = re.sub(r"^\s*\d+[.)]\s+", "", md, flags=re.MULTILINE)  # numbered

    kept = []
    for line in md.split("\n"):
        stripped = line.strip()
        if stripped and set(stripped) <= set("|-: "):
            continue  # table separator row
        kept.append(line)
    md = "\n".join(kept).replace("|", " ")  # remaining pipes: cell gaps

    md = re.sub(r"</?[a-zA-Z][^>]*>", "", md)  # html tags
    md = re.sub(r"\$\$([^$]+)\$\$", r"\1", md)  # display math
    md = re.sub(r"\$([^$]+)\$", r"\1", md)  # inline math

    lines = [line.strip() for line in md.split("\n")]
    spoken = [
        line if line[-1] in ".!?…¡¿" else line + "." for line in lines if line
    ]
    return re.sub(r"\s+", " ", " ".join(spoken)).strip()


_pipelines = {}


def get_pipeline(code: str):
    """Return the cached KPipeline for a lang_code ("a", "b", or "e")."""
    if code not in _pipelines:
        # Imported lazily so importing this module stays cheap.
        from kokoro import KPipeline

        _pipelines[code] = KPipeline(
            lang_code=code, repo_id="hexgrad/Kokoro-82M", **PIPELINE_KWARGS
        )
    return _pipelines[code]


def generate(clean: str, code: str, voice: str, speed: float) -> str:
    """Speak ``clean`` and return the written WAV path."""
    chunks = []
    for _graphemes, _phonemes, audio in get_pipeline(code)(
        clean, voice=voice, speed=speed
    ):
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        chunks.append(np.asarray(audio, dtype=np.float32))
    if not chunks:
        raise RuntimeError("no audio produced")
    wav = np.concatenate(chunks)
    path = OUT_DIR / f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{code}.wav"
    sf.write(str(path), wav, SAMPLE_RATE)
    return str(path)


def speak(text: str, lang_choice: str, voice_choice: str, speed: float) -> tuple[str, str, str]:
    """Full UI action: clean the text, speak it, and report what happened."""
    if not text or not text.strip():
        return (None, "", "Paste some text first.")
    detected = detect_language(text)
    target = {"english": "en", "spanish": "es"}.get(
        lang_choice.strip().lower(), detected
    )
    if voice_choice != "Auto":
        code = voice_choice[0]
        voice = voice_choice
    else:
        code = {"en": "a", "es": "e"}[target]
        voice = DEFAULT_VOICE[code]
    clean = preprocess(text, target)
    path = generate(clean, code, voice, float(speed))
    label = f"lang={target} (detected={detected}) | voice={voice} | device={DEVICE}"
    return (path, clean, label)


demo = gr.Blocks(title="Kokoro TTS test")
with demo:
    gr.Markdown(
        "# Kokoro-82M TTS test\n"
        "Paste an LLM response. English or Spanish is detected automatically."
    )
    text = gr.Textbox(label="LLM response", lines=14, placeholder="Paste here...")
    with gr.Row():
        lang = gr.Radio(["Auto", "English", "Spanish"], value="Auto", label="Language")
        voice = gr.Dropdown(
            choices=["Auto"] + ALL_VOICES,
            value="Auto",
            label="Voice",
            info="Picking a voice overrides the language",
        )
        speed = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.05, label="Speed")
    btn = gr.Button("Generate", variant="primary")
    with gr.Row():
        audio = gr.Audio(label="Speech", type="filepath")
        clean_view = gr.Textbox(
            label="Preprocessed text (what will be spoken)", lines=8, interactive=False
        )
        status = gr.Textbox(label="Status", interactive=False)
    btn.click(
        speak,
        inputs=[text, lang, voice, speed],
        outputs=[audio, clean_view, status],
        api_name="speak",
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
