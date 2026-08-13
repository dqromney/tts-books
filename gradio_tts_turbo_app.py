import random
import re
import numpy as np
import torch
import torch as th
import fitz  # pymupdf
import gradio as gr
from chatterbox.tts_turbo import ChatterboxTurboTTS

DEVICE = "cpu"  # Quadro M2200 has only 4GB VRAM, Turbo model needs more

EVENT_TAGS = [
    "[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]",
    "[sniff]", "[gasp]", "[chuckle]", "[laugh]"
]

# --- REFINED CSS ---
# 1. tag-container: Forces the row to wrap items instead of scrolling. Removes borders/backgrounds.
# 2. tag-btn: Sets the specific look (indigo theme) and stops them from stretching.
CUSTOM_CSS = """
.tag-container {
    display: flex !important;
    flex-wrap: wrap !important; /* This fixes the one-per-line issue */
    gap: 8px !important;
    margin-top: 5px !important;
    margin-bottom: 10px !important;
    border: none !important;
    background: transparent !important;
}

.tag-btn {
    min-width: fit-content !important;
    width: auto !important;
    height: 32px !important;
    font-size: 13px !important;
    background: #eef2ff !important;
    border: 1px solid #c7d2fe !important;
    color: #3730a3 !important;
    border-radius: 6px !important;
    padding: 0 10px !important;
    margin: 0 !important;
    box-shadow: none !important;
}

.tag-btn:hover {
    background: #c7d2fe !important;
    transform: translateY(-1px);
}
"""

INSERT_TAG_JS = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#main_textbox textarea');
    if (!textarea) return current_text + " " + tag_val; 

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;

    let prefix = " ";
    let suffix = " ";

    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";

    if (end < current_text.length && current_text[end] === ' ') suffix = "";

    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model():
    print(f"Loading Chatterbox-Turbo on {DEVICE}...")
    model = ChatterboxTurboTTS.from_pretrained(DEVICE)
    return model


def load_text_file(file):
    """Read uploaded .txt, .pdf, or .epub file and return its content."""
    if file is None:
        return ""
    path = file if isinstance(file, str) else file.name
    if path.lower().endswith('.pdf'):
        doc = fitz.open(path)
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    elif path.lower().endswith('.epub'):
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
        book = epub.read_epub(path)
        texts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_body_content(), 'html.parser')
            texts.append(soup.get_text())
        return '\n\n'.join(texts)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()


def split_text(text, max_chars=200):
    """Split text into chunks at sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) <= max_chars:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks if chunks else [text]


def generate(
        model,
        text,
        audio_prompt_path,
        temperature,
        seed_num,
        min_p,
        top_p,
        top_k,
        repetition_penalty,
        norm_loudness,
        progress=gr.Progress(),
):
    if model is None:
        model = ChatterboxTurboTTS.from_pretrained(DEVICE)

    if seed_num != 0:
        set_seed(int(seed_num))

    chunks = split_text(text)
    total = len(chunks)
    print(f"Generating {total} chunk(s)...")

    import tempfile, os
    import torchaudio

    tmpdir = tempfile.mkdtemp(prefix="tts_chunks_")
    chunk_files = []

    try:
        for i, chunk in enumerate(chunks):
            progress(i / total, desc=f"Chunk {i+1}/{total} ({len(chunk)} chars)")
            wav = model.generate(
                chunk,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                min_p=min_p,
                top_p=top_p,
                top_k=int(top_k),
                repetition_penalty=repetition_penalty,
                norm_loudness=norm_loudness,
            )
            wav = wav.squeeze(0).cpu()
            fpath = os.path.join(tmpdir, f"chunk_{i:06d}.wav")
            torchaudio.save(fpath, wav.unsqueeze(0), model.sr)
            chunk_files.append(fpath)
            del wav  # free memory immediately

        progress(1.0, desc="Concatenating...")
        # Concatenate with 0.3s silence between chunks
        if len(chunk_files) == 1:
            final, sr = torchaudio.load(chunk_files[0])
        else:
            silence = th.zeros(1, int(model.sr * 0.3))
            waveforms = []
            for fpath in chunk_files:
                w, sr = torchaudio.load(fpath)
                waveforms.append(w)
                waveforms.append(silence)
            final = th.cat(waveforms[:-1], dim=1)  # drop trailing silence

        # Auto-save to ~/tts_output/
        import os as _os
        from datetime import datetime as _dt
        outdir = _os.path.expanduser("~/tts_output")
        _os.makedirs(outdir, exist_ok=True)
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        outpath = _os.path.join(outdir, f"tts_{ts}.wav")
        torchaudio.save(outpath, final, sr)
        print(f"Saved: {outpath}")

        return (sr, final.squeeze(0).numpy())

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


with gr.Blocks(title="Chatterbox Turbo") as demo:
    gr.Markdown("# ⚡ Chatterbox Turbo")

    model_state = gr.State(None)

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. It's the SkyNet T-800 series and it's got basically everything. Including AI integration with ChatGPT and um all that jazz. Would you like me to get some prices for you?",
                label="Text to synthesize",
                max_lines=15,
                elem_id="main_textbox"
            )

            text_file = gr.File(
                label="Or upload a text file (.txt, .pdf, .epub)",
                file_types=[".txt", ".pdf", ".epub"],
                type="filepath"
            )
            text_file.upload(fn=load_text_file, inputs=text_file, outputs=text)

            # --- Event Tags ---
            # Switched back to Row, but applied specific CSS to force wrapping
            with gr.Row(elem_classes=["tag-container"]):
                for tag in EVENT_TAGS:
                    # elem_classes targets the button specifically
                    btn = gr.Button(tag, elem_classes=["tag-btn"])

                    btn.click(
                        fn=None,
                        inputs=[btn, text],
                        outputs=text,
                        js=INSERT_TAG_JS
                    )

            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Audio File",
                value="https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_random_podcast.wav"
            )

            run_btn = gr.Button("Generate ⚡", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

            with gr.Accordion("Advanced Options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 2.0, step=.05, label="Temperature", value=0.8)
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="Top P", value=0.95)
                top_k = gr.Slider(0, 1000, step=10, label="Top K", value=1000)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.05, label="Repetition Penalty", value=1.2)
                min_p = gr.Slider(0.00, 1.00, step=0.01, label="Min P (Set to 0 to disable)", value=0.00)
                norm_loudness = gr.Checkbox(value=True, label="Normalize Loudness (-27 LUFS)")

    demo.load(fn=load_model, inputs=[], outputs=model_state)

    run_btn.click(
        fn=generate,
        inputs=[
            model_state,
            text,
            ref_wav,
            temp,
            seed_num,
            min_p,
            top_p,
            top_k,
            repetition_penalty,
            norm_loudness,
        ],
        outputs=audio_output,
    )

if __name__ == "__main__":
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(share=True, inbrowser=True, css=CUSTOM_CSS)
