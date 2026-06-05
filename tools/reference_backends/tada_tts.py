"""TADA-3B-ML TTS reference dump backend for crispasr-diff.

Uses the official model.generate() API with a reference audio prompt
for voice conditioning. This produces audible, intelligible speech
that can be validated via ASR roundtrip.

Pipeline:
  1. Encoder: reference audio → aligned acoustic features (voice prompt)
  2. model.generate(prompt, text): AR + FM → acoustic features + time durations
  3. Decoder: expand + local-attention + DAC upsampler → 24 kHz PCM

Environment variables:
  TADA_SYN_TEXT       — text to synthesize (default: "Please call Stella.")
  TADA_PROMPT_TEXT    — transcript of the reference audio (optional, for alignment)
  TADA_DEVICE         — "cpu" or "cuda" (default: "cpu")
  TADA_SEED           — random seed (default: 42)
  TADA_CODEC_DIR      — local path to HumeAI/tada-codec (optional)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "text_tokens",
    "prompt_token_values",
    "prompt_token_positions",
    "prompt_time_before",
    "prompt_time_after",
    "acoustic_features",
    "time_before",
    "codec_pcm",
]

DEFAULT_SYN_TEXT = "Please call Stella."


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int = 256) -> Dict[str, np.ndarray]:
    import torch
    import torchaudio
    torch.set_grad_enabled(False)

    out: Dict[str, np.ndarray] = {}

    device = os.environ.get("TADA_DEVICE", "cpu")
    syn_text = os.environ.get("TADA_SYN_TEXT", DEFAULT_SYN_TEXT)
    seed = int(os.environ.get("TADA_SEED", "42"))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Load model ──
    print(f"  loading TADA from {model_dir}")
    from transformers import AutoTokenizer
    from tada.modules.tada import TadaForCausalLM, InferenceOptions
    from tada.modules.encoder import Encoder
    from tada.modules.decoder import Decoder

    # Patch tokenizer for gated Llama
    _orig = AutoTokenizer.from_pretrained.__func__
    @classmethod
    def _patched(cls, name, *a, **kw):
        try:
            return _orig(cls, name, *a, **kw)
        except Exception as e:
            if "gated" in str(e).lower() or "401" in str(e):
                return _orig(cls, "unsloth/Llama-3.2-1B", *a, **kw)
            raise
    AutoTokenizer.from_pretrained = _patched

    codec_dir = os.environ.get("TADA_CODEC_DIR")
    if codec_dir:
        _orig_dec = Decoder.from_pretrained.__func__
        @classmethod
        def _patched_dec(cls, name, *a, **kw):
            if "tada-codec" in str(name):
                return _orig_dec(cls, codec_dir, *a, **kw)
            return _orig_dec(cls, name, *a, **kw)
        Decoder.from_pretrained = _patched_dec

    model = TadaForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    tokenizer = model.tokenizer

    # ── Load Encoder for voice conditioning ──
    print(f"  loading Encoder")
    codec_path = codec_dir or "HumeAI/tada-codec"
    encoder = Encoder.from_pretrained(codec_path, subfolder="encoder").to(device)
    encoder.eval()

    # ── Process reference audio ──
    # audio is 16 kHz mono float32 from dump_reference.py
    audio_t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(device)
    # Resample 16k → 24k for the encoder
    audio_24k = torchaudio.functional.resample(audio_t, 16000, 24000)

    prompt_text = os.environ.get("TADA_PROMPT_TEXT")
    if prompt_text:
        prompt_texts = [prompt_text]
    else:
        prompt_texts = None  # auto-transcribe via parakeet

    print(f"  encoding reference audio ({audio_24k.shape[-1]/24000:.1f}s @ 24kHz)")
    with torch.no_grad():
        prompt = encoder(
            audio_24k,
            text=prompt_texts,
            sample_rate=24000,
        )

    print(f"  prompt: {prompt.token_values.shape[1]} aligned tokens")

    # ── Dump prompt features for C++ to use as pre-computed input ──
    if "prompt_token_values" in stages:
        out["prompt_token_values"] = prompt.token_values[0].cpu().float().numpy()
    if "prompt_token_positions" in stages:
        out["prompt_token_positions"] = prompt.token_positions[0].cpu().float().numpy()

    # ── Tokenize synth text ──
    text_tokens_raw = tokenizer.encode(syn_text, add_special_tokens=False)
    if "text_tokens" in stages:
        out["text_tokens"] = np.array(text_tokens_raw, dtype=np.float32)

    # ── Generate with official API ──
    print(f"  generating: {syn_text!r}")
    opts = InferenceOptions(
        text_do_sample=False,
        acoustic_cfg_scale=1.6,
        duration_cfg_scale=1.0,
        noise_temperature=0.9,
        num_flow_matching_steps=10,
        num_acoustic_candidates=1,
        cfg_schedule="cosine",
        time_schedule="logsnr",
    )

    with torch.no_grad():
        gen_output = model.generate(
            prompt=prompt,
            text=syn_text,
            inference_options=opts,
            verbose=True,
        )

    # ── Extract intermediates ──
    if gen_output.input_text_ids is not None:
        print(f"  input_ids: {gen_output.input_text_ids.shape}")
        if "input_ids" in stages:
            out["input_ids"] = gen_output.input_text_ids[0].cpu().float().numpy()

    if gen_output.acoustic_features is not None and "acoustic_features" in stages:
        af = gen_output.acoustic_features[0].cpu().float().numpy()
        out["acoustic_features"] = af

    if gen_output.time_before is not None and "time_before" in stages:
        tb = gen_output.time_before[0].cpu().float().numpy()
        out["time_before"] = tb

    if gen_output.audio is not None and gen_output.audio[0] is not None:
        pcm = gen_output.audio[0].cpu().float().numpy()
        if "codec_pcm" in stages:
            out["codec_pcm"] = pcm
        # Save WAV for ASR roundtrip
        import wave, struct
        wav_path = Path(os.environ.get("TADA_WAV_OUTPUT", "/tmp/tada-ref-output.wav"))
        pcm16 = (pcm * 32767).clip(-32767, 32767).astype(np.int16)
        with wave.open(str(wav_path), "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(pcm16.tobytes())
        print(f"  WAV saved: {wav_path} ({len(pcm16)} samples, "
              f"RMS={np.sqrt(np.mean(pcm.astype(float)**2)):.4f})")

    generated_text = ""
    if gen_output.output_str:
        generated_text = gen_output.output_str[0]
    print(f"  generated text: {generated_text!r}")

    return out
