#!/usr/bin/env python3
"""
Parler TTS reference backend -- loads the HuggingFace model, runs
inference, dumps intermediates for diff-testing the C++ port.

Usage:
    python tools/reference_backends/parler_tts.py \\
        --text "Hello, this is a test." \\
        --description "A female speaker with a warm voice in a quiet room" \\
        --output /mnt/storage/parler-tts/ref_output.wav \\
        --dump-dir /mnt/storage/parler-tts/ref_intermediates

Requires: pip install parler-tts torch soundfile numpy
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Parler TTS reference inference")
    parser.add_argument("--text", type=str, default="Hello, this is a test of Parler TTS.",
                        help="Text to synthesize")
    parser.add_argument("--description", type=str,
                        default="A female speaker with a warm, natural voice delivers her words at a moderate pace in a quiet environment.",
                        help="Voice description prompt")
    parser.add_argument("--model", type=str, default="parler-tts/parler-tts-mini-v1.1",
                        help="HuggingFace model ID")
    parser.add_argument("--output", type=str, default=None,
                        help="Output WAV file path")
    parser.add_argument("--dump-dir", type=str, default=None,
                        help="Directory to dump intermediate tensors for diff-testing")
    parser.add_argument("--max-length", type=int, default=2580,
                        help="Max generation length")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (1.0 = default)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    try:
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("pip install parler-tts transformers")

    try:
        import soundfile as sf
    except ImportError:
        sys.exit("pip install soundfile")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float32  # CPU-safe

    print(f"Loading model {args.model}...", file=sys.stderr)
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch_dtype
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Also load the description tokenizer if available
    desc_tokenizer = None
    try:
        desc_tokenizer = AutoTokenizer.from_pretrained(args.model, subfolder="description_tokenizer")
    except Exception:
        # Some versions don't have a separate description tokenizer
        desc_tokenizer = tokenizer

    print(f"Tokenizing description: '{args.description[:60]}...'", file=sys.stderr)
    description_ids = desc_tokenizer(args.description, return_tensors="pt").input_ids.to(device)

    print(f"Tokenizing text: '{args.text[:60]}...'", file=sys.stderr)
    prompt_ids = tokenizer(args.text, return_tensors="pt").input_ids.to(device)

    # Dump tokenizer info
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        np.save(os.path.join(args.dump_dir, "description_ids.npy"),
                description_ids.cpu().numpy())
        np.save(os.path.join(args.dump_dir, "prompt_ids.npy"),
                prompt_ids.cpu().numpy())

    # Run T5 encoder on description
    print("Running T5 encoder on description...", file=sys.stderr)
    with torch.no_grad():
        # Get encoder outputs manually
        encoder_outputs = model.text_encoder(
            input_ids=description_ids,
        )
        encoder_hidden = encoder_outputs.last_hidden_state
        print(f"  T5 encoder output shape: {encoder_hidden.shape}", file=sys.stderr)

        if args.dump_dir:
            np.save(os.path.join(args.dump_dir, "t5_encoder_output.npy"),
                    encoder_hidden.cpu().numpy())

    # Run full generation
    print("Generating audio codes...", file=sys.stderr)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    with torch.no_grad():
        generation = model.generate(
            input_ids=description_ids,
            prompt_input_ids=prompt_ids,
            max_length=args.max_length,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
        )

    print(f"Generated codes shape: {generation.shape}", file=sys.stderr)

    if args.dump_dir:
        np.save(os.path.join(args.dump_dir, "generated_codes.npy"),
                generation.cpu().numpy())

    # Decode audio
    print("Decoding audio via DAC...", file=sys.stderr)
    with torch.no_grad():
        audio = model.audio_encoder.decode(
            audio_codes=generation,
            audio_scales=[None],
            padding_mask=None,
        ).audio_values

    audio_np = audio.squeeze().cpu().numpy()
    sample_rate = model.config.audio_encoder.sampling_rate

    print(f"Audio shape: {audio_np.shape}, sample rate: {sample_rate}", file=sys.stderr)
    print(f"Duration: {len(audio_np) / sample_rate:.2f}s", file=sys.stderr)

    if args.dump_dir:
        np.save(os.path.join(args.dump_dir, "audio_pcm.npy"), audio_np)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        sf.write(args.output, audio_np, sample_rate)
        print(f"Saved to {args.output}", file=sys.stderr)

    # Dump model architecture info
    print("\n=== Model Architecture ===", file=sys.stderr)
    print(f"T5 encoder: d_model={model.config.text_encoder.d_model}, "
          f"n_heads={model.config.text_encoder.num_heads}, "
          f"n_layers={model.config.text_encoder.num_layers}, "
          f"d_kv={model.config.text_encoder.d_kv}, "
          f"d_ff={model.config.text_encoder.d_ff}", file=sys.stderr)
    print(f"Decoder: hidden={model.config.decoder.hidden_size}, "
          f"n_heads={model.config.decoder.num_attention_heads}, "
          f"n_layers={model.config.decoder.num_hidden_layers}, "
          f"ffn_dim={model.config.decoder.ffn_dim}, "
          f"num_codebooks={model.config.decoder.num_codebooks}, "
          f"vocab_size={model.config.decoder.vocab_size}", file=sys.stderr)
    print(f"DAC: sample_rate={model.config.audio_encoder.sampling_rate}, "
          f"n_codebooks={model.config.audio_encoder.n_codebooks}, "
          f"codebook_size={model.config.audio_encoder.codebook_size}", file=sys.stderr)

    # Dump some weight stats for diff-testing
    if args.dump_dir:
        stats = {}
        for name, param in model.named_parameters():
            stats[name] = {
                "shape": list(param.shape),
                "mean": float(param.float().mean()),
                "std": float(param.float().std()),
                "min": float(param.float().min()),
                "max": float(param.float().max()),
            }

        import json
        with open(os.path.join(args.dump_dir, "weight_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Dumped weight stats to {args.dump_dir}/weight_stats.json", file=sys.stderr)


if __name__ == "__main__":
    main()
