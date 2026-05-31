#!/usr/bin/env python3
"""
Convert parler-tts/parler-tts-mini-v1.1 (or -large-v1) HuggingFace
safetensors -> GGUF F16 for the CrispASR `parler-tts` backend.

Parler TTS is a prompt-conditioned TTS: you describe the voice in natural
language ("A female speaker with a warm voice in a quiet room") and the
model generates speech matching that description.

Architecture (parler-tts-mini-v1.1/config.json):

  TEXT ENCODER: T5 (flan-t5-large)
    d_model         = 1024
    d_kv            = 64
    d_ff            = 2816
    n_heads         = 16
    num_layers      = 24  (encoder only -- decoder layers not used)
    feed_forward_proj = "gated-gelu"
    vocab_size      = 32128
    relative_attention_num_buckets = 32
    relative_attention_max_distance = 128

  DECODER: MusicGen-style causal transformer
    hidden_size     = 1024
    num_hidden_layers = 24
    num_attention_heads = 16
    ffn_dim         = 4096
    activation      = "gelu"
    num_codebooks   = 9
    vocab_size      = 1088  (1024 audio + pad + bos + extras)
    max_position_embeddings = 4096
    sinusoidal positional embeddings (not RoPE)
    layer norm with bias (not RMS norm)
    bos_token_id    = 1025
    eos_token_id    = 1024  (= pad_token_id)

  AUDIO ENCODER: DAC 44 kHz
    9 codebooks x 1024 entries x 8 dim
    upsampling_ratios = [8, 8, 4, 2] => hop_length = 512
    sample_rate = 44100

  PROMPT TOKENIZER: unigram (from LLaMA-2 tokenizer, vocab ~90714)

Usage:
    python models/convert-parler-to-gguf.py \\
        --input parler-tts/parler-tts-mini-v1.1 \\
        --output /mnt/storage/parler-tts/parler-tts-mini-v1.1-f16.gguf

    # Large variant:
    python models/convert-parler-to-gguf.py \\
        --input parler-tts/parler-tts-large-v1 \\
        --output /mnt/storage/parler-tts/parler-tts-large-v1-f16.gguf
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")

try:
    from safetensors import safe_open
except ImportError:
    sys.exit("pip install safetensors")

try:
    import torch
except ImportError:
    sys.exit("pip install torch")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("pip install huggingface_hub")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_model_dir(model_id: str) -> Path:
    p = Path(model_id)
    if p.is_dir():
        return p
    print(f"Downloading {model_id}...", file=sys.stderr)
    return Path(snapshot_download(model_id, allow_patterns=[
        "*.safetensors", "*.json", "*.txt", "tokenizer*", "*.model",
    ]))


def load_safetensors(model_dir: Path) -> dict[str, np.ndarray]:
    """Load all safetensor shards into a flat dict of numpy arrays."""
    tensors = {}
    for st_file in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for name in f.keys():
                tensors[name] = f.get_tensor(name).numpy()
    return tensors


def to_f16(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        return arr.astype(np.float16)
    return arr


def to_f32(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32) if arr.dtype != np.float32 else arr


# ---------------------------------------------------------------------------
# Weight normalization (DAC uses weight_norm)
# ---------------------------------------------------------------------------


def fuse_weight_norm(tensors: dict, prefix: str) -> np.ndarray:
    """Compute fused weight from weight_v and weight_g (weight normalization)."""
    v = tensors[f"{prefix}weight_v"]
    g = tensors[f"{prefix}weight_g"]
    # weight = g * v / ||v||
    # For Conv1d: v is (out, in, kernel), g is (out, 1, 1)
    norm_dims = tuple(range(1, v.ndim))
    norm = np.sqrt(np.sum(v ** 2, axis=norm_dims, keepdims=True))
    return g * v / (norm + 1e-12)


# ---------------------------------------------------------------------------
# T5 encoder tensor mapping
# ---------------------------------------------------------------------------


def map_t5_encoder_tensors(tensors: dict) -> list[tuple[str, np.ndarray]]:
    """Map HuggingFace T5 encoder tensors to our GGUF naming."""
    mapped = []

    # Shared embedding
    for name in ["text_encoder.shared.weight", "text_encoder.encoder.embed_tokens.weight"]:
        if name in tensors:
            mapped.append(("t5enc.embed.weight", to_f16(tensors[name])))
            break

    # Relative attention bias (only in layer 0, shared across all layers)
    rel_key = "text_encoder.encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    if rel_key in tensors:
        mapped.append(("t5enc.rel_bias.weight", to_f32(tensors[rel_key])))

    # Final layer norm
    fn_key = "text_encoder.encoder.final_layer_norm.weight"
    if fn_key in tensors:
        mapped.append(("t5enc.final_rms.weight", to_f16(tensors[fn_key])))

    # Encoder layers
    layer_re = re.compile(r"text_encoder\.encoder\.block\.(\d+)\.(.*)")
    layer_tensors: dict[int, dict[str, np.ndarray]] = {}
    for name, data in tensors.items():
        m = layer_re.match(name)
        if not m:
            continue
        layer_idx = int(m.group(1))
        suffix = m.group(2)
        if layer_idx not in layer_tensors:
            layer_tensors[layer_idx] = {}
        layer_tensors[layer_idx][suffix] = data

    for layer_idx in sorted(layer_tensors.keys()):
        lt = layer_tensors[layer_idx]
        pfx = f"t5enc.blk.{layer_idx}"

        # Self-attention
        sa_map = {
            "layer.0.SelfAttention.q.weight": f"{pfx}.attn_q.weight",
            "layer.0.SelfAttention.k.weight": f"{pfx}.attn_k.weight",
            "layer.0.SelfAttention.v.weight": f"{pfx}.attn_v.weight",
            "layer.0.SelfAttention.o.weight": f"{pfx}.attn_o.weight",
            "layer.0.layer_norm.weight": f"{pfx}.attn_rms.weight",
        }
        for src, dst in sa_map.items():
            if src in lt:
                mapped.append((dst, to_f16(lt[src])))

        # FFN: gated-gelu has wi_0 (gate), wi_1 (up), wo (down)
        ffn_map = {
            "layer.1.DenseReluDense.wi_0.weight": f"{pfx}.ffn_gate.weight",
            "layer.1.DenseReluDense.wi_1.weight": f"{pfx}.ffn_up.weight",
            "layer.1.DenseReluDense.wo.weight": f"{pfx}.ffn_down.weight",
            "layer.1.layer_norm.weight": f"{pfx}.ffn_rms.weight",
        }
        for src, dst in ffn_map.items():
            if src in lt:
                mapped.append((dst, to_f16(lt[src])))

    return mapped


# ---------------------------------------------------------------------------
# Decoder tensor mapping
# ---------------------------------------------------------------------------


def map_decoder_tensors(tensors: dict, config: dict) -> list[tuple[str, np.ndarray]]:
    """Map HuggingFace Parler decoder tensors to GGUF naming."""
    mapped = []
    num_codebooks = config["decoder"]["num_codebooks"]

    # Per-codebook embeddings (embed_tokens.0 .. embed_tokens.8)
    for i in range(num_codebooks):
        key = f"decoder.model.decoder.embed_tokens.{i}.weight"
        if key in tensors:
            mapped.append((f"dec.embed.{i}.weight", to_f16(tensors[key])))

    # Prompt embedding (for the text prompt tokens)
    prompt_key = "embed_prompts.weight"
    if prompt_key in tensors:
        mapped.append(("dec.embed_prompts.weight", to_f16(tensors[prompt_key])))

    # Sinusoidal positional embedding
    pos_key = "decoder.model.decoder.embed_positions.weights"
    if pos_key not in tensors:
        pos_key = "decoder.model.decoder.embed_positions.weight"
    if pos_key in tensors:
        mapped.append(("dec.pos_embed.weight", to_f32(tensors[pos_key])))

    # Final layer norm
    for name_pair in [
        ("decoder.model.decoder.layer_norm.weight", "dec.final_norm.weight"),
        ("decoder.model.decoder.layer_norm.bias", "dec.final_norm.bias"),
    ]:
        if name_pair[0] in tensors:
            mapped.append((name_pair[1], to_f16(tensors[name_pair[0]])))

    # LM heads: fused or per-codebook
    use_fused = config["decoder"].get("use_fused_lm_heads", False)
    if use_fused:
        fused_key = "decoder.lm_heads.weight"
        if fused_key in tensors:
            # Fused head: (num_codebooks * vocab_size, hidden_size)
            # Split into per-codebook heads
            w = tensors[fused_key]
            vocab_size = config["decoder"]["vocab_size"]
            for i in range(num_codebooks):
                chunk = w[i * vocab_size : (i + 1) * vocab_size]
                mapped.append((f"dec.lm_head.{i}.weight", to_f16(chunk)))
    else:
        for i in range(num_codebooks):
            key = f"decoder.lm_heads.{i}.weight"
            if key in tensors:
                mapped.append((f"dec.lm_head.{i}.weight", to_f16(tensors[key])))

    # Decoder layers
    layer_re = re.compile(r"decoder\.model\.decoder\.layers\.(\d+)\.(.*)")
    layer_tensors: dict[int, dict[str, np.ndarray]] = {}
    for name, data in tensors.items():
        m = layer_re.match(name)
        if not m:
            continue
        layer_idx = int(m.group(1))
        suffix = m.group(2)
        if layer_idx not in layer_tensors:
            layer_tensors[layer_idx] = {}
        layer_tensors[layer_idx][suffix] = data

    for layer_idx in sorted(layer_tensors.keys()):
        lt = layer_tensors[layer_idx]
        pfx = f"dec.blk.{layer_idx}"

        # Self-attention
        sa_map = {
            "self_attn.q_proj.weight": f"{pfx}.self_attn_q.weight",
            "self_attn.k_proj.weight": f"{pfx}.self_attn_k.weight",
            "self_attn.v_proj.weight": f"{pfx}.self_attn_v.weight",
            "self_attn.out_proj.weight": f"{pfx}.self_attn_o.weight",
            "self_attn_layer_norm.weight": f"{pfx}.self_attn_norm.weight",
            "self_attn_layer_norm.bias": f"{pfx}.self_attn_norm.bias",
        }
        for src, dst in sa_map.items():
            if src in lt:
                mapped.append((dst, to_f16(lt[src])))

        # Cross-attention (encoder_attn)
        ca_map = {
            "encoder_attn.q_proj.weight": f"{pfx}.cross_attn_q.weight",
            "encoder_attn.k_proj.weight": f"{pfx}.cross_attn_k.weight",
            "encoder_attn.v_proj.weight": f"{pfx}.cross_attn_v.weight",
            "encoder_attn.out_proj.weight": f"{pfx}.cross_attn_o.weight",
            "encoder_attn_layer_norm.weight": f"{pfx}.cross_attn_norm.weight",
            "encoder_attn_layer_norm.bias": f"{pfx}.cross_attn_norm.bias",
        }
        for src, dst in ca_map.items():
            if src in lt:
                mapped.append((dst, to_f16(lt[src])))

        # FFN
        ffn_map = {
            "fc1.weight": f"{pfx}.fc1.weight",
            "fc2.weight": f"{pfx}.fc2.weight",
            "final_layer_norm.weight": f"{pfx}.ffn_norm.weight",
            "final_layer_norm.bias": f"{pfx}.ffn_norm.bias",
        }
        for src, dst in ffn_map.items():
            if src in lt:
                mapped.append((dst, to_f16(lt[src])))

    return mapped


# ---------------------------------------------------------------------------
# DAC tensor mapping
# ---------------------------------------------------------------------------


def map_dac_tensors(tensors: dict) -> list[tuple[str, np.ndarray]]:
    """Map HuggingFace DAC audio encoder tensors to GGUF naming.

    DAC uses weight normalization (weight_v + weight_g) which we fuse
    into a single weight tensor.
    """
    mapped = []

    # Build set of weight_norm prefixes for fusing
    wn_prefixes = set()
    for name in tensors:
        if name.endswith(".weight_v"):
            wn_prefixes.add(name[:-len("weight_v")])

    # Quantizer: codebooks + out_proj
    for name, data in tensors.items():
        if not name.startswith("audio_encoder.model.quantizer."):
            continue
        # Skip in_proj (encoder-only)
        if "in_proj" in name:
            continue
        # Skip weight_v (handled via weight_g fusing)
        if name.endswith(".weight_v"):
            continue

        suffix = name[len("audio_encoder.model.quantizer."):]

        if name.endswith(".weight_g"):
            # Fuse weight norm
            prefix = name[:-len("weight_g")]
            fused = fuse_weight_norm(tensors, prefix)
            out_name = f"dac.quant.{suffix[:-len('.weight_g')]}.weight"
            mapped.append((out_name, to_f16(fused)))
        elif (name[:-len("weight")] if name.endswith("weight") else name + "NOMATCH") not in wn_prefixes:
            # Regular tensor (not part of weight_norm pair)
            out_name = f"dac.quant.{suffix}"
            mapped.append((out_name, to_f16(data)))
        else:
            # codebook embeddings (no weight norm)
            out_name = f"dac.quant.{suffix}"
            mapped.append((out_name, to_f16(data)))

    # Decoder
    for name, data in tensors.items():
        if not name.startswith("audio_encoder.model.decoder."):
            continue
        if name.endswith(".weight_v"):
            continue

        suffix = name[len("audio_encoder.model.decoder."):]

        if name.endswith(".weight_g"):
            prefix = name[:-len("weight_g")]
            fused = fuse_weight_norm(tensors, prefix)
            out_name = f"dac.dec.{suffix[:-len('.weight_g')]}.weight"
            mapped.append((out_name, to_f16(fused)))
        elif (name[:-len("weight")] if name.endswith("weight") else "") in wn_prefixes:
            # Part of a weight_norm pair, skip (handled by weight_g branch)
            continue
        else:
            out_name = f"dac.dec.{suffix}"
            mapped.append((out_name, to_f16(data)))

    return mapped


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def load_tokenizer_vocab(model_dir: Path) -> tuple[list[str], list[float]]:
    """Load the unigram tokenizer vocabulary and scores."""
    tok_json = model_dir / "tokenizer.json"
    if not tok_json.exists():
        print("WARNING: tokenizer.json not found, skipping vocab", file=sys.stderr)
        return [], []

    with open(tok_json) as f:
        tok = json.load(f)

    model = tok.get("model", {})
    vocab_items = model.get("vocab", [])

    tokens = []
    scores = []
    for item in vocab_items:
        if isinstance(item, list) and len(item) == 2:
            token, score = item
            # Replace sentencepiece underscore with space
            token = token.replace("\u2581", " ")
            tokens.append(token)
            scores.append(float(score))
        else:
            tokens.append(str(item))
            scores.append(0.0)

    return tokens, scores


# ---------------------------------------------------------------------------
# Description tokenizer (T5 / flan-t5 sentencepiece)
# ---------------------------------------------------------------------------


def load_description_tokenizer(model_dir: Path) -> tuple[list[str], list[float]]:
    """Load the description tokenizer (T5 sentencepiece) vocab + scores.

    Parler v1.1 uses two tokenizers:
    - Main tokenizer (unigram, from LLaMA) for text prompts
    - Description tokenizer (T5 sentencepiece) for voice descriptions

    We embed both in the GGUF. The description tokenizer is used by the
    T5 encoder.
    """
    # Look for the description tokenizer in various places
    desc_tok_json = model_dir / "description_tokenizer" / "tokenizer.json"
    if not desc_tok_json.exists():
        # Might be stored alongside main tokenizer with a prefix
        desc_tok_json = model_dir / "tokenizer.json"
        # In this case, the main tokenizer IS the description tokenizer
        # for older models

    # For the T5 encoder, we load the sentencepiece model directly
    # Try spm first
    spm_path = model_dir / "description_tokenizer" / "spiece.model"
    if not spm_path.exists():
        spm_path = model_dir / "spiece.model"

    # Fall back to tokenizer.json
    if desc_tok_json.exists():
        with open(desc_tok_json) as f:
            tok = json.load(f)
        model = tok.get("model", {})
        vocab_items = model.get("vocab", [])
        tokens = []
        scores = []
        for item in vocab_items:
            if isinstance(item, list) and len(item) == 2:
                token, score = item
                token = token.replace("\u2581", " ")
                tokens.append(token)
                scores.append(float(score))
        return tokens, scores

    return [], []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Convert Parler TTS to GGUF")
    parser.add_argument("--input", type=str, required=True,
                        help="HuggingFace model ID or local directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output GGUF file path")
    parser.add_argument("--use-f32", action="store_true",
                        help="Store weights as F32 instead of F16")
    args = parser.parse_args()

    model_dir = load_model_dir(args.input)
    print(f"Model directory: {model_dir}", file=sys.stderr)

    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        sys.exit(f"config.json not found in {model_dir}")
    with open(config_path) as f:
        config = json.load(f)

    # Load generation config for max_length
    gen_config_path = model_dir / "generation_config.json"
    gen_config = {}
    if gen_config_path.exists():
        with open(gen_config_path) as f:
            gen_config = json.load(f)

    # Load all safetensor weights
    print("Loading safetensors...", file=sys.stderr)
    tensors = load_safetensors(model_dir)
    print(f"Loaded {len(tensors)} tensors", file=sys.stderr)

    # Create GGUF writer
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(output_path), "parler-tts")

    # ----- Metadata -----
    dec_cfg = config["decoder"]
    t5_cfg = config["text_encoder"]
    dac_cfg = config["audio_encoder"]

    writer.add_name("Parler TTS")
    writer.add_description(f"Parler TTS ({args.input})")

    # Decoder config
    writer.add_uint32("parler.decoder.hidden_size", dec_cfg["hidden_size"])
    writer.add_uint32("parler.decoder.num_layers", dec_cfg["num_hidden_layers"])
    writer.add_uint32("parler.decoder.num_heads", dec_cfg["num_attention_heads"])
    writer.add_uint32("parler.decoder.num_kv_heads",
                      dec_cfg.get("num_key_value_heads", dec_cfg["num_attention_heads"]))
    writer.add_uint32("parler.decoder.num_cross_kv_heads",
                      dec_cfg.get("num_cross_attention_key_value_heads",
                                  dec_cfg["num_attention_heads"]))
    writer.add_uint32("parler.decoder.ffn_dim", dec_cfg["ffn_dim"])
    writer.add_uint32("parler.decoder.vocab_size", dec_cfg["vocab_size"])
    writer.add_uint32("parler.decoder.num_codebooks", dec_cfg["num_codebooks"])
    writer.add_uint32("parler.decoder.max_position_embeddings",
                      dec_cfg["max_position_embeddings"])
    writer.add_uint32("parler.decoder.bos_token_id", dec_cfg["bos_token_id"])
    writer.add_uint32("parler.decoder.eos_token_id", dec_cfg["eos_token_id"])
    writer.add_uint32("parler.decoder.pad_token_id", dec_cfg["pad_token_id"])
    writer.add_bool("parler.decoder.use_fused_lm_heads",
                    dec_cfg.get("use_fused_lm_heads", False))
    writer.add_bool("parler.decoder.rope_embeddings",
                    dec_cfg.get("rope_embeddings", False))

    max_gen = gen_config.get("max_length", 2580)
    writer.add_uint32("parler.decoder.max_generation", max_gen)

    # T5 encoder config
    writer.add_uint32("parler.t5enc.d_model", t5_cfg["d_model"])
    writer.add_uint32("parler.t5enc.d_kv", t5_cfg["d_kv"])
    writer.add_uint32("parler.t5enc.d_ff", t5_cfg.get("d_ff", 2816))
    writer.add_uint32("parler.t5enc.n_heads", t5_cfg["num_heads"])
    writer.add_uint32("parler.t5enc.n_layers", t5_cfg["num_layers"])
    writer.add_uint32("parler.t5enc.vocab_size", t5_cfg["vocab_size"])
    writer.add_uint32("parler.t5enc.rel_attn_num_buckets",
                      t5_cfg.get("relative_attention_num_buckets", 32))
    writer.add_uint32("parler.t5enc.rel_attn_max_distance",
                      t5_cfg.get("relative_attention_max_distance", 128))
    writer.add_string("parler.t5enc.feed_forward_proj",
                      t5_cfg.get("feed_forward_proj", "gated-gelu"))

    # DAC config
    writer.add_uint32("parler.dac.n_codebooks", dac_cfg["n_codebooks"])
    writer.add_uint32("parler.dac.codebook_size", dac_cfg["codebook_size"])
    writer.add_uint32("parler.dac.codebook_dim", dac_cfg.get("codebook_dim", 8))
    writer.add_uint32("parler.dac.hidden_size", dac_cfg.get("hidden_size", 1024))
    writer.add_uint32("parler.dac.sample_rate", dac_cfg.get("sampling_rate", 44100))
    writer.add_uint32("parler.dac.hop_length", dac_cfg.get("hop_length", 512))

    # Overall
    writer.add_uint32("parler.prompt_vocab_size", config.get("vocab_size", 90714))

    # File type
    if args.use_f32:
        writer.add_file_type(gguf.GGMLQuantizationType.F32)
    else:
        writer.add_file_type(gguf.GGMLQuantizationType.F16)

    # ----- Prompt tokenizer -----
    print("Loading prompt tokenizer...", file=sys.stderr)
    tokens, scores = load_tokenizer_vocab(model_dir)
    if tokens:
        writer.add_token_list(tokens)
        writer.add_token_scores(scores)
        print(f"  prompt tokenizer: {len(tokens)} tokens", file=sys.stderr)

    # ----- Description tokenizer (T5) -----
    print("Loading description tokenizer...", file=sys.stderr)
    desc_tok_dir = model_dir / "description_tokenizer"
    if desc_tok_dir.exists():
        desc_tokens, desc_scores = load_description_tokenizer(desc_tok_dir)
        if desc_tokens:
            # Store as separate arrays with desc_ prefix
            writer.add_array("parler.desc_tokenizer.tokens",
                             desc_tokens)
            writer.add_array("parler.desc_tokenizer.scores",
                             np.array(desc_scores, dtype=np.float32))
            print(f"  description tokenizer: {len(desc_tokens)} tokens", file=sys.stderr)
    else:
        print("  no separate description tokenizer found", file=sys.stderr)

    # ----- Tensors -----

    # T5 encoder
    print("Mapping T5 encoder tensors...", file=sys.stderr)
    t5_tensors = map_t5_encoder_tensors(tensors)
    print(f"  T5 encoder: {len(t5_tensors)} tensors", file=sys.stderr)

    # Decoder
    print("Mapping decoder tensors...", file=sys.stderr)
    dec_tensors = map_decoder_tensors(tensors, config)
    print(f"  Decoder: {len(dec_tensors)} tensors", file=sys.stderr)

    # DAC
    print("Mapping DAC tensors...", file=sys.stderr)
    dac_tensors = map_dac_tensors(tensors)
    print(f"  DAC: {len(dac_tensors)} tensors", file=sys.stderr)

    # Write all tensors
    all_tensors = t5_tensors + dec_tensors + dac_tensors
    for name, data in all_tensors:
        if args.use_f32 and data.dtype == np.float16:
            data = data.astype(np.float32)
        writer.add_tensor(name, data)

    # ----- Finalize -----
    print(f"Writing {len(all_tensors)} tensors to {args.output}...", file=sys.stderr)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
