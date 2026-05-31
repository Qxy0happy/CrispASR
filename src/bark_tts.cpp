// bark_tts.cpp -- Suno Bark TTS (MIT) 3-stage hierarchical TTS backend.
//
// Architecture:
//   Stage 1 (text/semantic): GPT-2 causal transformer, BERT-tokenized text +
//     optional speaker semantic history -> semantic tokens (vocab 10000).
//     Input: 256 text tokens (padded, offset by TEXT_ENCODING_OFFSET) +
//            256 semantic history tokens (padded) + SEMANTIC_INFER_TOKEN.
//     merge_context=true: first 256 text + first 256 semantic are summed
//     in embedding space, then remaining tokens appended.
//
//   Stage 2 (coarse): GPT-2 causal transformer, semantic tokens -> coarse
//     EnCodec codes (2 codebooks, 1024 entries each, interleaved).
//     Sliding window decode with semantic context padding.
//
//   Stage 3 (fine): Non-causal (bidirectional) GPT, coarse codes -> full
//     8-codebook EnCodec codes. Multiple token embeddings (one per codebook),
//     multiple lm_heads. Predicts codebooks 2-7 iteratively.
//
//   Decode: EnCodec 24 kHz SEANet decoder converts fine codes -> PCM.
//
// Key constants from bark/generation.py:
//   SEMANTIC_VOCAB_SIZE     = 10000
//   CODEBOOK_SIZE           = 1024
//   N_COARSE_CODEBOOKS      = 2
//   N_FINE_CODEBOOKS         = 8
//   TEXT_ENCODING_OFFSET    = 10048
//   SEMANTIC_PAD_TOKEN      = 10000
//   TEXT_PAD_TOKEN           = 129595
//   SEMANTIC_INFER_TOKEN    = 129599
//   COARSE_SEMANTIC_PAD_TOKEN = 12048
//   COARSE_INFER_TOKEN      = 12050
//   SAMPLE_RATE             = 24000
//   SEMANTIC_RATE_HZ        = 49.9
//   COARSE_RATE_HZ          = 75

#include "bark_tts.h"
#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

namespace {

// Hyperparams for each GPT sub-model
struct bark_gpt_hp {
    uint32_t n_layer = 12;
    uint32_t n_head = 12;
    uint32_t n_embd = 768;
    uint32_t block_size = 1024;
    uint32_t input_vocab = 10048;
    uint32_t output_vocab = 10048;
    // Fine-model specific
    uint32_t n_codes_total = 8;
    uint32_t n_codes_given = 1;
};

// One transformer layer (causal or non-causal -- same weight layout)
struct bark_gpt_layer {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_norm_b = nullptr;
    ggml_tensor* attn_qkv_w = nullptr; // fused Q+K+V projection
    ggml_tensor* attn_qkv_b = nullptr;
    ggml_tensor* attn_out_w = nullptr;
    ggml_tensor* attn_out_b = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_norm_b = nullptr;
    ggml_tensor* ffn_up_w = nullptr; // c_fc (d -> 4d)
    ggml_tensor* ffn_up_b = nullptr;
    ggml_tensor* ffn_down_w = nullptr; // c_proj (4d -> d)
    ggml_tensor* ffn_down_b = nullptr;
};

// Text and Coarse sub-models share the same GPT structure
struct bark_gpt_model {
    ggml_tensor* token_embd = nullptr; // (input_vocab, n_embd)
    ggml_tensor* pos_embd = nullptr;   // (block_size, n_embd)
    std::vector<bark_gpt_layer> layers;
    ggml_tensor* output_norm_w = nullptr;
    ggml_tensor* output_norm_b = nullptr;
    ggml_tensor* output_w = nullptr; // lm_head
};

// Fine model has multiple embeddings and heads
struct bark_fine_model {
    std::vector<ggml_tensor*> token_embds; // [n_codes_total] embeddings
    ggml_tensor* pos_embd = nullptr;
    std::vector<bark_gpt_layer> layers;
    ggml_tensor* output_norm_w = nullptr;
    ggml_tensor* output_norm_b = nullptr;
    std::vector<ggml_tensor*> output_heads; // [n_codes_total - n_codes_given] lm heads
};

// Pipeline-level constants
struct bark_pipeline_params {
    uint32_t sample_rate = 24000;
    uint32_t semantic_vocab_size = 10000;
    uint32_t codebook_size = 1024;
    uint32_t n_coarse_codebooks = 2;
    uint32_t n_fine_codebooks = 8;
    uint32_t text_encoding_offset = 10048;
    uint32_t semantic_pad_token = 10000;
    uint32_t text_pad_token = 129595;
    uint32_t semantic_infer_token = 129599;
    uint32_t coarse_semantic_pad_token = 12048;
    uint32_t coarse_infer_token = 12050;
};

// Speaker prompt (loaded from .npz)
struct bark_speaker_prompt {
    std::vector<int32_t> semantic_prompt;
    std::vector<int32_t> coarse_prompt; // flattened (n_coarse_codebooks, T)
    int coarse_prompt_cols = 0;         // T dimension
    std::vector<int32_t> fine_prompt;   // flattened (n_fine_codebooks, T)
    int fine_prompt_cols = 0;
    bool loaded = false;
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct bark_context {
    bark_context_params params{};
    int n_threads = 4;

    bark_gpt_hp text_hp;
    bark_gpt_hp coarse_hp;
    bark_gpt_hp fine_hp;
    bark_pipeline_params pp;

    bark_gpt_model text_model;
    bark_gpt_model coarse_model;
    bark_fine_model fine_model;

    bark_speaker_prompt speaker;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    std::mt19937_64 rng;

    ~bark_context() {
        if (sched)
            ggml_backend_sched_free(sched);
        if (ctx_w)
            ggml_free(ctx_w);
        if (buf_w)
            ggml_backend_buffer_free(buf_w);
        if (backend && backend != backend_cpu)
            ggml_backend_free(backend);
        if (backend_cpu)
            ggml_backend_free(backend_cpu);
    }
};

// ---------------------------------------------------------------------------
// Tensor binding helpers
// ---------------------------------------------------------------------------

namespace {

static ggml_tensor* get_tensor(bark_context* c, const char* name) {
    auto it = c->tensors.find(name);
    if (it == c->tensors.end())
        return nullptr;
    return it->second;
}

static ggml_tensor* require_tensor(bark_context* c, const char* name) {
    ggml_tensor* t = get_tensor(c, name);
    if (!t) {
        fprintf(stderr, "bark: missing tensor '%s'\n", name);
    }
    return t;
}

// Bind a causal GPT model (text or coarse)
static bool bind_gpt_model(bark_context* c, bark_gpt_model& m, const bark_gpt_hp& hp, const char* prefix) {
    char key[128];
    auto fmt = [&](const char* suffix) {
        std::snprintf(key, sizeof(key), "%s.%s", prefix, suffix);
        return key;
    };

    m.token_embd = require_tensor(c, fmt("token_embd.weight"));
    m.pos_embd = require_tensor(c, fmt("pos_embd.weight"));
    m.output_norm_w = require_tensor(c, fmt("output_norm.weight"));
    m.output_norm_b = get_tensor(c, fmt("output_norm.bias"));
    m.output_w = get_tensor(c, fmt("output.weight"));

    // If output head not present, tie to token_embd
    if (!m.output_w)
        m.output_w = m.token_embd;

    if (!m.token_embd || !m.pos_embd || !m.output_norm_w)
        return false;

    m.layers.resize(hp.n_layer);
    for (uint32_t i = 0; i < hp.n_layer; i++) {
        auto& l = m.layers[i];
        auto lkey = [&](const char* s) {
            std::snprintf(key, sizeof(key), "%s.blk.%u.%s", prefix, i, s);
            return key;
        };
        l.attn_norm_w = require_tensor(c, lkey("attn_norm.weight"));
        l.attn_norm_b = get_tensor(c, lkey("attn_norm.bias"));
        l.attn_qkv_w = require_tensor(c, lkey("attn_qkv.weight"));
        l.attn_qkv_b = get_tensor(c, lkey("attn_qkv.bias"));
        l.attn_out_w = require_tensor(c, lkey("attn_output.weight"));
        l.attn_out_b = get_tensor(c, lkey("attn_output.bias"));
        l.ffn_norm_w = require_tensor(c, lkey("ffn_norm.weight"));
        l.ffn_norm_b = get_tensor(c, lkey("ffn_norm.bias"));
        l.ffn_up_w = require_tensor(c, lkey("ffn_up.weight"));
        l.ffn_up_b = get_tensor(c, lkey("ffn_up.bias"));
        l.ffn_down_w = require_tensor(c, lkey("ffn_down.weight"));
        l.ffn_down_b = get_tensor(c, lkey("ffn_down.bias"));

        if (!l.attn_norm_w || !l.attn_qkv_w || !l.attn_out_w || !l.ffn_norm_w || !l.ffn_up_w || !l.ffn_down_w) {
            fprintf(stderr, "bark: missing tensor in %s layer %u\n", prefix, i);
            return false;
        }
    }
    return true;
}

// Bind the fine model (multiple embeddings + heads)
static bool bind_fine_model(bark_context* c, bark_fine_model& m, const bark_gpt_hp& hp) {
    char key[128];
    m.pos_embd = require_tensor(c, "fine.pos_embd.weight");
    m.output_norm_w = require_tensor(c, "fine.output_norm.weight");
    m.output_norm_b = get_tensor(c, "fine.output_norm.bias");

    if (!m.pos_embd || !m.output_norm_w)
        return false;

    // Multiple token embeddings
    m.token_embds.resize(hp.n_codes_total);
    for (uint32_t i = 0; i < hp.n_codes_total; i++) {
        std::snprintf(key, sizeof(key), "fine.token_embd.%u.weight", i);
        m.token_embds[i] = require_tensor(c, key);
        if (!m.token_embds[i])
            return false;
    }

    // Multiple output heads (for codebooks n_codes_given..n_codes_total-1)
    uint32_t n_heads = hp.n_codes_total - hp.n_codes_given;
    m.output_heads.resize(n_heads);
    for (uint32_t i = 0; i < n_heads; i++) {
        std::snprintf(key, sizeof(key), "fine.output.%u.weight", i);
        m.output_heads[i] = require_tensor(c, key);
        if (!m.output_heads[i])
            return false;
    }

    // Layers
    m.layers.resize(hp.n_layer);
    for (uint32_t i = 0; i < hp.n_layer; i++) {
        auto& l = m.layers[i];
        auto lkey = [&](const char* s) {
            std::snprintf(key, sizeof(key), "fine.blk.%u.%s", i, s);
            return key;
        };
        l.attn_norm_w = require_tensor(c, lkey("attn_norm.weight"));
        l.attn_norm_b = get_tensor(c, lkey("attn_norm.bias"));
        l.attn_qkv_w = require_tensor(c, lkey("attn_qkv.weight"));
        l.attn_qkv_b = get_tensor(c, lkey("attn_qkv.bias"));
        l.attn_out_w = require_tensor(c, lkey("attn_output.weight"));
        l.attn_out_b = get_tensor(c, lkey("attn_output.bias"));
        l.ffn_norm_w = require_tensor(c, lkey("ffn_norm.weight"));
        l.ffn_norm_b = get_tensor(c, lkey("ffn_norm.bias"));
        l.ffn_up_w = require_tensor(c, lkey("ffn_up.weight"));
        l.ffn_up_b = get_tensor(c, lkey("ffn_up.bias"));
        l.ffn_down_w = require_tensor(c, lkey("ffn_down.weight"));
        l.ffn_down_b = get_tensor(c, lkey("ffn_down.bias"));

        if (!l.attn_norm_w || !l.attn_qkv_w || !l.attn_out_w || !l.ffn_norm_w || !l.ffn_up_w || !l.ffn_down_w) {
            fprintf(stderr, "bark: missing tensor in fine layer %u\n", i);
            return false;
        }
    }
    return true;
}

// Load metadata from GGUF KV pairs
static void load_metadata(bark_context* c, gguf_context* g) {
    auto& th = c->text_hp;
    th.n_layer = core_gguf::kv_u32(g, "bark.text.n_layer", th.n_layer);
    th.n_head = core_gguf::kv_u32(g, "bark.text.n_head", th.n_head);
    th.n_embd = core_gguf::kv_u32(g, "bark.text.n_embd", th.n_embd);
    th.block_size = core_gguf::kv_u32(g, "bark.text.block_size", th.block_size);
    th.input_vocab = core_gguf::kv_u32(g, "bark.text.input_vocab_size", th.input_vocab);
    th.output_vocab = core_gguf::kv_u32(g, "bark.text.output_vocab_size", th.output_vocab);

    auto& ch = c->coarse_hp;
    ch.n_layer = core_gguf::kv_u32(g, "bark.coarse.n_layer", ch.n_layer);
    ch.n_head = core_gguf::kv_u32(g, "bark.coarse.n_head", ch.n_head);
    ch.n_embd = core_gguf::kv_u32(g, "bark.coarse.n_embd", ch.n_embd);
    ch.block_size = core_gguf::kv_u32(g, "bark.coarse.block_size", ch.block_size);
    ch.input_vocab = core_gguf::kv_u32(g, "bark.coarse.input_vocab_size", ch.input_vocab);
    ch.output_vocab = core_gguf::kv_u32(g, "bark.coarse.output_vocab_size", ch.output_vocab);

    auto& fh = c->fine_hp;
    fh.n_layer = core_gguf::kv_u32(g, "bark.fine.n_layer", fh.n_layer);
    fh.n_head = core_gguf::kv_u32(g, "bark.fine.n_head", fh.n_head);
    fh.n_embd = core_gguf::kv_u32(g, "bark.fine.n_embd", fh.n_embd);
    fh.block_size = core_gguf::kv_u32(g, "bark.fine.block_size", fh.block_size);
    fh.input_vocab = core_gguf::kv_u32(g, "bark.fine.input_vocab_size", fh.input_vocab);
    fh.output_vocab = core_gguf::kv_u32(g, "bark.fine.output_vocab_size", fh.output_vocab);
    fh.n_codes_total = core_gguf::kv_u32(g, "bark.fine.n_codes_total", fh.n_codes_total);
    fh.n_codes_given = core_gguf::kv_u32(g, "bark.fine.n_codes_given", fh.n_codes_given);

    auto& pp = c->pp;
    pp.sample_rate = core_gguf::kv_u32(g, "bark.sample_rate", pp.sample_rate);
    pp.semantic_vocab_size = core_gguf::kv_u32(g, "bark.semantic_vocab_size", pp.semantic_vocab_size);
    pp.codebook_size = core_gguf::kv_u32(g, "bark.codebook_size", pp.codebook_size);
    pp.n_coarse_codebooks = core_gguf::kv_u32(g, "bark.n_coarse_codebooks", pp.n_coarse_codebooks);
    pp.n_fine_codebooks = core_gguf::kv_u32(g, "bark.n_fine_codebooks", pp.n_fine_codebooks);
    pp.text_encoding_offset = core_gguf::kv_u32(g, "bark.text_encoding_offset", pp.text_encoding_offset);
    pp.semantic_pad_token = core_gguf::kv_u32(g, "bark.semantic_pad_token", pp.semantic_pad_token);
    pp.text_pad_token = core_gguf::kv_u32(g, "bark.text_pad_token", pp.text_pad_token);
    pp.semantic_infer_token = core_gguf::kv_u32(g, "bark.semantic_infer_token", pp.semantic_infer_token);
    pp.coarse_semantic_pad_token = core_gguf::kv_u32(g, "bark.coarse_semantic_pad_token", pp.coarse_semantic_pad_token);
    pp.coarse_infer_token = core_gguf::kv_u32(g, "bark.coarse_infer_token", pp.coarse_infer_token);
}

// ---------------------------------------------------------------------------
// GPT-2 forward pass (shared by text and coarse models)
// ---------------------------------------------------------------------------

// Build a ggml graph for one GPT-2 forward pass.
// For text model: merge_context=true on first call (sums first 256+256 embeddings).
// Returns logits for the last position.
//
// This is a skeleton -- full graph building requires the ggml compute
// infrastructure (ggml_new_graph, add ops, schedule, compute). The
// pattern follows orpheus.cpp's run_talker_kv.
//
// TODO(#134): implement full ggml graph for GPT-2 forward.

static int sample_from_logits(const float* logits, int vocab_size, float temperature, std::mt19937_64& rng) {
    if (temperature <= 0.0f) {
        // Greedy
        int best = 0;
        float mx = logits[0];
        for (int k = 1; k < vocab_size; k++) {
            if (logits[k] > mx) {
                mx = logits[k];
                best = k;
            }
        }
        return best;
    }
    // Temperature sampling
    float inv_t = 1.0f / temperature;
    float mx = logits[0] * inv_t;
    for (int k = 1; k < vocab_size; k++) {
        float s = logits[k] * inv_t;
        if (s > mx)
            mx = s;
    }
    std::vector<double> probs((size_t)vocab_size);
    double sum = 0.0;
    for (int k = 0; k < vocab_size; k++) {
        double e = std::exp((double)(logits[k] * inv_t - mx));
        probs[(size_t)k] = e;
        sum += e;
    }
    if (sum <= 0.0) {
        // fallback to greedy
        int best = 0;
        float mxf = logits[0];
        for (int k = 1; k < vocab_size; k++) {
            if (logits[k] > mxf) {
                mxf = logits[k];
                best = k;
            }
        }
        return best;
    }
    std::uniform_real_distribution<double> unif(0.0, sum);
    double r = unif(rng);
    double acc = 0.0;
    for (int k = 0; k < vocab_size; k++) {
        acc += probs[(size_t)k];
        if (r <= acc)
            return k;
    }
    return vocab_size - 1;
}

// ---------------------------------------------------------------------------
// Stage 1: Generate semantic tokens from text
// ---------------------------------------------------------------------------

// TODO(#134): Implement BERT tokenizer (or embed vocab in GGUF).
// For now, this is a stub that documents the pipeline.
static std::vector<int32_t> generate_text_semantic(bark_context* ctx, const char* text) {
    (void)text; // TODO: tokenize with BERT

    auto& pp = ctx->pp;
    int max_steps = ctx->params.max_semantic_tokens > 0 ? ctx->params.max_semantic_tokens : 768;

    // The semantic generation loop:
    //   1. Tokenize text with BERT, add TEXT_ENCODING_OFFSET
    //   2. Pad to 256 tokens with TEXT_PAD_TOKEN
    //   3. Prepare semantic history (from speaker prompt or all-PAD)
    //   4. Concatenate: [text_tokens(256) | semantic_history(256) | SEMANTIC_INFER_TOKEN]
    //   5. merge_context=true on first forward: sum embeddings of text[0:256] + sem_hist[0:256]
    //   6. AR decode up to max_steps, sampling from logits[0:SEMANTIC_VOCAB_SIZE]
    //   7. Stop on EOS (token == SEMANTIC_VOCAB_SIZE) or min_eos_p >= 0.2

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "bark: stage 1 (semantic) - max_steps=%d\n", max_steps);
    }

    // Placeholder: return empty for now
    // Full implementation requires ggml graph building for the GPT-2 forward.
    std::vector<int32_t> out;

    (void)pp;
    (void)max_steps;

    return out;
}

// ---------------------------------------------------------------------------
// Stage 2: Generate coarse codes from semantic tokens
// ---------------------------------------------------------------------------

static std::vector<int32_t> generate_coarse(bark_context* ctx, const std::vector<int32_t>& semantic_tokens) {
    auto& pp = ctx->pp;

    // semantic_to_coarse_ratio = COARSE_RATE_HZ / SEMANTIC_RATE_HZ * N_COARSE_CODEBOOKS
    //                          = 75.0 / 49.9 * 2 = ~3.006
    constexpr double SEMANTIC_RATE_HZ = 49.9;
    constexpr double COARSE_RATE_HZ = 75.0;
    double semantic_to_coarse_ratio = COARSE_RATE_HZ / SEMANTIC_RATE_HZ * pp.n_coarse_codebooks;

    int n_steps = (int)(std::floor((double)semantic_tokens.size() * semantic_to_coarse_ratio / pp.n_coarse_codebooks) *
                        pp.n_coarse_codebooks);
    if (n_steps <= 0 || n_steps % (int)pp.n_coarse_codebooks != 0) {
        return {};
    }

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "bark: stage 2 (coarse) - %d semantic -> %d coarse steps\n", (int)semantic_tokens.size(),
                n_steps);
    }

    // The coarse generation loop:
    //   1. Prepare semantic input with optional history
    //   2. Sliding window of 60 steps at a time
    //   3. For each window:
    //      a. Pad semantic context to 256 with COARSE_SEMANTIC_PAD_TOKEN
    //      b. Append COARSE_INFER_TOKEN + coarse history
    //      c. AR decode, alternating codebook 0/1 (even/odd steps)
    //      d. logit slice: SEMANTIC_VOCAB_SIZE + cb_offset : +CODEBOOK_SIZE
    //   4. Output: interleaved codes, then reshape to (2, T)

    // Placeholder
    std::vector<int32_t> out;
    (void)semantic_to_coarse_ratio;
    (void)n_steps;
    return out;
}

// ---------------------------------------------------------------------------
// Stage 3: Generate fine codes from coarse codes
// ---------------------------------------------------------------------------

static std::vector<int32_t> generate_fine(bark_context* ctx, const int32_t* coarse_codes, int n_coarse_codebooks,
                                          int n_timesteps) {
    auto& pp = ctx->pp;
    auto& fh = ctx->fine_hp;

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "bark: stage 3 (fine) - %d timesteps, %d->%d codebooks\n", n_timesteps, n_coarse_codebooks,
                (int)pp.n_fine_codebooks);
    }

    // The fine generation loop:
    //   1. Build input array (n_fine_codebooks, T) -- coarse codes in top rows,
    //      padding (CODEBOOK_SIZE) in remaining rows
    //   2. Optionally prepend fine history (max 512 timesteps)
    //   3. Pad to at least 1024 timesteps
    //   4. Sliding window of 1024, step 512:
    //      For each codebook nn in [n_coarse..n_fine-1]:
    //        a. Sum embeddings of codebooks 0..nn -> input
    //        b. Non-causal forward (full attention, no mask)
    //        c. Sample from logits -> fill codebook nn
    //   5. Output: (n_fine_codebooks, T) array

    // Placeholder: copy coarse codes, fill fine with zeros
    int n_fine = (int)pp.n_fine_codebooks;
    std::vector<int32_t> out((size_t)(n_fine * n_timesteps), 0);
    for (int cb = 0; cb < n_coarse_codebooks && cb < n_fine; cb++) {
        for (int t = 0; t < n_timesteps; t++) {
            out[(size_t)(cb * n_timesteps + t)] = coarse_codes[cb * n_timesteps + t];
        }
    }

    (void)fh;
    return out;
}

// ---------------------------------------------------------------------------
// EnCodec decoder stub
// ---------------------------------------------------------------------------

// TODO(#134): Implement SEANet decoder (Conv1d + LSTM + ConvTranspose1d stack).
// The EnCodec 24 kHz decoder has:
//   - Quantizer dequantize: 8 codebooks, each (1024, 128), sum embeddings
//   - Conv1d(128, 512, k=7, p=3)
//   - LSTM(512, 512, 2 layers)
//   - 4x upsample blocks: ELU + ConvTranspose1d + ResBlock
//     ratios=[8,5,4,2], filters double each stage
//   - Final: ELU + Conv1d(32, 1, k=7, p=3)
//
// Similar to SNAC decoder (orpheus_snac.cpp) but with:
//   - ELU activation instead of Snake1d
//   - LSTM layers
//   - Different stride/filter pattern
//   - Weight norm (pre-folded at export time)
//   - More codebooks (8 vs 3)

static std::vector<float> encodec_decode(bark_context* ctx, const int32_t* fine_tokens, int n_codebooks,
                                         int n_timesteps) {
    (void)ctx;
    (void)fine_tokens;

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "bark: EnCodec decode - %d codebooks x %d timesteps\n", n_codebooks, n_timesteps);
    }

    // Placeholder: return silence
    // hop_length = product of ratios = 8*5*4*2 = 320
    int hop_length = 320;
    int n_samples = n_timesteps * hop_length;
    return std::vector<float>((size_t)n_samples, 0.0f);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public C ABI
// ---------------------------------------------------------------------------

extern "C" {

struct bark_context_params bark_context_default_params(void) {
    bark_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.temperature_semantic = 0.7f;
    p.temperature_coarse = 0.7f;
    p.temperature_fine = 0.5f;
    p.seed = 0;
    p.max_semantic_tokens = 0;
    p.flash_attn = false;
    return p;
}

struct bark_context* bark_init_from_file(const char* path_model, struct bark_context_params params) {
    if (!path_model)
        return nullptr;

    bark_context* ctx = new bark_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    // Seed RNG
    uint64_t seed = params.seed;
    if (seed == 0) {
        seed = (uint64_t)std::chrono::system_clock::now().time_since_epoch().count();
    }
    ctx->rng.seed(seed);

    // Backend setup
    ctx->backend_cpu = ggml_backend_cpu_init();
    ctx->backend = ctx->backend_cpu; // CPU-only for now

    // Load GGUF
    core_gguf::LoadResult lr = core_gguf::load_gguf(path_model, ctx->backend_cpu, ctx->backend);
    if (!lr.ok) {
        fprintf(stderr, "bark: failed to load '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }

    ctx->ctx_w = lr.ctx_w;
    ctx->buf_w = lr.buf_w;
    ctx->tensors = std::move(lr.tensors);

    // Load metadata
    gguf_context* g = gguf_init_from_file(path_model, {/*.no_alloc=*/true, /*.ctx=*/nullptr});
    if (g) {
        load_metadata(ctx, g);
        gguf_free(g);
    }

    // Bind models
    if (!bind_gpt_model(ctx, ctx->text_model, ctx->text_hp, "text")) {
        fprintf(stderr, "bark: failed to bind text model\n");
        delete ctx;
        return nullptr;
    }

    if (!bind_gpt_model(ctx, ctx->coarse_model, ctx->coarse_hp, "coarse")) {
        fprintf(stderr, "bark: failed to bind coarse model\n");
        delete ctx;
        return nullptr;
    }

    if (!bind_fine_model(ctx, ctx->fine_model, ctx->fine_hp)) {
        fprintf(stderr, "bark: failed to bind fine model\n");
        delete ctx;
        return nullptr;
    }

    // Setup scheduler
    ggml_backend_t backends[] = {ctx->backend};
    ctx->sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false);

    if (params.verbosity >= 1) {
        fprintf(stderr, "bark: loaded from '%s'\n", path_model);
        fprintf(stderr, "bark: text   %uL %ud %uh vocab=%u/%u\n", ctx->text_hp.n_layer, ctx->text_hp.n_embd,
                ctx->text_hp.n_head, ctx->text_hp.input_vocab, ctx->text_hp.output_vocab);
        fprintf(stderr, "bark: coarse %uL %ud %uh vocab=%u/%u\n", ctx->coarse_hp.n_layer, ctx->coarse_hp.n_embd,
                ctx->coarse_hp.n_head, ctx->coarse_hp.input_vocab, ctx->coarse_hp.output_vocab);
        fprintf(stderr, "bark: fine   %uL %ud %uh codes=%u/%u\n", ctx->fine_hp.n_layer, ctx->fine_hp.n_embd,
                ctx->fine_hp.n_head, ctx->fine_hp.n_codes_total, ctx->fine_hp.n_codes_given);
    }

    return ctx;
}

uint32_t bark_sample_rate(const struct bark_context* ctx) {
    return ctx ? ctx->pp.sample_rate : 24000;
}

int bark_set_speaker_npz(struct bark_context* ctx, const char* npz_path) {
    if (!ctx || !npz_path)
        return -1;

    // TODO(#134): Load .npz file and populate ctx->speaker
    // The .npz contains:
    //   "semantic_prompt" -> int64 array
    //   "coarse_prompt"   -> int64 array (2, T)
    //   "fine_prompt"     -> int64 array (8, T)
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "bark: loading speaker from '%s'\n", npz_path);
    }

    ctx->speaker.loaded = false; // Not yet implemented
    return -1;                   // TODO
}

void bark_clear_speaker(struct bark_context* ctx) {
    if (ctx) {
        ctx->speaker = bark_speaker_prompt{};
    }
}

float* bark_synthesize(struct bark_context* ctx, const char* text, int* out_n_samples) {
    if (!ctx || !text || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;

    // Stage 1: text -> semantic tokens
    std::vector<int32_t> semantic = generate_text_semantic(ctx, text);
    if (semantic.empty()) {
        fprintf(stderr, "bark: stage 1 produced no semantic tokens (not yet implemented)\n");
        return nullptr;
    }

    // Stage 2: semantic -> coarse codes (2 codebooks)
    std::vector<int32_t> coarse = generate_coarse(ctx, semantic);
    if (coarse.empty()) {
        fprintf(stderr, "bark: stage 2 produced no coarse codes\n");
        return nullptr;
    }

    // coarse is interleaved: reshape to (2, T)
    int n_coarse_timesteps = (int)coarse.size() / (int)ctx->pp.n_coarse_codebooks;

    // Stage 3: coarse -> fine codes (8 codebooks)
    std::vector<int32_t> fine = generate_fine(ctx, coarse.data(), (int)ctx->pp.n_coarse_codebooks, n_coarse_timesteps);
    if (fine.empty()) {
        fprintf(stderr, "bark: stage 3 produced no fine codes\n");
        return nullptr;
    }

    // Decode: fine codes -> PCM
    int n_fine_timesteps = (int)fine.size() / (int)ctx->pp.n_fine_codebooks;
    std::vector<float> pcm = encodec_decode(ctx, fine.data(), (int)ctx->pp.n_fine_codebooks, n_fine_timesteps);
    if (pcm.empty()) {
        fprintf(stderr, "bark: EnCodec decode produced no audio\n");
        return nullptr;
    }

    // Copy to caller-owned buffer
    float* out = (float*)malloc(pcm.size() * sizeof(float));
    if (!out)
        return nullptr;
    std::memcpy(out, pcm.data(), pcm.size() * sizeof(float));
    *out_n_samples = (int)pcm.size();
    return out;
}

void bark_pcm_free(float* pcm) {
    free(pcm);
}

void bark_free(struct bark_context* ctx) {
    delete ctx;
}

void bark_set_n_threads(struct bark_context* ctx, int n) {
    if (ctx)
        ctx->n_threads = n > 0 ? n : 1;
}

void bark_set_temperature_semantic(struct bark_context* ctx, float t) {
    if (ctx)
        ctx->params.temperature_semantic = t;
}

void bark_set_temperature_coarse(struct bark_context* ctx, float t) {
    if (ctx)
        ctx->params.temperature_coarse = t;
}

void bark_set_temperature_fine(struct bark_context* ctx, float t) {
    if (ctx)
        ctx->params.temperature_fine = t;
}

void bark_set_seed(struct bark_context* ctx, uint64_t seed) {
    if (!ctx)
        return;
    ctx->params.seed = seed;
    if (seed == 0) {
        seed = (uint64_t)std::chrono::system_clock::now().time_since_epoch().count();
    }
    ctx->rng.seed(seed);
}

} // extern "C"
