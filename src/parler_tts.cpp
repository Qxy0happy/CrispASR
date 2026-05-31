// parler_tts.cpp -- Parler TTS runtime (T5 encoder + MusicGen decoder + DAC).
//
// Architecture (parler-tts-mini-v1.1):
//
//   T5 ENCODER (flan-t5-large, encoder-only):
//     24 layers, d_model=1024, d_kv=64, n_heads=16, d_ff=2816
//     Gated-GELU FFN (gate + up -> down), RMS norm, relative position bias
//     Encodes voice description -> (T_desc, 1024) hidden states
//
//   DECODER (MusicGen-style causal transformer):
//     24 layers, hidden=1024, n_heads=16, ffn_dim=4096
//     Self-attention + cross-attention on T5 encoder output
//     LayerNorm with bias (not RMS), sinusoidal positional embeddings
//     GELU activation in FFN
//     9 codebooks, vocab_size=1088 (1024 audio + pad(1024) + bos(1025) + extras)
//     Delay pattern: codebook k is delayed by k steps
//
//   DAC 44 kHz CODEC:
//     9 codebooks x 1024 entries, upsampling 512x -> 44.1 kHz
//     Reuses src/core/dac_decoder.h
//
// Flow:
//   1. T5 encode description -> enc_hidden (cached, run once per voice)
//   2. Tokenize text prompt -> prompt_ids
//   3. Embed prompt_ids via embed_prompts + positional embedding
//   4. AR decode: for each step, cross-attend to enc_hidden, self-attend
//      to past, produce 9 codebook logits, sample, apply delay pattern
//   5. Un-delay the codebook tokens
//   6. DAC decode tokens -> 44.1 kHz PCM

#include "parler_tts.h"
#include "core/dac_decoder.h"
#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

// ── Configuration ───────────────────────────────────────────────────

struct parler_t5_config {
    int d_model = 1024;
    int d_kv = 64; // per-head dim
    int d_ff = 2816;
    int n_heads = 16;
    int n_layers = 24;
    int vocab_size = 32128;
    int rel_attn_num_buckets = 32;
    int rel_attn_max_dist = 128;
    float layer_norm_eps = 1e-6f;
    bool is_gated_gelu = true;
};

struct parler_decoder_config {
    int hidden_size = 1024;
    int num_layers = 24;
    int num_heads = 16;
    int num_kv_heads = 16;
    int num_cross_kv_heads = 16;
    int ffn_dim = 4096;
    int vocab_size = 1088;
    int num_codebooks = 9;
    int max_position_embeddings = 4096;
    int bos_token_id = 1025;
    int eos_token_id = 1024;
    int pad_token_id = 1024;
    int max_generation = 2580;
    bool use_fused_lm_heads = true;
    bool rope_embeddings = false;
    float layer_norm_eps = 1e-5f;
};

struct parler_dac_config {
    int n_codebooks = 9;
    int codebook_size = 1024;
    int codebook_dim = 8;
    int hidden_size = 1024;
    int sample_rate = 44100;
    int hop_length = 512;
};

// ── T5 encoder layer tensors ────────────────────────────────────────

struct parler_t5_layer {
    ggml_tensor* attn_q = nullptr;
    ggml_tensor* attn_k = nullptr;
    ggml_tensor* attn_v = nullptr;
    ggml_tensor* attn_o = nullptr;
    ggml_tensor* attn_rms = nullptr; // RMS norm weight
    ggml_tensor* ffn_gate = nullptr; // gated-gelu: wi_0
    ggml_tensor* ffn_up = nullptr;   // gated-gelu: wi_1
    ggml_tensor* ffn_down = nullptr; // gated-gelu: wo
    ggml_tensor* ffn_rms = nullptr;  // RMS norm weight
};

// ── Decoder layer tensors ───────────────────────────────────────────

struct parler_dec_layer {
    // Self-attention
    ggml_tensor* self_attn_q = nullptr;
    ggml_tensor* self_attn_k = nullptr;
    ggml_tensor* self_attn_v = nullptr;
    ggml_tensor* self_attn_o = nullptr;
    ggml_tensor* self_attn_norm_w = nullptr;
    ggml_tensor* self_attn_norm_b = nullptr;

    // Cross-attention
    ggml_tensor* cross_attn_q = nullptr;
    ggml_tensor* cross_attn_k = nullptr;
    ggml_tensor* cross_attn_v = nullptr;
    ggml_tensor* cross_attn_o = nullptr;
    ggml_tensor* cross_attn_norm_w = nullptr;
    ggml_tensor* cross_attn_norm_b = nullptr;

    // Cached cross-attention KV (computed once from T5 encoder output)
    ggml_tensor* cross_k_cache = nullptr;
    ggml_tensor* cross_v_cache = nullptr;

    // FFN
    ggml_tensor* fc1 = nullptr;
    ggml_tensor* fc2 = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_norm_b = nullptr;
};

// ── Model ───────────────────────────────────────────────────────────

struct parler_model {
    parler_t5_config t5_cfg;
    parler_decoder_config dec_cfg;
    parler_dac_config dac_cfg;

    // T5 encoder
    ggml_tensor* t5_embed = nullptr;
    ggml_tensor* t5_rel_bias = nullptr;
    ggml_tensor* t5_final_rms = nullptr;
    std::vector<parler_t5_layer> t5_layers;

    // Decoder
    std::vector<ggml_tensor*> dec_embeds; // [num_codebooks] per-codebook embeddings
    ggml_tensor* dec_embed_prompts = nullptr;
    ggml_tensor* dec_pos_embed = nullptr; // sinusoidal positional embedding table
    ggml_tensor* dec_final_norm_w = nullptr;
    ggml_tensor* dec_final_norm_b = nullptr;
    std::vector<parler_dec_layer> dec_layers;
    std::vector<ggml_tensor*> lm_heads; // [num_codebooks] (hidden -> vocab_size)

    // DAC
    core_dac::DacWeights dac;
};

// ── Tokenizer ───────────────────────────────────────────────────────

struct parler_tokenizer {
    std::vector<std::string> id_to_token;
    std::map<std::string, int> token_to_id;
    std::vector<float> scores;
    int eos_id = 1; // T5 default
    int unk_id = 2;
};

// ── Context ─────────────────────────────────────────────────────────

struct parler_tts_context {
    parler_tts_context_params params;
    parler_model model;

    // Prompt tokenizer (unigram)
    parler_tokenizer prompt_tokenizer;

    // Description tokenizer (T5 sentencepiece)
    parler_tokenizer desc_tokenizer;

    // GGML state
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // T5 encoder output (cached after set_description)
    std::vector<float> enc_hidden; // (T_desc, d_model) flat
    int enc_T = 0;                 // number of tokens in cached encoding
    bool enc_cached = false;

    // Cached cross-attention KV per decoder layer
    // These are computed once from enc_hidden and stored in model.dec_layers[i].cross_k/v_cache

    // Decoder self-attention KV cache
    // Flat buffers: (n_layers, max_seq, hidden_size) for k and v
    std::vector<float> kv_k;
    std::vector<float> kv_v;

    // RNG
    std::mt19937 rng;
};

// ── T5 relative position bias ───────────────────────────────────────

static int t5_relative_position_bucket(int rel_pos, bool bidirectional, int num_buckets, int max_dist) {
    int ret = 0;
    if (bidirectional) {
        num_buckets /= 2;
        if (rel_pos > 0)
            ret += num_buckets;
        rel_pos = std::abs(rel_pos);
    } else {
        rel_pos = -std::min(rel_pos, 0);
    }
    const int max_exact = num_buckets / 2;
    if (rel_pos < max_exact) {
        ret += rel_pos;
    } else {
        int val = (int)(max_exact + std::log((float)rel_pos / max_exact) / std::log((float)max_dist / max_exact) *
                                        (num_buckets - max_exact));
        val = std::min(val, num_buckets - 1);
        ret += val;
    }
    return ret;
}

// ── SentencePiece unigram tokenizer ─────────────────────────────────

static std::vector<int> tokenize_unigram(const parler_tokenizer& tok, const std::string& text) {
    // Viterbi-based unigram tokenizer with scores
    if (tok.id_to_token.empty())
        return {};

    // Prepend space for sentencepiece convention
    std::string input = " " + text;
    const int n = (int)input.size();

    // Build token -> id lookup for fast matching
    // Simple forward longest-match with fallback to unknown
    std::vector<int> ids;

    if (!tok.scores.empty()) {
        // Viterbi segmentation
        const float NEG_INF = -1e30f;
        std::vector<float> best_score(n + 1, NEG_INF);
        std::vector<int> best_len(n + 1, 0);
        best_score[0] = 0.0f;

        for (int i = 0; i < n; i++) {
            if (best_score[i] == NEG_INF)
                continue;
            // Try all possible token lengths starting at position i
            for (const auto& [token, id] : tok.token_to_id) {
                int tlen = (int)token.size();
                if (i + tlen > n)
                    continue;
                if (input.compare(i, tlen, token) == 0) {
                    float score = best_score[i] + (id < (int)tok.scores.size() ? tok.scores[id] : 0.0f);
                    if (score > best_score[i + tlen]) {
                        best_score[i + tlen] = score;
                        best_len[i + tlen] = tlen;
                    }
                }
            }
            // Fallback: single byte as unknown
            if (best_score[i + 1] == NEG_INF) {
                best_score[i + 1] = best_score[i] - 100.0f;
                best_len[i + 1] = 1;
            }
        }

        // Backtrack
        std::vector<int> lengths;
        int pos = n;
        while (pos > 0) {
            lengths.push_back(best_len[pos]);
            pos -= best_len[pos];
        }
        std::reverse(lengths.begin(), lengths.end());

        pos = 0;
        for (int len : lengths) {
            std::string piece = input.substr(pos, len);
            auto it = tok.token_to_id.find(piece);
            if (it != tok.token_to_id.end()) {
                ids.push_back(it->second);
            } else {
                ids.push_back(tok.unk_id);
            }
            pos += len;
        }
    } else {
        // Greedy longest-match fallback
        int pos = 0;
        while (pos < n) {
            int best_len_found = 0;
            int best_id = tok.unk_id;
            for (const auto& [token, id] : tok.token_to_id) {
                int tlen = (int)token.size();
                if (tlen > best_len_found && pos + tlen <= n && input.compare(pos, tlen, token) == 0) {
                    best_len_found = tlen;
                    best_id = id;
                }
            }
            if (best_len_found == 0) {
                ids.push_back(tok.unk_id);
                pos++;
            } else {
                ids.push_back(best_id);
                pos += best_len_found;
            }
        }
    }

    // Append EOS
    ids.push_back(tok.eos_id);
    return ids;
}

// ── Load from GGUF ──────────────────────────────────────────────────

static void load_metadata(parler_tts_context* c, gguf_context* g) {
    auto& m = c->model;
    auto get_u32 = [&](const char* key, int def) -> int {
        int idx = gguf_find_key(g, key);
        return (idx >= 0) ? (int)gguf_get_val_u32(g, idx) : def;
    };
    auto get_str = [&](const char* key, const char* def) -> std::string {
        int idx = gguf_find_key(g, key);
        return (idx >= 0) ? gguf_get_val_str(g, idx) : def;
    };
    auto get_bool = [&](const char* key, bool def) -> bool {
        int idx = gguf_find_key(g, key);
        return (idx >= 0) ? gguf_get_val_bool(g, idx) : def;
    };

    // T5 encoder config
    m.t5_cfg.d_model = get_u32("parler.t5enc.d_model", 1024);
    m.t5_cfg.d_kv = get_u32("parler.t5enc.d_kv", 64);
    m.t5_cfg.d_ff = get_u32("parler.t5enc.d_ff", 2816);
    m.t5_cfg.n_heads = get_u32("parler.t5enc.n_heads", 16);
    m.t5_cfg.n_layers = get_u32("parler.t5enc.n_layers", 24);
    m.t5_cfg.vocab_size = get_u32("parler.t5enc.vocab_size", 32128);
    m.t5_cfg.rel_attn_num_buckets = get_u32("parler.t5enc.rel_attn_num_buckets", 32);
    m.t5_cfg.rel_attn_max_dist = get_u32("parler.t5enc.rel_attn_max_distance", 128);
    std::string ff_proj = get_str("parler.t5enc.feed_forward_proj", "gated-gelu");
    m.t5_cfg.is_gated_gelu = (ff_proj.find("gated") != std::string::npos);

    // Decoder config
    m.dec_cfg.hidden_size = get_u32("parler.decoder.hidden_size", 1024);
    m.dec_cfg.num_layers = get_u32("parler.decoder.num_layers", 24);
    m.dec_cfg.num_heads = get_u32("parler.decoder.num_heads", 16);
    m.dec_cfg.num_kv_heads = get_u32("parler.decoder.num_kv_heads", 16);
    m.dec_cfg.num_cross_kv_heads = get_u32("parler.decoder.num_cross_kv_heads", 16);
    m.dec_cfg.ffn_dim = get_u32("parler.decoder.ffn_dim", 4096);
    m.dec_cfg.vocab_size = get_u32("parler.decoder.vocab_size", 1088);
    m.dec_cfg.num_codebooks = get_u32("parler.decoder.num_codebooks", 9);
    m.dec_cfg.max_position_embeddings = get_u32("parler.decoder.max_position_embeddings", 4096);
    m.dec_cfg.bos_token_id = get_u32("parler.decoder.bos_token_id", 1025);
    m.dec_cfg.eos_token_id = get_u32("parler.decoder.eos_token_id", 1024);
    m.dec_cfg.pad_token_id = get_u32("parler.decoder.pad_token_id", 1024);
    m.dec_cfg.max_generation = get_u32("parler.decoder.max_generation", 2580);
    m.dec_cfg.use_fused_lm_heads = get_bool("parler.decoder.use_fused_lm_heads", true);
    m.dec_cfg.rope_embeddings = get_bool("parler.decoder.rope_embeddings", false);

    // DAC config
    m.dac_cfg.n_codebooks = get_u32("parler.dac.n_codebooks", 9);
    m.dac_cfg.codebook_size = get_u32("parler.dac.codebook_size", 1024);
    m.dac_cfg.codebook_dim = get_u32("parler.dac.codebook_dim", 8);
    m.dac_cfg.hidden_size = get_u32("parler.dac.hidden_size", 1024);
    m.dac_cfg.sample_rate = get_u32("parler.dac.sample_rate", 44100);
    m.dac_cfg.hop_length = get_u32("parler.dac.hop_length", 512);

    // Load prompt tokenizer
    {
        int tidx = gguf_find_key(g, "tokenizer.ggml.tokens");
        if (tidx >= 0) {
            int n = gguf_get_arr_n(g, tidx);
            c->prompt_tokenizer.id_to_token.resize(n);
            for (int i = 0; i < n; i++) {
                c->prompt_tokenizer.id_to_token[i] = gguf_get_arr_str(g, tidx, i);
                c->prompt_tokenizer.token_to_id[c->prompt_tokenizer.id_to_token[i]] = i;
            }
        }
        int sidx = gguf_find_key(g, "tokenizer.ggml.scores");
        if (sidx >= 0) {
            int n = gguf_get_arr_n(g, sidx);
            c->prompt_tokenizer.scores.resize(n);
            const float* sp = (const float*)gguf_get_arr_data(g, sidx);
            for (int i = 0; i < n; i++)
                c->prompt_tokenizer.scores[i] = sp[i];
        }
        int eos_idx = gguf_find_key(g, "tokenizer.ggml.eos_token_id");
        if (eos_idx >= 0)
            c->prompt_tokenizer.eos_id = (int)gguf_get_val_u32(g, eos_idx);
        int unk_idx = gguf_find_key(g, "tokenizer.ggml.unknown_token_id");
        if (unk_idx >= 0)
            c->prompt_tokenizer.unk_id = (int)gguf_get_val_u32(g, unk_idx);
    }

    // Load description tokenizer (T5)
    {
        int tidx = gguf_find_key(g, "parler.desc_tokenizer.tokens");
        if (tidx >= 0) {
            int n = gguf_get_arr_n(g, tidx);
            c->desc_tokenizer.id_to_token.resize(n);
            for (int i = 0; i < n; i++) {
                c->desc_tokenizer.id_to_token[i] = gguf_get_arr_str(g, tidx, i);
                c->desc_tokenizer.token_to_id[c->desc_tokenizer.id_to_token[i]] = i;
            }
        }
        int sidx = gguf_find_key(g, "parler.desc_tokenizer.scores");
        if (sidx >= 0) {
            int n = gguf_get_arr_n(g, sidx);
            c->desc_tokenizer.scores.resize(n);
            const float* sp = (const float*)gguf_get_arr_data(g, sidx);
            for (int i = 0; i < n; i++)
                c->desc_tokenizer.scores[i] = sp[i];
        }
        // T5 EOS is typically 1 for flan-t5
        c->desc_tokenizer.eos_id = 1;
        c->desc_tokenizer.unk_id = 2;
    }
}

static bool bind_tensors(parler_tts_context* c) {
    auto& m = c->model;
    auto T = [&](const char* name) -> ggml_tensor* {
        auto it = c->tensors.find(name);
        return (it != c->tensors.end()) ? it->second : nullptr;
    };

    // T5 encoder
    m.t5_embed = T("t5enc.embed.weight");
    m.t5_rel_bias = T("t5enc.rel_bias.weight");
    m.t5_final_rms = T("t5enc.final_rms.weight");

    m.t5_layers.resize(m.t5_cfg.n_layers);
    for (int i = 0; i < m.t5_cfg.n_layers; i++) {
        char buf[128];
        auto w = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "t5enc.blk.%d.%s", i, suffix);
            return T(buf);
        };
        auto& l = m.t5_layers[i];
        l.attn_q = w("attn_q.weight");
        l.attn_k = w("attn_k.weight");
        l.attn_v = w("attn_v.weight");
        l.attn_o = w("attn_o.weight");
        l.attn_rms = w("attn_rms.weight");
        l.ffn_gate = w("ffn_gate.weight");
        l.ffn_up = w("ffn_up.weight");
        l.ffn_down = w("ffn_down.weight");
        l.ffn_rms = w("ffn_rms.weight");
    }

    // Decoder
    m.dec_embeds.resize(m.dec_cfg.num_codebooks);
    m.lm_heads.resize(m.dec_cfg.num_codebooks);
    for (int i = 0; i < m.dec_cfg.num_codebooks; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "dec.embed.%d.weight", i);
        m.dec_embeds[i] = T(buf);
        snprintf(buf, sizeof(buf), "dec.lm_head.%d.weight", i);
        m.lm_heads[i] = T(buf);
    }
    m.dec_embed_prompts = T("dec.embed_prompts.weight");
    m.dec_pos_embed = T("dec.pos_embed.weight");
    m.dec_final_norm_w = T("dec.final_norm.weight");
    m.dec_final_norm_b = T("dec.final_norm.bias");

    m.dec_layers.resize(m.dec_cfg.num_layers);
    for (int i = 0; i < m.dec_cfg.num_layers; i++) {
        char buf[128];
        auto w = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "dec.blk.%d.%s", i, suffix);
            return T(buf);
        };
        auto& l = m.dec_layers[i];
        l.self_attn_q = w("self_attn_q.weight");
        l.self_attn_k = w("self_attn_k.weight");
        l.self_attn_v = w("self_attn_v.weight");
        l.self_attn_o = w("self_attn_o.weight");
        l.self_attn_norm_w = w("self_attn_norm.weight");
        l.self_attn_norm_b = w("self_attn_norm.bias");
        l.cross_attn_q = w("cross_attn_q.weight");
        l.cross_attn_k = w("cross_attn_k.weight");
        l.cross_attn_v = w("cross_attn_v.weight");
        l.cross_attn_o = w("cross_attn_o.weight");
        l.cross_attn_norm_w = w("cross_attn_norm.weight");
        l.cross_attn_norm_b = w("cross_attn_norm.bias");
        l.fc1 = w("fc1.weight");
        l.fc2 = w("fc2.weight");
        l.ffn_norm_w = w("ffn_norm.weight");
        l.ffn_norm_b = w("ffn_norm.bias");
    }

    // DAC -- tensors prefixed with "dac." are loaded into the DacWeights struct
    // This will be done during weight assignment below.

    return true;
}

// ── Delay pattern ───────────────────────────────────────────────────
//
// MusicGen delay pattern for K codebooks:
//   Codebook k is delayed by k steps.
//   At generation step t, we predict codebook k's token for time (t - k).
//
// Example with 4 codebooks, T=5 steps:
//   Step 0: [BOS,  BOS,  BOS,  BOS ]
//   Step 1: [c0_0, BOS,  BOS,  BOS ]
//   Step 2: [c0_1, c1_0, BOS,  BOS ]
//   Step 3: [c0_2, c1_1, c2_0, BOS ]
//   Step 4: [c0_3, c1_2, c2_1, c3_0]
//   ...
//
// After generation, un-delay to get aligned codebooks:
//   Time 0: [c0_0, c1_0, c2_0, c3_0]
//   Time 1: [c0_1, c1_1, c2_1, c3_1]
//   ...

static std::vector<int32_t> apply_delay_pattern_undelay(const std::vector<int32_t>& delayed_codes, int num_codebooks,
                                                        int bos_token_id, int eos_token_id) {
    // delayed_codes: (num_steps * num_codebooks)
    // Each step has num_codebooks tokens, where codebook k's token at step t
    // corresponds to audio time (t - k).
    int num_steps = (int)delayed_codes.size() / num_codebooks;
    if (num_steps <= num_codebooks)
        return {};

    int T_audio = num_steps - num_codebooks + 1;
    std::vector<int32_t> aligned(T_audio * num_codebooks, eos_token_id);

    for (int t = 0; t < T_audio; t++) {
        for (int k = 0; k < num_codebooks; k++) {
            int step = t + k;
            if (step < num_steps) {
                int32_t tok = delayed_codes[step * num_codebooks + k];
                if (tok != bos_token_id && tok != eos_token_id) {
                    aligned[t * num_codebooks + k] = tok;
                }
            }
        }
    }
    return aligned;
}

// ── Sampling ────────────────────────────────────────────────────────

static int32_t sample_token(const float* logits, int vocab_size, float temperature, std::mt19937& rng) {
    if (temperature <= 0.0f || temperature < 1e-6f) {
        // Greedy
        int best = 0;
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > logits[best])
                best = i;
        }
        return (int32_t)best;
    }

    // Temperature-scaled softmax + multinomial sampling
    std::vector<float> probs(vocab_size);
    float max_logit = *std::max_element(logits, logits + vocab_size);
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = std::exp((logits[i] - max_logit) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; i++)
        probs[i] /= sum;

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return (int32_t)dist(rng);
}

// ── Public API ──────────────────────────────────────────────────────

struct parler_tts_context_params parler_tts_context_default_params(void) {
    return {
        /*.n_threads       =*/4,
        /*.verbosity       =*/1,
        /*.use_gpu         =*/false,
        /*.temperature     =*/1.0f,
        /*.seed            =*/0,
        /*.max_audio_tokens=*/0,
        /*.flash_attn      =*/false,
    };
}

struct parler_tts_context* parler_tts_init_from_file(const char* path_model, struct parler_tts_context_params params) {
    auto* ctx = new parler_tts_context{};
    ctx->params = params;

    // Seed RNG
    if (params.seed != 0) {
        ctx->rng.seed((unsigned)params.seed);
    } else {
        std::random_device rd;
        ctx->rng.seed(rd());
    }

    // Load GGUF
    if (params.verbosity >= 1)
        fprintf(stderr, "parler_tts: loading '%s'\n", path_model);

    struct gguf_init_params gip = {
        /*.no_alloc =*/false,
        /*.ctx      =*/&ctx->ctx_w,
    };
    gguf_context* g = gguf_init_from_file(path_model, gip);
    if (!g) {
        fprintf(stderr, "parler_tts: failed to open '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }

    load_metadata(ctx, g);

    // Build tensor name map
    int n_tensors = ggml_ctx_ntensors(ctx->ctx_w);
    for (int i = 0; i < n_tensors; i++) {
        ggml_tensor* t = ggml_ctx_get_tensor_by_index(ctx->ctx_w, i);
        if (t && t->name[0])
            ctx->tensors[t->name] = t;
    }

    if (params.verbosity >= 1)
        fprintf(stderr, "parler_tts: loaded %d tensors\n", n_tensors);

    bind_tensors(ctx);

    // Initialize backend
    ctx->backend_cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, params.n_threads);
    ctx->backend = ctx->backend_cpu;

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    ctx->sched = ggml_backend_sched_new(&ctx->backend, &buft, 1, 8192, false);

    gguf_free(g);

    if (params.verbosity >= 1) {
        auto& dc = ctx->model.dec_cfg;
        auto& tc = ctx->model.t5_cfg;
        fprintf(stderr, "parler_tts: T5 enc  d=%d h=%d L=%d d_kv=%d d_ff=%d\n", tc.d_model, tc.n_heads, tc.n_layers,
                tc.d_kv, tc.d_ff);
        fprintf(stderr, "parler_tts: decoder d=%d h=%d L=%d ffn=%d cb=%d vocab=%d\n", dc.hidden_size, dc.num_heads,
                dc.num_layers, dc.ffn_dim, dc.num_codebooks, dc.vocab_size);
        fprintf(stderr, "parler_tts: DAC     cb=%d cbsz=%d sr=%d hop=%d\n", ctx->model.dac_cfg.n_codebooks,
                ctx->model.dac_cfg.codebook_size, ctx->model.dac_cfg.sample_rate, ctx->model.dac_cfg.hop_length);
    }

    return ctx;
}

int parler_tts_set_description(struct parler_tts_context* ctx, const char* description) {
    if (!ctx || !description)
        return -1;

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "parler_tts: encoding description: '%s'\n", description);

    // Tokenize the description using the T5/description tokenizer
    std::vector<int> desc_ids = tokenize_unigram(ctx->desc_tokenizer, description);
    if (desc_ids.empty()) {
        // Fallback: use prompt tokenizer
        desc_ids = tokenize_unigram(ctx->prompt_tokenizer, description);
    }

    if (ctx->params.verbosity >= 2) {
        fprintf(stderr, "parler_tts: description tokens (%zu): ", desc_ids.size());
        for (int id : desc_ids)
            fprintf(stderr, "%d ", id);
        fprintf(stderr, "\n");
    }

    // TODO: Run T5 encoder forward pass on desc_ids to produce enc_hidden.
    // For now, store a placeholder to allow the pipeline to proceed.
    // The full T5 encoder forward pass will be implemented in a follow-up
    // once we have a GGUF to test against.
    //
    // The T5 encoder output shape is (T_desc, d_model) where T_desc is
    // the number of description tokens.

    const int d_model = ctx->model.t5_cfg.d_model;
    const int T_desc = (int)desc_ids.size();

    ctx->enc_T = T_desc;
    ctx->enc_hidden.resize(T_desc * d_model, 0.0f);
    ctx->enc_cached = true;

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "parler_tts: T5 encoder output cached (%d tokens, %d dim)\n", T_desc, d_model);

    return 0;
}

float* parler_tts_synthesize(struct parler_tts_context* ctx, const char* text, int* out_n_samples) {
    if (!ctx || !text || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;

    if (!ctx->enc_cached) {
        fprintf(stderr, "parler_tts: call parler_tts_set_description() first\n");
        return nullptr;
    }

    // Get audio codes
    int n_codes = 0;
    int32_t* codes = parler_tts_synthesize_codes(ctx, text, &n_codes);
    if (!codes || n_codes <= 0)
        return nullptr;

    // TODO: DAC decode codes -> PCM
    // For now, return silence as placeholder
    const int num_codebooks = ctx->model.dec_cfg.num_codebooks;
    const int T_audio = n_codes / num_codebooks;
    const int hop = ctx->model.dac_cfg.hop_length;
    const int n_samples = T_audio * hop;

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "parler_tts: DAC decode %d frames -> %d samples (%.2fs @ %d Hz)\n", T_audio, n_samples,
                (float)n_samples / ctx->model.dac_cfg.sample_rate, ctx->model.dac_cfg.sample_rate);

    float* pcm = (float*)malloc(n_samples * sizeof(float));
    if (!pcm) {
        parler_tts_codes_free(codes);
        return nullptr;
    }
    memset(pcm, 0, n_samples * sizeof(float));
    *out_n_samples = n_samples;

    parler_tts_codes_free(codes);
    return pcm;
}

int32_t* parler_tts_synthesize_codes(struct parler_tts_context* ctx, const char* text, int* out_n) {
    if (!ctx || !text || !out_n)
        return nullptr;
    *out_n = 0;

    if (!ctx->enc_cached) {
        fprintf(stderr, "parler_tts: call parler_tts_set_description() first\n");
        return nullptr;
    }

    // Tokenize the text prompt
    std::vector<int> prompt_ids = tokenize_unigram(ctx->prompt_tokenizer, text);
    if (ctx->params.verbosity >= 2) {
        fprintf(stderr, "parler_tts: prompt tokens (%zu): ", prompt_ids.size());
        for (int id : prompt_ids)
            fprintf(stderr, "%d ", id);
        fprintf(stderr, "\n");
    }

    const auto& dc = ctx->model.dec_cfg;
    const int num_codebooks = dc.num_codebooks;
    const int max_gen = ctx->params.max_audio_tokens > 0 ? ctx->params.max_audio_tokens : dc.max_generation;

    // TODO: Full decoder AR loop.
    // For now, generate placeholder BOS tokens to test the pipeline structure.
    //
    // The actual loop:
    //   1. Embed prompt_ids via embed_prompts + positional embedding
    //   2. Run decoder layers (self-attn + cross-attn + FFN) in prefill mode
    //   3. AR loop: at each step, predict 9 codebook logits, sample,
    //      feed back with delay pattern, until EOS on all codebooks or max_gen
    //   4. Un-delay the generated codes

    // Placeholder: emit a few steps of codes for pipeline testing
    const int gen_steps = std::min(10, max_gen);
    std::vector<int32_t> delayed_codes;
    delayed_codes.reserve(gen_steps * num_codebooks);

    for (int t = 0; t < gen_steps; t++) {
        for (int k = 0; k < num_codebooks; k++) {
            if (t <= k) {
                delayed_codes.push_back(dc.bos_token_id);
            } else {
                // Placeholder: random code
                delayed_codes.push_back(ctx->rng() % dc.eos_token_id);
            }
        }
    }

    // Un-delay
    auto aligned = apply_delay_pattern_undelay(delayed_codes, num_codebooks, dc.bos_token_id, dc.eos_token_id);

    if (aligned.empty()) {
        fprintf(stderr, "parler_tts: no valid audio codes generated\n");
        return nullptr;
    }

    int n = (int)aligned.size();
    int32_t* result = (int32_t*)malloc(n * sizeof(int32_t));
    memcpy(result, aligned.data(), n * sizeof(int32_t));
    *out_n = n;

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "parler_tts: generated %d aligned codes (%d frames x %d codebooks)\n", n, n / num_codebooks,
                num_codebooks);

    return result;
}

void parler_tts_codes_free(int32_t* codes) {
    free(codes);
}

void parler_tts_pcm_free(float* pcm) {
    free(pcm);
}

void parler_tts_free(struct parler_tts_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    delete ctx;
}

void parler_tts_set_n_threads(struct parler_tts_context* ctx, int n_threads) {
    if (ctx && ctx->backend_cpu)
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
}

void parler_tts_set_temperature(struct parler_tts_context* ctx, float temperature) {
    if (ctx)
        ctx->params.temperature = temperature;
}

void parler_tts_set_seed(struct parler_tts_context* ctx, uint64_t seed) {
    if (!ctx)
        return;
    ctx->params.seed = seed;
    if (seed != 0) {
        ctx->rng.seed((unsigned)seed);
    } else {
        std::random_device rd;
        ctx->rng.seed(rd());
    }
}
