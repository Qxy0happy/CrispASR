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

// BERT WordPiece tokenizer (cased, no lowering -- bert-base-multilingual-cased)
struct bark_wordpiece_tokenizer {
    std::vector<std::string> id_to_token;
    std::map<std::string, int> token_to_id;
    int unk_id = 100; // [UNK]
    bool loaded = false;

    void build_map() {
        token_to_id.clear();
        for (int i = 0; i < (int)id_to_token.size(); i++) {
            if (!id_to_token[(size_t)i].empty()) {
                token_to_id[id_to_token[(size_t)i]] = i;
            }
        }
        loaded = !id_to_token.empty();
    }

    // BERT WordPiece tokenization (cased -- NO lowering)
    // Splits on whitespace + punctuation, then greedily matches longest subword.
    std::vector<int> tokenize(const std::string& text) const {
        std::vector<int> ids;
        // Step 1: split into words on whitespace and punctuation boundaries
        std::vector<std::string> words;
        std::string cur;
        for (size_t i = 0; i < text.size(); i++) {
            char c = text[i];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                if (!cur.empty()) {
                    words.push_back(cur);
                    cur.clear();
                }
            } else if ((c >= '!' && c <= '/') || (c >= ':' && c <= '@') || (c >= '[' && c <= '`') ||
                       (c >= '{' && c <= '~')) {
                // Punctuation gets its own word
                if (!cur.empty()) {
                    words.push_back(cur);
                    cur.clear();
                }
                words.push_back(std::string(1, c));
            } else {
                cur += c;
            }
        }
        if (!cur.empty())
            words.push_back(cur);

        // Step 2: WordPiece each word
        for (const auto& word : words) {
            // Try whole word first
            auto it = token_to_id.find(word);
            if (it != token_to_id.end()) {
                ids.push_back(it->second);
                continue;
            }
            // Greedy longest-match from left
            size_t start = 0;
            bool failed = false;
            while (start < word.size()) {
                size_t end = word.size();
                int best_id = -1;
                while (end > start) {
                    std::string sub = (start == 0) ? word.substr(0, end) : ("##" + word.substr(start, end - start));
                    auto sit = token_to_id.find(sub);
                    if (sit != token_to_id.end()) {
                        best_id = sit->second;
                        break;
                    }
                    // Don't split mid-UTF8
                    end--;
                    while (end > start && (word[end] & 0xC0) == 0x80)
                        end--;
                }
                if (best_id < 0) {
                    ids.push_back(unk_id);
                    failed = true;
                    break;
                }
                ids.push_back(best_id);
                start = end;
            }
            (void)failed;
        }
        return ids;
    }
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

// EnCodec decoder model (weight-normalized convolutions + LSTM)
struct bark_encodec_model {
    // Quantizer embeddings (8 codebooks)
    std::vector<ggml_tensor*> codebooks; // (128, 1024) each

    // Pre conv (index 0)
    ggml_tensor* pre_conv_w_g = nullptr;
    ggml_tensor* pre_conv_w_v = nullptr;
    ggml_tensor* pre_conv_b = nullptr;

    // LSTM (index 1)
    int lstm_layers = 2;
    std::vector<ggml_tensor*> lstm_wih;
    std::vector<ggml_tensor*> lstm_whh;
    std::vector<ggml_tensor*> lstm_bih;
    std::vector<ggml_tensor*> lstm_bhh;

    // 4 upsample blocks
    struct UpsampleBlock {
        ggml_tensor* convtr_w_g = nullptr;
        ggml_tensor* convtr_w_v = nullptr;
        ggml_tensor* convtr_b = nullptr;
        ggml_tensor* res_conv0_w_g = nullptr;
        ggml_tensor* res_conv0_w_v = nullptr;
        ggml_tensor* res_conv0_b = nullptr;
        ggml_tensor* res_conv1_w_g = nullptr;
        ggml_tensor* res_conv1_w_v = nullptr;
        ggml_tensor* res_conv1_b = nullptr;
        ggml_tensor* res_short_w_g = nullptr;
        ggml_tensor* res_short_w_v = nullptr;
        ggml_tensor* res_short_b = nullptr;
    };
    UpsampleBlock blocks[4];

    // Post conv (index 15)
    ggml_tensor* post_conv_w_g = nullptr;
    ggml_tensor* post_conv_w_v = nullptr;
    ggml_tensor* post_conv_b = nullptr;

    bool loaded = false;
};

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
    bark_encodec_model encodec;

    bark_speaker_prompt speaker;
    bark_wordpiece_tokenizer tokenizer;

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
// KV cache management
// ---------------------------------------------------------------------------

struct bark_kv_cache {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor* k = nullptr; // (hd, max_ctx, n_h, n_layer)
    ggml_tensor* v = nullptr;
    int max_ctx = 0;
};

static void kv_cache_free(bark_kv_cache& kv) {
    if (kv.buf)
        ggml_backend_buffer_free(kv.buf);
    if (kv.ctx)
        ggml_free(kv.ctx);
    kv = {};
}

static bool kv_cache_alloc(bark_kv_cache& kv, ggml_backend_t backend, int max_ctx, int n_head, int head_dim,
                           int n_layer) {
    if (kv.ctx && max_ctx <= kv.max_ctx)
        return true;
    kv_cache_free(kv);
    kv.max_ctx = max_ctx;
    struct ggml_init_params ip = {2 * ggml_tensor_overhead(), nullptr, true};
    kv.ctx = ggml_init(ip);
    if (!kv.ctx)
        return false;
    kv.k = ggml_new_tensor_4d(kv.ctx, GGML_TYPE_F32, head_dim, max_ctx, n_head, n_layer);
    kv.v = ggml_new_tensor_4d(kv.ctx, GGML_TYPE_F32, head_dim, max_ctx, n_head, n_layer);
    kv.buf = ggml_backend_alloc_ctx_tensors(kv.ctx, backend);
    if (!kv.buf) {
        ggml_free(kv.ctx);
        kv = {};
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// GPT-2 forward pass (shared by text and coarse models)
// ---------------------------------------------------------------------------

// Build ggml graph for causal GPT-2 forward with KV cache.
// Input: token embeddings (D, T) already computed by caller.
// Output: logits for last position.
static ggml_cgraph* build_gpt2_graph(bark_context* c, const bark_gpt_model& m, const bark_gpt_hp& hp, bark_kv_cache& kv,
                                     int n_past, int n_tokens) {
    const int D = (int)hp.n_embd;
    const int n_h = (int)hp.n_head;
    const int hd = D / n_h;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const float ln_eps = 1e-5f;
    const int T = n_tokens;
    const int Lk = n_past + T;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);

    ggml_tensor* causal_mask = nullptr;
    if (T > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    ggml_tensor* cur = embeds;

    for (uint32_t il = 0; il < hp.n_layer; il++) {
        const auto& l = m.layers[il];
        ggml_tensor* residual = cur;

        // Pre-attention LayerNorm
        ggml_tensor* x = ggml_norm(ctx0, cur, ln_eps);
        x = ggml_mul(ctx0, x, l.attn_norm_w);
        if (l.attn_norm_b)
            x = ggml_add(ctx0, x, l.attn_norm_b);

        // Fused QKV
        ggml_tensor* qkv = ggml_mul_mat(ctx0, l.attn_qkv_w, x);
        if (l.attn_qkv_b)
            qkv = ggml_add(ctx0, qkv, l.attn_qkv_b);

        // Split Q, K, V
        const size_t ts = ggml_type_size(qkv->type);
        ggml_tensor* Q = ggml_view_2d(ctx0, qkv, D, T, qkv->nb[1], 0);
        ggml_tensor* K = ggml_view_2d(ctx0, qkv, D, T, qkv->nb[1], D * ts);
        ggml_tensor* V = ggml_view_2d(ctx0, qkv, D, T, qkv->nb[1], 2 * D * ts);
        if (T > 1) {
            Q = ggml_cont(ctx0, Q);
            K = ggml_cont(ctx0, K);
            V = ggml_cont(ctx0, V);
        }

        // Reshape to (hd, n_h, T)
        Q = ggml_reshape_3d(ctx0, Q, hd, n_h, T);
        K = ggml_reshape_3d(ctx0, K, hd, n_h, T);
        V = ggml_reshape_3d(ctx0, V, hd, n_h, T);

        // Permute K/V to (hd, T, n_h) for cache write
        ggml_tensor* K_perm = ggml_permute(ctx0, K, 0, 2, 1, 3);
        ggml_tensor* V_perm = ggml_permute(ctx0, V, 0, 2, 1, 3);

        // Write into KV cache
        ggml_tensor* k_view = ggml_view_4d(ctx0, kv.k, hd, T, n_h, 1, kv.k->nb[1], kv.k->nb[2], kv.k->nb[3],
                                           (size_t)il * kv.k->nb[3] + (size_t)n_past * kv.k->nb[1]);
        ggml_tensor* v_view = ggml_view_4d(ctx0, kv.v, hd, T, n_h, 1, kv.v->nb[1], kv.v->nb[2], kv.v->nb[3],
                                           (size_t)il * kv.v->nb[3] + (size_t)n_past * kv.v->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, K_perm, k_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, V_perm, v_view));

        // Read full K/V
        ggml_tensor* Kfull =
            ggml_cont(ctx0, ggml_view_3d(ctx0, kv.k, hd, Lk, n_h, kv.k->nb[1], kv.k->nb[2], (size_t)il * kv.k->nb[3]));
        ggml_tensor* Vfull =
            ggml_cont(ctx0, ggml_view_3d(ctx0, kv.v, hd, Lk, n_h, kv.v->nb[1], kv.v->nb[2], (size_t)il * kv.v->nb[3]));

        // Permute Q to (hd, T, n_h)
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));

        // Flash attention
        ggml_tensor* attn =
            ggml_flash_attn_ext(ctx0, Q, Kfull, Vfull, (T == 1) ? nullptr : causal_mask, attn_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        attn = ggml_reshape_2d(ctx0, attn, D, T);

        // Output projection + residual
        attn = ggml_mul_mat(ctx0, l.attn_out_w, attn);
        if (l.attn_out_b)
            attn = ggml_add(ctx0, attn, l.attn_out_b);
        cur = ggml_add(ctx0, residual, attn);

        // FFN
        residual = cur;
        x = ggml_norm(ctx0, cur, ln_eps);
        x = ggml_mul(ctx0, x, l.ffn_norm_w);
        if (l.ffn_norm_b)
            x = ggml_add(ctx0, x, l.ffn_norm_b);

        ggml_tensor* mlp = ggml_mul_mat(ctx0, l.ffn_up_w, x);
        if (l.ffn_up_b)
            mlp = ggml_add(ctx0, mlp, l.ffn_up_b);
        mlp = ggml_gelu(ctx0, mlp);
        mlp = ggml_mul_mat(ctx0, l.ffn_down_w, mlp);
        if (l.ffn_down_b)
            mlp = ggml_add(ctx0, mlp, l.ffn_down_b);

        cur = ggml_add(ctx0, residual, mlp);
    }

    // Final LayerNorm
    cur = ggml_norm(ctx0, cur, ln_eps);
    cur = ggml_mul(ctx0, cur, m.output_norm_w);
    if (m.output_norm_b)
        cur = ggml_add(ctx0, cur, m.output_norm_b);

    // Take last token
    if (T > 1) {
        cur = ggml_view_2d(ctx0, cur, D, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }

    // LM head
    cur = ggml_mul_mat(ctx0, m.output_w, cur);
    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Run GPT-2 forward for n_tokens embeddings, return logits (caller must free).
static float* run_gpt2_forward(bark_context* c, const bark_gpt_model& m, const bark_gpt_hp& hp, bark_kv_cache& kv,
                               const float* embeds, int n_tokens, int n_past, int output_vocab) {
    const int D = (int)hp.n_embd;
    const int Lk = n_past + n_tokens;

    // Build causal mask
    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        mask.assign((size_t)Lk * n_tokens, ggml_fp32_to_fp16(0.0f));
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < n_tokens; q++) {
            for (int k = n_past + q + 1; k < Lk; k++) {
                mask[(size_t)q * Lk + k] = neg_inf;
            }
        }
    }

    ggml_cgraph* gf = build_gpt2_graph(c, m, hp, kv, n_past, n_tokens);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "bark: failed to alloc GPT-2 graph\n");
        return nullptr;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"), embeds, 0,
                            (size_t)D * n_tokens * sizeof(float));
    if (n_tokens > 1) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                mask.size() * sizeof(ggml_fp16_t));
    }
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "bark: GPT-2 compute failed\n");
        return nullptr;
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    float* r = (float*)malloc((size_t)output_vocab * sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)output_vocab * sizeof(float));
    return r;
}

// Compute embeddings: token_embd[token_ids] + pos_embd[positions]
// Returns (D, T) float array. Handles merge_context summing for text model.
//
// When merge_context=true, follows Bark's exact logic:
//   1. Compute token embeddings only (no pos)
//   2. Merge: tok_emb[0:256] += tok_emb[256:512], then append tok_emb[512:]
//   3. Add position embeddings to the MERGED sequence (positions 0..new_T-1)
// This ensures the INFER_TOKEN gets pos_embd[256], not pos_embd[512].
static std::vector<float> compute_embeddings(bark_context* /*c*/, const bark_gpt_model& m, const bark_gpt_hp& hp,
                                             const std::vector<int32_t>& tokens, int pos_offset, bool merge_context,
                                             int merge_len) {
    const int D = (int)hp.n_embd;
    const int T = (int)tokens.size();

    // Read embedding and position tables from backend
    // token_embd: (D, vocab) — row i = embedding for token i
    // pos_embd: (D, block_size)
    const size_t embd_row = (size_t)D * sizeof(ggml_fp16_t);
    std::vector<float> result((size_t)D * T, 0.0f);
    std::vector<ggml_fp16_t> row_buf((size_t)D);

    // Step 1: Compute token embeddings
    for (int t = 0; t < T; t++) {
        int tok = tokens[(size_t)t];
        ggml_backend_tensor_get(m.token_embd, row_buf.data(), (size_t)tok * embd_row, embd_row);
        for (int d = 0; d < D; d++) {
            result[(size_t)t * D + d] = ggml_fp16_to_fp32(row_buf[(size_t)d]);
        }
    }

    // Step 2: merge_context — sum token embeddings[0:merge_len] += [merge_len:2*merge_len]
    int out_T = T;
    if (merge_context && merge_len > 0 && T >= 2 * merge_len) {
        for (int t = 0; t < merge_len; t++) {
            for (int d = 0; d < D; d++) {
                result[(size_t)t * D + d] += result[(size_t)(t + merge_len) * D + d];
            }
        }
        // Remove the merged semantic history tokens (shift rest down)
        out_T = T - merge_len;
        for (int t = merge_len; t < out_T; t++) {
            std::memcpy(&result[(size_t)t * D], &result[(size_t)(t + merge_len) * D], (size_t)D * sizeof(float));
        }
        result.resize((size_t)out_T * D);
    }

    // Step 3: Add position embeddings to the (possibly merged) sequence
    for (int t = 0; t < out_T; t++) {
        int pos = pos_offset + t;
        ggml_backend_tensor_get(m.pos_embd, row_buf.data(), (size_t)pos * embd_row, embd_row);
        for (int d = 0; d < D; d++) {
            result[(size_t)t * D + d] += ggml_fp16_to_fp32(row_buf[(size_t)d]);
        }
    }

    return result;
}

// Build bidirectional GPT-2 graph (no causal mask) for fine model.
// Processes all positions at once, outputs logits for all positions.
static ggml_cgraph* build_fine_graph(bark_context* c, int n_tokens, int codebook_idx) {
    const auto& hp = c->fine_hp;
    const auto& m = c->fine_model;
    const int D = (int)hp.n_embd;
    const int n_h = (int)hp.n_head;
    const int hd = D / n_h;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const float ln_eps = 1e-5f;
    const int T = n_tokens;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);

    ggml_tensor* cur = embeds;

    for (uint32_t il = 0; il < hp.n_layer; il++) {
        const auto& l = m.layers[il];
        ggml_tensor* residual = cur;

        // Pre-attention LayerNorm
        ggml_tensor* x = ggml_norm(ctx0, cur, ln_eps);
        x = ggml_mul(ctx0, x, l.attn_norm_w);
        if (l.attn_norm_b)
            x = ggml_add(ctx0, x, l.attn_norm_b);

        // Fused QKV
        ggml_tensor* qkv = ggml_mul_mat(ctx0, l.attn_qkv_w, x);
        if (l.attn_qkv_b)
            qkv = ggml_add(ctx0, qkv, l.attn_qkv_b);

        const size_t ts = ggml_type_size(qkv->type);
        ggml_tensor* Q = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, D, T, qkv->nb[1], 0));
        ggml_tensor* K = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, D, T, qkv->nb[1], D * ts));
        ggml_tensor* V = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, D, T, qkv->nb[1], 2 * D * ts));

        // Reshape to (hd, T, n_h) for flash_attn_ext
        Q = ggml_reshape_3d(ctx0, Q, hd, n_h, T);
        K = ggml_reshape_3d(ctx0, K, hd, n_h, T);
        V = ggml_reshape_3d(ctx0, V, hd, n_h, T);
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        // Bidirectional: no mask (nullptr)
        ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, attn_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        attn = ggml_reshape_2d(ctx0, attn, D, T);

        // Output projection + residual
        attn = ggml_mul_mat(ctx0, l.attn_out_w, attn);
        if (l.attn_out_b)
            attn = ggml_add(ctx0, attn, l.attn_out_b);
        cur = ggml_add(ctx0, residual, attn);

        // FFN
        residual = cur;
        x = ggml_norm(ctx0, cur, ln_eps);
        x = ggml_mul(ctx0, x, l.ffn_norm_w);
        if (l.ffn_norm_b)
            x = ggml_add(ctx0, x, l.ffn_norm_b);

        ggml_tensor* mlp = ggml_mul_mat(ctx0, l.ffn_up_w, x);
        if (l.ffn_up_b)
            mlp = ggml_add(ctx0, mlp, l.ffn_up_b);
        mlp = ggml_gelu(ctx0, mlp);
        mlp = ggml_mul_mat(ctx0, l.ffn_down_w, mlp);
        if (l.ffn_down_b)
            mlp = ggml_add(ctx0, mlp, l.ffn_down_b);

        cur = ggml_add(ctx0, residual, mlp);
    }

    // Final LayerNorm
    cur = ggml_norm(ctx0, cur, ln_eps);
    cur = ggml_mul(ctx0, cur, m.output_norm_w);
    if (m.output_norm_b)
        cur = ggml_add(ctx0, cur, m.output_norm_b);

    // Per-codebook lm_head (codebook_idx is relative: 0 = first predicted cb)
    cur = ggml_mul_mat(ctx0, m.output_heads[(size_t)codebook_idx], cur);
    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// ---------------------------------------------------------------------------
// EnCodec decoder tensor binding
// ---------------------------------------------------------------------------

static bool bind_encodec_model(bark_context* c, bark_encodec_model& enc) {
    char key[128];

    // Quantizer codebooks
    enc.codebooks.resize(8);
    for (int i = 0; i < 8; i++) {
        std::snprintf(key, sizeof(key), "encodec.quantizer.%d.embed", i);
        enc.codebooks[(size_t)i] = require_tensor(c, key);
        if (!enc.codebooks[(size_t)i])
            return false;
    }

    // Pre conv
    enc.pre_conv_w_g = require_tensor(c, "encodec.decoder.0.conv.conv.weight_g");
    enc.pre_conv_w_v = require_tensor(c, "encodec.decoder.0.conv.conv.weight_v");
    enc.pre_conv_b = require_tensor(c, "encodec.decoder.0.conv.conv.bias");
    if (!enc.pre_conv_w_g || !enc.pre_conv_w_v)
        return false;

    // LSTM
    enc.lstm_wih.resize(2);
    enc.lstm_whh.resize(2);
    enc.lstm_bih.resize(2);
    enc.lstm_bhh.resize(2);
    for (int i = 0; i < 2; i++) {
        std::snprintf(key, sizeof(key), "encodec.decoder.1.lstm.weight_ih_l%d", i);
        enc.lstm_wih[(size_t)i] = require_tensor(c, key);
        std::snprintf(key, sizeof(key), "encodec.decoder.1.lstm.weight_hh_l%d", i);
        enc.lstm_whh[(size_t)i] = require_tensor(c, key);
        std::snprintf(key, sizeof(key), "encodec.decoder.1.lstm.bias_ih_l%d", i);
        enc.lstm_bih[(size_t)i] = require_tensor(c, key);
        std::snprintf(key, sizeof(key), "encodec.decoder.1.lstm.bias_hh_l%d", i);
        enc.lstm_bhh[(size_t)i] = require_tensor(c, key);
        if (!enc.lstm_wih[(size_t)i] || !enc.lstm_whh[(size_t)i])
            return false;
    }

    // 4 upsample blocks
    // Decoder layout: [0=pre_conv, 1=lstm, 2=skip, 3=convtr0, 4=res0, 5=skip, 6=convtr1, ...]
    // Actually from GGUF: indices 3, 6, 9, 12 are ConvTranspose; 4, 7, 10, 13 are ResBlocks
    const int convtr_idx[] = {3, 6, 9, 12};
    const int res_idx[] = {4, 7, 10, 13};
    for (int b = 0; b < 4; b++) {
        auto& blk = enc.blocks[b];
        std::snprintf(key, sizeof(key), "encodec.decoder.%d.convtr.convtr.weight_g", convtr_idx[b]);
        blk.convtr_w_g = require_tensor(c, key);
        std::snprintf(key, sizeof(key), "encodec.decoder.%d.convtr.convtr.weight_v", convtr_idx[b]);
        blk.convtr_w_v = require_tensor(c, key);
        std::snprintf(key, sizeof(key), "encodec.decoder.%d.convtr.convtr.bias", convtr_idx[b]);
        blk.convtr_b = get_tensor(c, key);

        std::snprintf(key, sizeof(key), "encodec.decoder.%d.block.1.conv.conv.weight_g", res_idx[b]);
        blk.res_conv0_w_g = require_tensor(c, key);
        std::snprintf(key, sizeof(key), "encodec.decoder.%d.block.1.conv.conv.weight_v", res_idx[b]);
        blk.res_conv0_w_v = require_tensor(c, key);
        std::snprintf(key, sizeof(key), "encodec.decoder.%d.block.1.conv.conv.bias", res_idx[b]);
        blk.res_conv0_b = get_tensor(c, key);

        std::snprintf(key, sizeof(key), "encodec.decoder.%d.block.3.conv.conv.weight_g", res_idx[b]);
        blk.res_conv1_w_g = require_tensor(c, key);
        std::snprintf(key, sizeof(key), "encodec.decoder.%d.block.3.conv.conv.weight_v", res_idx[b]);
        blk.res_conv1_w_v = require_tensor(c, key);
        std::snprintf(key, sizeof(key), "encodec.decoder.%d.block.3.conv.conv.bias", res_idx[b]);
        blk.res_conv1_b = get_tensor(c, key);

        std::snprintf(key, sizeof(key), "encodec.decoder.%d.shortcut.conv.conv.weight_g", res_idx[b]);
        blk.res_short_w_g = require_tensor(c, key);
        std::snprintf(key, sizeof(key), "encodec.decoder.%d.shortcut.conv.conv.weight_v", res_idx[b]);
        blk.res_short_w_v = require_tensor(c, key);
        std::snprintf(key, sizeof(key), "encodec.decoder.%d.shortcut.conv.conv.bias", res_idx[b]);
        blk.res_short_b = get_tensor(c, key);

        if (!blk.convtr_w_g || !blk.convtr_w_v || !blk.res_conv0_w_g || !blk.res_conv0_w_v)
            return false;
    }

    // Post conv
    enc.post_conv_w_g = require_tensor(c, "encodec.decoder.15.conv.conv.weight_g");
    enc.post_conv_w_v = require_tensor(c, "encodec.decoder.15.conv.conv.weight_v");
    enc.post_conv_b = require_tensor(c, "encodec.decoder.15.conv.conv.bias");
    if (!enc.post_conv_w_g || !enc.post_conv_w_v)
        return false;

    enc.loaded = true;
    return true;
}

// Fold weight_g * weight_v / ||weight_v|| into effective weight.
// weight_g: (1, 1, Cout), weight_v: (K, Cin, Cout)
// Returns folded weight as F32 vector.
static std::vector<float> fold_weight_norm(ggml_tensor* w_g, ggml_tensor* w_v) {
    // w_v shape: ne[0]=K, ne[1]=Cin, ne[2]=Cout (for conv1d)
    // w_g shape: ne[0]=1, ne[1]=1, ne[2]=Cout
    const int64_t K = w_v->ne[0];
    const int64_t Cin = w_v->ne[1];
    const int64_t Cout = w_v->ne[2];
    const size_t total = (size_t)(K * Cin * Cout);

    // Read w_v
    std::vector<float> v(total);
    if (w_v->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> buf(total);
        ggml_backend_tensor_get(w_v, buf.data(), 0, total * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < total; i++)
            v[i] = ggml_fp16_to_fp32(buf[i]);
    } else {
        ggml_backend_tensor_get(w_v, v.data(), 0, total * sizeof(float));
    }

    // Read w_g
    std::vector<float> g((size_t)Cout);
    if (w_g->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> buf((size_t)Cout);
        ggml_backend_tensor_get(w_g, buf.data(), 0, (size_t)Cout * sizeof(ggml_fp16_t));
        for (int i = 0; i < Cout; i++)
            g[(size_t)i] = ggml_fp16_to_fp32(buf[(size_t)i]);
    } else {
        ggml_backend_tensor_get(w_g, g.data(), 0, (size_t)Cout * sizeof(float));
    }

    // For each output channel, compute norm of v and scale
    std::vector<float> result(total);
    for (int64_t co = 0; co < Cout; co++) {
        float norm_sq = 0.0f;
        size_t off = (size_t)(co * K * Cin);
        for (int64_t i = 0; i < K * Cin; i++) {
            norm_sq += v[off + (size_t)i] * v[off + (size_t)i];
        }
        float scale = g[(size_t)co] / (std::sqrt(norm_sq) + 1e-12f);
        for (int64_t i = 0; i < K * Cin; i++) {
            result[off + (size_t)i] = v[off + (size_t)i] * scale;
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Sampling helper
// ---------------------------------------------------------------------------

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

// Tokenize text using BERT WordPiece (cased), offset by TEXT_ENCODING_OFFSET,
// then pad to max_len with text_pad_token.
static std::vector<int32_t> tokenize_text_bert(bark_context* ctx, const char* text, uint32_t text_encoding_offset,
                                               uint32_t text_pad_token, int max_len) {
    std::vector<int32_t> tokens;

    if (ctx->tokenizer.loaded) {
        // Use proper BERT WordPiece tokenization (add_special_tokens=False)
        std::vector<int> bert_ids = ctx->tokenizer.tokenize(std::string(text));
        for (int id : bert_ids) {
            if ((int)tokens.size() >= max_len)
                break;
            tokens.push_back((int32_t)id + (int32_t)text_encoding_offset);
        }
    } else {
        // Fallback: byte-level (should not happen if GGUF has vocab)
        fprintf(stderr, "bark: WARNING - no BERT vocab loaded, using byte-level fallback\n");
        const uint8_t* p = (const uint8_t*)text;
        while (*p && (int)tokens.size() < max_len) {
            tokens.push_back((int32_t)(*p) + (int32_t)text_encoding_offset);
            p++;
        }
    }

    // Truncate if too long
    if ((int)tokens.size() > max_len) {
        tokens.resize((size_t)max_len);
    }
    // Pad to max_len
    while ((int)tokens.size() < max_len) {
        tokens.push_back((int32_t)text_pad_token);
    }
    return tokens;
}

static std::vector<int32_t> generate_text_semantic(bark_context* ctx, const char* text) {
    auto& pp = ctx->pp;
    auto& hp = ctx->text_hp;
    int max_steps = ctx->params.max_semantic_tokens > 0 ? ctx->params.max_semantic_tokens : 768;
    const int ctx_len = 256;

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "bark: stage 1 (semantic) - max_steps=%d\n", max_steps);
    }

    // 1. Tokenize text -> padded to 256 (BERT WordPiece, cased, + TEXT_ENCODING_OFFSET)
    std::vector<int32_t> text_tokens =
        tokenize_text_bert(ctx, text, pp.text_encoding_offset, pp.text_pad_token, ctx_len);

    if (ctx->params.verbosity >= 2) {
        fprintf(stderr, "bark: text tokens (first non-pad):");
        for (int i = 0; i < ctx_len && text_tokens[(size_t)i] != (int32_t)pp.text_pad_token; i++) {
            fprintf(stderr, " %d", text_tokens[(size_t)i]);
        }
        fprintf(stderr, "\n");
    }

    // 2. Semantic history (from speaker or all-PAD)
    std::vector<int32_t> sem_hist(ctx_len, (int32_t)pp.semantic_pad_token);
    if (ctx->speaker.loaded && !ctx->speaker.semantic_prompt.empty()) {
        int copy_len = std::min((int)ctx->speaker.semantic_prompt.size(), ctx_len);
        for (int i = 0; i < copy_len; i++) {
            sem_hist[(size_t)(ctx_len - copy_len + i)] = ctx->speaker.semantic_prompt[(size_t)i];
        }
    }

    // 3. Build input: [text(256) | sem_hist(256) | SEMANTIC_INFER_TOKEN]
    std::vector<int32_t> input_tokens;
    input_tokens.reserve(ctx_len * 2 + 1);
    input_tokens.insert(input_tokens.end(), text_tokens.begin(), text_tokens.end());
    input_tokens.insert(input_tokens.end(), sem_hist.begin(), sem_hist.end());
    input_tokens.push_back((int32_t)pp.semantic_infer_token);

    // 4. Allocate KV cache
    const int n_h = (int)hp.n_head;
    const int hd = (int)hp.n_embd / n_h;
    int total_ctx = (int)input_tokens.size() + max_steps;
    if (total_ctx > (int)hp.block_size)
        total_ctx = (int)hp.block_size;

    bark_kv_cache kv{};
    if (!kv_cache_alloc(kv, ctx->backend, total_ctx, n_h, hd, (int)hp.n_layer)) {
        fprintf(stderr, "bark: failed to alloc text KV cache\n");
        return {};
    }

    // 5. Compute initial embeddings with merge_context
    std::vector<float> embeds =
        compute_embeddings(ctx, ctx->text_model, hp, input_tokens, 0, /*merge_context=*/true, ctx_len);
    int n_input = (int)(embeds.size() / (size_t)hp.n_embd); // after merge: 256 + 1 = 257

    // 6. Prefill
    float* logits = run_gpt2_forward(ctx, ctx->text_model, hp, kv, embeds.data(), n_input, 0, (int)hp.output_vocab);
    if (!logits) {
        kv_cache_free(kv);
        return {};
    }

    int n_past = n_input;
    std::vector<int32_t> out;
    out.reserve((size_t)max_steps);
    float temperature = ctx->params.temperature_semantic;

    // 7. AR decode — sample from logits[0:10000] + logits[SEMANTIC_PAD_TOKEN] as EOS
    // This matches Python: relevant_logits = hstack([logits[:10000], logits[10000]])
    const int sample_vocab = (int)pp.semantic_vocab_size + 1; // 10001: 10000 semantic + 1 EOS
    std::vector<float> sample_logits((size_t)sample_vocab);

    for (int step = 0; step < max_steps; step++) {
        // Build sampling distribution: first 10000 logits + EOS logit (at index SEMANTIC_PAD_TOKEN)
        std::memcpy(sample_logits.data(), logits, (size_t)pp.semantic_vocab_size * sizeof(float));
        sample_logits[(size_t)pp.semantic_vocab_size] = logits[pp.semantic_pad_token];

        int tok = sample_from_logits(sample_logits.data(), sample_vocab, temperature, ctx->rng);
        free(logits);
        logits = nullptr;

        // Check EOS (sampled index == 10000 means EOS token was selected)
        if (tok == (int)pp.semantic_vocab_size) {
            break;
        }
        out.push_back(tok);

        // Check max context
        if (n_past >= total_ctx - 1) {
            break;
        }

        // Prepare next token embedding
        std::vector<int32_t> next_tok = {tok};
        std::vector<float> next_emb = compute_embeddings(ctx, ctx->text_model, hp, next_tok, n_past, false, 0);
        logits = run_gpt2_forward(ctx, ctx->text_model, hp, kv, next_emb.data(), 1, n_past, (int)hp.output_vocab);
        n_past++;

        if (!logits)
            break;
    }

    if (logits)
        free(logits);
    kv_cache_free(kv);

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "bark: stage 1 produced %d semantic tokens\n", (int)out.size());
    }

    return out;
}

// ---------------------------------------------------------------------------
// Stage 2: Generate coarse codes from semantic tokens
// ---------------------------------------------------------------------------

static std::vector<int32_t> generate_coarse(bark_context* ctx, const std::vector<int32_t>& semantic_tokens) {
    auto& pp = ctx->pp;
    auto& hp = ctx->coarse_hp;

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

    const int n_h = (int)hp.n_head;
    const int hd = (int)hp.n_embd / n_h;
    const int max_semantic_ctx = 256;
    (void)0; // sliding_window=60 used in full implementation

    // Build semantic token sequence with RIGHT-side padding (tokens first, then pad)
    std::vector<int32_t> semantic_padded(max_semantic_ctx, (int32_t)pp.coarse_semantic_pad_token);
    int sem_len = std::min((int)semantic_tokens.size(), max_semantic_ctx);
    for (int i = 0; i < sem_len; i++) {
        semantic_padded[(size_t)i] = semantic_tokens[(size_t)i];
    }

    // Coarse generation: slide through semantic tokens in windows
    std::vector<int32_t> coarse_out;
    coarse_out.reserve((size_t)n_steps);

    float temperature = ctx->params.temperature_coarse;

    // Process all steps (simplified: single context window)
    // Build input: [semantic_padded(256) | COARSE_INFER_TOKEN | coarse_history...]
    std::vector<int32_t> input_tokens;
    input_tokens.reserve(max_semantic_ctx + 1 + (size_t)n_steps);
    input_tokens.insert(input_tokens.end(), semantic_padded.begin(), semantic_padded.end());
    input_tokens.push_back((int32_t)pp.coarse_infer_token);

    int total_ctx = (int)input_tokens.size() + n_steps;
    if (total_ctx > (int)hp.block_size)
        total_ctx = (int)hp.block_size;

    bark_kv_cache kv{};
    if (!kv_cache_alloc(kv, ctx->backend, total_ctx, n_h, hd, (int)hp.n_layer)) {
        fprintf(stderr, "bark: failed to alloc coarse KV cache\n");
        return {};
    }

    // Prefill with context
    std::vector<float> embeds = compute_embeddings(ctx, ctx->coarse_model, hp, input_tokens, 0, false, 0);
    int n_input = (int)input_tokens.size();

    float* logits = run_gpt2_forward(ctx, ctx->coarse_model, hp, kv, embeds.data(), n_input, 0, (int)hp.output_vocab);
    if (!logits) {
        kv_cache_free(kv);
        return {};
    }

    int n_past = n_input;

    // AR decode
    for (int step = 0; step < n_steps && n_past < total_ctx; step++) {
        // Determine which codebook we're generating (alternating cb0/cb1)
        int cb = step % (int)pp.n_coarse_codebooks;
        int cb_offset = (int)pp.semantic_vocab_size + cb * (int)pp.codebook_size;

        // Sample from logits[cb_offset : cb_offset + codebook_size]
        int tok = sample_from_logits(logits + cb_offset, (int)pp.codebook_size, temperature, ctx->rng);
        free(logits);
        logits = nullptr;

        // Store as full vocab token (offset by semantic_vocab + cb*codebook_size)
        int full_tok = tok + cb_offset;
        coarse_out.push_back(tok); // store raw codebook index

        // Next token
        std::vector<int32_t> next_tok = {(int32_t)full_tok};
        std::vector<float> next_emb = compute_embeddings(ctx, ctx->coarse_model, hp, next_tok, n_past, false, 0);
        logits = run_gpt2_forward(ctx, ctx->coarse_model, hp, kv, next_emb.data(), 1, n_past, (int)hp.output_vocab);
        n_past++;

        if (!logits)
            break;
    }

    if (logits)
        free(logits);
    kv_cache_free(kv);

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "bark: stage 2 produced %d coarse tokens\n", (int)coarse_out.size());
    }

    // Ensure even number (n_coarse_codebooks = 2)
    if ((int)coarse_out.size() % (int)pp.n_coarse_codebooks != 0) {
        coarse_out.resize(coarse_out.size() - coarse_out.size() % pp.n_coarse_codebooks);
    }

    return coarse_out;
}

// ---------------------------------------------------------------------------
// Stage 3: Generate fine codes from coarse codes
// ---------------------------------------------------------------------------

static std::vector<int32_t> generate_fine(bark_context* ctx, const int32_t* coarse_codes, int n_coarse_codebooks,
                                          int n_timesteps) {
    auto& pp = ctx->pp;
    auto& fh = ctx->fine_hp;
    auto& m = ctx->fine_model;

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "bark: stage 3 (fine) - %d timesteps, %d->%d codebooks\n", n_timesteps, n_coarse_codebooks,
                (int)pp.n_fine_codebooks);
    }

    int n_fine = (int)pp.n_fine_codebooks;
    const int D = (int)fh.n_embd;
    const int window_size = (int)fh.block_size; // 1024
    float temperature = ctx->params.temperature_fine;

    // Build codes array (n_fine, T) — coarse filled, rest = codebook_size (padding)
    std::vector<int32_t> codes((size_t)(n_fine * n_timesteps), (int32_t)pp.codebook_size);
    // The coarse codes come interleaved (cb0_t0, cb1_t0, cb0_t1, cb1_t1, ...)
    for (int t = 0; t < n_timesteps; t++) {
        for (int cb = 0; cb < n_coarse_codebooks && cb < n_fine; cb++) {
            codes[(size_t)(cb * n_timesteps + t)] = coarse_codes[t * n_coarse_codebooks + cb];
        }
    }

    // For each fine codebook to predict
    for (int nn = n_coarse_codebooks; nn < n_fine; nn++) {
        if (ctx->params.verbosity >= 2) {
            fprintf(stderr, "bark: fine codebook %d/%d\n", nn, n_fine);
        }

        // Sliding window with step = window_size/2
        int step_size = window_size / 2;
        for (int win_start = 0; win_start < n_timesteps; win_start += step_size) {
            int win_end = std::min(win_start + window_size, n_timesteps);
            int win_len = win_end - win_start;

            // Sum embeddings of codebooks 0..nn-1, add position embedding
            std::vector<float> embeds((size_t)(D * win_len), 0.0f);
            const size_t embd_row = (size_t)D * sizeof(ggml_fp16_t);
            std::vector<ggml_fp16_t> row_buf((size_t)D);

            for (int t = 0; t < win_len; t++) {
                // Sum token embeddings for codebooks 0..nn
                for (int cb = 0; cb <= nn; cb++) {
                    int tok = codes[(size_t)(cb * n_timesteps + win_start + t)];
                    if (tok >= (int)m.token_embds[(size_t)cb]->ne[1])
                        tok = 0; // clamp
                    ggml_backend_tensor_get(m.token_embds[(size_t)cb], row_buf.data(), (size_t)tok * embd_row,
                                            embd_row);
                    for (int d = 0; d < D; d++) {
                        embeds[(size_t)(t * D + d)] += ggml_fp16_to_fp32(row_buf[(size_t)d]);
                    }
                }
                // Add position embedding
                ggml_backend_tensor_get(m.pos_embd, row_buf.data(), (size_t)t * embd_row, embd_row);
                for (int d = 0; d < D; d++) {
                    embeds[(size_t)(t * D + d)] += ggml_fp16_to_fp32(row_buf[(size_t)d]);
                }
            }

            // Build and run fine graph (bidirectional)
            int cb_idx = nn - (int)fh.n_codes_given; // index into output_heads
            ggml_cgraph* gf = build_fine_graph(ctx, win_len, cb_idx);
            ggml_backend_sched_reset(ctx->sched);
            if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
                fprintf(stderr, "bark: failed to alloc fine graph\n");
                break;
            }
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"), embeds.data(), 0,
                                    (size_t)(D * win_len) * sizeof(float));
            if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "bark: fine compute failed\n");
                break;
            }

            // Get logits for all positions: (output_vocab, win_len)
            ggml_tensor* out_t = ggml_graph_get_tensor(gf, "logits");
            int out_vocab = (int)fh.output_vocab;
            std::vector<float> all_logits((size_t)(out_vocab * win_len));
            ggml_backend_tensor_get(out_t, all_logits.data(), 0, all_logits.size() * sizeof(float));

            // Sample for each position in the window
            for (int t = 0; t < win_len; t++) {
                int tok = sample_from_logits(&all_logits[(size_t)(t * out_vocab)], (int)pp.codebook_size, temperature,
                                             ctx->rng);
                codes[(size_t)(nn * n_timesteps + win_start + t)] = tok;
            }
        }
    }

    return codes;
}

// ---------------------------------------------------------------------------
// EnCodec decoder stub
// ---------------------------------------------------------------------------

// ELU activation: alpha * (exp(x) - 1) for x < 0, x for x >= 0
static inline float elu_f(float x, float alpha = 1.0f) {
    return x >= 0.0f ? x : alpha * (std::exp(x) - 1.0f);
}

// CPU-based EnCodec decoder implementation.
// Uses weight-normalized convolutions (folded at decode time).
// Architecture: pre_conv -> LSTM -> 4x(ELU + ConvTranspose + ResBlock) -> ELU + post_conv -> tanh

// Helper: Conv1D with same-padding on CPU vectors. Layout: (C, T) row-major.
static std::vector<float> cpu_conv1d(const float* input, int Cin, int T, const float* weight, const float* bias,
                                     int Cout, int K, int stride, int padding, int dilation) {
    int T_out = (T + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    std::vector<float> out((size_t)(Cout * T_out), 0.0f);

    for (int co = 0; co < Cout; co++) {
        for (int t = 0; t < T_out; t++) {
            float sum = bias ? bias[co] : 0.0f;
            for (int ci = 0; ci < Cin; ci++) {
                for (int k = 0; k < K; k++) {
                    int t_in = t * stride - padding + k * dilation;
                    if (t_in >= 0 && t_in < T) {
                        // weight layout: (K, Cin, Cout) — matching GGUF ne order
                        sum += weight[(size_t)(co * Cin * K + ci * K + k)] * input[(size_t)(ci * T + t_in)];
                    }
                }
            }
            out[(size_t)(co * T_out + t)] = sum;
        }
    }
    return out;
}

// ConvTranspose1d on CPU
static std::vector<float> cpu_conv_transpose1d(const float* input, int Cin, int T, const float* weight,
                                               const float* bias, int Cout, int K, int stride, int padding) {
    int T_out = (T - 1) * stride + K - 2 * padding;
    std::vector<float> out((size_t)(Cout * T_out), 0.0f);

    for (int co = 0; co < Cout; co++) {
        for (int ti = 0; ti < T; ti++) {
            for (int ci = 0; ci < Cin; ci++) {
                float val = input[(size_t)(ci * T + ti)];
                for (int k = 0; k < K; k++) {
                    int t_out = ti * stride + k - padding;
                    if (t_out >= 0 && t_out < T_out) {
                        // weight layout: (K, Cout, Cin) for ConvTranspose
                        out[(size_t)(co * T_out + t_out)] += val * weight[(size_t)(ci * Cout * K + co * K + k)];
                    }
                }
            }
        }
        if (bias) {
            for (int t = 0; t < T_out; t++) {
                out[(size_t)(co * T_out + t)] += bias[co];
            }
        }
    }
    return out;
}

// LSTM forward on CPU: processes (C, T) sequence through LSTM
static std::vector<float> cpu_lstm_forward(const float* input, int C, int T, const bark_encodec_model& enc) {
    std::vector<float> output((size_t)(C * T), 0.0f);
    std::vector<float> h((size_t)C, 0.0f);
    std::vector<float> cell((size_t)C, 0.0f);

    for (int layer = 0; layer < enc.lstm_layers; layer++) {
        // Read weights from tensors
        const size_t ws = (size_t)(4 * C * C);
        std::vector<float> wih(ws), whh(ws);
        std::vector<float> bih((size_t)(4 * C)), bhh((size_t)(4 * C));

        auto read_f = [](ggml_tensor* t, float* dst, size_t n) {
            if (t->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> buf(n);
                ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(ggml_fp16_t));
                for (size_t i = 0; i < n; i++)
                    dst[i] = ggml_fp16_to_fp32(buf[i]);
            } else {
                ggml_backend_tensor_get(t, dst, 0, n * sizeof(float));
            }
        };

        read_f(enc.lstm_wih[(size_t)layer], wih.data(), ws);
        read_f(enc.lstm_whh[(size_t)layer], whh.data(), ws);
        read_f(enc.lstm_bih[(size_t)layer], bih.data(), (size_t)(4 * C));
        read_f(enc.lstm_bhh[(size_t)layer], bhh.data(), (size_t)(4 * C));

        std::fill(h.begin(), h.end(), 0.0f);
        std::fill(cell.begin(), cell.end(), 0.0f);

        const float* x_ptr = (layer == 0) ? input : output.data();

        for (int t = 0; t < T; t++) {
            // gates = wih @ x + bih + whh @ h + bhh
            std::vector<float> gates((size_t)(4 * C), 0.0f);
            for (int g = 0; g < 4 * C; g++) {
                float sum_ih = bih[(size_t)g];
                float sum_hh = bhh[(size_t)g];
                for (int c = 0; c < C; c++) {
                    sum_ih += wih[(size_t)(g * C + c)] * x_ptr[(size_t)(c * T + t)];
                    sum_hh += whh[(size_t)(g * C + c)] * h[(size_t)c];
                }
                gates[(size_t)g] = sum_ih + sum_hh;
            }

            // i, f, g, o gates
            for (int c = 0; c < C; c++) {
                float i_gate = 1.0f / (1.0f + std::exp(-gates[(size_t)(0 * C + c)]));
                float f_gate = 1.0f / (1.0f + std::exp(-gates[(size_t)(1 * C + c)]));
                float g_gate = std::tanh(gates[(size_t)(2 * C + c)]);
                float o_gate = 1.0f / (1.0f + std::exp(-gates[(size_t)(3 * C + c)]));

                cell[(size_t)c] = f_gate * cell[(size_t)c] + i_gate * g_gate;
                h[(size_t)c] = o_gate * std::tanh(cell[(size_t)c]);
            }

            // Write output for this timestep
            for (int c = 0; c < C; c++) {
                output[(size_t)(c * T + t)] = h[(size_t)c];
            }
        }

        // For multi-layer LSTM, output of layer i is input to layer i+1
        // output already contains the result for next layer
    }

    // Add residual (LSTM in EnCodec has skip connection)
    for (size_t i = 0; i < (size_t)(C * T); i++) {
        output[i] += input[i];
    }

    return output;
}

static std::vector<float> encodec_decode(bark_context* ctx, const int32_t* fine_tokens, int n_codebooks,
                                         int n_timesteps) {
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "bark: EnCodec decode - %d codebooks x %d timesteps\n", n_codebooks, n_timesteps);
    }

    if (!ctx->encodec.loaded) {
        fprintf(stderr, "bark: EnCodec model not loaded\n");
        return {};
    }

    auto& enc = ctx->encodec;
    const int cb_dim = 128; // codebook dimension
    const int ratios[] = {8, 5, 4, 2};
    const int channels[] = {512, 256, 128, 64, 32}; // pre -> after each upsample

    // 1. RVQ dequantize: sum codebook embeddings
    std::vector<float> z((size_t)(cb_dim * n_timesteps), 0.0f);
    std::vector<ggml_fp16_t> emb_buf((size_t)cb_dim);
    for (int cb = 0; cb < n_codebooks && cb < 8; cb++) {
        for (int t = 0; t < n_timesteps; t++) {
            int tok = fine_tokens[cb * n_timesteps + t];
            if (tok < 0 || tok >= 1024)
                tok = 0;
            ggml_backend_tensor_get(enc.codebooks[(size_t)cb], emb_buf.data(),
                                    (size_t)tok * cb_dim * sizeof(ggml_fp16_t), (size_t)cb_dim * sizeof(ggml_fp16_t));
            for (int d = 0; d < cb_dim; d++) {
                z[(size_t)(d * n_timesteps + t)] += ggml_fp16_to_fp32(emb_buf[(size_t)d]);
            }
        }
    }

    // 2. Pre-conv (128 -> 512, k=7, p=3)
    std::vector<float> pre_w = fold_weight_norm(enc.pre_conv_w_g, enc.pre_conv_w_v);
    std::vector<float> pre_b((size_t)channels[0]);
    if (enc.pre_conv_b) {
        ggml_backend_tensor_get(enc.pre_conv_b, pre_b.data(), 0, (size_t)channels[0] * sizeof(float));
    }
    std::vector<float> h =
        cpu_conv1d(z.data(), cb_dim, n_timesteps, pre_w.data(), pre_b.data(), channels[0], 7, 1, 3, 1);
    int T = n_timesteps; // time dimension stays same after same-padding conv

    // 3. LSTM
    h = cpu_lstm_forward(h.data(), channels[0], T, enc);

    // 4. Upsample blocks
    for (int b = 0; b < 4; b++) {
        auto& blk = enc.blocks[b];
        int Cin = channels[b];
        int Cout = channels[b + 1];
        int stride = ratios[b];
        int K_up = stride * 2; // kernel = 2*stride for EnCodec ConvTranspose

        // ELU activation
        for (size_t i = 0; i < (size_t)(Cin * T); i++) {
            h[i] = elu_f(h[i]);
        }

        // ConvTranspose1d (Cin -> Cout)
        std::vector<float> up_w = fold_weight_norm(blk.convtr_w_g, blk.convtr_w_v);
        std::vector<float> up_b((size_t)Cout, 0.0f);
        if (blk.convtr_b) {
            ggml_backend_tensor_get(blk.convtr_b, up_b.data(), 0, (size_t)Cout * sizeof(float));
        }
        // Symmetric padding: pad = stride/2
        int pad = stride / 2;
        h = cpu_conv_transpose1d(h.data(), Cin, T, up_w.data(), up_b.data(), Cout, K_up, stride, pad);
        T = (int)h.size() / Cout;

        // ResBlock: ELU -> Conv(k=3, dilation=1) -> ELU -> Conv(k=1) + shortcut
        std::vector<float> residual = h;

        // First conv (Cout -> Cout/2, k=3, dilation=1, same-pad)
        int Cmid = Cout / 2;
        for (size_t i = 0; i < h.size(); i++)
            h[i] = elu_f(h[i]);
        std::vector<float> rw0 = fold_weight_norm(blk.res_conv0_w_g, blk.res_conv0_w_v);
        std::vector<float> rb0((size_t)Cmid, 0.0f);
        if (blk.res_conv0_b)
            ggml_backend_tensor_get(blk.res_conv0_b, rb0.data(), 0, (size_t)Cmid * sizeof(float));
        h = cpu_conv1d(h.data(), Cout, T, rw0.data(), rb0.data(), Cmid, 3, 1, 1, 1);

        // Second conv (Cout/2 -> Cout, k=1)
        for (size_t i = 0; i < h.size(); i++)
            h[i] = elu_f(h[i]);
        std::vector<float> rw1 = fold_weight_norm(blk.res_conv1_w_g, blk.res_conv1_w_v);
        std::vector<float> rb1((size_t)Cout, 0.0f);
        if (blk.res_conv1_b)
            ggml_backend_tensor_get(blk.res_conv1_b, rb1.data(), 0, (size_t)Cout * sizeof(float));
        h = cpu_conv1d(h.data(), Cmid, T, rw1.data(), rb1.data(), Cout, 1, 1, 0, 1);

        // Shortcut (Cout -> Cout, k=1)
        std::vector<float> sw = fold_weight_norm(blk.res_short_w_g, blk.res_short_w_v);
        std::vector<float> sb((size_t)Cout, 0.0f);
        if (blk.res_short_b)
            ggml_backend_tensor_get(blk.res_short_b, sb.data(), 0, (size_t)Cout * sizeof(float));
        std::vector<float> shortcut = cpu_conv1d(residual.data(), Cout, T, sw.data(), sb.data(), Cout, 1, 1, 0, 1);

        // Add residual
        for (size_t i = 0; i < h.size(); i++) {
            h[i] += shortcut[i];
        }
    }

    // 5. Final: ELU -> Conv1d(32, 1, k=7, p=3) -> tanh
    for (size_t i = 0; i < h.size(); i++)
        h[i] = elu_f(h[i]);

    std::vector<float> post_w = fold_weight_norm(enc.post_conv_w_g, enc.post_conv_w_v);
    std::vector<float> post_b(1, 0.0f);
    if (enc.post_conv_b)
        ggml_backend_tensor_get(enc.post_conv_b, post_b.data(), 0, sizeof(float));
    std::vector<float> pcm = cpu_conv1d(h.data(), channels[4], T, post_w.data(), post_b.data(), 1, 7, 1, 3, 1);

    // tanh
    for (size_t i = 0; i < pcm.size(); i++) {
        pcm[i] = std::tanh(pcm[i]);
    }

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "bark: EnCodec produced %d samples (%.2f sec)\n", (int)pcm.size(),
                (float)pcm.size() / 24000.0f);
    }

    return pcm;
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
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend, "bark", wl)) {
        fprintf(stderr, "bark: failed to load '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }

    ctx->ctx_w = wl.ctx;
    ctx->buf_w = wl.buf;
    ctx->tensors = std::move(wl.tensors);

    // Load metadata + tokenizer vocab
    gguf_context* g = gguf_init_from_file(path_model, {/*.no_alloc=*/true, /*.ctx=*/nullptr});
    if (g) {
        load_metadata(ctx, g);
        // Load BERT vocab from GGUF
        ctx->tokenizer.id_to_token = core_gguf::kv_str_array(g, "tokenizer.ggml.tokens");
        if (!ctx->tokenizer.id_to_token.empty()) {
            ctx->tokenizer.build_map();
            if (params.verbosity >= 1) {
                fprintf(stderr, "bark: loaded BERT vocab (%d tokens)\n", (int)ctx->tokenizer.id_to_token.size());
            }
        } else {
            fprintf(stderr, "bark: WARNING - no tokenizer vocab in GGUF, will use byte fallback\n");
        }
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

    if (!bind_encodec_model(ctx, ctx->encodec)) {
        fprintf(stderr, "bark: failed to bind EnCodec model\n");
        delete ctx;
        return nullptr;
    }

    // Allocate compute metadata buffer for graph building
    ctx->compute_meta.resize(GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead());

    // Setup scheduler
    ggml_backend_t backends[] = {ctx->backend};
    ctx->sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, false);

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
