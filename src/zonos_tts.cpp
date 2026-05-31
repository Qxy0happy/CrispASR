// zonos_tts.cpp -- Zonos TTS backend (Zyphra/Zonos-v0.1-transformer).
//
// PLAN #130: Apache 2.0 licensed ~500M-param TTS with rich conditioning.
//
// Pipeline:
//   1. Text -> eSpeak phoneme IDs -> phoneme embedding lookup
//   2. Build conditioning prefix: concat(phoneme_embs, speaker_emb,
//      emotion_fourier, fmax_fourier, pitch_std_fourier, rate_fourier,
//      language_int_emb) -> LayerNorm -> linear projection
//   3. CFG: stack [cond_prefix; uncond_prefix] for classifier-free guidance
//   4. Prefill: prefix -> backbone -> first 9 codebook logits
//   5. AR decode loop: each step generates 9 codebook tokens via delay
//      pattern, sampling with min_p=0.1 from upstream
//   6. Revert delay pattern -> 9 codebook streams
//   7. DAC decoder: codes -> 44.1 kHz PCM (separate GGUF)
//
// Weight layout (from convert-zonos-to-gguf.py):
//   backbone.blk.N.attn_qkv.weight    [3072, 2048]  fused Q/K/V
//   backbone.blk.N.attn_output.weight  [2048, 2048]
//   backbone.blk.N.ffn_gate_up.weight  [16384, 2048] fused SwiGLU gate+up
//   backbone.blk.N.ffn_down.weight     [2048, 8192]
//   backbone.blk.N.attn_norm.{weight,bias}  [2048]   LayerNorm
//   backbone.blk.N.ffn_norm.{weight,bias}   [2048]   LayerNorm
//   backbone.output_norm.{weight,bias}      [2048]   final LayerNorm
//   embeddings.K.weight                [1026, 2048]   per-codebook
//   heads.K.weight                     [1025, 2048]   per-codebook
//   prefix_conditioner.*               conditioning weights

#include "zonos_tts.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// -----------------------------------------------------------------------
// Internal structures
// -----------------------------------------------------------------------

namespace {

struct zonos_hp {
    uint32_t d_model = 2048;
    uint32_t n_layer = 26;
    uint32_t n_heads = 16;
    uint32_t n_kv_heads = 4;
    uint32_t head_dim = 128;
    uint32_t ff_dim = 8192;
    float norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    uint32_t rope_dim = 128;

    uint32_t eos_token_id = 1024;
    uint32_t masked_token_id = 1025;
    uint32_t n_codebooks = 9;
    uint32_t codebook_size = 1024;
    uint32_t sample_rate = 44100;

    // Conditioner config
    uint32_t n_conditioners = 7;
    uint32_t phoneme_vocab_size = 189; // 4 special + 185 symbols
    uint32_t n_languages = 100;

    // Derived
    uint32_t emb_vocab_size = 1026;  // codebook_size + 2 (eos + mask)
    uint32_t head_vocab_size = 1025; // codebook_size + 1 (eos)
};

struct zonos_layer {
    // Pre-attention LayerNorm
    ggml_tensor* attn_norm_w = nullptr; // (d_model,)
    ggml_tensor* attn_norm_b = nullptr; // (d_model,)
    // Fused Q/K/V projection
    ggml_tensor* attn_qkv_w = nullptr;    // (d_model, (n_heads+2*n_kv_heads)*head_dim)
    ggml_tensor* attn_output_w = nullptr; // (n_heads*head_dim, d_model)
    // Pre-FFN LayerNorm
    ggml_tensor* ffn_norm_w = nullptr; // (d_model,)
    ggml_tensor* ffn_norm_b = nullptr; // (d_model,)
    // Fused gate+up (SwiGLU)
    ggml_tensor* ffn_gate_up_w = nullptr; // (d_model, 2*ff_dim)
    ggml_tensor* ffn_down_w = nullptr;    // (ff_dim, d_model)
};

struct zonos_backbone {
    std::vector<zonos_layer> layers;
    ggml_tensor* output_norm_w = nullptr; // (d_model,)
    ggml_tensor* output_norm_b = nullptr; // (d_model,)
};

struct zonos_conditioner_weights {
    // Phoneme embedder (conditioner 0 - EspeakPhonemeConditioner)
    ggml_tensor* phoneme_emb_w = nullptr; // (phoneme_vocab_size, d_model)

    // Speaker projection (conditioner 1 - PassthroughConditioner + linear)
    ggml_tensor* speaker_proj_w = nullptr; // (cond_dim=128, d_model)
    ggml_tensor* speaker_proj_b = nullptr; // (d_model,)
    ggml_tensor* speaker_uncond = nullptr; // (d_model,)

    // Fourier conditioners (2=emotion, 3=fmax, 4=pitch_std, 5=speaking_rate)
    // Each has a random Fourier feature weight and an uncond vector
    struct fourier_cond {
        ggml_tensor* weight = nullptr;     // (d_model/2, input_dim)
        ggml_tensor* uncond_vec = nullptr; // (d_model,)
    };
    fourier_cond emotion;       // input_dim=8
    fourier_cond fmax;          // input_dim=1
    fourier_cond pitch_std;     // input_dim=1
    fourier_cond speaking_rate; // input_dim=1

    // Integer conditioner (6 - language_id)
    ggml_tensor* lang_emb_w = nullptr;  // (n_languages, d_model)
    ggml_tensor* lang_uncond = nullptr; // (d_model,)

    // Prefix conditioner top-level
    ggml_tensor* norm_w = nullptr; // (d_model,)
    ggml_tensor* norm_b = nullptr; // (d_model,)
    ggml_tensor* proj_w = nullptr; // (d_model, d_model)
    ggml_tensor* proj_b = nullptr; // (d_model,)
};

// Conditioning state (set by the user before synthesis)
struct zonos_cond_state {
    // Speaker embedding (128-d LDA projected)
    std::vector<float> speaker_emb; // 128 floats

    // Emotion vector (8 floats, normalized to sum=1)
    float emotion[8] = {0.3077f, 0.0256f, 0.0256f, 0.0256f, 0.0256f, 0.0256f, 0.2564f, 0.3077f};

    float fmax = 22050.0f;
    float pitch_std = 20.0f;
    float speaking_rate = 15.0f;
    int language_id = 25; // default: en-us (index 25 in the language list)

    // Language code strings for lookup
    std::vector<std::string> language_codes;
};

} // namespace

// -----------------------------------------------------------------------
// Context
// -----------------------------------------------------------------------

struct zonos_tts_context {
    zonos_tts_params params{};
    int n_threads = 4;

    zonos_hp hp;
    zonos_backbone backbone;
    zonos_conditioner_weights cond_w;
    zonos_cond_state cond_state;

    // Per-codebook embeddings and heads
    std::vector<ggml_tensor*> emb_w;  // [n_codebooks] each (emb_vocab_size, d_model)
    std::vector<ggml_tensor*> head_w; // [n_codebooks] each (head_vocab_size, d_model)

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Compute scheduler
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // KV cache
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr; // (head_dim, max_ctx, n_kv_heads, n_layer)
    ggml_tensor* kv_v = nullptr; // (head_dim, max_ctx, n_kv_heads, n_layer)
    int kv_max_ctx = 0;

    // DAC codec path (loaded lazily)
    std::string dac_codec_path;

    // Sampler RNG
    uint64_t rng_state = 0xdeadbeefcafebabeULL;
};

// -----------------------------------------------------------------------
// Defaults
// -----------------------------------------------------------------------

struct zonos_tts_params zonos_tts_default_params(void) {
    struct zonos_tts_params p;
    std::memset(&p, 0, sizeof(p));
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.temperature = 0.0f; // greedy by default; upstream uses min_p=0.1
    p.seed = 0;
    p.max_audio_tokens = 0; // 0 = default (86*30=2580)
    p.flash_attn = false;
    p.cfg_scale = 2.0f;
    return p;
}

// -----------------------------------------------------------------------
// GGUF loading
// -----------------------------------------------------------------------

static bool load_hparams(struct gguf_context* gguf_ctx, zonos_hp& hp) {
    auto get_u32 = [&](const char* key, uint32_t def) -> uint32_t {
        int idx = gguf_find_key(gguf_ctx, key);
        return idx >= 0 ? (uint32_t)gguf_get_val_u32(gguf_ctx, idx) : def;
    };
    auto get_f32 = [&](const char* key, float def) -> float {
        int idx = gguf_find_key(gguf_ctx, key);
        return idx >= 0 ? gguf_get_val_f32(gguf_ctx, idx) : def;
    };

    hp.d_model = get_u32("zonos.d_model", 2048);
    hp.n_layer = get_u32("zonos.n_layer", 26);
    hp.n_heads = get_u32("zonos.n_heads", 16);
    hp.n_kv_heads = get_u32("zonos.n_kv_heads", 4);
    hp.head_dim = get_u32("zonos.head_dim", 128);
    hp.ff_dim = get_u32("zonos.ff_dim", 8192);
    hp.norm_eps = get_f32("zonos.norm_eps", 1e-5f);
    hp.rope_theta = get_f32("zonos.rope_theta", 10000.0f);
    hp.rope_dim = get_u32("zonos.rope_dim", 128);
    hp.eos_token_id = get_u32("zonos.eos_token_id", 1024);
    hp.masked_token_id = get_u32("zonos.masked_token_id", 1025);
    hp.n_codebooks = get_u32("zonos.n_codebooks", 9);
    hp.codebook_size = get_u32("zonos.codebook_size", 1024);
    hp.sample_rate = get_u32("zonos.sample_rate", 44100);
    hp.n_conditioners = get_u32("zonos.n_conditioners", 7);
    hp.phoneme_vocab_size = get_u32("zonos.phoneme_vocab_size", 189);
    hp.n_languages = get_u32("zonos.n_languages", 100);

    hp.emb_vocab_size = hp.codebook_size + 2;  // eos + mask
    hp.head_vocab_size = hp.codebook_size + 1; // eos

    return true;
}

static bool load_language_codes(struct gguf_context* gguf_ctx, zonos_cond_state& state) {
    int idx = gguf_find_key(gguf_ctx, "zonos.language_codes");
    if (idx < 0)
        return false;
    const char* str = gguf_get_val_str(gguf_ctx, idx);
    if (!str)
        return false;

    // Parse newline-separated language codes
    std::string s(str);
    size_t pos = 0;
    while (pos < s.size()) {
        size_t next = s.find('\n', pos);
        if (next == std::string::npos)
            next = s.size();
        if (next > pos) {
            state.language_codes.push_back(s.substr(pos, next - pos));
        }
        pos = next + 1;
    }
    return true;
}

// -----------------------------------------------------------------------
// Init
// -----------------------------------------------------------------------

struct zonos_tts_context* zonos_tts_init_from_file(const char* path_model, struct zonos_tts_params params) {
    if (!path_model)
        return nullptr;

    struct gguf_init_params gguf_params = {
        /*.no_alloc =*/true,
        /*.ctx      =*/nullptr,
    };

    struct gguf_context* gguf_ctx = gguf_init_from_file(path_model, gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "zonos_tts: failed to load %s\n", path_model);
        return nullptr;
    }

    auto* ctx = new zonos_tts_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    if (!load_hparams(gguf_ctx, ctx->hp)) {
        fprintf(stderr, "zonos_tts: failed to load hyperparameters\n");
        gguf_free(gguf_ctx);
        delete ctx;
        return nullptr;
    }

    load_language_codes(gguf_ctx, ctx->cond_state);

    const auto& hp = ctx->hp;
    if (params.verbosity >= 1) {
        fprintf(stderr, "zonos_tts: d_model=%u n_layer=%u n_heads=%u n_kv=%u\n", hp.d_model, hp.n_layer, hp.n_heads,
                hp.n_kv_heads);
        fprintf(stderr, "zonos_tts: ff_dim=%u head_dim=%u n_codebooks=%u\n", hp.ff_dim, hp.head_dim, hp.n_codebooks);
    }

    // Count tensors
    const int n_tensors = gguf_get_n_tensors(gguf_ctx);

    // Create ggml context for weight metadata
    size_t ctx_size = ggml_tensor_overhead() * (size_t)(n_tensors + 1);
    struct ggml_init_params gp = {
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ctx->ctx_w = ggml_init(gp);
    if (!ctx->ctx_w) {
        fprintf(stderr, "zonos_tts: ggml_init failed\n");
        gguf_free(gguf_ctx);
        delete ctx;
        return nullptr;
    }

    // Create tensors from GGUF metadata
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf_ctx, i);
        ggml_tensor* t = ggml_get_tensor(ctx->ctx_w, name);
        if (!t) {
            // Need to create it
            enum ggml_type type = gguf_get_tensor_type(gguf_ctx, i);
            int n_dims = gguf_get_tensor_n_dims(gguf_ctx, i);
            int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
            for (int d = 0; d < n_dims; d++) {
                ne[d] = gguf_get_tensor_ne(gguf_ctx, i, d);
            }
            t = ggml_new_tensor(ctx->ctx_w, type, n_dims, ne);
            ggml_set_name(t, name);
        }
        ctx->tensors[name] = t;
    }

    // Allocate backend and buffer
    ctx->backend_cpu = ggml_backend_cpu_init();
    ctx->backend = ctx->backend_cpu; // CPU-only for now

    ctx->buf_w = ggml_backend_alloc_ctx_tensors(ctx->ctx_w, ctx->backend);
    if (!ctx->buf_w) {
        fprintf(stderr, "zonos_tts: buffer allocation failed\n");
        ggml_free(ctx->ctx_w);
        gguf_free(gguf_ctx);
        delete ctx;
        return nullptr;
    }

    // Load tensor data from GGUF
    FILE* fin = fopen(path_model, "rb");
    if (!fin) {
        fprintf(stderr, "zonos_tts: cannot open %s for reading\n", path_model);
        ggml_free(ctx->ctx_w);
        gguf_free(gguf_ctx);
        delete ctx;
        return nullptr;
    }

    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf_ctx, i);
        auto it = ctx->tensors.find(name);
        if (it == ctx->tensors.end())
            continue;
        ggml_tensor* t = it->second;
        size_t offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, i);
        size_t nbytes = ggml_nbytes(t);
        std::vector<uint8_t> buf(nbytes);
        fseek(fin, (long)offset, SEEK_SET);
        if (fread(buf.data(), 1, nbytes, fin) != nbytes) {
            fprintf(stderr, "zonos_tts: short read for tensor %s\n", name);
        }
        ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
    }
    fclose(fin);

    // Wire up weight pointers
    auto find_t = [&](const char* name) -> ggml_tensor* {
        auto it = ctx->tensors.find(name);
        return it != ctx->tensors.end() ? it->second : nullptr;
    };

    // Backbone layers
    ctx->backbone.layers.resize(hp.n_layer);
    for (uint32_t i = 0; i < hp.n_layer; i++) {
        auto& layer = ctx->backbone.layers[i];
        char buf[128];

        snprintf(buf, sizeof(buf), "backbone.blk.%u.attn_norm.weight", i);
        layer.attn_norm_w = find_t(buf);
        snprintf(buf, sizeof(buf), "backbone.blk.%u.attn_norm.bias", i);
        layer.attn_norm_b = find_t(buf);
        snprintf(buf, sizeof(buf), "backbone.blk.%u.attn_qkv.weight", i);
        layer.attn_qkv_w = find_t(buf);
        snprintf(buf, sizeof(buf), "backbone.blk.%u.attn_output.weight", i);
        layer.attn_output_w = find_t(buf);
        snprintf(buf, sizeof(buf), "backbone.blk.%u.ffn_norm.weight", i);
        layer.ffn_norm_w = find_t(buf);
        snprintf(buf, sizeof(buf), "backbone.blk.%u.ffn_norm.bias", i);
        layer.ffn_norm_b = find_t(buf);
        snprintf(buf, sizeof(buf), "backbone.blk.%u.ffn_gate_up.weight", i);
        layer.ffn_gate_up_w = find_t(buf);
        snprintf(buf, sizeof(buf), "backbone.blk.%u.ffn_down.weight", i);
        layer.ffn_down_w = find_t(buf);
    }
    ctx->backbone.output_norm_w = find_t("backbone.output_norm.weight");
    ctx->backbone.output_norm_b = find_t("backbone.output_norm.bias");

    // Codebook embeddings and heads
    ctx->emb_w.resize(hp.n_codebooks);
    ctx->head_w.resize(hp.n_codebooks);
    for (uint32_t k = 0; k < hp.n_codebooks; k++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "embeddings.%u.weight", k);
        ctx->emb_w[k] = find_t(buf);
        snprintf(buf, sizeof(buf), "heads.%u.weight", k);
        ctx->head_w[k] = find_t(buf);
    }

    // Conditioner weights
    auto& cw = ctx->cond_w;
    cw.phoneme_emb_w = find_t("prefix_conditioner.conditioners.0.phoneme_embedder.weight");
    cw.speaker_proj_w = find_t("prefix_conditioner.conditioners.1.project.weight");
    cw.speaker_proj_b = find_t("prefix_conditioner.conditioners.1.project.bias");
    cw.speaker_uncond = find_t("prefix_conditioner.conditioners.1.uncond_vector");

    cw.emotion.weight = find_t("prefix_conditioner.conditioners.2.weight");
    cw.emotion.uncond_vec = find_t("prefix_conditioner.conditioners.2.uncond_vector");
    cw.fmax.weight = find_t("prefix_conditioner.conditioners.3.weight");
    cw.fmax.uncond_vec = find_t("prefix_conditioner.conditioners.3.uncond_vector");
    cw.pitch_std.weight = find_t("prefix_conditioner.conditioners.4.weight");
    cw.pitch_std.uncond_vec = find_t("prefix_conditioner.conditioners.4.uncond_vector");
    cw.speaking_rate.weight = find_t("prefix_conditioner.conditioners.5.weight");
    cw.speaking_rate.uncond_vec = find_t("prefix_conditioner.conditioners.5.uncond_vector");

    cw.lang_emb_w = find_t("prefix_conditioner.conditioners.6.int_embedder.weight");
    cw.lang_uncond = find_t("prefix_conditioner.conditioners.6.uncond_vector");

    cw.norm_w = find_t("prefix_conditioner.norm.weight");
    cw.norm_b = find_t("prefix_conditioner.norm.bias");
    cw.proj_w = find_t("prefix_conditioner.project.weight");
    cw.proj_b = find_t("prefix_conditioner.project.bias");

    // Set default speaker embedding (random, will be overwritten by set_voice)
    ctx->cond_state.speaker_emb.resize(128, 0.0f);

    // Initialize RNG
    if (params.seed != 0) {
        ctx->rng_state = params.seed;
    }

    if (params.verbosity >= 1) {
        fprintf(stderr, "zonos_tts: loaded %d tensors from %s\n", n_tensors, path_model);
    }

    gguf_free(gguf_ctx);

    // Create compute scheduler
    ctx->sched = ggml_backend_sched_new(&ctx->backend, nullptr, 1, 4096, false);
    ctx->compute_meta.resize(ggml_tensor_overhead() * 4096);

    return ctx;
}

// -----------------------------------------------------------------------
// Conditioning setters
// -----------------------------------------------------------------------

void zonos_tts_set_pitch_std(struct zonos_tts_context* ctx, float pitch_std) {
    if (ctx)
        ctx->cond_state.pitch_std = pitch_std;
}

void zonos_tts_set_speaking_rate(struct zonos_tts_context* ctx, float rate) {
    if (ctx)
        ctx->cond_state.speaking_rate = rate;
}

void zonos_tts_set_emotion(struct zonos_tts_context* ctx, const float* emotion, int len) {
    if (!ctx || !emotion || len < 1)
        return;
    float sum = 0.0f;
    for (int i = 0; i < 8 && i < len; i++) {
        ctx->cond_state.emotion[i] = emotion[i];
        sum += emotion[i];
    }
    // Normalize
    if (sum > 0.0f) {
        for (int i = 0; i < 8; i++) {
            ctx->cond_state.emotion[i] /= sum;
        }
    }
}

void zonos_tts_set_fmax(struct zonos_tts_context* ctx, float fmax) {
    if (ctx)
        ctx->cond_state.fmax = fmax;
}

int zonos_tts_set_language(struct zonos_tts_context* ctx, const char* lang_code) {
    if (!ctx || !lang_code)
        return -1;
    for (size_t i = 0; i < ctx->cond_state.language_codes.size(); i++) {
        if (ctx->cond_state.language_codes[i] == lang_code) {
            ctx->cond_state.language_id = (int)i;
            return 0;
        }
    }
    fprintf(stderr, "zonos_tts: unknown language code '%s'\n", lang_code);
    return -1;
}

void zonos_tts_set_speaker_embedding(struct zonos_tts_context* ctx, const float* emb, int dim) {
    if (!ctx || !emb || dim < 1)
        return;
    ctx->cond_state.speaker_emb.assign(emb, emb + dim);
}

int zonos_tts_set_codec_path(struct zonos_tts_context* ctx, const char* path) {
    if (!ctx || !path)
        return -1;
    ctx->dac_codec_path = path;
    return 0;
}

int zonos_tts_set_voice(struct zonos_tts_context* ctx, const char* wav_path) {
    if (!ctx || !wav_path)
        return -1;
    // TODO: implement ResNet293 speaker encoder in C++ or load pre-computed
    // embeddings. For now, this is a stub that requires pre-computed embeddings
    // via zonos_tts_set_speaker_embedding().
    fprintf(stderr, "zonos_tts: set_voice not yet implemented; use set_speaker_embedding\n");
    return -1;
}

// -----------------------------------------------------------------------
// Phoneme tokenization (simplified -- maps IPA characters to embedding IDs)
// -----------------------------------------------------------------------

namespace {

// The phoneme symbol table from conditioning.py, indexed starting at 4
// (after PAD=0, UNK=1, BOS=2, EOS=3).
static const char* const PHONEME_SPECIAL_NAMES[] = {"<pad>", "<unk>", "<bos>", "<eos>"};

// Build a simple char -> ID map. The full eSpeak phonemizer runs
// externally; this is a fallback for when phonemes are pre-computed
// or for simple Latin-script inputs.
static std::unordered_map<uint32_t, int> build_phoneme_map() {
    // These must match the symbols list in conditioning.py exactly
    static const char punctuation[] = ";:,.!?\xc2\xa1\xc2\xbf\xe2\x80\x94\xe2\x80\xa6\""
                                      "\xc2\xab\xc2\xbb\xe2\x80\x9c\xe2\x80\x9d() *~-/\\&";
    static const char letters[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

    std::unordered_map<uint32_t, int> map;
    int id = 4; // start after special tokens

    // We just map ASCII printable + common IPA to sequential IDs.
    // The converter script embeds the full symbol table as GGUF metadata.
    // For now, map basic ASCII characters.
    for (const char* p = punctuation; *p;) {
        // UTF-8 decode
        uint32_t cp = 0;
        uint8_t c = (uint8_t)*p;
        if (c < 0x80) {
            cp = c;
            p++;
        } else if ((c & 0xE0) == 0xC0) {
            cp = (c & 0x1F) << 6 | ((uint8_t)p[1] & 0x3F);
            p += 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = (c & 0x0F) << 12 | ((uint8_t)p[1] & 0x3F) << 6 | ((uint8_t)p[2] & 0x3F);
            p += 3;
        } else {
            cp = (c & 0x07) << 18 | ((uint8_t)p[1] & 0x3F) << 12 | ((uint8_t)p[2] & 0x3F) << 6 | ((uint8_t)p[3] & 0x3F);
            p += 4;
        }
        map[cp] = id++;
    }
    for (const char* p = letters; *p; p++) {
        map[(uint32_t)*p] = id++;
    }
    // IPA symbols would continue here (id keeps incrementing).
    // The full IPA table is embedded in GGUF metadata for production use.
    return map;
}

static std::vector<int32_t> tokenize_text_simple(const char* text) {
    static auto map = build_phoneme_map();

    std::vector<int32_t> ids;
    ids.push_back(2); // BOS

    for (const char* p = text; *p;) {
        uint32_t cp = 0;
        uint8_t c = (uint8_t)*p;
        if (c < 0x80) {
            cp = c;
            p++;
        } else if ((c & 0xE0) == 0xC0) {
            cp = (c & 0x1F) << 6 | ((uint8_t)p[1] & 0x3F);
            p += 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = (c & 0x0F) << 12 | ((uint8_t)p[1] & 0x3F) << 6 | ((uint8_t)p[2] & 0x3F);
            p += 3;
        } else {
            cp = (c & 0x07) << 18 | ((uint8_t)p[1] & 0x3F) << 12 | ((uint8_t)p[2] & 0x3F) << 6 | ((uint8_t)p[3] & 0x3F);
            p += 4;
        }
        auto it = map.find(cp);
        if (it != map.end()) {
            ids.push_back(it->second);
        } else {
            ids.push_back(1); // UNK
        }
    }

    ids.push_back(3); // EOS
    return ids;
}

} // namespace

// -----------------------------------------------------------------------
// Delay pattern (from codebook_pattern.py)
// -----------------------------------------------------------------------

namespace {

// Apply delay pattern: shift codebook k by (k+1) positions, fill with mask_token.
// Input codes: (n_codebooks, seq_len), output: (n_codebooks, seq_len + n_codebooks)
static void apply_delay_pattern(const std::vector<std::vector<int32_t>>& codes, int32_t mask_token,
                                std::vector<std::vector<int32_t>>& out) {
    int n_cb = (int)codes.size();
    int seq_len = codes.empty() ? 0 : (int)codes[0].size();
    int out_len = seq_len + n_cb;
    out.resize(n_cb);
    for (int k = 0; k < n_cb; k++) {
        out[k].resize(out_len, mask_token);
        // Roll by (k+1): codes[k][i] goes to position (i + k + 1) % out_len
        for (int i = 0; i < seq_len; i++) {
            int dest = (i + k + 1) % out_len;
            out[k][dest] = codes[k][i];
        }
    }
}

// Revert delay pattern: undo the shift.
// Input: (n_codebooks, delayed_len), output: (n_codebooks, delayed_len - n_codebooks)
static void revert_delay_pattern(const std::vector<std::vector<int32_t>>& delayed,
                                 std::vector<std::vector<int32_t>>& out) {
    int n_cb = (int)delayed.size();
    if (n_cb == 0)
        return;
    int delayed_len = (int)delayed[0].size();
    int out_len = delayed_len - n_cb;
    if (out_len <= 0) {
        out.clear();
        return;
    }
    out.resize(n_cb);
    for (int k = 0; k < n_cb; k++) {
        out[k].resize(out_len);
        for (int i = 0; i < out_len; i++) {
            out[k][i] = delayed[k][k + 1 + i];
        }
    }
}

} // namespace

// -----------------------------------------------------------------------
// Stub synthesis (skeleton -- graph building is next step)
// -----------------------------------------------------------------------

float* zonos_tts_synthesize(struct zonos_tts_context* ctx, const char* text, int* out_n_samples) {
    if (!ctx || !text || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "zonos_tts: synthesize '%s'\n", text);
    }

    // Step 1: Tokenize text to phoneme IDs
    auto phoneme_ids = tokenize_text_simple(text);
    if (ctx->params.verbosity >= 2) {
        fprintf(stderr, "zonos_tts: %zu phoneme tokens\n", phoneme_ids.size());
    }

    // Step 2: Build conditioning prefix
    // TODO: build ggml graph for prefix conditioning
    //   - phoneme_emb = phoneme_embedder(phoneme_ids)  -> (n_phonemes, d_model)
    //   - speaker_emb = project(speaker_128d)           -> (1, d_model)
    //   - emotion_emb = fourier(emotion_8d)             -> (1, d_model)
    //   - fmax_emb    = fourier(fmax_1d)                -> (1, d_model)
    //   - pitch_emb   = fourier(pitch_std_1d)           -> (1, d_model)
    //   - rate_emb    = fourier(speaking_rate_1d)        -> (1, d_model)
    //   - lang_emb    = int_embedder(language_id)        -> (1, d_model)
    //   - cond_concat = cat(phoneme, speaker, emotion, fmax, pitch, rate, lang)
    //   - cond_prefix = project(norm(cond_concat))       -> (cond_len, d_model)
    //
    //   For CFG: also build uncond prefix using uncond_vectors
    //   - uncond_prefix = project(norm(cat(phoneme, uncond_speaker, uncond_emotion, ...)))

    // Step 3: AR decode with delay pattern
    // TODO: build backbone transformer graph
    //   - prefill: [cond_prefix; uncond_prefix] + first delay frame
    //   - decode loop: 1 token per step, 9 codebook logits per step
    //   - sampling with min_p or temperature
    //   - stop when codebook 0 emits EOS, then drain remaining codebooks

    // Step 4: DAC decode
    // TODO: load DAC GGUF and run decoder graph
    //   - codes (9, seq_len) -> quantizer lookup + sum -> decoder conv stack -> PCM

    fprintf(stderr, "zonos_tts: synthesis graph not yet implemented (skeleton)\n");
    return nullptr;
}

int32_t* zonos_tts_synthesize_codes(struct zonos_tts_context* ctx, const char* text, int* out_n_codes,
                                    int* out_n_codebooks) {
    if (!ctx || !text)
        return nullptr;
    if (out_n_codes)
        *out_n_codes = 0;
    if (out_n_codebooks)
        *out_n_codebooks = 0;

    fprintf(stderr, "zonos_tts: synthesize_codes not yet implemented (skeleton)\n");
    return nullptr;
}

// -----------------------------------------------------------------------
// Cleanup
// -----------------------------------------------------------------------

void zonos_tts_codes_free(int32_t* codes) {
    free(codes);
}
void zonos_tts_pcm_free(float* pcm) {
    free(pcm);
}

void zonos_tts_free(struct zonos_tts_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->buf_w)
        ggml_backend_buffer_free(ctx->buf_w);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

void zonos_tts_set_n_threads(struct zonos_tts_context* ctx, int n_threads) {
    if (ctx)
        ctx->n_threads = n_threads > 0 ? n_threads : 1;
}

void zonos_tts_set_temperature(struct zonos_tts_context* ctx, float temperature) {
    if (ctx)
        ctx->params.temperature = temperature;
}

void zonos_tts_set_seed(struct zonos_tts_context* ctx, uint64_t seed) {
    if (ctx) {
        ctx->params.seed = seed;
        ctx->rng_state = seed ? seed : 0xdeadbeefcafebabeULL;
    }
}

void zonos_tts_set_cfg_scale(struct zonos_tts_context* ctx, float cfg_scale) {
    if (ctx)
        ctx->params.cfg_scale = cfg_scale;
}
