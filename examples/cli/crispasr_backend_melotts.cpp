// crispasr_backend_melotts.cpp — MeloTTS (VITS2) backend adapter.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "melotts.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

class MelottsBackend : public CrispasrBackend {
public:
    const char * name() const override { return "melotts"; }

    uint32_t capabilities() const override {
        return CAP_TTS;
    }

    bool init(const whisper_params & p) override {
        struct melotts_params mp = melotts_default_params();
        mp.n_threads  = p.n_threads;
        mp.verbosity  = p.no_prints ? 0 : 1;
        mp.use_gpu    = crispasr_backend_should_use_gpu(p);
        mp.speaker_id = p.tts_speaker_id;
        if (p.seed > 0) mp.seed = (uint32_t)p.seed;

        ctx_ = melotts_init_from_file(p.model.c_str(), mp);
        return ctx_ != nullptr;
    }

    std::vector<float> synthesize(const std::string & text,
                                  const whisper_params & params) override {
        if (!ctx_) return {};

        // Apply runtime params
        if (params.tts_speed > 0) {
            melotts_set_length_scale(ctx_, 1.0f / params.tts_speed);
        }

        float * pcm = nullptr;
        int n = melotts_synthesize(ctx_, text.c_str(), &pcm, nullptr);
        if (!pcm || n <= 0) return {};

        std::vector<float> out(pcm, pcm + n);
        melotts_pcm_free(pcm);
        return out;
    }

    int tts_sample_rate() const override {
        return ctx_ ? melotts_sample_rate(ctx_) : 44100;
    }

    std::vector<crispasr_segment> transcribe(
        const float *, int, int64_t, const whisper_params &) override {
        return {}; // TTS-only
    }

    void shutdown() override {
        if (ctx_) {
            melotts_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    melotts_context * ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_melotts_backend() {
    return std::unique_ptr<CrispasrBackend>(new MelottsBackend());
}
