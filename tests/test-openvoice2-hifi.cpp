// Minimal test: load OpenVoice2 GGUF, feed known z to HiFi-GAN, check output range.
#include "openvoice2.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <openvoice2-tcc.gguf>\n", argv[0]);
        return 1;
    }

    auto cp = openvoice2_context_default_params();
    cp.verbosity = 2;
    cp.tau = 0.3f;
    auto* ctx = openvoice2_init_from_file(argv[1], cp);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Test: convert JFK audio with itself as reference (identity test)
    // Read JFK WAV
    const char* jfk_path = "samples/jfk.wav";
    FILE* f = fopen(jfk_path, "rb");
    if (!f) {
        // Try from build dir
        jfk_path = "../samples/jfk.wav";
        f = fopen(jfk_path, "rb");
    }
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", jfk_path);
        openvoice2_free(ctx);
        return 1;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> wav_data(fsize);
    fread(wav_data.data(), 1, fsize, f);
    fclose(f);

    // Parse WAV — find data chunk
    int sr = 16000, bits = 16;
    std::vector<float> pcm;
    size_t pos = 12;
    while (pos < wav_data.size() - 8) {
        uint32_t csz = *(uint32_t*)(wav_data.data() + pos + 4);
        if (memcmp(wav_data.data() + pos, "fmt ", 4) == 0) {
            sr = *(int32_t*)(wav_data.data() + pos + 12);
            bits = *(int16_t*)(wav_data.data() + pos + 22);
        } else if (memcmp(wav_data.data() + pos, "data", 4) == 0) {
            int n = csz / (bits / 8);
            pcm.resize(n);
            const int16_t* samples = (const int16_t*)(wav_data.data() + pos + 8);
            for (int i = 0; i < n; i++) pcm[i] = samples[i] / 32768.0f;
            break;
        }
        pos += 8 + csz;
    }
    fprintf(stderr, "Loaded %zu samples @ %d Hz\n", pcm.size(), sr);

    float* out = nullptr;
    int n_out = 0;
    bool ok = openvoice2_convert(ctx, pcm.data(), (int)pcm.size(), sr,
                                  pcm.data(), (int)pcm.size(), sr,
                                  &out, &n_out);

    if (ok && out && n_out > 0) {
        float mn = out[0], mx = out[0];
        double sum = 0;
        for (int i = 0; i < n_out; i++) {
            if (out[i] < mn) mn = out[i];
            if (out[i] > mx) mx = out[i];
            sum += out[i];
        }
        fprintf(stderr, "Output: %d samples, min=%.6f max=%.6f mean=%.6f\n",
                n_out, mn, mx, (float)(sum / n_out));

        if (mx > 0.1f || mn < -0.1f) {
            fprintf(stderr, "PASS: audio has content (range > 0.1)\n");
        } else {
            fprintf(stderr, "FAIL: audio is near-silent (range < 0.1)\n");
            free(out);
            openvoice2_free(ctx);
            return 1;
        }

        // Write WAV for ASR roundtrip
        {
            const char* wav_out = "/tmp/ov2-hifi-test.wav";
            FILE* wf = fopen(wav_out, "wb");
            if (wf) {
                int16_t* pcm16 = (int16_t*)malloc(n_out * 2);
                for (int i = 0; i < n_out; i++) {
                    float v = out[i] * 32767.0f;
                    if (v > 32767.0f) v = 32767.0f;
                    if (v < -32768.0f) v = -32768.0f;
                    pcm16[i] = (int16_t)v;
                }
                uint32_t data_size = n_out * 2;
                uint32_t riff_size = 36 + data_size;
                fwrite("RIFF", 1, 4, wf);
                fwrite(&riff_size, 4, 1, wf);
                fwrite("WAVE", 1, 4, wf);
                fwrite("fmt ", 1, 4, wf);
                uint32_t fmt_size = 16; fwrite(&fmt_size, 4, 1, wf);
                uint16_t audio_fmt = 1; fwrite(&audio_fmt, 2, 1, wf);
                uint16_t channels = 1; fwrite(&channels, 2, 1, wf);
                uint32_t sr_out = 22050; fwrite(&sr_out, 4, 1, wf);
                uint32_t byte_rate = 44100; fwrite(&byte_rate, 4, 1, wf);
                uint16_t block_align = 2; fwrite(&block_align, 2, 1, wf);
                uint16_t bps = 16; fwrite(&bps, 2, 1, wf);
                fwrite("data", 1, 4, wf);
                fwrite(&data_size, 4, 1, wf);
                fwrite(pcm16, 2, n_out, wf);
                fclose(wf);
                free(pcm16);
                fprintf(stderr, "Wrote %s\n", wav_out);
            }
        }
        free(out);
    } else {
        fprintf(stderr, "FAIL: openvoice2_convert returned false\n");
        openvoice2_free(ctx);
        return 1;
    }

    openvoice2_free(ctx);
    return 0;
}
