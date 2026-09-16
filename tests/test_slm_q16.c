/* Independent Q6 packing and int64 dot oracle, plus activation error bounds. */
#include <stdio.h>
#include <string.h>
#include "../src/slm_q6_dot.h"
int main(void) {
    uint32_t seed = 84371;
    for (int trial = 0; trial < 16384; trial++) {
        unsigned char storage[49] = {0}, *w = storage + (trial & 1);
        int16_t inputs[65], *a = inputs + (trial & 1), quantized[64];
        float x[64], scale;
        int64_t expected = 0;
        double magnitude = 0, exact = 0;
        for (int j = 0; j < 64; j++) {
            seed = seed * 1664525u + 1013904223u;
            int q = (int)(seed >> 26) - 32;
            seed = seed * 1664525u + 1013904223u;
            int v = (int)(seed >> 16) - 32768;
            if (trial < 4) { q = (trial & 1) ? 31 : -32; v = (trial & 2) ? 32767 : -32768; }
            unsigned packed = (unsigned)(q + 32);
            w[j / 2] |= (unsigned char)((packed & 15) << (4 * (j % 2)));
            w[32 + j / 4] |= (unsigned char)((packed >> 4) << (2 * (j % 4)));
            a[j] = (int16_t)v;
            expected += (int64_t)q * v;
            x[j] = (float)v / 1091.0f;
            if (trial == 4) x[j] = 0;
            if (trial == 5) x[j] *= FLT_MIN;
            if (trial == 6) x[j] *= FLT_MAX / 64.0f;
            exact += (double)q * x[j];
            magnitude += fabs((double)q);
        }
        if (slm_dot_q6_q16(w, a) != expected) {
            fprintf(stderr, "integer dot mismatch at %d\n", trial); return 1;
        }
        slm_quantize_q16_64(x, quantized, &scale);
        if (!isfinite(scale) || scale <= 0) return 2;
        for (int j = 0; j < 64; j++) {
            double error = fabs((double)x[j] - (double)quantized[j] * scale);
            if (error > (double)scale * .501 + fabs((double)x[j]) * FLT_EPSILON * 4) {
                fprintf(stderr, "activation error at %d/%d\n", trial, j); return 3;
            }
        }
        double actual = (double)slm_dot_q6_q16(w, quantized) * scale;
        if (fabs(actual - exact) > magnitude * (double)scale * .51 + fabs(exact) * FLT_EPSILON * 8) return 4;
        /* Exercise the retained float kernel with ordinary finite magnitudes. */
        if (trial > 6 && fabs((double)slm_dot_float(w, x, 6, 64) - exact) > magnitude * 32 * FLT_EPSILON * 32) return 5;
    }
    printf("16384 Q6/Q16 integer oracle and activation error checks passed (%s).\n", SLM_FLOAT_DOT_SIMD ? "SSE2" : "scalar");
    return 0;
}
