/* Independent double-precision oracle, including unaligned inputs and tails. */
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../src/slm_float_dot.h"

int main(void) {
    uint32_t seed = 183467;
    unsigned char storage[130], *weights = storage + 1;
    float input_storage[130], *x = input_storage + 1;
    int count = 0;
    double worst = 0.0;
    for (int bits = 4; bits <= 8; bits += 4) {
        for (int length = 1; length <= 128; length++) {
            for (int trial = 0; trial < 64; trial++) {
                memset(storage, 0, sizeof(storage));
                double expected = 0.0, magnitude = 0.0;
                for (int j = 0; j < length; j++) {
                    seed = seed * 1664525u + 1013904223u;
                    int q = bits == 4 ? (int)(seed % 16u) - 8 : (int)(seed % 255u) - 127;
                    seed = seed * 1664525u + 1013904223u;
                    x[j] = ((int)(seed % 65535u) - 32767) / 1009.0f;
                    if (trial == 0) x[j] = 0;
                    if (trial == 1) q = bits == 4 ? -8 : -127;
                    if (trial == 2) q = bits == 4 ? 7 : 127;
                    if (bits == 4) weights[j / 2] |= (unsigned char)((q + 8) << (4 * (j % 2)));
                    else ((int8_t *)weights)[j] = (int8_t)q;
                    expected += (double)q * (double)x[j];
                    magnitude += fabs((double)q * (double)x[j]);
                }
                double actual = slm_dot_float(weights, x, bits, length);
                double error = fabs(actual - expected);
                if (!isfinite(actual) || error > magnitude * FLT_EPSILON * 32.0 + 1e-7) {
                    fprintf(stderr, "bits%d length%d trial%d: %.12g != %.12g\n", bits, length, trial, actual, expected);
                    return 1;
                }
                if (error > worst) worst = error;
                count++;
            }
        }
    }
    printf("%d Q4/Q8-F32 dot checks passed; mode=%s; max_abs_error=%.9g\n", count,
        SLM_FLOAT_DOT_SIMD ? "SSE2" : "scalar", worst);
    return 0;
}
