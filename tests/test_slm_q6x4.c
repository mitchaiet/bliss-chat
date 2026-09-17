/* Independent row-major int64 oracle for the lossless four-output layout. */
#include <stdio.h>
#include <string.h>
#include "../src/slm_q6x4.h"
int main(void) {
    enum { ROWS = 12, GROUPS = 3, COLS = GROUPS * 64 };
    uint32_t seed = 712937;
    for (int trial = 0; trial < 1024; trial++) {
        int weights[ROWS][COLS];
        unsigned char storage[ROWS * COLS + 1], *packed = storage + (trial & 1);
        float x[COLS], activation_scales[GROUPS], out[ROWS];
        _Alignas(16) float scales[ROWS * GROUPS];
        int16_t q[COLS];
        _Alignas(16) int16_t pairs[COLS * 4];
        int correction[GROUPS];
        for (int j = 0; j < COLS; j++) {
            seed = seed * 1664525u + 1013904223u;
            x[j] = trial == 0 ? 0 : ((int)(seed >> 16) - 32768) / 701.0f;
        }
        slm_q6x4_prepare(x, q, pairs, activation_scales, correction, GROUPS);
        for (int r = 0; r < ROWS; r++) for (int j = 0; j < COLS; j++) {
            seed = seed * 1664525u + 1013904223u;
            int value = (int)(seed >> 26) - 32;
            if (trial < 3) value = trial == 1 ? -32 : 31;
            weights[r][j] = value;
            size_t at = ((size_t)(r / 4) * GROUPS + j / 64) * 256 + (j % 64 / 2) * 8 + r % 4 * 2 + j % 2;
            packed[at] = (unsigned char)(value + 32);
            if (slm_q6x4_value(packed, GROUPS, r, j) != value) return 1;
        }
        for (int r = 0; r < ROWS; r++) for (int g = 0; g < GROUPS; g++)
            scales[(r / 4 * GROUPS + g) * 4 + r % 4] = (float)(r * 3 + g + 1) / 1001.0f;
        slm_q6x4_linear(out, packed, scales, ROWS, GROUPS, pairs, activation_scales, correction);
        for (int r = 0; r < ROWS; r++) {
            float expected = 0;
            for (int g = 0; g < GROUPS; g++) {
                int64_t dot = 0;
                for (int j = 0; j < 64; j++) dot += (int64_t)weights[r][g * 64 + j] * q[g * 64 + j];
                expected += (float)dot * scales[(r / 4 * GROUPS + g) * 4 + r % 4] * activation_scales[g];
            }
            if (memcmp(&out[r], &expected, sizeof(float))) {
                fprintf(stderr, "Q6X4 mismatch trial=%d row=%d: %.9g != %.9g\n", trial, r, out[r], expected);
                return 2;
            }
        }
    }
    /* Retain a direct check of the packed Q6 helper included by this header. */
    { unsigned char p[48] = {0}; int16_t q[64] = {0};
      if (slm_dot_q6_q16(p, q) || slm_dot_float(p, (float[64]){0}, 6, 64)) return 3; }
    printf("12288 outputs (including the four-row tail) match the independent int64 oracle (%s).\n", SLM_FLOAT_DOT_SIMD ? "SSE2" : "scalar");
    return 0;
}
