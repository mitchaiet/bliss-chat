#ifndef SLM_Q6_DOT_H
#define SLM_Q6_DOT_H

#include <float.h>
#include <math.h>
#include <stdint.h>
#include "slm_float_dot.h"

/* Keep the released Q6 weights unchanged. Quantize only the current activation
 * group to signed 16-bit values. SSE2 then needs two integer multiply/adds per
 * 16 weights, instead of four int-to-float conversions and float multiplies.
 * One group's worst possible dot is 64 * 32 * 32768, safely inside int32.
 * Norms, convolution, attention, residuals and group scales remain FP32.
 */
static void slm_quantize_q16_64(const float *x, int16_t *out, float *scale_out) {
    float maximum = 0;
    for (int j = 0; j < 64; j++) {
        float v = fabsf(x[j]);
        if (v > maximum) maximum = v;
    }
    float scale = maximum > 0 ? maximum / 32767.0f : 1.0f;
    /* A normal scale keeps its reciprocal finite even for subnormal inputs. */
    if (scale < FLT_MIN) scale = FLT_MIN;
    *scale_out = scale;
    float inverse = 1.0f / scale;
#if SLM_FLOAT_DOT_SIMD
    __m128 multiplier = _mm_set1_ps(inverse);
    for (int j = 0; j < 64; j += 8) {
        __m128i low = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(x + j), multiplier));
        __m128i high = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(x + j + 4), multiplier));
        _mm_storeu_si128((__m128i *)(out + j), _mm_packs_epi32(low, high));
    }
#else
    for (int j = 0; j < 64; j++) {
        long q = lrintf(x[j] * inverse);
        if (q > 32767) q = 32767;
        if (q < -32767) q = -32767;
        out[j] = (int16_t)q;
    }
#endif
}

static int slm_dot_q6_q16(const unsigned char *weights, const int16_t *x) {
#if SLM_FLOAT_DOT_SIMD
    __m128i sum = _mm_setzero_si128(), zero = sum;
    for (int j = 0; j < 64; j += 16) {
        __m128i q = slm_unpack_q6_16(weights, j);
        __m128i sign = _mm_cmpgt_epi8(zero, q);
        sum = _mm_add_epi32(sum, _mm_madd_epi16(_mm_unpacklo_epi8(q, sign),
            _mm_loadu_si128((const __m128i *)(x + j))));
        sum = _mm_add_epi32(sum, _mm_madd_epi16(_mm_unpackhi_epi8(q, sign),
            _mm_loadu_si128((const __m128i *)(x + j + 8))));
    }
    int lanes[4];
    _mm_storeu_si128((__m128i *)lanes, sum);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
#else
    int sum = 0;
    for (int j = 0; j < 64; j++) sum += slm_q6_value(weights, j) * x[j];
    return sum;
#endif
}
#endif
