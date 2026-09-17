#ifndef SLM_Q6X4_H
#define SLM_Q6X4_H
#include <stddef.h>
#include "slm_q6_dot.h"

/* Lossless Q6 layout: four output rows, one 64-column group, then
 * 32 input pairs. Each pair has eight biased bytes (two per output).
 * pmaddwd produces four independent outputs, avoiding horizontal sums.
 * Activation pairs and scales must be 16-byte aligned.
 * All 64 values and their original FP32 scale are preserved per row.
 */
static int slm_q6x4_value(const unsigned char *weights, int groups, int row, int col) {
    size_t block = ((size_t)(row / 4) * groups + col / 64) * 256;
    return (int)weights[block + (col % 64 / 2) * 8 + row % 4 * 2 + col % 2] - 32;
}

static void slm_q6x4_prepare(const float *x, int16_t *q, int16_t *pairs,
                            float *scales, int *correction, int groups) {
    for (int g = 0; g < groups; g++) {
        slm_quantize_q16_64(x + g * 64, q + g * 64, scales + g);
        int sum = 0;
        for (int j = 0; j < 64; j++) sum += q[g * 64 + j];
        correction[g] = sum * 32;
        for (int j = 0; j < 64; j += 2) for (int r = 0; r < 4; r++) {
            pairs[g * 256 + j * 4 + r * 2] = q[g * 64 + j];
            pairs[g * 256 + j * 4 + r * 2 + 1] = q[g * 64 + j + 1];
        }
    }
}

static void slm_q6x4_linear(float *out, const unsigned char *weights,
                           const float *scales, int rows, int groups,
                           const int16_t *pairs, const float *activation_scales,
                           const int *correction) {
    int r=0;
#if SLM_FLOAT_DOT_SIMD
    for(;r+8<=rows;r+=8){
        __m128 total0=_mm_setzero_ps(),total1=total0;
        for(int g=0;g<groups;g++){
            const unsigned char *w0=weights+((size_t)(r/4)*groups+g)*256;
            const unsigned char *w1=w0+(size_t)groups*256;
            const int16_t *x=pairs+g*256;
            __m128i sum0=_mm_setzero_si128(),sum1=sum0,zero=sum0;
            /* Fixed-size groups remove repeated branch and offset steps.
             * Each output accumulates groups in its original order. */
#if defined(__clang__)
#pragma clang loop unroll_count(32)
#elif defined(__GNUC__)
#pragma GCC unroll 32
#endif
            for(int j=0;j<32;j++){
                __m128i a=_mm_load_si128((const __m128i*)(x+j*8));
                __m128i q0=_mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(w0+j*8)),zero);
                __m128i q1=_mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(w1+j*8)),zero);
                sum0=_mm_add_epi32(sum0,_mm_madd_epi16(q0,a));
                sum1=_mm_add_epi32(sum1,_mm_madd_epi16(q1,a));
            }
            __m128i correction4=_mm_set1_epi32(correction[g]);
            __m128 a=_mm_set1_ps(activation_scales[g]);
            __m128 v0=_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(sum0,correction4)),_mm_load_ps(scales+(size_t)r*groups+g*4));
            __m128 v1=_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(sum1,correction4)),_mm_load_ps(scales+(size_t)(r+4)*groups+g*4));
            total0=_mm_add_ps(total0,_mm_mul_ps(v0,a));
            total1=_mm_add_ps(total1,_mm_mul_ps(v1,a));
        }
        _mm_storeu_ps(out+r,total0);_mm_storeu_ps(out+r+4,total1);
    }
#endif
    for (; r < rows; r += 4) {
#if SLM_FLOAT_DOT_SIMD
        __m128 total = _mm_setzero_ps();
        for (int g = 0; g < groups; g++) {
            const unsigned char *w = weights + ((size_t)(r / 4) * groups + g) * 256;
            const int16_t *x = pairs + g * 256;
            __m128i sum = _mm_setzero_si128(), zero = sum;
#if defined(__clang__)
#pragma clang loop unroll_count(8)
#elif defined(__GNUC__)
#pragma GCC unroll 8
#endif
            for (int j = 0; j < 32; j++) {
                __m128i q = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(w + j * 8)), zero);
                sum = _mm_add_epi32(sum, _mm_madd_epi16(q, _mm_load_si128((const __m128i *)(x + j * 8))));
            }
            /* 64 * 63 * 32768 and the offset correction both fit int32. */
            sum = _mm_sub_epi32(sum, _mm_set1_epi32(correction[g]));
            __m128 value = _mm_mul_ps(_mm_cvtepi32_ps(sum), _mm_load_ps(scales + (size_t)r * groups + g * 4));
            total = _mm_add_ps(total, _mm_mul_ps(value, _mm_set1_ps(activation_scales[g])));
        }
        _mm_storeu_ps(out + r, total);
#else
        for (int k = 0; k < 4; k++) {
            float total = 0;
            for (int g = 0; g < groups; g++) {
                const unsigned char *w = weights + ((size_t)(r / 4) * groups + g) * 256;
                int sum = 0;
                for (int j = 0; j < 64; j++)
                    sum += ((int)w[(j / 2) * 8 + k * 2 + j % 2] - 32) * pairs[g * 256 + (j / 2) * 8 + j % 2];
                total += (float)sum * scales[(size_t)r * groups + g * 4 + k] * activation_scales[g];
            }
            out[r + k] = total;
        }
        (void)correction;
#endif
    }
}
#endif
