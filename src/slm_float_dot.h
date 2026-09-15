#ifndef SLM_FLOAT_DOT_H
#define SLM_FLOAT_DOT_H
/* Grouped Q4/Q6/Q8 weights times unquantized FP32 activations.
   Q4 uses consecutive low/high nibbles with an offset of8. Scales are
   applied by the caller once per group. Q6 only supports group64:32 low
   nibble bytes followed by16 high2-bit bytes, signed offset32. */
#include <stdint.h>
#include <string.h>
#if defined(__SSE2__)
#include <emmintrin.h>
/* Unpack16 Q6 values starting at j=0,16,32,48 in a48-byte group. */
static inline __m128i slm_unpack_q6_16(const unsigned char *w, int j) {
    const __m128i mask4=_mm_set1_epi8(15), mask2=_mm_set1_epi8(3);
    __m128i lo=_mm_loadl_epi64((const __m128i *)(w+j/2));
    lo=_mm_unpacklo_epi8(_mm_and_si128(lo,mask4),_mm_and_si128(_mm_srli_epi16(lo,4),mask4));
    uint32_t word;memcpy(&word,w+32+j/4,4);
    __m128i packed=_mm_cvtsi32_si128((int)word);
    __m128i h0=_mm_and_si128(packed,mask2),h1=_mm_and_si128(_mm_srli_epi16(packed,2),mask2);
    __m128i h2=_mm_and_si128(_mm_srli_epi16(packed,4),mask2),h3=_mm_and_si128(_mm_srli_epi16(packed,6),mask2);
    __m128i hi=_mm_unpacklo_epi16(_mm_unpacklo_epi8(h0,h1),_mm_unpacklo_epi8(h2,h3));
    return _mm_sub_epi8(_mm_or_si128(lo,_mm_slli_epi16(hi,4)),_mm_set1_epi8(32));
}
#endif
static inline int slm_q6_value(const unsigned char *w, int j) {
    int lo=(w[j/2]>>(4*(j%2)))&15;
    int hi=(w[32+j/4]>>(2*(j%4)))&3;
    return (lo|(hi<<4))-32;
}
#if defined(__SSE2__) && !defined(SLM_FLOAT_DOT_SCALAR)
#include <emmintrin.h>
#define SLM_FLOAT_DOT_SIMD 1
#else
#define SLM_FLOAT_DOT_SIMD 0
#endif

static float slm_dot_float(const unsigned char *w, const float *x, int bits, int n) {
    int j = 0;
    float total = 0.0f;
#if SLM_FLOAT_DOT_SIMD
    __m128 a = _mm_setzero_ps(), b = a, c = a, d = a;
    const __m128i zero = _mm_setzero_si128();
    const __m128i mask = _mm_set1_epi8(15), offset = _mm_set1_epi8(8);
    for (; j + 16 <= n; j += 16) {
        __m128i q;
        if (bits == 4) {
            __m128i packed = _mm_loadl_epi64((const __m128i *)(w + j / 2));
            q = _mm_unpacklo_epi8(_mm_and_si128(packed, mask),
                _mm_and_si128(_mm_srli_epi16(packed, 4), mask));
            q = _mm_sub_epi8(q, offset);
        } else if (bits == 6) {
            q = slm_unpack_q6_16(w, j);
        } else {
            q = _mm_loadu_si128((const __m128i *)(w + j));
        }
        __m128i sign = _mm_cmpgt_epi8(zero, q);
        __m128i low = _mm_unpacklo_epi8(q, sign), high = _mm_unpackhi_epi8(q, sign);
        __m128i low_sign = _mm_cmpgt_epi16(zero, low), high_sign = _mm_cmpgt_epi16(zero, high);
        __m128 qa = _mm_cvtepi32_ps(_mm_unpacklo_epi16(low, low_sign));
        __m128 qb = _mm_cvtepi32_ps(_mm_unpackhi_epi16(low, low_sign));
        __m128 qc = _mm_cvtepi32_ps(_mm_unpacklo_epi16(high, high_sign));
        __m128 qd = _mm_cvtepi32_ps(_mm_unpackhi_epi16(high, high_sign));
        a = _mm_add_ps(a, _mm_mul_ps(qa, _mm_loadu_ps(x + j)));
        b = _mm_add_ps(b, _mm_mul_ps(qb, _mm_loadu_ps(x + j + 4)));
        c = _mm_add_ps(c, _mm_mul_ps(qc, _mm_loadu_ps(x + j + 8)));
        d = _mm_add_ps(d, _mm_mul_ps(qd, _mm_loadu_ps(x + j + 12)));
    }
    float lanes[4];
    _mm_storeu_ps(lanes, _mm_add_ps(_mm_add_ps(a, b), _mm_add_ps(c, d)));
    total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
#endif
    for (; j < n; j++) {
        int q = bits == 6 ? slm_q6_value(w,j) : bits == 4 ? ((w[j / 2] >> (4 * (j % 2))) & 15) - 8 : ((const int8_t *)w)[j];
        total += (float)q * x[j];
    }
    return total;
}
#endif
