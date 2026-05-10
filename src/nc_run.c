// nc_run: native nanochat inference for Windows XP (and Mac/Linux for testing).
// Single-file C99, no dependencies beyond libc + winsock for the XPCHAT pipe protocol.
//
// Usage (CLI):
//   nc_run MODEL.NCB TOKENIZER.NCT [-c ctx] [-t temp] [-p top_p] [-s seed]
// In CLI mode, reads lines from stdin and prints generated tokens. Same sentinel
// protocol as our existing chat-backend: \x01READY\n, \x01EOT\n, \x01INFO ...\n.
//
// File format documented in export_ncb.py. Tokenizer file format in nc_tokenizer.c.

#define _CRT_SECURE_NO_WARNINGS
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#endif

// ===== model header (must match export_ncb.py) =====
#pragma pack(push, 1)
typedef struct {
    char     magic[8];        // "NCB1\0\0\0\0"
    int32_t  version;
    int32_t  dtype_code;      // 0=fp32, 1=int8
    int32_t  vocab_size;
    int32_t  pad_vocab_size;
    int32_t  n_layer;
    int32_t  n_head;
    int32_t  n_kv_head;
    int32_t  n_embd;
    int32_t  head_dim;
    int32_t  sequence_len;
    int32_t  rotary_base;
    int32_t  rotary_seq_len;
    uint64_t ve_layer_mask;
    uint64_t window_pattern_mask;
    int32_t  short_window;
    int32_t  long_window;
    int32_t  smear_gate_in;   // 24
    int32_t  ve_gate_in;      // 12
    int32_t  softcap;         // 15
} ncb_header_t;
#pragma pack(pop)

// ===== forward declarations: tokenizer in nc_tokenizer.c =====
typedef struct nct nct;
nct *nct_load(const char *path);
void nct_free(nct *t);
int  nct_vocab_size(const nct *t);
int  nct_special_id(const nct *t, const char *name);  // returns -1 if not found
// encode UTF-8 text into ids; returns number of tokens written; caps at out_max.
int  nct_encode(const nct *t, const char *text, int *out_ids, int out_max);
// decode one id to UTF-8; returns bytes written (no NUL); 0 on error or special.
int  nct_decode_one(const nct *t, int id, char *out, int out_max);

// ===== utilities =====
static double now_sec(void) {
#ifdef _WIN32
    return (double)GetTickCount() / 1000.0;
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
#endif
}

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "[nc_run] OOM allocating %zu bytes\n", n); exit(2); }
    return p;
}

static void *xcalloc(size_t n, size_t s) {
    void *p = calloc(n, s);
    if (!p) { fprintf(stderr, "[nc_run] OOM allocating %zu*%zu bytes\n", n, s); exit(2); }
    return p;
}

// ===== model =====
typedef struct {
    int n_layer, n_head, n_kv_head, n_embd, head_dim;
    int vocab_size, pad_vocab_size, sequence_len;
    int kv_dim;            // n_kv_head * head_dim
    int rotary_seq_len;
    int short_window, long_window;
    uint64_t ve_layer_mask;
    uint64_t window_pattern_mask;
    int dtype_code;        // 0=fp32, 1=int8
    int softcap;

    // Owned buffer for the whole file (we read it in)
    void   *blob;
    size_t  blob_len;

    // Pointers into blob (each weight is EITHER fp32 OR (int8 + per-row scales))
    float  *wte;           void *wte_q; float *wte_scales;
    float  *lm_head;       void *lm_head_q; float *lm_head_scales;

    // Per-layer
    struct {
        // Either fp32 weights or (int8 + scales). We keep both pointer types and a flag.
        void  *q_w;  float *q_scale;  float *q_fp32;
        void  *k_w;  float *k_scale;  float *k_fp32;
        void  *v_w;  float *v_scale;  float *v_fp32;
        void  *o_w;  float *o_scale;  float *o_fp32;   // attn output proj
        void  *fc_w; float *fc_scale; float *fc_fp32;  // mlp c_fc
        void  *pj_w; float *pj_scale; float *pj_fp32;  // mlp c_proj
        // VE gate (n_kv_head, 12) fp32 — small, never quantized
        float *ve_gate_w;   // NULL if no VE on this layer
        // Value embedding for this layer (pad_vocab, kv_dim). NULL if no VE.
        float *ve_fp32;  void *ve_q;  float *ve_scales;
        int   window;       // (effective left window) — long_window if -1 or unbounded
    } *L;

    // Per-layer scalars
    float *resid_lambdas;  // (n_layer)
    float *x0_lambdas;     // (n_layer)
    float *smear_gate_w;   // (1, 24) fp32
    float  smear_lambda;
    float  backout_lambda;

    // Rotary
    float *cos;            // (rotary_seq_len, head_dim/2)
    float *sin;            // (rotary_seq_len, head_dim/2)
} nc_model;

static int has_ve_layer(int i, uint64_t mask) {
    return (mask >> i) & 1ULL ? 1 : 0;
}

static int read_file(const char *path, void **out_blob, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = xmalloc((size_t)sz);
    size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) { free(buf); return 0; }
    *out_blob = buf;
    *out_len = (size_t)sz;
    return 1;
}

// Parse header + set up pointers into blob. Returns 1 on success.
static int model_load(nc_model *m, const char *path) {
    if (!read_file(path, &m->blob, &m->blob_len)) {
        fprintf(stderr, "[nc_run] could not read %s\n", path);
        return 0;
    }
    if (m->blob_len < sizeof(ncb_header_t)) {
        fprintf(stderr, "[nc_run] file too small for header\n");
        return 0;
    }
    ncb_header_t *h = (ncb_header_t *)m->blob;
    if (memcmp(h->magic, "NCB1\0\0\0\0", 8) != 0) {
        fprintf(stderr, "[nc_run] bad magic\n");
        return 0;
    }
    if (h->version != 1) {
        fprintf(stderr, "[nc_run] unsupported version %d\n", (int)h->version);
        return 0;
    }
    m->dtype_code      = h->dtype_code;
    m->vocab_size      = h->vocab_size;
    m->pad_vocab_size  = h->pad_vocab_size;
    m->n_layer         = h->n_layer;
    m->n_head          = h->n_head;
    m->n_kv_head       = h->n_kv_head;
    m->n_embd          = h->n_embd;
    m->head_dim        = h->head_dim;
    m->sequence_len    = h->sequence_len;
    m->ve_layer_mask   = h->ve_layer_mask;
    m->window_pattern_mask = h->window_pattern_mask;
    m->short_window    = h->short_window;
    m->long_window     = h->long_window;
    m->rotary_seq_len  = h->rotary_seq_len;
    m->kv_dim          = m->n_kv_head * m->head_dim;
    m->softcap         = h->softcap;

    if (m->dtype_code != 0 && m->dtype_code != 1) {
        fprintf(stderr, "[nc_run] unsupported dtype_code=%d\n", m->dtype_code);
        return 0;
    }

    // Walk the blob: header is 256 bytes, weights follow in the order documented in export_ncb.py.
    char *p = (char *)m->blob + 256;
    char *end = (char *)m->blob + m->blob_len;

    int V = m->pad_vocab_size, D = m->n_embd, H = m->n_head, KH = m->n_kv_head;
    int HD = m->head_dim, KD = m->kv_dim, FF = 4 * D;

    // Helper to read N fp32 values
    #define TAKE_FP32(n_floats) ({                                            \
        size_t bytes = (size_t)(n_floats) * sizeof(float);                    \
        if (p + bytes > end) { fprintf(stderr,"[nc_run] truncated at %s:%d\n", __FILE__, __LINE__); return 0; } \
        float *r = (float *)p; p += bytes; r;                                 \
    })

    // Helper for matrix weights honoring dtype_code (macro because plain C99 has no nested fns).
    // For matrix W of shape (rows, cols):
    //   fp32: rows*cols floats
    //   int8: rows*cols int8 + rows fp32 scales (per-row symmetric quant)
    #define TAKE_MATRIX(rows_, cols_, qslot_, sslot_, fpslot_)                    \
        do {                                                                       \
            size_t _r = (size_t)(rows_), _c = (size_t)(cols_);                     \
            if (m->dtype_code == 0) {                                              \
                size_t _b = _r * _c * sizeof(float);                               \
                if (p + _b > end) { fprintf(stderr,"[nc_run] truncated matrix\n"); return 0; } \
                (fpslot_) = (float *)p; p += _b;                                   \
                (qslot_) = NULL; (sslot_) = NULL;                                  \
            } else {                                                               \
                size_t _b = _r * _c;                                               \
                size_t _bs = _r * sizeof(float);                                   \
                if (p + _b + _bs > end) { fprintf(stderr,"[nc_run] truncated matrix\n"); return 0; } \
                (qslot_) = (void *)p; p += _b;                                     \
                (sslot_) = (float *)p; p += _bs;                                   \
                (fpslot_) = NULL;                                                  \
            }                                                                      \
        } while (0)

    // wte (matrix, dtype-aware)
    TAKE_MATRIX(V, D, m->wte_q, m->wte_scales, m->wte);

    // Per-layer
    m->L = xcalloc(m->n_layer, sizeof(*m->L));
    for (int i = 0; i < m->n_layer; i++) {
        TAKE_MATRIX(H * HD, D, m->L[i].q_w,  m->L[i].q_scale,  m->L[i].q_fp32);
        TAKE_MATRIX(KH * HD, D, m->L[i].k_w, m->L[i].k_scale,  m->L[i].k_fp32);
        TAKE_MATRIX(KH * HD, D, m->L[i].v_w, m->L[i].v_scale,  m->L[i].v_fp32);
        TAKE_MATRIX(D, D,    m->L[i].o_w,    m->L[i].o_scale,  m->L[i].o_fp32);
        TAKE_MATRIX(FF, D,   m->L[i].fc_w,   m->L[i].fc_scale, m->L[i].fc_fp32);
        TAKE_MATRIX(D, FF,   m->L[i].pj_w,   m->L[i].pj_scale, m->L[i].pj_fp32);
        if (has_ve_layer(i, m->ve_layer_mask)) {
            // ve_gate (n_kv_head, 12) fp32
            m->L[i].ve_gate_w = TAKE_FP32((size_t)KH * 12);
        } else {
            m->L[i].ve_gate_w = NULL;
        }
        // Compute window
        int is_long = (m->window_pattern_mask >> i) & 1ULL;
        m->L[i].window = is_long ? m->long_window : m->short_window;
    }

    // Value embeddings (matrix, dtype-aware) for VE layers
    for (int i = 0; i < m->n_layer; i++) {
        if (has_ve_layer(i, m->ve_layer_mask)) {
            TAKE_MATRIX(V, KD, m->L[i].ve_q, m->L[i].ve_scales, m->L[i].ve_fp32);
        } else {
            m->L[i].ve_fp32 = NULL; m->L[i].ve_q = NULL; m->L[i].ve_scales = NULL;
        }
    }

    // Per-layer scalars
    m->resid_lambdas = TAKE_FP32(m->n_layer);
    m->x0_lambdas    = TAKE_FP32(m->n_layer);
    m->smear_gate_w  = TAKE_FP32(24);
    float *sl = TAKE_FP32(1); m->smear_lambda = sl[0];
    float *bl = TAKE_FP32(1); m->backout_lambda = bl[0];

    // LM head (matrix)
    TAKE_MATRIX(V, D, m->lm_head_q, m->lm_head_scales, m->lm_head);
    #undef TAKE_MATRIX

    // RoPE cos/sin
    m->cos = TAKE_FP32((size_t)m->rotary_seq_len * (HD / 2));
    m->sin = TAKE_FP32((size_t)m->rotary_seq_len * (HD / 2));

    if (p > end) { fprintf(stderr, "[nc_run] read past end\n"); return 0; }
    return 1;

    #undef TAKE_FP32
}

// ===== math kernels =====

// y = RMSNorm(x), no learnable scale (matches nanochat's norm())
// y[i] = x[i] / sqrt(mean(x^2) + eps)
static void rmsnorm(float *y, const float *x, int d) {
    double s = 0.0;
    for (int i = 0; i < d; i++) s += (double)x[i] * x[i];
    s /= (double)d;
    float scale = (float)(1.0 / sqrt(s + 1e-6));
    for (int i = 0; i < d; i++) y[i] = x[i] * scale;
}

// y = W @ x where W is (rows, cols) fp32 row-major, x is (cols), y is (rows).
// For us, "x" is the activation, W is the weight matrix.
// In PyTorch nn.Linear: y = x @ W.T; W stored as (out_features, in_features) row-major.
// So y[r] = sum_c W[r,c] * x[c]
static void matmul_fp32(float *y, const float *W, const float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float *wr = W + (size_t)r * cols;
        double acc = 0.0;
        for (int c = 0; c < cols; c++) acc += (double)wr[c] * x[c];
        y[r] = (float)acc;
    }
}

// Same as matmul_fp32 but W is int8 (rows*cols) with per-row fp32 scales[rows].
static void matmul_int8(float *y, const int8_t *W, const float *scales,
                        const float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const int8_t *wr = W + (size_t)r * cols;
        double acc = 0.0;
        for (int c = 0; c < cols; c++) acc += (double)wr[c] * x[c];
        y[r] = (float)(acc * scales[r]);
    }
}

static inline void linear(float *y, const float *x, int rows, int cols,
                          void *q, float *scale, float *fp32) {
    if (fp32) matmul_fp32(y, fp32, x, rows, cols);
    else      matmul_int8(y, (const int8_t *)q, scale, x, rows, cols);
}

// Apply RoPE (nanochat variant: split second-half rotates against first-half).
// d = head_dim; cos/sin: (head_dim/2)
// Layout: x[..head] is (head_dim) per head.
static void apply_rope_head(float *x, const float *cosrow, const float *sinrow, int head_dim) {
    int d = head_dim / 2;
    for (int i = 0; i < d; i++) {
        float x1 = x[i], x2 = x[i + d];
        x[i]     =  x1 * cosrow[i] + x2 * sinrow[i];
        x[i + d] = -x1 * sinrow[i] + x2 * cosrow[i];
    }
}

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

// ===== runtime state =====
typedef struct {
    nc_model *m;
    int       seq_pos;             // current cache position (next write index)
    float    *kcache;              // (n_layer, sequence_len, kv_dim) fp32
    float    *vcache;              // (n_layer, sequence_len, kv_dim) fp32 (post-VE blend)
    float    *prev_x_norm;         // (n_embd) — last token's pre-smear normed embedding for KV-decode smear
    int       has_prev_x;
    // Working buffers
    float    *x;                   // (n_embd)
    float    *x0;                  // (n_embd)
    float    *xb;                  // (n_embd) scratch
    float    *xb2;                 // (n_embd)
    float    *q_h;                 // (n_head * head_dim)
    float    *k_h;                 // (n_kv_head * head_dim)
    float    *v_h;                 // (n_kv_head * head_dim)
    float    *attn_scores;         // (n_head, sequence_len)
    float    *ffn_h;               // (4 * n_embd)
    float    *logits;              // (pad_vocab_size)
    int      *ranking;             // (vocab_size) reusable buffer for top-p
    float    *probs;               // (vocab_size)
    float    *x_backout;           // (n_embd) — saved residual for backout
    int       backout_layer;
} nc_state;

static void state_init(nc_state *s, nc_model *m) {
    int D = m->n_embd, T = m->sequence_len;
    int FF = 4 * D, KD = m->kv_dim, V = m->vocab_size, PV = m->pad_vocab_size;
    s->m = m;
    s->seq_pos = 0;
    s->has_prev_x = 0;
    s->kcache = (float *)xcalloc((size_t)m->n_layer * T * KD, sizeof(float));
    s->vcache = (float *)xcalloc((size_t)m->n_layer * T * KD, sizeof(float));
    s->prev_x_norm = (float *)xcalloc(D, sizeof(float));
    s->x   = (float *)xcalloc(D, sizeof(float));
    s->x0  = (float *)xcalloc(D, sizeof(float));
    s->xb  = (float *)xcalloc(D, sizeof(float));
    s->xb2 = (float *)xcalloc(D, sizeof(float));
    s->q_h = (float *)xcalloc((size_t)m->n_head * m->head_dim, sizeof(float));
    s->k_h = (float *)xcalloc(KD, sizeof(float));
    s->v_h = (float *)xcalloc(KD, sizeof(float));
    s->attn_scores = (float *)xcalloc((size_t)m->n_head * T, sizeof(float));
    s->ffn_h = (float *)xcalloc(FF, sizeof(float));
    s->logits = (float *)xcalloc(PV, sizeof(float));
    s->ranking = (int *)xcalloc(V, sizeof(int));
    s->probs = (float *)xcalloc(V, sizeof(float));
    s->x_backout = (float *)xcalloc(D, sizeof(float));
    s->backout_layer = m->n_layer / 2;
}

static void state_free(nc_state *s) {
    free(s->kcache); free(s->vcache); free(s->prev_x_norm);
    free(s->x); free(s->x0); free(s->xb); free(s->xb2);
    free(s->q_h); free(s->k_h); free(s->v_h);
    free(s->attn_scores); free(s->ffn_h); free(s->logits);
    free(s->ranking); free(s->probs); free(s->x_backout);
}

static void state_reset(nc_state *s) {
    s->seq_pos = 0;
    s->has_prev_x = 0;
}

// ===== forward pass for a single token =====
// Reads from state.x (which the caller has prepared from token id), writes back to state.x.
// Updates KV cache. Computes logits at end.
static void forward_one(nc_state *s, int token_id) {
    nc_model *m = s->m;
    int D = m->n_embd, H = m->n_head, KH = m->n_kv_head, HD = m->head_dim, KD = m->kv_dim;
    int T = m->sequence_len, FF = 4 * D;
    int pos = s->seq_pos;
    if (pos >= T) {
        fprintf(stderr, "[nc_run] context overflow at pos=%d\n", pos);
        return;
    }

    // 1) Embed token (look up wte row)
    if (m->wte) {
        float *wte_row = m->wte + (size_t)token_id * D;
        memcpy(s->x, wte_row, sizeof(float) * D);
    } else {
        const int8_t *row = (const int8_t *)m->wte_q + (size_t)token_id * D;
        float scale = m->wte_scales[token_id];
        for (int i = 0; i < D; i++) s->x[i] = (float)row[i] * scale;
    }

    // 2) Norm after embedding
    rmsnorm(s->xb, s->x, D);
    memcpy(s->x, s->xb, sizeof(float) * D);

    // 3) Smear: x = x + smear_lambda * sigmoid(smear_gate(x[:24])) * prev_x
    if (s->has_prev_x) {
        // smear_gate_w is (1, 24) fp32: gate = sum(smear_gate_w[0,:] * x[:24])
        double g = 0.0;
        for (int i = 0; i < 24; i++) g += (double)m->smear_gate_w[i] * s->x[i];
        float gate = m->smear_lambda * sigmoidf((float)g);
        for (int i = 0; i < D; i++) s->x[i] += gate * s->prev_x_norm[i];
    }
    // Save the (post-norm, pre-smear) x as prev for next token
    // NOTE: nanochat stores PRE-smear embedding for the cache (kv_cache.prev_embedding = x[:, -1:, :]
    // is set BEFORE the smear is applied — see line 444 of gpt.py). We saved that at step 2.
    memcpy(s->prev_x_norm, s->xb, sizeof(float) * D);  // s->xb still has the pre-smear normed embedding
    s->has_prev_x = 1;

    // 4) Save x0 for residual blending
    memcpy(s->x0, s->x, sizeof(float) * D);

    // 5) Transformer blocks
    for (int li = 0; li < m->n_layer; li++) {
        // Pre-block residual mix: x = resid[i]*x + x0[i]*x0
        {
            float r = m->resid_lambdas[li], x0l = m->x0_lambdas[li];
            for (int i = 0; i < D; i++) s->x[i] = r * s->x[i] + x0l * s->x0[i];
        }

        // ---- attention ----
        rmsnorm(s->xb, s->x, D);  // norm input

        linear(s->q_h, s->xb, H * HD, D, m->L[li].q_w, m->L[li].q_scale, m->L[li].q_fp32);
        linear(s->k_h, s->xb, KH * HD, D, m->L[li].k_w, m->L[li].k_scale, m->L[li].k_fp32);
        linear(s->v_h, s->xb, KH * HD, D, m->L[li].v_w, m->L[li].v_scale, m->L[li].v_fp32);

        // Value embedding mix (only on VE layers).
        // Lookup ve[token_id] -> KD floats, dequantizing if int8.
        if (m->L[li].ve_fp32 || m->L[li].ve_q) {
            float ve_buf[1024];  // KD <= 1024 expected
            const float *ve;
            if (m->L[li].ve_fp32) {
                ve = m->L[li].ve_fp32 + (size_t)token_id * KD;
            } else {
                const int8_t *row = (const int8_t *)m->L[li].ve_q + (size_t)token_id * KD;
                float scale = m->L[li].ve_scales[token_id];
                for (int i = 0; i < KD; i++) ve_buf[i] = (float)row[i] * scale;
                ve = ve_buf;
            }
            // gate[head] = 3 * sigmoid(ve_gate_w[head, :] @ x[:12])
            for (int h = 0; h < KH; h++) {
                const float *gw = m->L[li].ve_gate_w + (size_t)h * 12;
                double a = 0.0;
                for (int i = 0; i < 12; i++) a += (double)gw[i] * s->xb[i];
                float gate = 3.0f * sigmoidf((float)a);
                for (int d = 0; d < HD; d++) {
                    s->v_h[h * HD + d] += gate * ve[h * HD + d];
                }
            }
        }

        // RoPE on each Q head (H heads) and each K head (KH heads), and write K/V cache
        const float *cosrow = m->cos + (size_t)pos * (HD / 2);
        const float *sinrow = m->sin + (size_t)pos * (HD / 2);
        for (int h = 0; h < H; h++) apply_rope_head(s->q_h + h * HD, cosrow, sinrow, HD);
        for (int h = 0; h < KH; h++) apply_rope_head(s->k_h + h * HD, cosrow, sinrow, HD);

        // QK norm: rmsnorm per head (over head_dim)
        for (int h = 0; h < H; h++) rmsnorm(s->q_h + h * HD, s->q_h + h * HD, HD);
        for (int h = 0; h < KH; h++) rmsnorm(s->k_h + h * HD, s->k_h + h * HD, HD);
        // Q*1.2, K*1.2
        for (int i = 0; i < H * HD; i++) s->q_h[i] *= 1.2f;
        for (int i = 0; i < KH * HD; i++) s->k_h[i] *= 1.2f;

        // Write current K/V into cache at position `pos`
        {
            float *kc = s->kcache + (size_t)li * T * KD + (size_t)pos * KD;
            float *vc = s->vcache + (size_t)li * T * KD + (size_t)pos * KD;
            memcpy(kc, s->k_h, sizeof(float) * KD);
            memcpy(vc, s->v_h, sizeof(float) * KD);
        }

        // Sliding-window attention: positions [max(0, pos - window + 1) .. pos] inclusive
        int window = m->L[li].window;
        int p0 = window <= 0 ? 0 : pos - window + 1;
        if (p0 < 0) p0 = 0;
        int n_ctx = pos - p0 + 1;

        // Per head, compute scores, softmax, weighted sum
        // GQA: each query head uses kv head h_kv = h * KH / H
        float scale = 1.0f / sqrtf((float)HD);
        memset(s->xb2, 0, sizeof(float) * D);  // reuse xb2 as attention output (D = H*HD)
        for (int h = 0; h < H; h++) {
            int h_kv = (KH == H) ? h : (h * KH / H);
            const float *qh = s->q_h + h * HD;
            float *scores = s->attn_scores + (size_t)h * T;
            for (int t = p0; t <= pos; t++) {
                const float *kh = s->kcache + (size_t)li * T * KD
                                + (size_t)t * KD + h_kv * HD;
                double dot = 0.0;
                for (int d = 0; d < HD; d++) dot += (double)qh[d] * kh[d];
                scores[t] = (float)(dot * scale);
            }
            // softmax over [p0..pos]
            float maxv = scores[p0];
            for (int t = p0 + 1; t <= pos; t++) if (scores[t] > maxv) maxv = scores[t];
            double ssum = 0.0;
            for (int t = p0; t <= pos; t++) {
                scores[t] = expf(scores[t] - maxv);
                ssum += scores[t];
            }
            float inv = (float)(1.0 / ssum);
            // weighted sum of V into xb2 (head h slice)
            float *out = s->xb2 + (size_t)h * HD;
            for (int t = p0; t <= pos; t++) {
                float w = scores[t] * inv;
                const float *vh = s->vcache + (size_t)li * T * KD
                                + (size_t)t * KD + h_kv * HD;
                for (int d = 0; d < HD; d++) out[d] += w * vh[d];
            }
        }

        // attn output projection: x_out = c_proj(xb2)
        linear(s->xb, s->xb2, D, D, m->L[li].o_w, m->L[li].o_scale, m->L[li].o_fp32);
        // residual
        for (int i = 0; i < D; i++) s->x[i] += s->xb[i];

        // ---- mlp ----
        rmsnorm(s->xb, s->x, D);
        linear(s->ffn_h, s->xb, FF, D, m->L[li].fc_w, m->L[li].fc_scale, m->L[li].fc_fp32);
        // ReLU squared
        for (int i = 0; i < FF; i++) {
            float v = s->ffn_h[i];
            if (v < 0.0f) v = 0.0f;
            s->ffn_h[i] = v * v;
        }
        linear(s->xb, s->ffn_h, D, FF, m->L[li].pj_w, m->L[li].pj_scale, m->L[li].pj_fp32);
        for (int i = 0; i < D; i++) s->x[i] += s->xb[i];

        // Backout snapshot at midpoint
        if (li == s->backout_layer) {
            memcpy(s->x_backout, s->x, sizeof(float) * D);
        }
    }

    // 6) Subtract backout, then norm
    if (m->n_layer > 0) {
        for (int i = 0; i < D; i++) s->x[i] -= m->backout_lambda * s->x_backout[i];
    }
    rmsnorm(s->xb, s->x, D);

    // 7) lm_head -> logits, softcap
    linear(s->logits, s->xb, m->pad_vocab_size, D,
           m->lm_head_q, m->lm_head_scales, m->lm_head);
    float cap = (float)m->softcap;
    for (int i = 0; i < m->vocab_size; i++) {
        float v = s->logits[i] / cap;
        s->logits[i] = cap * tanhf(v);
    }

    s->seq_pos = pos + 1;
}

// ===== sampling =====
static uint64_t rng_state;
static uint32_t xorshift32(void) {
    uint64_t x = rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    rng_state = x;
    return (uint32_t)x;
}
static float urand(void) { return (xorshift32() & 0xffffff) / 16777216.0f; }

static int cmp_desc(const void *a, const void *b, void *ctx) {
    const float *p = (const float *)ctx;
    int ia = *(const int *)a, ib = *(const int *)b;
    if (p[ia] > p[ib]) return -1;
    if (p[ia] < p[ib]) return 1;
    return 0;
}

#ifdef _WIN32
// MSVC/MinGW don't have qsort_r. Use a thread-local pointer.
static const float *g_cmp_p;
static int cmp_desc_w(const void *a, const void *b) {
    int ia = *(const int *)a, ib = *(const int *)b;
    if (g_cmp_p[ia] > g_cmp_p[ib]) return -1;
    if (g_cmp_p[ia] < g_cmp_p[ib]) return 1;
    return 0;
}
#endif

static int sample(nc_state *s, float temperature, float top_p) {
    nc_model *m = s->m;
    int V = m->vocab_size;
    if (temperature <= 0.0f) {
        // greedy
        int best = 0;
        float bv = s->logits[0];
        for (int i = 1; i < V; i++) if (s->logits[i] > bv) { bv = s->logits[i]; best = i; }
        return best;
    }
    // apply temperature, softmax
    float maxv = s->logits[0];
    for (int i = 1; i < V; i++) if (s->logits[i] > maxv) maxv = s->logits[i];
    double ssum = 0.0;
    for (int i = 0; i < V; i++) {
        s->probs[i] = expf((s->logits[i] - maxv) / temperature);
        ssum += s->probs[i];
    }
    float inv = (float)(1.0 / ssum);
    for (int i = 0; i < V; i++) s->probs[i] *= inv;

    if (top_p >= 1.0f) {
        // direct sample
        float r = urand(), cdf = 0.0f;
        for (int i = 0; i < V; i++) { cdf += s->probs[i]; if (r < cdf) return i; }
        return V - 1;
    }

    // top-p filter: sort indices by probability desc
    for (int i = 0; i < V; i++) s->ranking[i] = i;
#ifdef _WIN32
    g_cmp_p = s->probs;
    qsort(s->ranking, V, sizeof(int), cmp_desc_w);
#else
    qsort_r(s->ranking, V, sizeof(int), cmp_desc, (void *)s->probs);
#endif
    float acc = 0.0f;
    int last = 0;
    for (int i = 0; i < V; i++) {
        acc += s->probs[s->ranking[i]];
        last = i;
        if (acc >= top_p) break;
    }
    // sample from top (last+1) by their original probs
    float r = urand() * acc, cdf = 0.0f;
    for (int i = 0; i <= last; i++) {
        cdf += s->probs[s->ranking[i]];
        if (r < cdf) return s->ranking[i];
    }
    return s->ranking[last];
}

// ===== sentinel I/O =====
static void emit_sentinel_str(const char *kind, const char *text) {
    fputc('\x01', stdout);
    fputs(kind, stdout);
    if (text && *text) { fputc(' ', stdout); fputs(text, stdout); }
    fputc('\n', stdout);
    fflush(stdout);
}

static void emit_text(const char *bytes, int n) {
    fwrite(bytes, 1, (size_t)n, stdout);
    fflush(stdout);
}

// ===== main loop =====
static int read_line(char *buf, int max) {
    int n = 0; int c;
    while ((c = getchar()) != EOF && c != '\n') {
        if (c == '\r') continue;
        if (n + 1 < max) buf[n++] = (char)c;
    }
    buf[n] = 0;
    return c == EOF && n == 0 ? -1 : n;
}

int main(int argc, char **argv) {
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stdin),  _O_BINARY);
#endif
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.NCB TOKENIZER.NCT [-c ctx] [-t temp] [-p top_p] [-s seed]\n", argv[0]);
        return 1;
    }
    const char *model_path = argv[1];
    const char *tok_path   = argv[2];
    float temp = 0.8f, top_p = 0.95f;
    int   ctx_max = 256;
    uint64_t seed = (uint64_t)time(NULL);
    for (int i = 3; i + 1 < argc; i += 2) {
        if      (!strcmp(argv[i], "-c")) ctx_max = atoi(argv[i+1]);
        else if (!strcmp(argv[i], "-t")) temp    = (float)atof(argv[i+1]);
        else if (!strcmp(argv[i], "-p")) top_p   = (float)atof(argv[i+1]);
        else if (!strcmp(argv[i], "-s")) seed    = (uint64_t)atoll(argv[i+1]);
    }
    rng_state = seed ? seed : 0xDEADBEEFCAFEBABEULL;

    nc_model M = {0};
    if (!model_load(&M, model_path)) return 2;
    if (M.sequence_len < ctx_max) ctx_max = M.sequence_len;

    nct *T = nct_load(tok_path);
    if (!T) { fprintf(stderr, "[nc_run] tokenizer load failed: %s\n", tok_path); return 3; }

    // Identify special tokens (best-effort)
    int bos_id          = nct_special_id(T, "<|bos|>");
    int eos_id          = nct_special_id(T, "<|endoftext|>");
    int user_start      = nct_special_id(T, "<|user_start|>");
    int user_end        = nct_special_id(T, "<|user_end|>");
    int assistant_start = nct_special_id(T, "<|assistant_start|>");
    int assistant_end   = nct_special_id(T, "<|assistant_end|>");

    nc_state S; state_init(&S, &M);

    // Report model info to GUI
    {
        char info[128];
        snprintf(info, sizeof(info), "nanochat-d%d %dM (fp32 %s)",
                 M.n_layer,
                 (int)((double)((size_t)M.pad_vocab_size * M.n_embd * 2
                                + (size_t)M.n_layer * 6 * M.n_embd * M.n_embd) / 1e6 + 0.5),
                 M.dtype_code == 1 ? "int8" : "fp32");
        emit_sentinel_str("INFO", info);
    }
    emit_sentinel_str("READY", NULL);

    char line[8192];
    int   prompt_ids[1024];

    while (read_line(line, sizeof(line)) >= 0) {
        if (line[0] == 0) continue;

        // Reset KV cache each turn for now (no multi-turn memory yet)
        state_reset(&S);

        // Build prompt: <bos> <user_start> user-text <user_end> <assistant_start>
        int n = 0;
        if (bos_id >= 0)          prompt_ids[n++] = bos_id;
        if (user_start >= 0)      prompt_ids[n++] = user_start;
        n += nct_encode(T, line, prompt_ids + n, (int)(sizeof(prompt_ids)/sizeof(int)) - n - 8);
        if (user_end >= 0)        prompt_ids[n++] = user_end;
        if (assistant_start >= 0) prompt_ids[n++] = assistant_start;

        // Prefill
        for (int i = 0; i < n; i++) forward_one(&S, prompt_ids[i]);

        // Generate
        int last_token = prompt_ids[n - 1];
        for (int gen = 0; gen < ctx_max - n; gen++) {
            int next = sample(&S, temp, top_p);
            if (next == eos_id || next == assistant_end) break;
            char piece[64];
            int pn = nct_decode_one(T, next, piece, sizeof(piece));
            if (pn > 0) emit_text(piece, pn);
            forward_one(&S, next);
            last_token = next;
            if (S.seq_pos >= ctx_max) break;
        }
        emit_sentinel_str("EOT", NULL);
    }

    state_free(&S);
    nct_free(T);
    return 0;
}
