// nc_tokenizer: minimal tiktoken-style BPE tokenizer for nanochat models.
// Loads .nct binary produced by export_tokenizer.py.
//
// Limitations vs upstream tiktoken:
//  - ASCII-only split pattern (Unicode \p{L}, \p{N} approximated as [A-Za-z], [0-9])
//  - Byte-level vocabulary handles non-ASCII by escaping into byte tokens, so
//    non-English UTF-8 still encodes (as raw bytes) but the regex won't form
//    optimal pre-tokens for it. For English chat that's totally fine.
//
// Algorithm:
//  encode(text):
//    1. Apply regex-style split to break text into chunks
//    2. For each chunk:
//       a. Initialize: each byte b becomes the token whose bytes are {b}
//       b. Repeatedly find adjacent token pair whose concatenated bytes are in
//          the vocabulary, choosing the pair with the SMALLEST rank (=token id).
//          Replace the pair with that token. Stop when no merges remain.
//       c. Emit the resulting token ids.

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>

// ===== struct visible to caller =====
typedef struct {
    uint8_t *bytes;   // pointer into vocab_bytes blob
    int      nbytes;
    int      id;
} nct_token;

typedef struct {
    char    *name;    // owned string
    int      id;
} nct_special;

typedef struct nct {
    int n_regular;
    int n_specials;
    int vocab_size;

    // Token bytes blob: all token byte sequences concatenated.
    uint8_t *vocab_blob;
    size_t   vocab_blob_len;

    // id -> token (owned arrays)
    nct_token  *id2tok;       // indexed by token id; nbytes==0 => empty/special
    int         id2tok_max;   // = vocab_size

    // Specials
    nct_special *specials;    // n_specials entries

    // Hash table: bytes -> id (for byte-sequence lookup during BPE)
    int      *hash_table;     // values are token IDs (or -1)
    int       hash_size;      // power of 2
    nct_token *hash_keys;     // pointer back to id2tok entry for collision check
} nct;

// ===== utilities =====
static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "[nct] OOM\n"); exit(2); }
    return p;
}
static void *xcalloc(size_t n, size_t s) {
    void *p = calloc(n, s);
    if (!p) { fprintf(stderr, "[nct] OOM\n"); exit(2); }
    return p;
}

static uint32_t fnv1a(const uint8_t *b, int n) {
    uint32_t h = 0x811C9DC5u;
    for (int i = 0; i < n; i++) { h ^= b[i]; h *= 0x01000193u; }
    return h;
}

static int next_pow2(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

// ===== load =====
nct *nct_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *all = (uint8_t *)xmalloc((size_t)sz);
    if (fread(all, 1, (size_t)sz, f) != (size_t)sz) { free(all); fclose(f); return NULL; }
    fclose(f);

    if (sz < 64 || memcmp(all, "NCT1\x00\x00\x00\x00", 8) != 0) { free(all); return NULL; }
    uint32_t version    = *(uint32_t *)(all + 8);
    uint32_t vocab_size = *(uint32_t *)(all + 12);
    uint32_t n_regular  = *(uint32_t *)(all + 16);
    uint32_t n_specials = *(uint32_t *)(all + 20);
    if (version != 1) { free(all); return NULL; }

    nct *T = (nct *)xcalloc(1, sizeof(nct));
    T->n_regular = n_regular;
    T->n_specials = n_specials;
    T->vocab_size = vocab_size;
    T->vocab_blob = all;          // we keep the whole file
    T->vocab_blob_len = (size_t)sz;
    T->id2tok = (nct_token *)xcalloc(vocab_size, sizeof(nct_token));
    T->id2tok_max = vocab_size;
    T->specials = (nct_special *)xcalloc(n_specials > 0 ? n_specials : 1, sizeof(nct_special));

    uint8_t *p = all + 64;
    uint8_t *end = all + sz;
    // Regular tokens
    for (uint32_t i = 0; i < n_regular; i++) {
        if (p + 8 > end) { fclose(f); /* fall through */ break; }
        uint32_t id = *(uint32_t *)p; p += 4;
        uint32_t nb = *(uint32_t *)p; p += 4;
        if (p + nb > end) break;
        if (id >= vocab_size) break;
        T->id2tok[id].bytes  = p;
        T->id2tok[id].nbytes = (int)nb;
        T->id2tok[id].id     = (int)id;
        p += nb;
    }
    // Specials
    for (uint32_t i = 0; i < n_specials; i++) {
        if (p + 8 > end) break;
        uint32_t id = *(uint32_t *)p; p += 4;
        uint32_t nb = *(uint32_t *)p; p += 4;
        if (p + nb > end) break;
        char *nm = (char *)xmalloc(nb + 1);
        memcpy(nm, p, nb); nm[nb] = 0;
        T->specials[i].id = (int)id;
        T->specials[i].name = nm;
        // Also stash in id2tok so decode can return the special name (rare path)
        if (id < vocab_size) {
            T->id2tok[id].bytes = p;
            T->id2tok[id].nbytes = (int)nb;
            T->id2tok[id].id = (int)id;
        }
        p += nb;
    }

    // Build hash table for byte-sequence -> token id lookup
    int hsz = next_pow2(n_regular * 2 + 16);
    T->hash_size = hsz;
    T->hash_table = (int *)xmalloc(sizeof(int) * hsz);
    for (int i = 0; i < hsz; i++) T->hash_table[i] = -1;
    for (uint32_t i = 0; i < n_regular; i++) {
        nct_token *t = &T->id2tok[i];   // tiktoken id space is 0..n_regular-1 for non-specials
        if (t->nbytes == 0) continue;
        uint32_t h = fnv1a(t->bytes, t->nbytes) & (hsz - 1);
        for (;;) {
            int slot = T->hash_table[h];
            if (slot == -1) { T->hash_table[h] = (int)i; break; }
            h = (h + 1) & (hsz - 1);
        }
    }
    return T;
}

void nct_free(nct *T) {
    if (!T) return;
    for (int i = 0; i < T->n_specials; i++) free(T->specials[i].name);
    free(T->specials);
    free(T->id2tok);
    free(T->hash_table);
    free(T->vocab_blob);
    free(T);
}

int nct_vocab_size(const nct *T) { return T ? T->vocab_size : 0; }

int nct_special_id(const nct *T, const char *name) {
    if (!T || !name) return -1;
    for (int i = 0; i < T->n_specials; i++) {
        if (strcmp(T->specials[i].name, name) == 0) return T->specials[i].id;
    }
    return -1;
}

// Returns id if bytes match a token, else -1.
static int lookup_id(const nct *T, const uint8_t *b, int n) {
    uint32_t h = fnv1a(b, n) & (T->hash_size - 1);
    for (int probes = 0; probes < T->hash_size; probes++) {
        int id = T->hash_table[h];
        if (id == -1) return -1;
        nct_token *t = &T->id2tok[id];
        if (t->nbytes == n && memcmp(t->bytes, b, n) == 0) return id;
        h = (h + 1) & (T->hash_size - 1);
    }
    return -1;
}

// ===== regex-style split (ASCII subset of tiktoken's pattern) =====
//
// Pattern (English-only approximation of tiktoken's GPT-4-style split):
//   '(?i:[sdmt]|ll|ve|re)         => contractions ('s 'd 'll 've ...)
//   [^\r\n A-Za-z0-9 punct etc]?+ [A-Za-z]+  => optional sep + letters
//   [0-9]{1,2}                      => 1 or 2 digits
//   ' ?'  punctuation chunk + newlines
//   newline runs
//   trailing whitespace
//   whitespace runs
//
// Returns the length of the next chunk starting at s[0..]. Always >= 1 unless
// at end-of-string.

static int is_letter(unsigned char c) { return isalpha(c); }
static int is_digit(unsigned char c)  { return isdigit(c); }
static int is_space(unsigned char c)  { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }

static int chunk_len(const unsigned char *s, int n) {
    if (n <= 0) return 0;
    unsigned char c = s[0];

    // Rule 1: contraction "'s" "'t" "'d" "'m" "'ll" "'ve" "'re" (case-insensitive after apostrophe)
    if (c == '\'') {
        if (n >= 2) {
            unsigned char c1 = (unsigned char)tolower(s[1]);
            if (c1 == 's' || c1 == 'd' || c1 == 'm' || c1 == 't') return 2;
            if (n >= 3) {
                unsigned char c2 = (unsigned char)tolower(s[2]);
                if ((c1 == 'l' && c2 == 'l') || (c1 == 'v' && c2 == 'e') || (c1 == 'r' && c2 == 'e')) return 3;
            }
        }
        // fallthrough: a lone apostrophe — handle as punctuation below
    }

    // Rule 2: optional non-letter/non-digit/non-newline + run of letters
    {
        int i = 0;
        // optional sep: not letter/digit, not newline (allowed: space, punctuation, etc.)
        if (i < n) {
            unsigned char x = s[i];
            if (!is_letter(x) && !is_digit(x) && x != '\r' && x != '\n') {
                // possessive lookahead: only consume sep if the NEXT char is a letter
                if (i + 1 < n && is_letter((unsigned char)s[i+1])) {
                    i++;
                }
            }
        }
        if (i < n && is_letter((unsigned char)s[i])) {
            int start = i;
            while (i < n && is_letter((unsigned char)s[i])) i++;
            // Match found: include everything from 0 to i
            (void)start;
            return i;
        }
    }

    // Rule 3: 1 or 2 digits
    if (is_digit(c)) {
        int i = 1;
        if (i < n && is_digit((unsigned char)s[i])) i++;
        return i;
    }

    // Rule 4: ' ?' + run of non-space/non-letter/non-digit chars + optional newlines
    {
        int i = 0;
        if (c == ' ') i = 1;  // optional leading space (single)
        if (i < n) {
            unsigned char x = s[i];
            if (!is_space(x) && !is_letter(x) && !is_digit(x)) {
                int start = i;
                while (i < n) {
                    unsigned char y = s[i];
                    if (is_space(y) || is_letter(y) || is_digit(y)) break;
                    i++;
                }
                // optional newlines after
                while (i < n && (s[i] == '\r' || s[i] == '\n')) i++;
                if (i > start) return i;
            }
        }
    }

    // Rule 5: whitespace + newline OR newlines: \s*[\r\n]
    {
        int i = 0;
        while (i < n && is_space((unsigned char)s[i]) && s[i] != '\r' && s[i] != '\n') i++;
        if (i < n && (s[i] == '\r' || s[i] == '\n')) {
            i++;
            return i;
        }
    }

    // Rule 6+7: any whitespace run
    if (is_space(c)) {
        int i = 0;
        while (i < n && is_space((unsigned char)s[i])) i++;
        return i;
    }

    // Fallback: just consume one byte (keeps progress)
    return 1;
}

// ===== BPE encode one chunk =====
//
// chunk is a byte sequence; we produce a sequence of token IDs.
// Working buffer: we represent the in-progress sequence as an array of "parts",
// where each part is (bytes_offset, nbytes) and corresponds to one token id.
// We store ID = -1 if the merged bytes haven't been resolved to an ID yet
// (we rebuild merged byte strings into a side buffer).

#define MAX_CHUNK_PARTS 1024

typedef struct {
    int id;           // current token id; -1 means "see scratch"
    const uint8_t *p; // pointer to bytes
    int nbytes;
} part_t;

static int bpe_encode_chunk(const nct *T, const uint8_t *bytes, int n,
                            int *out_ids, int out_max,
                            uint8_t *scratch, int scratch_max) {
    if (n == 0) return 0;
    if (n > MAX_CHUNK_PARTS) {
        // Fallback: emit each byte as its own token, no merging.
        int written = 0;
        for (int i = 0; i < n && written < out_max; i++) {
            int id = lookup_id(T, bytes + i, 1);
            if (id < 0) continue;  // shouldn't happen for byte tokens
            out_ids[written++] = id;
        }
        return written;
    }

    part_t parts[MAX_CHUNK_PARTS];
    int np = 0;
    for (int i = 0; i < n; i++) {
        int id = lookup_id(T, bytes + i, 1);
        if (id < 0) continue;
        parts[np].id = id;
        parts[np].p  = bytes + i;
        parts[np].nbytes = 1;
        np++;
    }

    // Merge loop
    int sp = 0; // scratch position
    for (;;) {
        int best_i = -1, best_id = 0x7fffffff;
        // Try every adjacent pair
        for (int i = 0; i + 1 < np; i++) {
            int total = parts[i].nbytes + parts[i+1].nbytes;
            if (sp + total > scratch_max) {
                sp = 0; // reset; we won't keep persistent storage so this is fine
            }
            // Build candidate bytes in scratch
            uint8_t *dst = scratch + sp;
            memcpy(dst, parts[i].p, parts[i].nbytes);
            memcpy(dst + parts[i].nbytes, parts[i+1].p, parts[i+1].nbytes);
            int id = lookup_id(T, dst, total);
            if (id >= 0 && id < best_id) {
                best_id = id;
                best_i = i;
            }
        }
        if (best_i < 0) break;
        // Apply merge at best_i: rebuild merged bytes into scratch (persistent until next merge cycle)
        int total = parts[best_i].nbytes + parts[best_i+1].nbytes;
        if (sp + total > scratch_max) sp = 0;
        uint8_t *dst = scratch + sp;
        memcpy(dst, parts[best_i].p, parts[best_i].nbytes);
        memcpy(dst + parts[best_i].nbytes, parts[best_i+1].p, parts[best_i+1].nbytes);
        parts[best_i].id = best_id;
        parts[best_i].p  = dst;
        parts[best_i].nbytes = total;
        sp += total;
        // Shift the rest left
        for (int j = best_i + 1; j + 1 < np; j++) parts[j] = parts[j+1];
        np--;
    }

    int written = 0;
    for (int i = 0; i < np && written < out_max; i++) out_ids[written++] = parts[i].id;
    return written;
}

// ===== public encode =====
int nct_encode(const nct *T, const char *text, int *out_ids, int out_max) {
    if (!T || !text) return 0;
    static uint8_t scratch[8192];
    int written = 0;
    const unsigned char *s = (const unsigned char *)text;
    int total = (int)strlen(text);
    int pos = 0;
    while (pos < total && written < out_max) {
        int len = chunk_len(s + pos, total - pos);
        if (len <= 0) len = 1;
        int n = bpe_encode_chunk(T, s + pos, len, out_ids + written, out_max - written,
                                 scratch, sizeof(scratch));
        written += n;
        pos += len;
    }
    return written;
}

// ===== decode =====
int nct_decode_one(const nct *T, int id, char *out, int out_max) {
    if (!T || id < 0 || id >= T->id2tok_max) return 0;
    nct_token *t = &T->id2tok[id];
    if (t->nbytes == 0) return 0;
    // Skip special tokens (their stored bytes are the human-readable name)
    for (int i = 0; i < T->n_specials; i++) {
        if (T->specials[i].id == id) return 0;
    }
    int n = t->nbytes;
    if (n > out_max) n = out_max;
    memcpy(out, t->bytes, n);
    return n;
}
