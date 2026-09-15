#ifndef SLM_TOKENIZER_H
#define SLM_TOKENIZER_H
#include <stddef.h>
typedef struct slm_tokenizer slm_tokenizer;
slm_tokenizer *slm_tokenizer_load(const char *path);
void slm_tokenizer_free(slm_tokenizer *t);
/* Returns token count or -1. Specials are recognized only when explicitly enabled.
 * Version 2 nonspecial added tokens always match, as in Hugging Face tokenizers.
 * No implicit BOS or other framing is inserted. */
int slm_encode(slm_tokenizer *t, const char *text, int specials, int *ids, int capacity);
const unsigned char *slm_token_bytes(slm_tokenizer *t, int id, int *length, int *special);
int slm_tokenizer_vocab(slm_tokenizer *t);
#endif
