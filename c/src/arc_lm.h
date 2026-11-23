/* Token n-gram language model for shallow fusion in Arc's decoder (arc_set_lm).
 *
 * The model is over the decoder's own SentencePiece ids. Only text pieces are scored: inline tags, context markers and
 * other specials are left out of the history and are neutral in fusion. Probabilities are an ARPA-style backoff model:
 * P(t | h) is the longest stored n-gram (h_k, t), times the backoff weights of the longer contexts that were missing.
 *
 * File "ARCLM1" (little endian, natural logs; docs/LIBARC.md#n-gram-lm):
 *   char magic[8] = "ARCLM1\0\0"; int32 order (1-8); int32 vocab; float32 unscored; int32 cap (power of 2)
 *   uint8 scored[vocab], zero padded to a multiple of 8 bytes
 *   cap x {uint64 key; float32 logp; float32 bow}   open addressing, key 0 = empty slot (at least one)
 * key = arc_lm_key(ids, n); its slot is (key ^ key >> 32) & (cap - 1), then linear probing. logp = ln P(last | rest);
 * a positive logp marks an entry that is only a context (it has a backoff weight but is not an n-gram itself).
 * bow = ln of the backoff weight of the entry as a context (0 if none). unscored (<= 0): a typical text piece's ln p;
 * a scored token with no unigram costs twice that.
 *
 * Self-contained (libc only; POSIX mmap, a plain read on Windows), so other tools can compile it on its own.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#ifndef ARC_LM_H
#define ARC_LM_H
#include <stddef.h>
#include <stdint.h>

typedef struct arc_lm arc_lm;

/* NULL and a message in err on any problem (missing file, bad magic or header, wrong size, bad values, full table) */
arc_lm* arc_lm_load(const char* path, char* err, size_t err_len);
void arc_lm_free(arc_lm* lm);
int arc_lm_order(const arc_lm* lm);
int arc_lm_vocab(const arc_lm* lm);
long arc_lm_entries(const arc_lm* lm);   /* used slots */
/* 1: the token is a text piece the model scores (0: tags, markers, ids beyond the vocabulary) */
int arc_lm_scored(const arc_lm* lm, int tok);
/* ln P(tok | hist[0..nh)) with hist the scored tokens so far, oldest first (only the last order-1 are used) */
float arc_lm_logp(const arc_lm* lm, const int* hist, int nh, int tok);
/* the key of an n-gram: FNV-1a 64 over the ids as little-endian int32 bytes, ^ n * 0x9E3779B97F4A7C15, | 1 */
uint64_t arc_lm_key(const int* ids, int n);

/* Shallow fusion on one row of decoder scores (row[V]; gen[ng]: the tokens generated so far, start token first).
 * A relative re-ranker: the row is log-softmaxed; of its topk best tokens (0 = 16, at most 64), each text piece within
 * ARC_LM_MARGIN of the best is charged weight * (ln P_lm(t) - ln P_lm(the LM's favourite among them)), capped at
 * weight * ARC_LM_CAP; tags are neutral; the end token gets what the best token gets (the LM never decides where
 * speech ends); every other token weight * ARC_LM_CAP. weight <= 0: the row is left as it is. */
#define ARC_LM_MARGIN 4.0f
#define ARC_LM_CAP 6.0f
void arc_lm_fuse(const arc_lm* lm, float weight, int topk, float* row, int V, const int* gen, int ng, int eos);

#endif
