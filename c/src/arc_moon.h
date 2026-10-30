/* Moonshine Streaming (tiny), the speech encoder and transcript decoder inside Arc's edge backbone: a C port of
 * transformers' MoonshineStreaming (encoder: per-5 ms-frame CMVN + asinh compression + linear, two stride-2 causal
 * convs, 6 sliding-window attention layers; decoder: 6 layers of causal self-attention with partial interleaved RoPE,
 * cross-attention over the encoder states plus learned positions, gated SiLU MLP; output tied to the embeddings),
 * and the decoding Arc runs on it (greedy, or HF-identical beam search, with Arc's logits processors).
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#ifndef ARC_MOON_H
#define ARC_MOON_H

#include <stdint.h>

#include "arc_base.h"
#include "arc_store.h"

#define ARC_MOON_D 320
#define ARC_MOON_LAYERS 6
#define ARC_MOON_HEADS 8
#define ARC_MOON_HD 40
#define ARC_MOON_FF 1280
#define ARC_MOON_VOCAB 32768
#define ARC_MOON_FRAME 80      /* samples per 5 ms frame */

/* A weight matrix: Arc's packed 4-bit form used in place when the tensor is stored that way, else float32. */
typedef struct {
  const float* f;
  arc_q4 q;
  int packed;
} arc_w;

typedef struct {
  const float *ln1, *ln2, *fc1b, *fc2b;
  arc_w q, k, v, o, fc1, fc2;
} arc_moon_enc_layer;

typedef struct {
  const float *ln_in, *ln_post, *ln_final, *fc1b, *fc2b;
  arc_w sq, sk, sv, so, cq, ck, cv, co, fc1, fc2;
} arc_moon_dec_layer;

typedef struct arc_moon {
  arc_store* st;
  arc_pool* pool;
  /* encoder */
  float comp_k;            /* exp(log_k) */
  const float *lin, *conv1, *conv1b, *conv2, *conv2b, *final_g;
  arc_moon_enc_layer enc[ARC_MOON_LAYERS];
  int win[ARC_MOON_LAYERS][2];
  /* decoder */
  arc_w embed;
  const float *pos_emb, *norm;
  arc_moon_dec_layer dec[ARC_MOON_LAYERS];
  float inv_freq[16];      /* RoPE on the first 32 of 40 head dims, interleaved pairs */
  int n_rot;               /* 32 */
  int bos, eos;
} arc_moon;

/* folder: the moonshine-streaming-tiny folder (model.q4.safetensors or model.safetensors). */
arc_moon* arc_moon_load(const char* folder, int threads);
void arc_moon_free(arc_moon* m);

/* Encoder over a whole utterance: pcm[n] at 16 kHz (n a multiple of 80, at least 80). Returns the states
 * [T, 320] (arc_free() them) and T, or NULL. */
float* arc_moon_encode(arc_moon* m, const float* pcm, int n, int* T);
/* The same, and (tap_out != NULL) the hidden states before layer `tap` [T, 320] (arc_free them). */
float* arc_moon_encode_ex(arc_moon* m, const float* pcm, int n, int* T, int tap, float** tap_out);

/* Logits processors Arc uses (arc_audio/transcribe.py): tag offsets (add, then log-softmax again), suppressed ids,
 * and the repeat guard (a sequence repeating a unit of 1..8 tokens `copies` times after `start` ends). */
/* Name biasing (arc_audio/bias.py): token sequences of listed names; a name's first token gets start_bonus, the next
 * token of a name in progress cont_bonus (on log-softmaxed scores), weighted by the name type's weight. */
typedef struct {
  int n;                   /* sequences */
  const int* off;          /* n + 1 offsets into ids */
  const int* ids;
  const float* w;          /* weight per sequence (its type's) */
  float start_bonus, cont_bonus;
  int maxlen;
  int prompt_len;          /* BiasProcessor.p0: the generated part starts here in the decoder input (1) */
} arc_moon_bias;

typedef struct {
  int n_off;
  const int* off_id;
  const float* off_val;
  int n_suppress;
  const int* suppress;
  int repeat_guard;        /* 1: on */
  int rg_max_unit, rg_copies, rg_min_tokens;
  const arc_moon_bias* bias;   /* NULL: off */
} arc_moon_procs;

/* Decode: prompt[n_prompt] (context tokens, may be 0) then the start token, up to max_length tokens in all
 * (prompt included, as HF's max_length), greedy (beams == 1) or beam search. Writes the generated tokens after the
 * prompt (start token first, as HF returns them) to out (cap entries) and returns their count, or -1. */
int arc_moon_decode(arc_moon* m, const float* enc, int T, const int* prompt, int n_prompt, int max_length, int beams,
                    const arc_moon_procs* procs, int* out, int cap);

/* Teacher-forced cross-attention (arc_audio align.cross_attention): tokens[n] -> weights [n][layers][heads][T]
 * (arc_free them), or NULL. */
float* arc_moon_cross_attn(arc_moon* m, const float* enc, int T, const int* tokens, int n);

/* The same, and the finished beams (best first, the first one is what arc_moon_decode returns): up to max_alts
 * sequences of up to cap tokens each in alt_ids[max_alts * cap] (start token first), their lengths and HF's
 * length-normalised beam scores; *n_alts their count (1 for greedy, score 0). */
int arc_moon_decode_nbest(arc_moon* m, const float* enc, int T, const int* prompt, int n_prompt, int max_length,
                          int beams, const arc_moon_procs* procs, int* out, int cap, int max_alts, int* alt_ids,
                          int* alt_len, float* alt_score, int* n_alts);

/* Teacher-forced logits (for tests): tokens[n] -> logits [n, 32768] of the next token at each position. */
float* arc_moon_logits(arc_moon* m, const float* enc, int T, const int* tokens, int n);

/* Teacher-forced log P(seq | audio) for n token sequences (each without the start token; EOS is appended and scored),
 * after prompt[n_prompt] and the start token, with no logits processors: out[i] = sum of log-softmax of the next token.
 * The cross-attention keys / values are computed once for all sequences, and sequences sharing a prefix share its
 * decoder steps (they are walked as a prefix tree, at most ARC_MOON_SCORE_GROUP leaves at a time). 0 or -1. */
#define ARC_MOON_SCORE_GROUP 32
int arc_moon_score(arc_moon* m, const float* enc, int T, const int* prompt, int n_prompt, const int* const* seqs,
                   const int* lens, int n, double* out);

/* Teacher-forced logits (for tests): tokens[n] -> logits [n, 32768] of the next token at each position. */
float* arc_moon_logits(arc_moon* m, const float* enc, int T, const int* tokens, int n);

#endif
