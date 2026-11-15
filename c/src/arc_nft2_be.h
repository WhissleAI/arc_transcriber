/* The NFT2 nets behind one small interface: the cache-aware streaming FastConformer encoder (one chunk at a time, with
 * its layer caches and the Arc tap) and the RNNT nets (prediction LSTM, joint). Everything around them (log-mel front
 * end, chunking, greedy decoding, biasing, the Arc layer) is libarc's own C in arc_nft2.c.
 *
 * Backends: arc_nft2_ort.c (ONNX Runtime C API on the release's encoder_onnx/ graphs, built with make ARC_ORT=...) and
 * arc_nft2_none.c (no backend: open fails with a message). A native C backend implements the same functions.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#ifndef ARC_NFT2_BE_H
#define ARC_NFT2_BE_H

#include "arc_base.h"

/* encoder_onnx/config.json, the shapes both sides agree on */
typedef struct {
  int n_layers, d_model, n_mels;
  int cache_ch;      /* last_channel_cache_size (70): attention keys / values kept per layer */
  int conv_cache;    /* conv_cache (8): causal depthwise conv state per layer */
  int first;         /* first_chunk_frames (49): mel frames of chunk 0 (no pre-encode cache); 0 = no first-chunk graph */
  int chunk;         /* shift_size_frames[1] (56): new mel frames per later chunk */
  int pre;           /* pre_encode_cache_size[1] (9): mel frames of the previous chunk fed again in front */
  int pred_layers, pred_hidden, joint_hidden, vocab, blank, max_sym;
} arc_nft2_dims;

typedef struct arc_nft2_be arc_nft2_be;
typedef struct arc_nft2_cache arc_nft2_cache;

/* onnx_dir: encoder_onnx/. tap: the encoder layer whose output is also returned (9: after layer 9), 0 = none.
 * threads: intra-op threads of the encoder (the RNNT nets run on one). NULL + arc_seterr on failure. */
arc_nft2_be* arc_nft2_be_open(const char* onnx_dir, const arc_nft2_dims* dims, int tap, int threads);
void arc_nft2_be_free(arc_nft2_be* b);
const char* arc_nft2_be_name(void);   /* "onnxruntime 1.23.2", "none" */

/* The encoder's layer caches, zero at the start of an utterance. */
arc_nft2_cache* arc_nft2_be_cache_new(arc_nft2_be* b);
void arc_nft2_be_cache_free(arc_nft2_cache* c);
/* One streaming chunk: x [n, n_mels] mel frames, time-major (chunk 0: the first `first` frames with first = 1; later:
 * the pre cached frames then the new ones). Writes *k encoder frames to out [k, d_model] and, when the backend was
 * opened with a tap, the tap's frames to tap [k, d_model] (may be NULL); at most cap frames. Updates the caches. */
int arc_nft2_be_chunk(arc_nft2_be* b, arc_nft2_cache* c, int first, const float* x, int n, float* out, float* tap,
                      int cap, int* k);

/* RNNT: joint encoder projection F [T, joint_hidden] = joint_enc(H [T, d_model]) */
int arc_nft2_be_joint_enc(arc_nft2_be* b, const float* H, int T, float* F);
/* prediction net step: token y (blank: the zero "start" input), state hc [2, pred_layers, pred_hidden] (h then c) in
 * and out, g [joint_hidden] out */
int arc_nft2_be_pred(arc_nft2_be* b, int y, float* hc, float* g);
/* joint: logits [n, vocab + 1] for rows F [n, joint_hidden] with the same g */
int arc_nft2_be_joint(arc_nft2_be* b, const float* F, int n, const float* g, float* logits);

#endif
