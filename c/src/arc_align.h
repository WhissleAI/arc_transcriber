/* Word timestamps for a transcript (arc_audio/align.py): the decoder run teacher-forced on the text over the
 * utterance's encoder states, the cross-attention of a few alignment heads normalised, median-filtered and averaged,
 * and a DTW path from tokens to 20 ms encoder frames. Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see
 * LICENSE. */
#ifndef ARC_ALIGN_H
#define ARC_ALIGN_H

#include "arc_moon.h"
#include "arc_tok.h"

typedef struct {
  char* w;              /* the word as the tokenizer's pieces spell it (punctuation attached) */
  double start, end;    /* seconds, rounded to 1 ms as align.py does */
} arc_aword;

typedef struct {
  int heads[16][2];     /* (layer, head); n_heads 0: every head of the upper half of the layers */
  int n_heads;
  int medfilt;          /* median filter width over frames (config alignment.medfilt, default 7) */
  double offset_s;      /* added to every time (alignment.offset_s) */
} arc_align_cfg;

/* Reads config.json "alignment" (NULL: defaults). */
struct arc_json;
void arc_align_cfg_load(arc_align_cfg* c, const struct arc_json* alignment);

/* align_eo: words of `text` over encoder states enc [T, 320] of n_samples of audio that start at t0 seconds. Returns
 * the count (0: nothing to align) and *out (arc_align_free it), or -1. */
int arc_align_words(arc_moon* m, const arc_tok* tok, const arc_align_cfg* c, const float* enc, int T, int n_samples,
                    const char* text, double t0, arc_aword** out);
void arc_align_free(arc_aword* w, int n);

/* Python's round(x, nd) (correctly rounded, half to even on the exact binary value). */
double arc_pyround(double x, int nd);

#endif
