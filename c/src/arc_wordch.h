/* Arc's word channel (arc_audio/wordch.py): the transcript's words, embedded from their pieces and the trunk states
 * under them, run through a small causal word encoder, and attended to from every 100 ms frame once each word has
 * "arrived" (offline: the rule arrival, the end of its utterance plus 1.5 s), added to the trunk output. Offline it
 * changes the tag head's inputs a little (the names pointer reads the fused states). Part of arc_transcriber
 * (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#ifndef ARC_WORDCH_H
#define ARC_WORDCH_H

#include "arc_base.h"
#include "arc_store.h"
#include "arc_tok.h"

typedef struct arc_wordch arc_wordch;

typedef struct {
  const char* w;   /* the word as transcribed ("Brandt.") */
  int f0, f1;      /* 100 ms frames (words.frames_of) */
  int src;         /* config word_channel.src_by_source of its source (moonshine: 1) */
} arc_wc_word;

/* NULL (no error) when the store has no word channel. */
arc_wordch* arc_wordch_load(arc_store* st);
void arc_wordch_free(arc_wordch* w);

/* Engine._fuse over a whole clip as Arc.predict feeds it (chunks of `chunk` frames, word_arrival 'rule'): words sorted
 * by time, Ha [N, 512] the trunk output, d the read-out delay in frames. Writes Hf [N, 512] (may be Ha: in place).
 * 0 / -1. */
int arc_wordch_fuse(arc_wordch* w, arc_pool* pool, const arc_tok* tok, const arc_wc_word* words, int nw, const float* Ha,
                    int N, int d, int chunk, float* Hf);

#endif
