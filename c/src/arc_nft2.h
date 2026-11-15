/* Arc Transcriber Space (NFT2) internals shared with the tools: the text side of the NFT2 finals (exposed for unit
 * tests) and a stage dump hook for parity checks. Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#ifndef ARC_NFT2_H
#define ARC_NFT2_H

#include "arc.h"

/* nft2.Pieces.norm: spoken-style lower case ("Apollo 12" -> "apollo twelve"); malloc'd (arc_free) */
char* arc_nft2_norm(const char* text);
/* nft2.num_words(n) for a non-negative decimal digit string; malloc'd */
char* arc_nft2_num_words(const char* digits);
/* nft2._wkey(w) as UTF-8; malloc'd */
char* arc_nft2_wkey(const char* w);

/* Stage dump for parity tools: called with each stage's arrays as they are made. name: "mel", "H", "Htap", "M"
 * (idx = the encoder pass / utterance), "X", "Xt" (pooled features) as floats [rows, cols]; "rnnt_ids",
 * "rnnt_frames", "prompt", "dec_ids" (idx = utterance) as ints [rows]. */
typedef void (*arc_nft2_dump_fn)(void* user, const char* name, int idx, const float* f, const int* ints, int rows,
                                 int cols);
void arc_nft2_set_dump(arc_nft2* m, arc_nft2_dump_fn fn, void* user);

#endif
