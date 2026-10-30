/* The text side of Arc's transcript (arc_audio/transcribe.py, tagdec.py, ctxprompt.py): utterance segmentation from
 * 100 ms frame levels, the loop fix, reading inline tags out of the decoded ids, and the decoder's context prompt.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#ifndef ARC_TEXT_H
#define ARC_TEXT_H

#include "arc_tok.h"

/* ---------------------------------------------------------------- utterance segmenter (UttSegmenter) */
typedef struct {
  float pause_s, min_s, max_s, cut_lookback_s, min_voiced_s;
} arc_seg_cfg;
void arc_seg_cfg_default(arc_seg_cfg* c);   /* 0.4, 1.0, 15.0, 3.0, 0.3 */

typedef struct arc_seg arc_seg;
arc_seg* arc_seg_new(const arc_seg_cfg* c);
void arc_seg_free(arc_seg* s);
/* Feed frame f (100 ms) with its level in dB. Returns 1 and sets *f0, *f1, *reason ("pause", "max_length") when an
 * utterance closes. */
int arc_seg_feed(arc_seg* s, int f, double db, int* f0, int* f1, const char** reason);
/* At the end: 1 and the open utterance ("end"), if any. */
int arc_seg_flush(arc_seg* s, int* f0, int* f1, const char** reason);
/* The frame level Arc uses: 20 log10(rms + 1e-6), -120 for an empty frame. */
double arc_frame_db(const float* a, int n);

/* ---------------------------------------------------------------- transcript text */
/* Collapse a word n-gram (n = 1..4) repeated more than K + 1 times to K copies. Returns malloc'd text. */
char* arc_loopfix(const char* text, int K);

/* An inline tag span over words [a, b). */
typedef struct {
  int tag;      /* tag id 32600.. (see arc_tag_name) */
  int a, b;
} arc_span;

/* Words and tag spans in decoded ids (tagdec.readout). words: malloc'd array of malloc'd strings. */
int arc_readout(const arc_tok* t, const int* ids, int n, char*** words, int* n_words, arc_span** spans, int* n_spans);
/* Spans over src words moved to dst words (difflib matching blocks on lowercased word keys). Returns the count. */
int arc_map_spans(char** src, int ns, const arc_span* spans, int nsp, char** dst, int nd, arc_span* out);
/* "ENTITY_PERSON", "NUMBER_TIME", ... for a tag id; NULL otherwise. */
const char* arc_tag_name(int id);
int arc_tag_id(const char* name);   /* -1 if unknown */
/* Split text on whitespace (Python str.split()); returns the count, words malloc'd. */
int arc_split_words(const char* text, char*** out);
void arc_free_words(char** w, int n);

/* ---------------------------------------------------------------- context prompt (ctxprompt.build_prompt) */
typedef struct {
  const char* dom;              /* domain words ("assistant"), may be NULL */
  const char* const* kb;        /* names / terms */
  const char* const* kb_type;   /* person team org place law product term id, or "" (parallel to kb; may be NULL) */
  int n_kb;
  const char* prev;             /* earlier text (already normalised, see arc_norm_prev), may be NULL */
} arc_ctx;
/* mode: "full" (kb + prev) or "prev". Returns the count of prompt ids written (<= max_tokens). */
int arc_ctx_prompt(const arc_tok* t, const arc_ctx* c, const char* mode, int max_tokens, int* out);
/* Earlier text as the prompt carries it: lower case, digits as words, only letters and inner apostrophes. */
char* arc_norm_prev(const char* text);
/* English cardinal words for an integer or decimal string ("4.30" -> "four point three"), as num2words. */
char* arc_num2words(const char* digits);

#endif
