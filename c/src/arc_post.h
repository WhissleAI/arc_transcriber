/* Arc Transcriber Space read-outs on an utterance's words (arc_audio/quantity.py, phonetic.py, nft2_post.py): typed
 * quantities, sound-alike keys, the no-speech / tail guard, disfluency tags, sound-alike name candidates and the
 * end-of-turn heuristic. Plain C over word texts; arc_nft2.c fills arc_result from them.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#ifndef ARC_POST_H
#define ARC_POST_H

#include "arc.h"

/* ---- quantities (quantity.quantities) */
typedef struct arc_qty {
  const char* type;   /* static: "number", "percent", "distance", ... */
  const char* unit;   /* static or NULL */
  double value;
  int is_int, relative, word_begin, word_end;
  char* text;         /* malloc'd: the words */
  char* written;      /* malloc'd: "75%", "2 km" */
} arc_qty;
/* words [n] (as said), tags [n] (inline tag of each word or NULL; may be NULL) -> *out (malloc'd, arc_post_qty_free) */
int arc_post_quantities(const char* const* words, const char* const* tags, int n, arc_qty** out);
void arc_post_qty_free(arc_qty* q, int n);

/* ---- sound-alike (phonetic.key / similarity); malloc'd key */
char* arc_phon_key(const char* text);
double arc_phon_similarity(const char* a, const char* b);

/* ---- guard (nft2_post.guard): keep[i] = 1 / 0 for words with start times and confidences (conf < 0: unknown) */
typedef struct {
  double silence_db, flat_db, flat_conf, rep_conf, tail_s, tail_conf;
} arc_guard_cfg;
void arc_guard_defaults(arc_guard_cfg* c);
/* frame levels of y (100 ms frames) */
int arc_post_frame_dbs(const float* y, long n, double** out);
/* returns the number of kept words; why[i] (may be NULL): NULL kept, else "no_speech" / "repeat" / "tail" */
int arc_post_guard(const char* const* words, const double* start, const float* conf, int n, const float* y, long ny,
                   double t0, const arc_guard_cfg* c, int* keep, const char** why);

/* ---- disfluency (nft2_post.disfluency): tag[i] = NULL, "filler", "repeat", "fragment" (static strings) */
void arc_post_disfluency(const char* const* words, int n, const char** tag);

/* ---- sound-alike names (nft2_post.almost / almost_p) */
typedef struct {
  double min_sim, min_p, a, b, c, d;
  int max_words, min_chars;
  double max_conf;
} arc_almost_cfg;
void arc_almost_defaults(arc_almost_cfg* c);
typedef struct {
  const char* form;   /* a spelling (name or alias) */
  int kb;             /* its entry */
} arc_form;
double arc_almost_p(const char* heard, const char* form, const arc_almost_cfg* c);
/* best list entry for heard words: kb index (-1 none), *p */
int arc_almost_best(const char* heard, const arc_form* forms, int nf, const arc_almost_cfg* c, double* p);
typedef struct {
  int word_begin, word_end, kb;
  double p;
} arc_almost_hit;
/* words [n] of one utterance, conf[n] (word confidences, < 0 unknown; may be NULL), common[n] (1: a whole-word piece of
 * the RNNT vocabulary, nft2_post.common_words; may be NULL), taken[n] (1: already in a linked name; may be NULL) -> hits
 * (malloc'd), sorted by position as almost() */
int arc_post_almost(const char* const* words, const float* conf, const int* common, int n, const int* taken,
                    const arc_form* forms, int nf, const arc_almost_cfg* c, arc_almost_hit** out);

/* ---- end of turn (nft2_post.eot_prob) */
double arc_post_eot(const char* const* words, int n, double pause_s);

/* 1: a word key in nft2_post.STOP */
int arc_post_is_stop(const char* key);
/* lower-case word key (nft2_post._wk): letters, digits, apostrophes; malloc'd */
char* arc_post_wk(const char* w);
/* difflib.SequenceMatcher(None, a, b).ratio() over bytes */
double arc_sm_ratio_bytes(const char* a, const char* b);

#endif
