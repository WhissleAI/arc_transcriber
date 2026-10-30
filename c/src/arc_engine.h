/* libarc internals shared by arc_api.c (transcript) and arc_engine.c (trunk, heads, readout, results).
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#ifndef ARC_ENGINE_H
#define ARC_ENGINE_H

#include "arc.h"
#include "arc_base.h"
#include "arc_store.h"
#include "arc_moon.h"
#include "arc_text.h"
#include "arc_tok.h"
#include "arc_align.h"

typedef struct arc_trunk arc_trunk;   /* arc_trunk.c */

/* A word of the transcript with its time (align.py) and inline tag (tagdec.annotate). */
typedef struct {
  char* w;
  double start, end;
  int tag;          /* tag id 32600.. or -1 */
  int tag_begin;
} arc_wordb;

typedef struct {
  int f0, f1;
  float start, end;
  arc_wordb* aw;    /* aligned words (when asked for) */
  int n_aw;
  char* text;
  arc_span* spans;
  int n_spans;
  char** words;
  int n_words;
  float dd, cc, dir_head;   /* -1: not computed */
  const char* intent;
  float intent_p;
  const char* act;
  float act_p;
} arc_utt_build;

/* A simple arena: everything a result points at lives in one, freed together. */
typedef struct arc_arena arc_arena;
arc_arena* arc_arena_new(void);
void* arc_arena_alloc(arc_arena* a, size_t n);   /* zeroed, 16-byte aligned */
char* arc_arena_strdup(arc_arena* a, const char* s);
void arc_arena_free(arc_arena* a);

typedef struct {
  arc_utt_build* u;
  int n;
  arc_event* ev;
  int n_ev, cap_ev;
  arc_arena* arena;
  arc_name* names;       /* in the arena */
  int n_names;
  arc_number* numbers;   /* in the arena */
  int n_numbers;
} arc_result_build;

arc_result_build* arc_result_new(int n_utts);
/* Packs the build into one arc_result (frees the build). */
arc_result* arc_result_finish(arc_result_build* b, float audio_s, float wall_ms);
void arc_result_add_event(arc_result_build* b, const char* label, float start, float end, float p);

struct arc_engine {
  char* dir;
  arc_json* cfg;
  arc_moon* moon;
  arc_tok* tok;
  /* transcript settings (config.json "transcription") */
  arc_seg_cfg seg;
  double pad_s, tokens_per_s, len_mult;
  int loop_guard, beams;
  int n_off, off_id[64];
  float off_val[64];
  int n_sup, sup[200];
  arc_moon_procs procs;
  /* context */
  char** ctx_kb;
  char** ctx_ty;
  int ctx_n;
  char* ctx_dom;
  char* ctx_prev;
  int auto_prev, has_ctx;
  /* name bias (bias.py Trie over bias_forms(names)) */
  int nb_seq, *nb_off, *nb_ids;
  float* nb_w;
  int nb_maxlen;
  float nb_start, nb_cont;   /* bias strength (arc_set_bias_strength; 0 = bias.py DEFAULT 2.0 / 1.5) */
  arc_moon_bias bias;
  struct {
    double s, e;
    char* text;
  } hist[64];
  int n_hist;
  arc_align_cfg align;
  /* trunk, heads, ontology (arc_engine.c) */
  arc_trunk* trunk;
};

/* Per-stage outputs for parity checks (scripts/check_parity.py): pooled features [N, 640], tapped features
 * [N, 640], event probabilities [N, 50], and each query's per-frame probabilities over its shipped columns. */
typedef struct {
  int N;
  float *X, *XT, *ev;
  int nq;
  struct {
    const char* q;
    int ncols;
    float* p;   /* [N, ncols] */
    const char** ids;   /* the columns' tag ids (owned by the engine; the array by the stages) */
  } fr[9];
} arc_stages;
int arc_engine_stages(arc_engine* e, const float* pcm, size_t n, const char* domain, arc_stages* out);
void arc_stages_free(arc_stages* s);

/* A weight matrix [O, K] of a store: packed 4-bit used in place, else float32 (0 / -1). */
int arc_w_load(arc_store* s, const char* name, int64_t n, arc_w* w);
/* y[T, O] = x[T, K] W^T + b */
void arc_w_linear(arc_pool* p, const float* x, int T, int K, const arc_w* w, int O, const float* b, float* y);

int arc_engine_load_trunk(arc_engine* e, const char* dir);
/* The names list for linking (arc_set_names). */
int arc_engine_set_names(arc_engine* e, const arc_name_entry* names, int n);
void arc_engine_set_pool(arc_engine* e, arc_pool* p);
void arc_engine_free_trunk(arc_engine* e);
/* Features over the whole clip, the trunk, the utterance readout and events into R. */
int arc_engine_run(arc_engine* e, const float* pcm, size_t n, const arc_options* o, arc_result_build* R);

/* For arc_link.c: tag ontology and head access. */
int arc_engine_tag_index(arc_engine* e, const char* id);   /* -1 if absent */
const char* arc_engine_tag_name(arc_engine* e, int i);
const char* arc_engine_tag_group(arc_engine* e, int i);    /* its exclusive group, or NULL */
const float* arc_engine_tag_key(arc_engine* e, int i);     /* the tag head's key [512] */
/* span_in probabilities (Readout.probs) over rows H [T, 512]: [T, ncols] (arc_free it); cols: the tag indices */
float* arc_engine_span_probs(arc_engine* e, const float* H, int T, int dom, int* ncols, const int** cols);

#include "arc_link.h"
/* The WordStore from the transcript's aligned words; arc_link_run reads names and numbers into R. */
void arc_wstore_init(arc_wstore* ws, const arc_result_build* R);
void arc_wstore_free(arc_wstore* ws);
int arc_link_run(arc_link* l, const arc_link_ctx* c, const arc_wstore* ws, arc_result_build* R);

#endif
