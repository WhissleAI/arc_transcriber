/* Names linked to a list and numbers with values (arc_audio/engine.py's decoder-sourced path: _dec_pd, _read_name_dec,
 * _ptr, _policy, _read_number; model.py EntityHeads / NameEncoder; api.py _post kb_snap and tidy_out), offline, over
 * the trunk output and the transcript's aligned words. Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see
 * LICENSE. */
#ifndef ARC_LINK_H
#define ARC_LINK_H

#include "arc.h"
#include "arc_base.h"
#include "arc_store.h"

struct arc_engine;
struct arc_result_build_s;

/* A word of the WordStore (words.norm_event): text, times rounded to 1 ms, 100 ms frames, inline tag. */
typedef struct {
  char* w;
  double start, end;
  int f0, f1;
  int tag;          /* 32600.. or -1 */
  int tag_begin;
  int utt;          /* transcript utterance it came from */
} arc_wsword;

typedef struct {
  arc_wsword* w;    /* finals sorted by (start, end) */
  int n;
} arc_wstore;

typedef struct arc_link arc_link;

typedef struct {
  struct arc_engine* e;
  arc_pool* pool;
  const float* H;   /* [N, 512] trunk output with the word channel (Engine.ring_h) */
  int N, dom, d;
  int names, numbers;
} arc_link_ctx;

arc_link* arc_link_load(struct arc_engine* e, arc_store* st);   /* NULL when the weights have no entity heads */
void arc_link_free(arc_link* l);
int arc_link_set_names(arc_link* l, const arc_name_entry* names, int n);

#endif
