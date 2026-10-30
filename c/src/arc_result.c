/* Results: built field by field, then packed into an arena the caller frees with arc_result_free. Part of
 * arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <stdlib.h>
#include <string.h>

#include "arc_engine.h"

/* ------------------------------------------------------------------ arena */

typedef struct chunk {
  struct chunk* next;
  size_t used, cap;
  /* data follows, 16-byte aligned */
} chunk;

struct arc_arena {
  chunk* head;
};

arc_arena* arc_arena_new(void) { return arc_calloc(sizeof(arc_arena)); }

void* arc_arena_alloc(arc_arena* a, size_t n) {
  n = (n + 15) & ~(size_t)15;
  if (!a->head || a->head->used + n > a->head->cap) {
    size_t cap = n > 16384 ? n : 16384;
    chunk* c = arc_malloc(sizeof(chunk) + 16 + cap);
    if (!c) return NULL;
    c->next = a->head;
    c->used = 0;
    c->cap = cap;
    a->head = c;
  }
  char* base = (char*)a->head + ((sizeof(chunk) + 15) & ~(size_t)15);
  void* p = base + a->head->used;
  a->head->used += n;
  memset(p, 0, n);
  return p;
}

char* arc_arena_strdup(arc_arena* a, const char* s) {
  if (!s) return NULL;
  size_t l = strlen(s);
  char* d = arc_arena_alloc(a, l + 1);
  if (d) memcpy(d, s, l + 1);
  return d;
}

void arc_arena_free(arc_arena* a) {
  if (!a) return;
  for (chunk* c = a->head; c;) {
    chunk* n = c->next;
    arc_free(c);
    c = n;
  }
  arc_free(a);
}

/* ------------------------------------------------------------------ build */

typedef struct {
  arc_result r;   /* first: arc_result_free gets this pointer */
  arc_arena* a;
} result_full;

arc_result_build* arc_result_new(int n_utts) {
  arc_result_build* b = arc_calloc(sizeof *b);
  if (!b) return NULL;
  b->n = n_utts;
  b->u = arc_calloc(sizeof(arc_utt_build) * (size_t)(n_utts + 1));
  b->arena = arc_arena_new();
  for (int i = 0; i < n_utts; ++i) {
    b->u[i].dd = b->u[i].cc = b->u[i].dir_head = -1.0f;
    b->u[i].intent_p = b->u[i].act_p = -1.0f;
  }
  return b;
}

void arc_result_add_event(arc_result_build* b, const char* label, float start, float end, float p) {
  if (b->n_ev == b->cap_ev) {
    int nc = b->cap_ev ? b->cap_ev * 2 : 16;
    arc_event* ne = arc_malloc(sizeof(arc_event) * (size_t)nc);
    if (!ne) return;
    if (b->n_ev) memcpy(ne, b->ev, sizeof(arc_event) * (size_t)b->n_ev);
    arc_free(b->ev);
    b->ev = ne;
    b->cap_ev = nc;
  }
  b->ev[b->n_ev++] = (arc_event){label, start, end, p};   /* labels are static strings */
}

static arc_word* pack_words(arc_arena* a, const arc_wordb* w, int n) {
  arc_word* o = arc_arena_alloc(a, sizeof(arc_word) * (size_t)(n + 1));
  for (int k = 0; k < n; ++k) {
    o[k].text = arc_arena_strdup(a, w[k].w);
    o[k].start = (float)w[k].start;
    o[k].end = (float)w[k].end;
    o[k].tag = w[k].tag >= 0 ? arc_tag_name(w[k].tag) : NULL;
    o[k].tag_begin = w[k].tag_begin;
  }
  return o;
}

arc_result* arc_result_finish(arc_result_build* b, float audio_s, float wall_ms) {
  if (!b) return NULL;
  arc_arena* a = b->arena;
  result_full* full = arc_arena_alloc(a, sizeof(result_full));
  full->a = a;
  arc_result* r = &full->r;
  arc_utt* utts = arc_arena_alloc(a, sizeof(arc_utt) * (size_t)(b->n + 1));
  r->n_utts = b->n;
  r->utts = utts;
  r->audio_s = audio_s;
  r->wall_ms = wall_ms;
  arc_event* evs = arc_arena_alloc(a, sizeof(arc_event) * (size_t)(b->n_ev + 1));
  if (b->n_ev) memcpy(evs, b->ev, sizeof(arc_event) * (size_t)b->n_ev);
  r->n_events = b->n_ev;
  r->events = evs;
  for (int i = 0; i < b->n; ++i) {
    arc_utt_build* u = &b->u[i];
    arc_utt* o = &utts[i];
    o->start = u->start;
    o->end = u->end;
    o->text = arc_arena_strdup(a, u->text ? u->text : "");
    arc_tag* tags = arc_arena_alloc(a, sizeof(arc_tag) * (size_t)(u->n_spans + 1));
    o->tags = tags;
    o->n_tags = u->n_spans;
    for (int s = 0; s < u->n_spans; ++s) {
      arc_tag* t = &tags[s];
      t->tag = arc_tag_name(u->spans[s].tag);
      t->word_begin = u->spans[s].a;
      t->word_end = u->spans[s].b;
      size_t len = 1;
      for (int w = u->spans[s].a; w < u->spans[s].b && w < u->n_words; ++w) len += strlen(u->words[w]) + 1;
      char* sp = arc_arena_alloc(a, len);
      t->text = sp;
      for (int w = u->spans[s].a; w < u->spans[s].b && w < u->n_words; ++w) {
        size_t wl = strlen(u->words[w]);
        if (w > u->spans[s].a) *sp++ = ' ';
        memcpy(sp, u->words[w], wl);
        sp += wl;
      }
      *sp = 0;
    }
    o->device_directed = u->dd;
    o->command_complete = u->cc;
    o->directed_head = u->dir_head;
    o->intent = (arc_label){arc_arena_strdup(a, u->intent), u->intent_p};
    o->act = (arc_label){arc_arena_strdup(a, u->act), u->act_p};
    o->n_words = u->n_aw;
    o->words = pack_words(a, u->aw, u->n_aw);
    const char** alts = arc_arena_alloc(a, sizeof(char*) * (size_t)(u->n_alts + 1));
    float* alt_sc = arc_arena_alloc(a, sizeof(float) * (size_t)(u->n_alts + 1));
    for (int k = 0; k < u->n_alts; ++k) {
      alts[k] = arc_arena_strdup(a, u->alts[k]);
      alt_sc[k] = u->alt_scores[k];
      arc_free(u->alts[k]);
    }
    o->n_alts = u->n_alts;
    o->alts = alts;
    o->alt_scores = alt_sc;
    arc_free(u->alts);
    arc_free(u->alt_scores);
    arc_free(u->text);
    arc_free(u->spans);
    arc_free_words(u->words, u->n_words);
    for (int k = 0; k < u->n_aw; ++k) arc_free(u->aw[k].w);
    arc_free(u->aw);
  }
  r->n_names = b->n_names;
  r->names = b->names;       /* built in the arena */
  r->n_numbers = b->n_numbers;
  r->numbers = b->numbers;
  arc_free(b->u);
  arc_free(b->ev);
  arc_free(b);
  return r;
}

void arc_result_free(arc_result* r) {
  if (r) arc_arena_free(((result_full*)r)->a);
}
