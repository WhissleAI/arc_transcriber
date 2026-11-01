/* Streams (arc_stream_*, see arc.h): audio pushed as it arrives through the streaming encoder (arc_moon_stream), the
 * streaming read-out (arc_live) and Arc's utterance segmenter; finals decoded as arc_transcribe decodes them,
 * partials read greedily from the streamed encoder states, or an external recogniser's words grouped into Arc's
 * utterances (arc_audio Stream with transcript="external": words whose midpoint lies within 0.3 s of the utterance,
 * once the recogniser's final words reach 0.3 s past its end or word_wait_s has passed).
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arc_engine.h"

enum { FR = 1600, ENC_HOP = 320 };   /* samples per 100 ms frame, per encoder frame */

typedef struct {
  char* w;
  float start, end;
  int used;
} xword;

typedef struct {
  int id;
  int f0, f1;          /* 100 ms frames, inclusive */
  int text_done;       /* FINAL sent */
  int untimed;         /* EXTERNAL: final text without times was given to it */
  int turn_end;        /* closed by arc_stream_end_turn */
  arc_result* r;       /* FINAL record (kept for the TAGS one) */
} putt;

struct arc_stream {
  arc_engine* e;
  arc_stream_config c;
  char domain[64];
  int need_enc;
  arc_moon_stream* ms;
  arc_live* live;
  arc_seg* seg;
  /* audio kept for finals: samples [a_base, a_base + a_n) */
  float* a;
  long a_base, a_n, a_cap;
  long total;          /* samples pushed */
  long nf;             /* 100 ms frames fed to the segmenter */
  /* streamed encoder states kept for partials: frames [h_base, h_base + h_n) */
  float* h;
  long h_base, h_n, h_cap;
  /* utterances closed by the segmenter and not done yet */
  putt* pu;
  int n_pu, cap_pu;
  int next_id, open_id, open_untimed;
  long last_partial;   /* samples at the last partial */
  /* EXTERNAL words */
  xword* fw;
  int n_fw, cap_fw;
  xword* iw;
  int n_iw;
  float covered;       /* the recogniser's final words reach this far */
  long pauses;         /* the read-out's pauses seen */
  int finished, failed;
};

/* ------------------------------------------------------------------ helpers */

static int grow(void** p, int* cap, int n, size_t sz) {
  if (n < *cap) return 0;
  int nc = *cap ? *cap * 2 : 16;
  void* q = arc_malloc(sz * (size_t)nc);
  if (!q) return -1;
  if (n) memcpy(q, *p, sz * (size_t)n);
  arc_free(*p);
  *p = q;
  *cap = nc;
  return 0;
}

static int lgrow(float** p, long* cap, long n, long add, int w) {
  if (n + add <= *cap) return 0;
  long nc = *cap ? *cap : 4096;
  while (nc < n + add) nc *= 2;
  float* q = arc_malloc(sizeof(float) * (size_t)nc * w);
  if (!q) return -1;
  if (n) memcpy(q, *p, sizeof(float) * (size_t)n * w);
  arc_free(*p);
  *p = q;
  *cap = nc;
  return 0;
}

static void emit(arc_stream* s, arc_out_kind k, int id, const arc_utt* u, int tags_final, const arc_event* ev,
                 const arc_frame* fr) {
  if (!s->c.on_output) return;
  arc_stream_out o = {k, id, u, tags_final, ev, fr, (double)s->total / ARC_SAMPLE_RATE};
  s->c.on_output(&o, s->c.user);
}

static void live_cb(void* user, const arc_frame* fr, const arc_event* ev) {
  arc_stream* s = user;
  if (fr && s->c.frames) emit(s, ARC_OUT_FRAME, -1, NULL, 0, NULL, fr);
  if (ev && s->c.events) emit(s, ARC_OUT_EVENT, -1, NULL, 0, ev, NULL);
}

/* tags (and the directed head) of [f0, f1] into u; returns settled */
static int fill_tags(arc_stream* s, int f0, int f1, arc_utt_build* u) {
  int settled = 1;
  u->dd = u->cc = u->dir_head = -1;
  u->intent_p = u->act_p = -1;
  if (!s->live) return 1;
  arc_live_tags(s->live, f0 / 10.0f, (f1 + 1) / 10.0f, u, &settled);
  float d = arc_live_directed(s->live, f0, f1 + 1);
  u->dir_head = d >= 0 ? d : -1;
  return settled;
}

static int close_utt(arc_stream* s, int f0, int f1, int turn_end) {
  if (grow((void**)&s->pu, &s->cap_pu, s->n_pu, sizeof(putt))) return -1;
  putt* p = &s->pu[s->n_pu++];
  memset(p, 0, sizeof *p);
  p->id = s->open_id >= 0 ? s->open_id : s->next_id++;
  p->f0 = f0;
  p->f1 = f1;
  p->turn_end = turn_end;
  p->untimed = s->open_untimed;
  s->open_id = -1;
  s->open_untimed = 0;
  if (s->c.on_output) {
    arc_utt u;
    memset(&u, 0, sizeof u);
    u.start = f0 / 10.0f;
    u.end = (f1 + 1) / 10.0f;
    u.text = "";
    u.device_directed = u.command_complete = u.directed_head = -1;
    u.intent.p = u.act.p = -1;
    int id = p->id;   /* the callback may push words (and so grow s->pu) */
    emit(s, ARC_OUT_ENDPOINT, id, &u, 0, NULL, NULL);
  }
  return 0;
}

/* ------------------------------------------------------------------ EXTERNAL words */

static int add_word(arc_stream* s, const char* w, float a, float b) {
  if (grow((void**)&s->fw, &s->cap_fw, s->n_fw, sizeof(xword))) return -1;
  /* sorted by start (a recogniser can revise earlier words: keep them in time order) */
  int i = s->n_fw;
  while (i > 0 && s->fw[i - 1].start > a) i--;
  memmove(s->fw + i + 1, s->fw + i, sizeof(xword) * (size_t)(s->n_fw - i));
  s->fw[i] = (xword){arc_strdup(w), a, b, 0};
  s->n_fw++;
  return 0;
}

static void clear_interim(arc_stream* s) {
  for (int i = 0; i < s->n_iw; ++i) arc_free(s->iw[i].w);
  arc_free(s->iw);
  s->iw = NULL;
  s->n_iw = 0;
}

/* words not yet in an utterance whose midpoint lies within [a - 0.3, b + 0.3] */
static int in_span(const xword* w, float a, float b) {
  float m = (w->start + w->end) / 2;
  return m >= a - 0.3f && m <= b + 0.3f;
}

/* the text of words (finals, then interim ones after them) from time a on, for a partial */
static char* join_from(arc_stream* s, float a, int with_interim) {
  size_t len = 1;
  for (int i = 0; i < s->n_fw; ++i)
    if (!s->fw[i].used && (s->fw[i].start + s->fw[i].end) / 2 >= a - 0.3f) len += strlen(s->fw[i].w) + 1;
  if (with_interim)
    for (int i = 0; i < s->n_iw; ++i) len += strlen(s->iw[i].w) + 1;
  char* t = arc_malloc(len);
  if (!t) return NULL;
  size_t k = 0;
  t[0] = 0;
  for (int pass = 0; pass < 2; ++pass) {
    int n = pass ? (with_interim ? s->n_iw : 0) : s->n_fw;
    for (int i = 0; i < n; ++i) {
      const xword* w = pass ? &s->iw[i] : &s->fw[i];
      if (!pass && (w->used || (w->start + w->end) / 2 < a - 0.3f)) continue;
      size_t l = strlen(w->w);
      if (k) t[k++] = ' ';
      memcpy(t + k, w->w, l);
      k += l;
      t[k] = 0;
    }
  }
  return t;
}

static void send_partial_text(arc_stream* s, char* text, int f0) {
  if (!text || !text[0]) {
    arc_free(text);
    return;
  }
  if (s->open_id < 0) s->open_id = s->next_id++;
  arc_result_build* R = arc_result_new(1);
  if (!R) {
    arc_free(text);
    return;
  }
  arc_utt_build* u = &R->u[0];
  u->f0 = f0;
  u->f1 = (int)(s->total / FR);
  u->start = f0 / 10.0f;
  u->end = (float)s->total / ARC_SAMPLE_RATE;
  u->text = text;
  u->n_words = arc_split_words(text, &u->words);
  fill_tags(s, u->f0, u->f1, u);
  arc_result* r = arc_result_finish(R, 0, 0);
  if (r) emit(s, ARC_OUT_PARTIAL, s->open_id, &r->utts[0], 0, NULL, NULL);
  arc_result_free(r);
}

static void external_partial(arc_stream* s) {
  int f0 = arc_seg_open(s->seg);
  float a = f0 >= 0 ? f0 / 10.0f : (s->n_pu ? s->pu[s->n_pu - 1].f0 / 10.0f : 0.0f);
  if (f0 < 0) f0 = s->n_pu ? s->pu[s->n_pu - 1].f0 : (int)(s->total / FR);
  send_partial_text(s, join_from(s, a, 1), f0);
}

int arc_stream_words(arc_stream* s, const arc_ext_word* w, int n, int final) {
  if (!s || s->c.mode != ARC_STREAM_EXTERNAL) {
    arc_seterr("arc_stream_words: not an ARC_STREAM_EXTERNAL stream");
    return -1;
  }
  if (!final) {
    clear_interim(s);
    if (n > 0 && !(s->iw = arc_malloc(sizeof(xword) * (size_t)n))) return -1;
    for (int i = 0; i < n; ++i)
      if (w[i].text && w[i].text[0]) s->iw[s->n_iw++] = (xword){arc_strdup(w[i].text), w[i].start, w[i].end, 0};
    external_partial(s);
    return 0;
  }
  /* untimed words: spread over the utterance open now, or the last one closed and not done */
  int untimed = 0;
  for (int i = 0; i < n; ++i) untimed |= w[i].start < 0;
  float a = 0, b = 0;
  if (untimed) {
    int f0 = arc_seg_open(s->seg);
    putt* last = NULL;
    for (int i = s->n_pu - 1; i >= 0 && !last; --i)
      if (!s->pu[i].text_done) last = &s->pu[i];
    if (f0 >= 0) {
      a = f0 / 10.0f;
      b = (float)s->total / ARC_SAMPLE_RATE;
      s->open_untimed = 1;
    } else if (last) {
      a = last->f0 / 10.0f;
      b = (last->f1 + 1) / 10.0f;
      last->untimed = 1;
    } else {
      b = (float)s->total / ARC_SAMPLE_RATE;
      a = b - 0.4f * n > 0 ? b - 0.4f * n : 0;
    }
  }
  for (int i = 0; i < n; ++i) {
    if (!w[i].text || !w[i].text[0]) continue;
    float ws = w[i].start, we = w[i].end;
    if (ws < 0) {
      ws = a + (b - a) * i / n;
      we = a + (b - a) * (i + 1) / n;
    }
    if (we < ws + 0.04f) we = ws + 0.04f;
    if (add_word(s, w[i].text, ws, we)) return -1;
    if (!untimed && we > s->covered) s->covered = we;
  }
  clear_interim(s);
  external_partial(s);
  return 0;
}

int arc_stream_text(arc_stream* s, const char* text, int final) {
  char** w;
  int n = arc_split_words(text ? text : "", &w);
  arc_ext_word* x = arc_malloc(sizeof(arc_ext_word) * (size_t)(n + 1));
  if (!x) {
    arc_free_words(w, n);
    return -1;
  }
  for (int i = 0; i < n; ++i) x[i] = (arc_ext_word){w[i], -1, -1};
  int r = arc_stream_words(s, x, n, final);
  arc_free(x);
  arc_free_words(w, n);
  return r;
}

/* ------------------------------------------------------------------ finals */

/* one utterance's FINAL, when it can be made; 1 when sent */
static int try_final(arc_stream* s, putt* p) {
  arc_engine* e = s->e;
  long a = (long)p->f0 * FR, b = (long)(p->f1 + 1) * FR;
  long pad = (long)(e->pad_s * ARC_SAMPLE_RATE);
  int end_known = s->finished || p->turn_end;
  if (s->live && arc_live_directed(s->live, p->f0, p->f1 + 1) == -2 && !s->finished) return 0;   /* features lag */
  arc_result_build* R = NULL;
  if (s->c.mode == ARC_STREAM_FULL) {
    if (s->total < b + pad && !end_known) return 0;   /* the decode reads 0.15 s past the end */
    long lo = a - pad > 0 ? a - pad : 0, hi = b + pad < s->total ? b + pad : s->total;
    if (lo < s->a_base) lo = s->a_base;
    if (!(R = arc_result_new(1))) return -1;
    arc_utt_build* u = &R->u[0];
    u->f0 = p->f0;
    u->f1 = p->f1;
    u->start = p->f0 / 10.0f;
    u->end = (p->f1 + 1) / 10.0f;
    int beams = s->c.beams > 0 ? s->c.beams : e->beams;
    if (arc_decode_segment(e, s->a + (lo - s->a_base), (int)(hi - lo), p->f0 / 10.0, beams, s->c.bias_names, 0,
                           lo / (double)ARC_SAMPLE_RATE, u)) {
      arc_result_free(arc_result_finish(R, 0, 0));
      return -1;
    }
    arc_hist_add(e, p->f0 / 10.0, (p->f1 + 1) / 10.0, u->text);
  } else if (s->c.mode == ARC_STREAM_EXTERNAL) {
    float st = p->f0 / 10.0f, en = (p->f1 + 1) / 10.0f;
    float wait = s->c.word_wait_s > 0 ? s->c.word_wait_s : 6.0f;
    float now = (float)s->total / ARC_SAMPLE_RATE;
    if (!s->finished && !p->untimed && s->covered < en + 0.3f && now < en + wait) return 0;
    if (!(R = arc_result_new(1))) return -1;
    arc_utt_build* u = &R->u[0];
    u->f0 = p->f0;
    u->f1 = p->f1;
    int nw = 0;
    for (int i = 0; i < s->n_fw; ++i) nw += !s->fw[i].used && in_span(&s->fw[i], st, en);
    u->aw = arc_calloc(sizeof(arc_wordb) * (size_t)(nw + 1));
    size_t len = 1;
    for (int i = 0; i < s->n_fw; ++i)
      if (!s->fw[i].used && in_span(&s->fw[i], st, en)) {
        xword* w = &s->fw[i];
        w->used = 1;
        u->aw[u->n_aw++] = (arc_wordb){arc_strdup(w->w), w->start, w->end, -1, 0};
        len += strlen(w->w) + 1;
      }
    char* t = arc_malloc(len);
    t[0] = 0;
    for (int i = 0; i < u->n_aw; ++i) {
      if (i) strcat(t, " ");
      strcat(t, u->aw[i].w);
    }
    u->text = t;
    u->n_words = arc_split_words(t, &u->words);
    /* the words' times, as arc_audio's group_words (the span when there are none) */
    u->start = u->n_aw ? (float)u->aw[0].start : st;
    u->end = u->n_aw ? (float)u->aw[u->n_aw - 1].end : en;
  } else {
    if (!(R = arc_result_new(1))) return -1;
    arc_utt_build* u = &R->u[0];
    u->f0 = p->f0;
    u->f1 = p->f1;
    u->start = p->f0 / 10.0f;
    u->end = (p->f1 + 1) / 10.0f;
  }
  int settled = fill_tags(s, p->f0, p->f1, &R->u[0]);
  p->r = arc_result_finish(R, 0, 0);
  if (!p->r) return -1;
  p->text_done = 1;
  emit(s, ARC_OUT_FINAL, p->id, &p->r->utts[0], settled, NULL, NULL);
  if (settled) {
    arc_result_free(p->r);
    p->r = NULL;
  }
  return 1;
}

/* the TAGS record of an utterance whose FINAL went out with unsettled tags; 1 when sent */
static int try_tags(arc_stream* s, putt* p) {
  arc_utt_build u;
  memset(&u, 0, sizeof u);
  int settled = fill_tags(s, p->f0, p->f1, &u);
  if (!settled && !s->finished) return 0;
  arc_utt* o = (arc_utt*)&p->r->utts[0];   /* the result's own copy: update its read-out in place */
  o->device_directed = u.dd;
  o->command_complete = u.cc;
  o->intent = (arc_label){u.intent, u.intent_p};
  o->act = (arc_label){u.act, u.act_p};
  o->directed_head = u.dir_head;
  emit(s, ARC_OUT_TAGS, p->id, o, 1, NULL, NULL);
  arc_result_free(p->r);
  p->r = NULL;
  return 1;
}

static int pump(arc_stream* s) {
  int i = 0;
  while (i < s->n_pu) {
    putt* p = &s->pu[i];
    if (!p->text_done) {
      int r = try_final(s, p);
      if (r < 0) return -1;
      if (!r) break;   /* in order: a later utterance waits for this one */
    }
    if (p->r && !try_tags(s, p)) {
      ++i;
      continue;
    }
    memmove(s->pu + i, s->pu + i + 1, sizeof(putt) * (size_t)(s->n_pu - i - 1));
    s->n_pu--;
  }
  /* audio still needed: from the first utterance not decoded (or the open one) less the pad; else the last 2 s */
  long keep = s->total - 2 * ARC_SAMPLE_RATE;
  long pad = (long)(s->e->pad_s * ARC_SAMPLE_RATE);
  for (int k = 0; k < s->n_pu; ++k)
    if (!s->pu[k].text_done && (long)s->pu[k].f0 * FR - pad < keep) keep = (long)s->pu[k].f0 * FR - pad;
  int f0 = arc_seg_open(s->seg);
  if (f0 >= 0 && (long)f0 * FR - pad < keep) keep = (long)f0 * FR - pad;
  if (keep > s->a_base && s->c.mode == ARC_STREAM_FULL) {
    long k = keep - s->a_base;
    if (k > s->a_n) k = s->a_n;
    memmove(s->a, s->a + k, sizeof(float) * (size_t)(s->a_n - k));
    s->a_n -= k;
    s->a_base += k;
  }
  /* external words used and older than a minute */
  if (s->c.mode == ARC_STREAM_EXTERNAL) {
    float old = (float)s->total / ARC_SAMPLE_RATE - 60.0f;
    int j = 0;
    for (int k = 0; k < s->n_fw; ++k) {
      if (s->fw[k].used && s->fw[k].end < old) arc_free(s->fw[k].w);
      else s->fw[j++] = s->fw[k];
    }
    s->n_fw = j;
  }
  return 0;
}

/* ------------------------------------------------------------------ partials (FULL) */

static int full_partial(arc_stream* s) {
  int f0 = arc_seg_open(s->seg);
  if (f0 < 0 || s->c.partial_ms < 0 || !s->h_n) return 0;
  long every = (long)(s->c.partial_ms > 0 ? s->c.partial_ms : 400) * ARC_SAMPLE_RATE / 1000;
  if (s->total - s->last_partial < every) return 0;
  s->last_partial = s->total;
  arc_engine* e = s->e;
  long pad = (long)(e->pad_s * ARC_SAMPLE_RATE);
  float win = s->c.partial_window_s > 0 ? s->c.partial_window_s : 7.0f;
  long lo = (long)f0 * FR - pad, w0 = s->total - (long)(win * ARC_SAMPLE_RATE);
  if (lo < w0) lo = w0;
  if (lo < 0) lo = 0;
  long g0 = lo / ENC_HOP, g1 = s->h_base + s->h_n;
  if (g0 < s->h_base) g0 = s->h_base;
  if (g1 - g0 < 5) return 0;   /* under 0.1 s of states */
  arc_utt_build u;
  memset(&u, 0, sizeof u);
  if (arc_decode_states(e, s->h + (size_t)(g0 - s->h_base) * ARC_MOON_D, (int)(g1 - g0),
                        (int)((g1 - g0) * ENC_HOP), f0 / 10.0, s->c.bias_names, &u))
    return -1;
  if (!u.text || !u.text[0]) {
    arc_free(u.text);
    arc_free(u.spans);
    arc_free_words(u.words, u.n_words);
    return 0;
  }
  if (s->open_id < 0) s->open_id = s->next_id++;
  arc_result_build* R = arc_result_new(1);
  if (!R) return -1;
  u.f0 = f0;
  u.f1 = (int)(s->total / FR);
  u.start = f0 / 10.0f;
  u.end = (float)s->total / ARC_SAMPLE_RATE;
  R->u[0] = u;
  fill_tags(s, u.f0, u.f1, &R->u[0]);
  arc_result* r = arc_result_finish(R, 0, 0);
  if (r) emit(s, ARC_OUT_PARTIAL, s->open_id, &r->utts[0], 0, NULL, NULL);
  arc_result_free(r);
  return 0;
}

/* ------------------------------------------------------------------ provisional tags */

/* At the read-out's first frame without speech after an utterance, its tags are what the utterance ends with unless
 * speech resumes: a TAGS record (tags_final = 0) for the utterance it belongs to (open, or the last not settled). */
static void provisional_tags(arc_stream* s) {
  if (!s->live || arc_live_pauses(s->live) == s->pauses) return;
  s->pauses = arc_live_pauses(s->live);
  long f0, f1;
  if (!arc_live_open(s->live, &f0, &f1)) return;
  int id = -1;
  for (int i = s->n_pu - 1; i >= 0 && id < 0; --i)
    if ((s->pu[i].f1 + 1) > f0 && s->pu[i].f0 <= f1 + 1) id = s->pu[i].id;   /* a closed utterance it overlaps */
  if (id < 0 && arc_seg_open(s->seg) >= 0) {
    if (s->open_id < 0) s->open_id = s->next_id++;
    id = s->open_id;
  }
  if (id < 0) return;
  arc_result_build* R = arc_result_new(1);
  if (!R) return;
  arc_utt_build* u = &R->u[0];
  u->f0 = (int)f0;
  u->f1 = (int)f1;
  u->start = f0 / 10.0f;
  u->end = (f1 + 1) / 10.0f;
  fill_tags(s, (int)f0, (int)f1, u);
  arc_result* r = arc_result_finish(R, 0, 0);
  if (r) emit(s, ARC_OUT_TAGS, id, &r->utts[0], 0, NULL, NULL);
  arc_result_free(r);
}

/* ------------------------------------------------------------------ the stream */

arc_stream* arc_stream_open(arc_engine* e, const arc_stream_config* c) {
  if (!e || !c) {
    arc_seterr("arc_stream_open: bad arguments");
    return NULL;
  }
  if (c->mode == ARC_STREAM_FULL && !arc_has_decoder(e)) {
    arc_seterr("arc_stream_open: ARC_STREAM_FULL needs the decoder (the engine was created with ARC_NO_DECODER)");
    return NULL;
  }
  if (c->mode != ARC_STREAM_FULL && c->mode != ARC_STREAM_HEADS && c->mode != ARC_STREAM_EXTERNAL) {
    arc_seterr("arc_stream_open: unknown mode %d", (int)c->mode);
    return NULL;
  }
  arc_stream* s = arc_calloc(sizeof *s);
  if (!s) return NULL;
  s->e = e;
  s->c = *c;
  snprintf(s->domain, sizeof s->domain, "%s", c->domain ? c->domain : "unknown");
  s->c.domain = s->domain;
  s->open_id = -1;
  s->seg = arc_seg_new(&e->seg);
  int delay = c->delay_ms == ARC_DELAY_NONE ? 0 : c->delay_ms > 0 ? (c->delay_ms + 50) / 100 : 3;
  if (e->trunk) s->live = arc_live_new(e, s->domain, 1, c->frames ? -1 : c->events, delay, live_cb, s);
  s->need_enc = s->live || (c->mode == ARC_STREAM_FULL && c->partial_ms >= 0);
  if (s->need_enc) s->ms = arc_moon_stream_new(e->moon, s->live && (c->events || c->frames) ? arc_live_tap(s->live) : -1);
  if (!s->seg || (e->trunk && !s->live) || (s->need_enc && !s->ms)) {
    arc_stream_close(s);
    return NULL;
  }
  return s;
}

void arc_stream_close(arc_stream* s) {
  if (!s) return;
  arc_moon_stream_free(s->ms);
  arc_live_free(s->live);
  arc_seg_free(s->seg);
  arc_free(s->a);
  arc_free(s->h);
  for (int i = 0; i < s->n_pu; ++i) arc_result_free(s->pu[i].r);
  arc_free(s->pu);
  for (int i = 0; i < s->n_fw; ++i) arc_free(s->fw[i].w);
  arc_free(s->fw);
  clear_interim(s);
  arc_free(s);
}

/* encoder frames to the read-out and the partials' window */
static int feed_encoder(arc_stream* s, const float* pcm, size_t n, int final) {
  if (!s->ms) return 0;
  if (arc_moon_stream_push(s->ms, pcm, n, final)) return -1;
  float *H = NULL, *tap = NULL;
  long ntap = 0;
  long k = arc_moon_stream_take(s->ms, &H, &tap, &ntap);
  int rc = 0;
  if (s->live && arc_live_push(s->live, H, k, tap, ntap, final)) rc = -1;
  if (!rc && k && s->c.mode == ARC_STREAM_FULL && s->c.partial_ms >= 0) {
    /* keep the last partial_window_s + 0.5 s of states */
    float win = s->c.partial_window_s > 0 ? s->c.partial_window_s : 7.0f;
    long keep = (long)((win + 0.5f) * ARC_SAMPLE_RATE / ENC_HOP);
    if (lgrow(&s->h, &s->h_cap, s->h_n, k, ARC_MOON_D)) rc = -1;
    else {
      memcpy(s->h + (size_t)s->h_n * ARC_MOON_D, H, sizeof(float) * (size_t)k * ARC_MOON_D);
      s->h_n += k;
      if (s->h_n > keep) {
        long d = s->h_n - keep;
        memmove(s->h, s->h + (size_t)d * ARC_MOON_D, sizeof(float) * (size_t)keep * ARC_MOON_D);
        s->h_n = keep;
        s->h_base += d;
      }
    }
  }
  arc_free(H);
  arc_free(tap);
  return rc;
}

int arc_stream_push(arc_stream* s, const float* pcm, size_t n) {
  if (!s || s->finished) {
    arc_seterr("arc_stream_push: the stream has finished");
    return -1;
  }
  if (s->failed) return -1;
  if (s->c.mode == ARC_STREAM_FULL) {
    if (lgrow(&s->a, &s->a_cap, s->a_n, (long)n, 1)) return -1;
    memcpy(s->a + s->a_n, pcm, sizeof(float) * n);
    s->a_n += (long)n;
  } else {
    /* the segmenter reads 100 ms frames: keep the samples of the frame not complete yet */
    long keep_from = s->nf * FR;
    if (keep_from > s->a_base) {
      long k = keep_from - s->a_base < s->a_n ? keep_from - s->a_base : s->a_n;
      memmove(s->a, s->a + k, sizeof(float) * (size_t)(s->a_n - k));
      s->a_n -= k;
      s->a_base += k;
    }
    if (lgrow(&s->a, &s->a_cap, s->a_n, (long)n, 1)) return -1;
    memcpy(s->a + s->a_n, pcm, sizeof(float) * n);
    s->a_n += (long)n;
  }
  s->total += (long)n;
  if (feed_encoder(s, pcm, n, 0)) goto fail;
  provisional_tags(s);
  /* the segmenter, frame by frame (frame levels as arc_transcribe computes them) */
  while ((s->nf + 1) * FR <= s->total) {
    int f0, f1;
    const char* why;
    const float* fr = s->a + (s->nf * FR - s->a_base);
    if (arc_seg_feed(s->seg, (int)s->nf, arc_frame_db(fr, FR), &f0, &f1, &why) && close_utt(s, f0, f1, 0)) goto fail;
    s->nf++;
  }
  if (s->c.mode == ARC_STREAM_FULL && full_partial(s)) goto fail;
  if (pump(s)) goto fail;
  return 0;
fail:
  s->failed = 1;
  return -1;
}

int arc_stream_end_turn(arc_stream* s) {
  if (!s || s->finished || s->failed) return -1;
  int f0, f1;
  const char* why;
  if (arc_seg_flush(s->seg, &f0, &f1, &why) && close_utt(s, f0, f1, 1)) return -1;
  s->open_id = -1;
  return pump(s);
}

int arc_stream_finish(arc_stream* s) {
  if (!s || s->finished) return s ? 0 : -1;
  if (s->failed) return -1;
  /* the last, partial 100 ms frame */
  if (s->nf * FR < s->total) {
    int f0, f1;
    const char* why;
    const float* fr = s->a + (s->nf * FR - s->a_base);
    if (arc_seg_feed(s->seg, (int)s->nf, arc_frame_db(fr, (int)(s->total - s->nf * FR)), &f0, &f1, &why) &&
        close_utt(s, f0, f1, 0))
      return -1;
    s->nf++;
  }
  int f0, f1;
  const char* why;
  if (arc_seg_flush(s->seg, &f0, &f1, &why) && close_utt(s, f0, f1, 0)) return -1;
  s->finished = 1;
  if (feed_encoder(s, NULL, 0, 1)) return -1;
  if (pump(s)) return -1;
  /* EXTERNAL: final words that fell in no utterance, grouped at gaps over 0.8 s (arc_audio group_words) */
  if (s->c.mode == ARC_STREAM_EXTERNAL) {
    int i = 0;
    while (i < s->n_fw) {
      if (s->fw[i].used) {
        ++i;
        continue;
      }
      int j = i + 1;
      while (j < s->n_fw && !s->fw[j].used && s->fw[j].start - s->fw[j - 1].end <= 0.8f) ++j;
      int f0w = (int)(s->fw[i].start * 10), f1w = (int)(s->fw[j - 1].end * 10) - 1;
      if (f1w < f0w) f1w = f0w;
      if (close_utt(s, f0w, f1w, 0) || pump(s)) return -1;
      i = j;
    }
  }
  return 0;
}
