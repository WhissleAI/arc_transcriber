/* Names and numbers: a C port of the offline, decoder-sourced path of arc_audio/engine.py (WordStore spans ->
 * _dec_pd -> _read_name_dec / _read_number, with _ptr, _policy, _head_p, _number_kind), model.py (NameEncoder,
 * EntityHeads.ptr_logits / span_embed / value_logits) and api.py (_post: kbsnap link and add_spans; tidy_out with
 * extend_number). Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include "arc_link.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arc_engine.h"
#include "arc_ntext.h"
#include "arc_num.h"
#include "arc_tidy.h"

#define DM 512
#define DN 128      /* name embedding */
#define CH 48       /* name characters */
#define NAME_MAX_BYTES 40
#define K_SLOTS 4
#define N_NWT 43
#define SPAN_G_MULT 20.0f
#define NEGV (-10000.0f)
#define NKIND 10

static const char* C_KINDS[NKIND] = {"person", "team", "organization", "place", "work_product",
                                     "event_name", "legal_ref", "game_entity", "game_map", "other"};

struct arc_link {
  struct arc_engine* e;
  /* NameEncoder */
  const float *emb, *cin_w, *cin_b, *cv_w[3], *cv_b[3], *kind, *o0_w, *o0_b, *o2_w, *o2_b;
  /* EntityHeads */
  const float *q0_w, *q0_b, *q2_w, *q2_b, *key_in, *special, *special_b, *sq_w, *sq_b, *span_g_src, *sim_w, *sim_new;
  const float *nwt_w, *nwt_b, *unit_w, *unit_b;
  float log_scale, span_g, span_new;
  int has_span, has_span_new, has_sim, n_units;
  /* the list */
  arc_kbent* kb;
  char** strs;   /* every string the list owns */
  int n_strs, nkb;
  int* ckind;
  float* E;      /* [nkb, 128] */
  arc_nt_candtab* ct;
  arc_nt_exact* exn;
  arc_nt_kbsnap* snap;
  /* config */
  const arc_json *nwt_vocab, *unit_vocab;
  int ptr_src;   /* ptr.src_by_source of the builtin transcript's words (moonshine) */
};

static float gelu1(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }

/* y[O] = W[O, K] x + b */
static void mv(const float* W, const float* b, const float* x, int O, int K, float* y) {
  for (int o = 0; o < O; ++o) {
    float s = 0;
    const float* w = W + (size_t)o * K;
    for (int k = 0; k < K; ++k) s += w[k] * x[k];
    y[o] = s + (b ? b[o] : 0.0f);
  }
}

static void l2norm(float* x, int n) {
  float s = 0;
  for (int i = 0; i < n; ++i) s += x[i] * x[i];
  float d = sqrtf(s);
  if (d < 1e-12f) d = 1e-12f;
  for (int i = 0; i < n; ++i) x[i] /= d;
}

arc_link* arc_link_load(struct arc_engine* e, arc_store* s) {
  if (!arc_store_has(s, "ent.names.emb.weight")) return NULL;
  arc_link* l = arc_calloc(sizeof *l);
  if (!l) return NULL;
  l->e = e;
  int bad = 0;
#define G(f, nm, n) if (!(l->f = arc_store_f32(s, nm, n))) bad = 1
  G(emb, "ent.names.emb.weight", 257 * CH);
  G(cin_w, "ent.names.conv_in.weight", DN * CH * 3);
  G(cin_b, "ent.names.conv_in.bias", DN);
  for (int i = 0; i < 3; ++i) {
    if (!(l->cv_w[i] = arc_store_f32(s, arc_namef("ent.names.convs.%d.weight", i), DN * DN * 3))) bad = 1;
    if (!(l->cv_b[i] = arc_store_f32(s, arc_namef("ent.names.convs.%d.bias", i), DN))) bad = 1;
  }
  G(kind, "ent.names.kind.weight", NKIND * DN);
  G(o0_w, "ent.names.out.0.weight", DN * 2 * DN);
  G(o0_b, "ent.names.out.0.bias", DN);
  G(o2_w, "ent.names.out.2.weight", DN * DN);
  G(o2_b, "ent.names.out.2.bias", DN);
  G(q0_w, "ent.query.0.weight", DN * DM);
  G(q0_b, "ent.query.0.bias", DN);
  G(q2_w, "ent.query.2.weight", DN * DN);
  G(q2_b, "ent.query.2.bias", DN);
  G(key_in, "ent.key_in.weight", DM * DM);
  G(special, "ent.special", 2 * DN);
  G(special_b, "ent.special_b", 2);
  G(nwt_w, "ent.nwt.weight", K_SLOTS * N_NWT * DM);
  G(nwt_b, "ent.nwt.bias", K_SLOTS * N_NWT);
  const float* ls = arc_store_f32(s, "ent.log_scale", 1);
  if (!ls) bad = 1;
  else l->log_scale = ls[0];
  int64_t sh[8];
  if (arc_store_shape(s, "ent.unit.weight", sh) == 2) {
    l->n_units = (int)sh[0];
    G(unit_w, "ent.unit.weight", (int64_t)l->n_units * DM);
    G(unit_b, "ent.unit.bias", l->n_units);
  } else bad = 1;
  if (arc_store_has(s, "ent.span_q.weight")) {
    l->has_span = 1;
    G(sq_w, "ent.span_q.weight", DN * DN);
    G(sq_b, "ent.span_q.bias", DN);
    const float* g = arc_store_f32(s, "ent.span_g", 1);
    l->span_g = g ? g[0] : 0.0f;
    if (arc_store_has(s, "ent.span_new")) {
      l->has_span_new = 1;
      const float* sn = arc_store_f32(s, "ent.span_new", 1);
      l->span_new = sn ? sn[0] : 0.0f;
    }
    if (arc_store_has(s, "ent.span_g_src")) G(span_g_src, "ent.span_g_src", 3);
  }
  if (arc_store_has(s, "ent.sim_w")) {
    l->has_sim = 1;
    G(sim_w, "ent.sim_w", 18);
    G(sim_new, "ent.sim_new", 9);
  }
#undef G
  if (bad) {
    arc_free(l);
    return NULL;
  }
  const arc_json* nm = arc_json_get(e->cfg, "numerals");
  l->nwt_vocab = arc_json_get(nm, "nwt_vocab");
  l->unit_vocab = arc_json_get(nm, "unit_vocab");
  l->ptr_src = (int)arc_json_num(arc_json_get(arc_json_get(arc_json_get(e->cfg, "ptr"), "src_by_source"), "moonshine"), 0);
  return l;
}

static void free_kb(arc_link* l) {
  for (int i = 0; i < l->n_strs; ++i) arc_free(l->strs[i]);
  arc_free(l->strs);
  if (l->kb)
    for (int i = 0; i < l->nkb; ++i) arc_free((void*)l->kb[i].aliases);
  arc_free(l->kb);
  arc_free(l->ckind);
  arc_free(l->E);
  arc_nt_candtab_free(l->ct);
  arc_nt_exact_free(l->exn);
  arc_nt_kbsnap_free(l->snap);
  l->kb = NULL;
  l->strs = NULL;
  l->n_strs = l->nkb = 0;
  l->ckind = NULL;
  l->E = NULL;
  l->ct = NULL;
  l->exn = NULL;
  l->snap = NULL;
}

void arc_link_free(arc_link* l) {
  if (!l) return;
  free_kb(l);
  arc_free(l);
}

/* NameEncoder over one string: encode_names (lower case, spaces collapsed, UTF-8 bytes, the first or last 40) */
static void name_embed(const arc_link* l, const char* text, int keep_tail, int kind, float* out) {
  /* ' '.join(lower().split()) */
  size_t n = strlen(text);
  char* t = arc_malloc(n + 1);
  size_t k = 0;
  int sp = 0;
  for (const char* p = text; *p; ++p) {
    if (isspace((unsigned char)*p)) {
      sp = k > 0;
      continue;
    }
    if (sp) t[k++] = ' ';
    sp = 0;
    t[k++] = (char)((*p >= 'A' && *p <= 'Z') ? *p + 32 : *p);
  }
  t[k] = 0;
  int L = NAME_MAX_BYTES, ids[NAME_MAX_BYTES] = {0};
  size_t nb = k < (size_t)L ? k : (size_t)L;
  const unsigned char* src = (const unsigned char*)t + (keep_tail ? k - nb : 0);
  for (size_t i = 0; i < nb; ++i) ids[i] = src[i] + 1;
  arc_free(t);
  /* x [CH, L], h [DN, L] */
  static __thread float x[CH][NAME_MAX_BYTES], h[DN][NAME_MAX_BYTES], h2[DN][NAME_MAX_BYTES];
  float m[NAME_MAX_BYTES];
  int cnt = 0;
  for (int i = 0; i < L; ++i) {
    m[i] = ids[i] > 0 ? 1.0f : 0.0f;
    cnt += ids[i] > 0;
    for (int c = 0; c < CH; ++c) x[c][i] = l->emb[ids[i] * CH + c];
  }
  for (int o = 0; o < DN; ++o)
    for (int i = 0; i < L; ++i) {
      float s = l->cin_b[o];
      for (int c = 0; c < CH; ++c)
        for (int kk = 0; kk < 3; ++kk) {
          int j = i + kk - 1;
          if (j >= 0 && j < L) s += l->cin_w[((size_t)o * CH + c) * 3 + kk] * x[c][j];
        }
      h[o][i] = gelu1(s) * m[i];
    }
  const int dil[3] = {1, 2, 4};
  for (int v = 0; v < 3; ++v) {
    for (int o = 0; o < DN; ++o)
      for (int i = 0; i < L; ++i) {
        float s = l->cv_b[v][o];
        for (int c = 0; c < DN; ++c)
          for (int kk = 0; kk < 3; ++kk) {
            int j = i + (kk - 1) * dil[v];
            if (j >= 0 && j < L) s += l->cv_w[v][((size_t)o * DN + c) * 3 + kk] * h[c][j];
          }
        h2[o][i] = gelu1(s) * m[i];
      }
    for (int o = 0; o < DN; ++o)
      for (int i = 0; i < L; ++i) h[o][i] += h2[o][i];
  }
  float cat[2 * DN], a[DN];
  for (int o = 0; o < DN; ++o) {
    float s = 0, mx = -INFINITY;
    for (int i = 0; i < L; ++i) {
      s += h[o][i];
      float v = m[i] > 0 ? h[o][i] : -10000.0f;
      if (v > mx) mx = v;
    }
    cat[o] = s / (float)(cnt > 0 ? cnt : 1);
    cat[DN + o] = mx * (cnt > 0 ? 1.0f : 0.0f);
  }
  mv(l->o0_w, l->o0_b, cat, DN, 2 * DN, a);
  for (int o = 0; o < DN; ++o) a[o] = gelu1(a[o]);
  mv(l->o2_w, l->o2_b, a, DN, DN, out);
  for (int o = 0; o < DN; ++o) out[o] += l->kind[kind * DN + o];
  l2norm(out, DN);
}

static int kind_index(const char* k) {
  if (!k) return 9;
  for (int i = 0; i < NKIND; ++i)
    if (!strcmp(C_KINDS[i], k)) return i;
  return 9;
}

static char* own(arc_link* l, const char* s) {
  char* d = arc_strdup(s ? s : "");
  l->strs[l->n_strs++] = d;
  return d;
}

int arc_link_set_names(arc_link* l, const arc_name_entry* names, int n) {
  free_kb(l);
  if (n <= 0) return 0;
  int ns = 0;
  for (int i = 0; i < n; ++i) ns += 3 + (names[i].n_aliases > 0 ? names[i].n_aliases : 0);
  l->strs = arc_calloc(sizeof(char*) * (size_t)ns);
  l->kb = arc_calloc(sizeof(arc_kbent) * (size_t)n);
  l->ckind = arc_calloc(sizeof(int) * (size_t)n);
  l->E = arc_calloc(sizeof(float) * (size_t)n * DN);
  const char** nm = arc_malloc(sizeof(char*) * (size_t)n);
  if (!l->strs || !l->kb || !l->ckind || !l->E || !nm) {
    arc_free(nm);
    free_kb(l);
    return -1;
  }
  l->nkb = n;
  for (int i = 0; i < n; ++i) {
    arc_kbent* c = &l->kb[i];
    c->name = own(l, names[i].name);
    c->kind = own(l, names[i].kind && names[i].kind[0] ? names[i].kind : "other");
    c->group = names[i].group && names[i].group[0] ? own(l, names[i].group) : NULL;
    int na = names[i].n_aliases > 0 ? names[i].n_aliases : 0;
    const char** al = arc_calloc(sizeof(char*) * (size_t)(na + 1));
    for (int j = 0; j < na; ++j) al[j] = own(l, names[i].aliases[j]);
    c->aliases = al;
    c->n_aliases = na;
    l->ckind[i] = kind_index(c->kind);
    name_embed(l, c->name, 0, l->ckind[i], l->E + (size_t)i * DN);   /* EntityHeads.names(encode_names(names), kinds) */
    nm[i] = c->name;
  }
  if (l->has_sim) l->ct = arc_nt_candtab_new(nm, n);
  arc_free(nm);
  l->exn = arc_nt_exact_new(l->kb, n);
  l->snap = arc_nt_kbsnap_new(l->kb, n);
  return 0;
}

/* ------------------------------------------------------------------ the word store */

static double round3(double x) { return arc_pyround(x, 3); }

static void frames_of(double s, double e, int* f0, int* f1) {
  *f0 = (int)floor(s * 10 + 1e-6);
  int b = (int)floor(e * 10 + 1e-6);
  *f1 = b > *f0 ? b : *f0;
}

static int cmp_ws(const void* a, const void* b) {
  const arc_wsword *x = a, *y = b;
  if (x->start != y->start) return x->start < y->start ? -1 : 1;
  if (x->end != y->end) return x->end < y->end ? -1 : 1;
  return x->utt != y->utt ? x->utt - y->utt : (x < y ? -1 : 1);   /* stable */
}

typedef struct {
  arc_wsword w;
  int ord;
} ordw;

static int cmp_ord(const void* a, const void* b) {
  const ordw *x = a, *y = b;
  int c = cmp_ws(&x->w, &y->w);
  if (x->w.start == y->w.start && x->w.end == y->w.end) return x->ord - y->ord;
  return c;
}

/* norm_event + WordStore.add over every utterance (finals sorted by (start, end), stable) */
void arc_wstore_init(arc_wstore* ws, const arc_result_build* R) {
  memset(ws, 0, sizeof *ws);
  int n = 0;
  for (int i = 0; i < R->n; ++i) n += R->u[i].n_aw;
  if (!n) return;
  ordw* o = arc_malloc(sizeof(ordw) * (size_t)n);
  int k = 0;
  for (int i = 0; i < R->n; ++i)
    for (int j = 0; j < R->u[i].n_aw; ++j) {
      const arc_wordb* a = &R->u[i].aw[j];
      const char* p = a->w;
      while (*p && isspace((unsigned char)*p)) p++;
      size_t len = strlen(p);
      while (len && isspace((unsigned char)p[len - 1])) len--;
      if (!len) continue;
      arc_wsword* w = &o[k].w;
      w->w = arc_malloc(len + 1);
      memcpy(w->w, p, len);
      w->w[len] = 0;
      double st = a->start, en = a->end > a->start + 0.04 ? a->end : a->start + 0.04;
      frames_of(st, en, &w->f0, &w->f1);
      w->start = round3(st);
      w->end = round3(en);
      w->tag = a->tag;
      w->tag_begin = a->tag_begin;
      w->utt = i;
      o[k].ord = k;
      k++;
    }
  qsort(o, (size_t)k, sizeof(ordw), cmp_ord);
  ws->w = arc_malloc(sizeof(arc_wsword) * (size_t)(k + 1));
  for (int i = 0; i < k; ++i) ws->w[i] = o[i].w;
  ws->n = k;
  arc_free(o);
}

void arc_wstore_free(arc_wstore* ws) {
  for (int i = 0; i < ws->n; ++i) arc_free(ws->w[i].w);
  arc_free(ws->w);
  ws->w = NULL;
  ws->n = 0;
}

/* WordStore.heard_tail(f): the last few words that ended before f (with a lag), lower case */
static char* heard_tail(const arc_wstore* W, int f) {
  const int n_words = 3, lag = 2, hold = 30;
  int last = -1;
  for (int k = 0; k < W->n; ++k)
    if (W->w[k].f1 + lag <= f) last = k;
  if (last < 0) return arc_strdup("");
  if (f - (W->w[last].f1 + lag) > hold) return arc_strdup("");
  int t0 = W->w[last].f1 + lag - hold;
  /* av[-3:]: the last three of the words that ended before f, in store order */
  int idx[3], ni = 0;
  for (int k = last; k >= 0 && ni < n_words; --k)
    if (W->w[k].f1 + lag <= f) idx[ni++] = k;
  size_t len = 1;
  for (int q = 0; q < ni; ++q) len += strlen(W->w[idx[q]].w) + 1;
  char* s = arc_malloc(len);
  size_t o = 0;
  int sp = 0;
  for (int q = ni - 1; q >= 0; --q) {
    const arc_wsword* w = &W->w[idx[q]];
    if (w->f1 + lag < t0) continue;
    for (const char* p = w->w; *p; ++p) {
      if (isspace((unsigned char)*p)) {
        sp = o > 0;
        continue;
      }
      if (sp) s[o++] = ' ';
      sp = 0;
      s[o++] = (char)((*p >= 'A' && *p <= 'Z') ? *p + 32 : *p);
    }
    sp = o > 0;
  }
  s[o] = 0;
  return s;
}

/* WordStore.text_between(t0, t1) */
static char* text_between(const arc_wstore* W, double t0, double t1) {
  size_t len = 1;
  for (int k = 0; k < W->n; ++k)
    if (W->w[k].end > t0 && W->w[k].start < t1) len += strlen(W->w[k].w) + 1;
  char* s = arc_malloc(len);
  size_t o = 0;
  for (int k = 0; k < W->n; ++k)
    if (W->w[k].end > t0 && W->w[k].start < t1) {
      if (o) s[o++] = ' ';
      size_t l = strlen(W->w[k].w);
      memcpy(s + o, W->w[k].w, l);
      o += l;
    }
  s[o] = 0;
  return s;
}

/* words.own_end */
static int own_end(const arc_wstore* W, int j) {
  int a = W->w[j].f0, b = W->w[j].f1;
  if (j + 1 < W->n && a < W->w[j + 1].f0 && W->w[j + 1].f0 <= b) b = W->w[j + 1].f0 - 1;
  return a > b ? a : b;
}

/* ------------------------------------------------------------------ word classes (engine.py) */

static void lw_key(const char* t, char* out, size_t cap) {   /* _w: lower, [a-z0-9'] only */
  size_t o = 0;
  for (const char* p = t; *p && o + 1 < cap; ++p) {
    char c = (char)((*p >= 'A' && *p <= 'Z') ? *p + 32 : *p);
    if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '\'') out[o++] = c;
  }
  out[o] = 0;
}

static int in_set(const char* w, const char* const* set) {
  for (int i = 0; set[i]; ++i)
    if (!strcmp(w, set[i])) return 1;
  return 0;
}

static const char* const FILLERS[] = {"uh", "um", "erm", "er", "ah", "eh", "hmm", "mm", "uhm", "mhm", "huh", NULL};
static const char* const BACKCHANNEL[] = {"mmhmm", "mhmm", "uhhuh", "uhuh", "mmm", "hm", "ok", "okay", "yeah", "yep", "nope", NULL};
static const char* const STOP[] = {
    "the", "and", "for", "but", "with", "out", "off", "from", "that", "this", "has", "had", "have", "was", "were",
    "are", "his", "her", "its", "our", "your", "their", "they", "them", "she", "him", "who", "what", "when", "then",
    "than", "there", "here", "not", "now", "all", "any", "can", "will", "just", "into", "onto", "over", "very", "yes",
    "yeah", "okay", "right", "well", "you", "one", "two", NULL};
static const char* const DISPLAY_EXTRA[] = {
    "a", "an", "of", "to", "in", "on", "at", "by", "is", "be", "been", "it", "he", "we", "i", "me", "my", "us", "so",
    "do", "did", "does", "if", "or", "as", "up", "no", "oh", "go", "get", "got", "let", "lets", "let's", "that's",
    "it's", "he's", "she's", "i'm", "you're", "we're", "they're", "there's", "what's", "don't", "can't", "didn't",
    "isn't", "about", "after", "again", "back", "down", "good", "great", "big", "little", "more", "most", "much",
    "some", "such", "these", "those", "too", "way", "where", "which", "while", "why", "how", "would", "could",
    "should", "being", "because", "through", "game", "guys", "man", "time", "thing", "things", "people", "really",
    "think", "know", "like", "going", "gonna", "want", "look", "see", "said", "says", "say", "make", "made", "take",
    "come", "came", "put", "nice", "play", "shot", "ball", "goal", "team", "point", "points", "year", "years", "today",
    "tonight", NULL};

static int is_filler(const char* t) {
  char k[256];
  lw_key(t, k, sizeof k);
  return in_set(k, FILLERS);
}

/* _read_name_dec's show: a word that is not a stop word, a backchannel, a number or a filler */
static int show_word(const char* t) {
  char k[256];
  lw_key(t, k, sizeof k);
  if (!k[0] || in_set(k, STOP) || in_set(k, DISPLAY_EXTRA) || in_set(k, BACKCHANNEL)) return 0;
  if (arc_num_word(t) || is_filler(t)) return 0;
  return 1;
}

/* ------------------------------------------------------------------ tags (tagdec.py) */

typedef struct {
  const char* dtag;   /* ENTITY_PERSON ... */
  const char* tid;    /* entity.mention.person, or NULL for a number */
  const char* nkind;  /* NUMBER_KIND, NULL for plain NUMBER */
  int is_name;
} tag_info;

static int tag_info_of(const char* t, tag_info* o) {
  static const char* NAME_TAG[][2] = {
      {"ENTITY_PERSON", "entity.mention.person"}, {"ENTITY_TEAM", "entity.mention.team"},
      {"ENTITY_ORG", "entity.mention.organization"}, {"ENTITY_PLACE", "entity.mention.place"},
      {"ENTITY_WORK", "entity.mention.work_product"}, {"ENTITY_EVENT", "entity.mention.event_name"},
      {"ENTITY_LEGAL", "entity.mention.legal_ref"}, {"ENTITY_GAME", "entity.mention.game_entity"},
      {"ENTITY_MAP", "entity.mention.game_map"}, {"ENTITY_LAW", "entity.mention.law"},
      {"ENTITY_PRODUCT", "entity.mention.product"}, {"ENTITY_TERM", "entity.mention.term"}};
  static const char* NUMBER_KIND[][2] = {
      {"NUMBER", NULL}, {"NUMBER_CLOCK", "clock"}, {"NUMBER_COUNT", "count"}, {"NUMBER_DATE", "date"},
      {"NUMBER_DURATION", "duration"}, {"NUMBER_FIELDPOS", "field_position"}, {"NUMBER_ID", "id_pii"},
      {"NUMBER_JERSEY", "jersey_number"}, {"NUMBER_MEASURE", "measure"}, {"NUMBER_MONEY", "money"},
      {"NUMBER_ODDS", "ratio_odds"}, {"NUMBER_ORDINAL", "ordinal"}, {"NUMBER_PERCENT", "percent"},
      {"NUMBER_REF", "ref_number"}, {"NUMBER_SCORE", "score"}, {"NUMBER_STAT", "stat"}, {"NUMBER_TIME", "time_of_day"}};
  memset(o, 0, sizeof *o);
  for (size_t i = 0; i < sizeof NAME_TAG / sizeof NAME_TAG[0]; ++i)
    if (!strcmp(t, NAME_TAG[i][0])) {
      o->dtag = NAME_TAG[i][0];
      o->tid = NAME_TAG[i][1];
      o->is_name = 1;
      return 1;
    }
  for (size_t i = 0; i < sizeof NUMBER_KIND / sizeof NUMBER_KIND[0]; ++i)
    if (!strcmp(t, NUMBER_KIND[i][0])) {
      o->dtag = NUMBER_KIND[i][0];
      o->nkind = NUMBER_KIND[i][1];
      return 1;
    }
  return 0;   /* contacts and anything else: no layer for names / numbers */
}

/* a span of tagged words (tagdec.spans_of_words), as Engine._dec_pd keeps it */
typedef struct {
  tag_info ti;
  int i;               /* tag index of the mention (names) */
  int a, b;            /* frames */
  double start, end;   /* rounded to 3 */
  int* keys;           /* word indices in the store that make the span (keys (start, w) resolved) */
  int nk;
} pend;

/* ------------------------------------------------------------------ the pointer head */

/* EntityHeads.ptr_logits for one frame: h [512] (trunk with words), the mention tag's key, the list's names E, the
 * allowed mask; heard text (may be ""), its sim features Sf [R, 6] (NULL: none). z [2 + R] */
static void ptr_logits(const arc_link* l, const float* h, const float* key, const unsigned char* ok, const float* Es,
                       const float* Sf, int src, float* z) {
  int R = l->nkb;
  float hk[DM], a[DN], qr[DN], q[DN];
  mv(l->key_in, NULL, key, DM, DM, hk);
  for (int c = 0; c < DM; ++c) hk[c] += h[c];
  mv(l->q0_w, l->q0_b, hk, DN, DM, a);
  for (int c = 0; c < DN; ++c) a[c] = gelu1(a[c]);
  mv(l->q2_w, l->q2_b, a, DN, DN, qr);
  int use_span = Es && l->has_span;
  if (use_span) {
    float sq[DN];
    mv(l->sq_w, l->sq_b, Es, DN, DN, sq);
    for (int c = 0; c < DN; ++c) qr[c] += sq[c];
  }
  memcpy(q, qr, sizeof q);
  l2norm(q, DN);
  float ls = l->log_scale < logf(50.0f) ? l->log_scale : logf(50.0f);
  float sc = expf(ls);
  float g = use_span ? l->span_g + (l->span_g_src ? l->span_g_src[src] : 0.0f) : 0.0f;
  int use_sim = Sf && l->has_sim;
  for (int r = 0; r < R; ++r) {
    const float* E = l->E + (size_t)r * DN;
    float s = 0;
    for (int c = 0; c < DN; ++c) s += q[c] * E[c];
    s *= sc;
    if (use_span) {
      float d = 0;
      for (int c = 0; c < DN; ++c) d += Es[c] * E[c];
      s += SPAN_G_MULT * g * sc * d;
    }
    if (use_sim) {
      float d = 0;
      for (int f = 0; f < 6; ++f) d += Sf[r * 6 + f] * l->sim_w[src * 6 + f];
      s += SPAN_G_MULT * sc * d;
    }
    z[2 + r] = ok[r] ? s : NEGV;
  }
  for (int k = 0; k < 2; ++k) {
    float sp[DN];
    memcpy(sp, l->special + k * DN, sizeof sp);
    l2norm(sp, DN);
    float s = 0;
    for (int c = 0; c < DN; ++c) s += q[c] * sp[c];
    z[k] = s * sc + l->special_b[k];
  }
  if (use_sim) {
    float tot = 0, mb = 0, msh = 0;
    int first = 1;
    for (int r = 0; r < R; ++r) {
      for (int f = 0; f < 6; ++f) tot += fabsf(Sf[r * 6 + f]);
      float v4 = ok[r] ? Sf[r * 6 + 4] : 0.0f, v5 = ok[r] ? Sf[r * 6 + 5] : 0.0f;
      if (first || v4 > mb) mb = v4;
      if (first || v5 > msh) msh = v5;
      first = 0;
    }
    float hs = tot > 0 ? 1.0f : 0.0f;
    const float* sn = l->sim_new + src * 3;
    float tn = sn[0] + sn[1] * (1.0f - mb) + sn[2] * (1.0f - msh);
    z[0] += SPAN_G_MULT * sc * hs * tn;
  }
  if (use_span && l->has_span_new) {
    float ms = -INFINITY, has = 0;
    for (int c = 0; c < DN; ++c) has += fabsf(Es[c]);
    for (int r = 0; r < R; ++r) {
      float d = -1.0f;
      if (ok[r]) {
        d = 0;
        for (int c = 0; c < DN; ++c) d += Es[c] * l->E[(size_t)r * DN + c];
      }
      if (d > ms) ms = d;
    }
    if (ms < -1.0f) ms = -1.0f;
    z[0] += SPAN_G_MULT * l->span_new * sc * (has > 0 ? 1.0f : 0.0f) * (1.0f - ms);
  }
}

/* ------------------------------------------------------------------ reading a name (_read_name_dec) */

typedef struct {
  arc_name n;          /* strings in the result arena */
  arc_wordb* w;        /* the name's words (malloc'd, texts borrowed from the store) */
  int nw;
  double start, end;
  double p, span_p, model_p, snap_score;   /* NAN: None */
  int has_p;
} name_rec;

static const int MENTION_KINDS_PERSON[] = {0, 9, -1}, MENTION_KINDS_TEAM[] = {1, 2, 9, -1},
                 MENTION_KINDS_ORG[] = {2, 1, 9, -1}, MENTION_KINDS_PLACE[] = {3, 9, -1},
                 MENTION_KINDS_WORK[] = {4, 9, -1}, MENTION_KINDS_EVENT[] = {5, 9, -1},
                 MENTION_KINDS_LEGAL[] = {6, 9, -1}, MENTION_KINDS_GAME[] = {7, 9, -1}, MENTION_KINDS_MAP[] = {8, 9, -1};

/* MENTION_KINDS.get(tid, all) (+ [9] for teams) */
static void allowed_kinds(const char* tid, unsigned char okk[NKIND]) {
  const int* k = NULL;
  if (!strcmp(tid, "entity.mention.person")) k = MENTION_KINDS_PERSON;
  else if (!strcmp(tid, "entity.mention.team")) k = MENTION_KINDS_TEAM;
  else if (!strcmp(tid, "entity.mention.organization")) k = MENTION_KINDS_ORG;
  else if (!strcmp(tid, "entity.mention.place")) k = MENTION_KINDS_PLACE;
  else if (!strcmp(tid, "entity.mention.work_product")) k = MENTION_KINDS_WORK;
  else if (!strcmp(tid, "entity.mention.event_name")) k = MENTION_KINDS_EVENT;
  else if (!strcmp(tid, "entity.mention.legal_ref")) k = MENTION_KINDS_LEGAL;
  else if (!strcmp(tid, "entity.mention.game_entity")) k = MENTION_KINDS_GAME;
  else if (!strcmp(tid, "entity.mention.game_map")) k = MENTION_KINDS_MAP;
  memset(okk, 0, NKIND);
  if (!k) {
    memset(okk, 1, NKIND);
    return;
  }
  for (int i = 0; k[i] >= 0; ++i) okk[k[i]] = 1;
}

typedef struct {
  arc_link* l;
  const arc_link_ctx* c;
  const arc_wstore* W;
  arc_nt_rescorer* resc;
  arc_nt_policy pol;
  arc_arena* A;
  int rescore;
} run_ctx;

/* Engine._head_p: the mean span_in probability of tag i over frames a..b */
static double head_p(run_ctx* x, int i, int a, int b, int* ok) {
  *ok = 0;
  if (b < a) return 0;
  int nc;
  const int* cols;
  int T = b - a + 1;
  float* H = arc_malloc(sizeof(float) * (size_t)T * DM);
  for (int f = a; f <= b; ++f) {
    int ff = f < 0 ? 0 : f >= x->c->N ? x->c->N - 1 : f;
    memcpy(H + (size_t)(f - a) * DM, x->c->H + (size_t)ff * DM, sizeof(float) * DM);
  }
  float* p = arc_engine_span_probs(x->c->e, H, T, x->c->dom, &nc, &cols);
  arc_free(H);
  int j = -1;
  for (int k = 0; k < nc; ++k)
    if (cols[k] == i) j = k;
  double s = 0;
  if (j >= 0) {
    for (int t = 0; t < T; ++t) s += p[(size_t)t * nc + j];
    s /= T;
    *ok = 1;
  }
  arc_free(p);
  return arc_pyround(s, 4);
}

static const float* h_at(const run_ctx* x, int f) {
  int ff = f < 0 ? 0 : f >= x->c->N ? x->c->N - 1 : f;
  return x->c->H + (size_t)ff * DM;
}

static char* join_words(const arc_wstore* W, const int* idx, int n, int skip_fillers) {
  size_t len = 1;
  for (int k = 0; k < n; ++k) len += strlen(W->w[idx[k]].w) + 1;
  char* s = arc_malloc(len);
  size_t o = 0;
  for (int k = 0; k < n; ++k) {
    const char* w = W->w[idx[k]].w;
    if (skip_fillers && is_filler(w)) continue;
    if (o) s[o++] = ' ';
    size_t l = strlen(w);
    memcpy(s + o, w, l);
    o += l;
  }
  s[o] = 0;
  return s;
}

/* Engine._policy */
static void policy(run_ctx* x, name_rec* r, const char* span_text, const int* allowed, int na) {
  const arc_nt_policy* pol = &x->pol;
  char** toks;
  int nt = arc_split_words(span_text, &toks);
  int nkeep = 0;
  for (int k = 0; k < nt; ++k)
    if (!is_filler(toks[k])) toks[nkeep++] = toks[k];
    else arc_free(toks[k]);
  if (pol->require_heard && r->n.state && !strcmp(r->n.state, "linked") && !nkeep) {
    r->n.entity = NULL;
    r->n.state = "abstain";
    r->n.kb_index = -1;
  }
  if (pol->fuzzy_tau > 0 && r->n.state && !strcmp(r->n.state, "abstain") && nkeep && na) {
    /* n-grams (1..3) of the normalised heard words against the allowed names (sim_feats best) */
    size_t len = 1;
    for (int k = 0; k < nkeep; ++k) len += strlen(toks[k]) + 1;
    char* joined = arc_malloc(len);
    joined[0] = 0;
    for (int k = 0; k < nkeep; ++k) {
      if (k) strcat(joined, " ");
      strcat(joined, toks[k]);
    }
    char* nn = arc_nt_snorm(joined);
    arc_free(joined);
    char** tk;
    int ntk = arc_split_words(nn, &tk);
    arc_free(nn);
    char** subs = arc_malloc(sizeof(char*) * (size_t)(3 * ntk + 1));
    int ns = 0;
    for (int n = 1; n <= 3; ++n)
      for (int i = 0; i + n <= ntk; ++i) {
        size_t sl = 1;
        for (int k = i; k < i + n; ++k) sl += strlen(tk[k]) + 1;
        char* s = arc_malloc(sl);
        s[0] = 0;
        for (int k = i; k < i + n; ++k) {
          if (k > i) strcat(s, " ");
          strcat(s, tk[k]);
        }
        int dup = 0;
        for (int q = 0; q < ns; ++q)
          if (!strcmp(subs[q], s)) dup = 1;
        if (dup) arc_free(s);
        else subs[ns++] = s;
      }
    for (int a = 0; a < ns; ++a)   /* sorted(set(...)) */
      for (int b = a + 1; b < ns; ++b)
        if (strcmp(subs[b], subs[a]) < 0) {
          char* t = subs[a];
          subs[a] = subs[b];
          subs[b] = t;
        }
    if (ns) {
      const char** nm = arc_malloc(sizeof(char*) * (size_t)na);
      for (int k = 0; k < na; ++k) nm[k] = x->l->kb[allowed[k]].name;
      arc_nt_candtab* ct = arc_nt_candtab_new(nm, na);
      float* F = arc_malloc(sizeof(float) * (size_t)ns * na * 6);
      arc_nt_sim_feats(ct, (const char* const*)subs, ns, F);
      /* F.max(0).argmax(): best over spans for each name, then the first best name */
      int j = 0;
      float best = -INFINITY;
      for (int k = 0; k < na; ++k) {
        float m = -INFINITY;
        for (int s = 0; s < ns; ++s)
          if (F[((size_t)s * na + k) * 6 + 4] > m) m = F[((size_t)s * na + k) * 6 + 4];
        if (m > best) {
          best = m;
          j = k;
        }
      }
      if (best >= pol->fuzzy_tau) {
        int row = allowed[j];
        r->n.entity = x->l->kb[row].name;
        r->n.state = "linked";
        r->n.kb_index = row;
        r->n.via = "text";
        r->p = arc_pyround(best, 4);
        r->has_p = 1;
      }
      arc_free(F);
      arc_free(nm);
      arc_nt_candtab_free(ct);
    }
    for (int k = 0; k < ns; ++k) arc_free(subs[k]);
    arc_free(subs);
    arc_free_words(tk, ntk);
  }
  for (int k = 0; k < nkeep; ++k) arc_free(toks[k]);
  arc_free(toks);
}

static int read_name(run_ctx* x, const pend* pd, name_rec* r) {
  arc_link* l = x->l;
  const arc_wstore* W = x->W;
  memset(r, 0, sizeof *r);
  if (!pd->nk) return 0;
  char* span_text = join_words(W, pd->keys, pd->nk, 1);
  int hok;
  double hp = head_p(x, pd->i, pd->a, pd->b, &hok);
  r->start = W->w[pd->keys[0]].start;
  r->end = W->w[pd->keys[pd->nk - 1]].end;
  r->n.tag = pd->ti.tid;
  r->n.src = "decoder";
  r->n.kb_index = -1;
  r->n.model_p = -1;
  r->p = NAN;
  r->snap_score = NAN;
  int R = l->nkb;
  unsigned char* ok = arc_calloc((size_t)R + 1);
  int* allowed = arc_malloc(sizeof(int) * (size_t)(R + 1));
  int na = 0;
  if (R) {
    unsigned char okk[NKIND];
    allowed_kinds(pd->ti.tid, okk);
    for (int k = 0; k < R; ++k) ok[k] = okk[l->ckind[k]];
    if (span_text[0]) {
      int* rows = arc_malloc(sizeof(int) * (size_t)R);
      int nr = arc_nt_exact_rows_in(l->exn, span_text, rows);
      for (int k = 0; k < nr; ++k) ok[rows[k]] = 1;
      arc_free(rows);
    }
    for (int k = 0; k < R; ++k)
      if (ok[k]) allowed[na++] = k;
  }
  if (R && na && span_text[0]) {
    const float* h = h_at(x, pd->b);
    char* heard = heard_tail(W, pd->b + x->c->d);
    float Es[DN], *Sf = NULL;
    int src = 0, has_heard = heard[0] != 0;
    if (has_heard) {
      if (l->has_span) name_embed(l, heard, 1, 0, Es);
      if (l->has_sim && l->ct) {
        Sf = arc_malloc(sizeof(float) * (size_t)R * 6);
        const char* sp[1] = {heard};
        arc_nt_sim_feats(l->ct, sp, 1, Sf);
      }
      src = l->ptr_src;
    }
    float* z = arc_malloc(sizeof(float) * (size_t)(R + 2));
    ptr_logits(l, h, arc_engine_tag_key(x->c->e, pd->i), ok, has_heard && l->has_span ? Es : NULL, Sf, src, z);
    z[1] = NEGV;
    float mx = z[0];
    int top = 0;
    for (int k = 1; k < R + 2; ++k)
      if (z[k] > mx) {
        mx = z[k];
        top = k;
      }
    double s = 0;
    for (int k = 0; k < R + 2; ++k) s += exp((double)z[k] - mx);
    r->n.model_entity = top == 0 ? NULL : l->kb[top - 2].name;
    r->n.model_p = (float)arc_pyround(1.0 / s, 4);
    if (x->rescore && x->resc) {
      char* past = text_between(W, fmax(0.0, pd->a / 10.0 - 30), pd->a / 10.0);
      arc_nt_resc_out ro;
      arc_nt_rescorer_score(x->resc, (pd->b + 1) / 10.0, z, R + 2, allowed, na, span_text, past, NULL, &ro);
      arc_free(past);
      r->n.entity = ro.link ? l->kb[ro.pred].name : NULL;
      r->p = ro.p;
      r->has_p = 1;
      r->n.state = ro.link ? "linked" : ro.pred < 0 ? "not_in_list" : "abstain";
      if (ro.link) r->n.kb_index = ro.pred;
      policy(x, r, span_text, allowed, na);
    } else {
      r->n.entity = r->n.model_entity;
      r->p = r->n.model_p;
      r->has_p = 1;
      r->n.state = top >= 2 ? "linked" : "not_in_list";
      if (top >= 2) r->n.kb_index = top - 2;
    }
    arc_free(z);
    arc_free(Sf);
    arc_free(heard);
  } else {
    r->n.entity = NULL;
    r->n.state = "not_in_list";
  }
  r->nw = pd->nk;
  r->w = arc_malloc(sizeof(arc_wordb) * (size_t)pd->nk);
  int show = 0;
  for (int k = 0; k < pd->nk; ++k) {
    const arc_wsword* w = &W->w[pd->keys[k]];
    r->w[k] = (arc_wordb){w->w, w->start, w->end, w->tag, w->tag_begin};
    show |= show_word(w->w);
  }
  r->span_p = hok ? hp : NAN;
  r->n.show = show;
  r->n.heard = span_text;   /* malloc'd for now; moved into the arena at the end */
  arc_free(ok);
  arc_free(allowed);
  return 1;
}

/* ------------------------------------------------------------------ reading a number (_read_number, decoder) */

typedef struct {
  arc_number n;       /* heard malloc'd until packed */
  double start, end;
  int has_vm, has_v2m;
  double vm;
} num_rec;

static const char* number_kind(run_ctx* x, int a, int b) {   /* Engine._number_kind */
  if (b < a) return NULL;
  int T = b - a + 1, nc;
  const int* cols;
  float* H = arc_malloc(sizeof(float) * (size_t)T * DM);
  for (int f = a; f <= b; ++f) memcpy(H + (size_t)(f - a) * DM, h_at(x, f), sizeof(float) * DM);
  float* p = arc_engine_span_probs(x->c->e, H, T, x->c->dom, &nc, &cols);
  arc_free(H);
  const char* best = NULL;
  double bv = -1;
  for (int k = 0; k < nc; ++k) {
    const char* g = arc_engine_tag_group(x->c->e, cols[k]);
    if (!g || strcmp(g, "entity.number")) continue;
    double s = 0;
    for (int t = 0; t < T; ++t) s += p[(size_t)t * nc + k];
    s /= T;
    if (s > bv) {
      bv = s;
      best = arc_engine_tag_name(x->c->e, cols[k]);
    }
  }
  arc_free(p);
  if (!best) return NULL;
  const char* d = strrchr(best, '.');
  return d ? d + 1 : best;
}

static int has_digit(const char* s) {
  for (; *s; ++s)
    if (*s >= '0' && *s <= '9') return 1;
  return 0;
}

static void read_number(run_ctx* x, const pend* pd, num_rec* r) {
  const arc_wstore* W = x->W;
  arc_link* l = x->l;
  memset(r, 0, sizeof *r);
  r->start = pd->start;
  r->end = pd->end;
  r->n.kind = pd->ti.nkind;
  /* ws: the span's number words, connectors off both ends */
  int* ws = arc_malloc(sizeof(int) * (size_t)(pd->nk + 1));
  int nws = 0;
  for (int k = 0; k < pd->nk; ++k)
    if (arc_num_word(W->w[pd->keys[k]].w)) ws[nws++] = pd->keys[k];
  int s0 = 0;
  while (s0 < nws && arc_num_connector(W->w[ws[s0]].w)) s0++;
  while (nws > s0 && arc_num_connector(W->w[ws[nws - 1]].w)) nws--;
  char* heard_span = join_words(W, pd->keys, pd->nk, 0);
  if (nws <= s0) {
    r->n.state = "no_words";
    r->n.heard = heard_span;
    arc_free(ws);
    return;
  }
  int* idx = ws + s0;
  int ni = nws - s0;
  /* value heads: nwt tokens per word (mean over its last two frames), the unit at the last word's end */
  const char* toks[256];
  int nt = 0;
  float logit[K_SLOTS * N_NWT];
  for (int q = 0; q < ni; ++q) {
    int j = idx[q], e = own_end(W, j), st = W->w[j].f0;
    int lo = st > e - 1 ? st : e - 1;
    float mean[K_SLOTS * N_NWT] = {0};
    int cnt = 0;
    for (int f = lo; f <= e; ++f) {
      mv(l->nwt_w, l->nwt_b, h_at(x, f), K_SLOTS * N_NWT, DM, logit);
      for (int k = 0; k < K_SLOTS * N_NWT; ++k) mean[k] += logit[k];
      cnt++;
    }
    for (int sl = 0; sl < K_SLOTS; ++sl) {
      int am = 0;
      for (int k = 1; k < N_NWT; ++k)
        if (mean[sl * N_NWT + k] / cnt > mean[sl * N_NWT + am] / cnt) am = k;
      if (am != 0 && nt < 256) toks[nt++] = l->nwt_vocab ? arc_json_str(arc_json_at(l->nwt_vocab, am), arc_nwt_token(am)) : arc_nwt_token(am);
    }
  }
  float* ul = arc_malloc(sizeof(float) * (size_t)l->n_units);
  mv(l->unit_w, l->unit_b, h_at(x, own_end(W, idx[ni - 1])), l->n_units, DM, ul);
  int ui = 0;
  for (int k = 1; k < l->n_units; ++k)
    if (ul[k] > ul[ui]) ui = k;
  arc_free(ul);
  int nuv = l->unit_vocab ? l->unit_vocab->n : 0;
  const char* unit = ui < nuv ? arc_json_str(arc_json_at(l->unit_vocab, ui), NULL) : NULL;
  const char* kind = pd->ti.nkind;
  arc_numval val = kind ? arc_num_finalize(kind, toks, nt, unit) : arc_num_compose(toks, nt, NULL);
  /* the value from the words: tw = the span's words, with the word after and before */
  int k0 = pd->keys[0], k1 = pd->keys[pd->nk - 1];
  const char** tw = arc_malloc(sizeof(char*) * (size_t)pd->nk);
  for (int k = 0; k < pd->nk; ++k) tw[k] = W->w[pd->keys[k]].w;
  const char* nxt[1] = {k1 + 1 < W->n ? W->w[k1 + 1].w : NULL};
  const char* unit_t = NULL;
  arc_numval vt = arc_text_value(kind, tw, pd->nk, nxt, k1 + 1 < W->n ? 1 : 0, k0 ? W->w[k0 - 1].w : "", NULL, &unit_t);
  if (!vt.ok && kind && !unit_t && unit) {
    const char* u2 = NULL;
    vt = arc_text_value(kind, tw, pd->nk, NULL, 0, "", unit, &u2);
    unit_t = vt.ok ? unit : NULL;
  }
  arc_free(tw);
  r->n.unit = unit;
  r->n.has_value = val.ok;
  r->n.value = val.v;
  r->n.value_int = val.v_int;
  r->n.has_value2 = val.ok && val.has_v2;
  r->n.value2 = val.v2;
  r->n.value2_int = val.v2_int;
  r->n.state = "read";
  r->n.has_value_model = val.ok;
  r->n.value_model = val.v;
  int digits = 0;
  for (int q = 0; q < ni; ++q) digits |= has_digit(W->w[idx[q]].w);
  if (vt.ok && (digits || (vt.has_v2 && !r->n.has_value2) || !r->n.has_value)) {
    r->n.has_value = 1;
    r->n.value = vt.v;
    r->n.value_int = vt.v_int;
    r->n.has_value2 = vt.has_v2;
    r->n.value2 = vt.v2;
    r->n.value2_int = vt.v2_int;
    r->n.state = "read_text";
    r->n.unit = unit_t ? unit_t : unit;
  }
  /* decoder-sourced: the tag head's kind too, heard = the span's words, and the words' value wins */
  r->n.kind_head = number_kind(x, pd->a, pd->b);
  if (vt.ok) {
    r->n.has_value = 1;
    r->n.value = vt.v;
    r->n.value_int = vt.v_int;
    r->n.has_value2 = vt.has_v2;
    r->n.value2 = vt.v2;
    r->n.value2_int = vt.v2_int;
    r->n.state = "read_text";
    r->n.unit = unit_t ? unit_t : unit;
  }
  r->n.heard = heard_span;
  arc_free(ws);
}

/* ------------------------------------------------------------------ extend_number (api.py) */

static void extend_number(run_ctx* x, num_rec* r, const arc_result_build* R) {
  /* the transcript's words sorted by start */
  int n = 0;
  for (int i = 0; i < R->n; ++i) n += R->u[i].n_aw;
  if (!n) return;
  const arc_wordb** Wd = arc_malloc(sizeof(arc_wordb*) * (size_t)n);
  int k = 0;
  for (int i = 0; i < R->n; ++i)
    for (int j = 0; j < R->u[i].n_aw; ++j) Wd[k++] = &R->u[i].aw[j];
  for (int a = 1; a < n; ++a) {   /* stable insertion sort by start */
    const arc_wordb* t = Wd[a];
    int b = a - 1;
    while (b >= 0 && Wd[b]->start > t->start) {
      Wd[b + 1] = Wd[b];
      b--;
    }
    Wd[b + 1] = t;
  }
  int i = -1, j = -1;
  for (int q = 0; q < n; ++q) {
    double m = (Wd[q]->start + Wd[q]->end) / 2;
    if (m >= r->start - 0.02 && m <= r->end + 0.02) {
      if (i < 0) i = q;
      j = q;
    }
  }
  if (i < 0) {
    arc_free(Wd);
    return;
  }
  static const char* const GROW[] = {"hundred", "thousand", "million", "billion", "and", "point", "to", "twenty",
                                     "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", NULL};
  int grew = 0;
  for (;;) {
    if (i <= 0) break;
    const char* prev = Wd[i - 1]->w;
    if (!arc_numeric_word(prev) || !(Wd[i]->start - Wd[i - 1]->end < 0.6)) break;
    char cw[256];
    arc_clean_word(Wd[i]->w, cw);
    for (char* p = cw; *p; ++p) *p = (char)tolower((unsigned char)*p);
    if (!in_set(cw, GROW)) break;
    size_t pl = strlen(prev);
    while (pl && isspace((unsigned char)prev[pl - 1])) pl--;
    if (pl && strchr(".,?!", prev[pl - 1])) break;
    i--;
    grew = 1;
  }
  if (grew) {
    const char** ws = arc_malloc(sizeof(char*) * (size_t)(j - i + 1));
    size_t len = 1;
    for (int q = i; q <= j; ++q) {
      ws[q - i] = Wd[q]->w;
      len += strlen(Wd[q]->w) + 1;
    }
    char* h = arc_malloc(len);
    h[0] = 0;
    for (int q = i; q <= j; ++q) {
      if (q > i) strcat(h, " ");
      strcat(h, Wd[q]->w);
    }
    arc_free((void*)r->n.heard);
    r->n.heard = h;
    r->start = arc_pyround(fmin(r->start, Wd[i]->start), 3);
    const char* ut = NULL;
    arc_numval vt = arc_text_value(r->n.kind, ws, j - i + 1, NULL, 0, "", NULL, &ut);
    if (vt.ok) {
      r->n.has_value = 1;
      r->n.value = vt.v;
      r->n.value_int = vt.v_int;
      r->n.has_value2 = vt.has_v2;
      r->n.value2 = vt.v2;
      r->n.value2_int = vt.v2_int;
      r->n.state = "read_text";
    }
    arc_free(ws);
  }
  arc_free(Wd);
  (void)x;
}

/* ------------------------------------------------------------------ run */

static arc_word* arena_words(arc_arena* A, const arc_wordb* w, int n) {
  arc_word* o = arc_arena_alloc(A, sizeof(arc_word) * (size_t)(n + 1));
  for (int k = 0; k < n; ++k) {
    o[k].text = arc_arena_strdup(A, w[k].w);
    o[k].start = (float)w[k].start;
    o[k].end = (float)w[k].end;
    o[k].tag = w[k].tag >= 0 ? arc_tag_name(w[k].tag) : NULL;
    o[k].tag_begin = w[k].tag_begin;
  }
  return o;
}

int arc_link_run(arc_link* l, const arc_link_ctx* c, const arc_wstore* W, arc_result_build* R) {
  run_ctx x = {l, c, W, NULL, {0}, R->arena, 1};
  const char* domain;
  static const char* DOMS[16] = {"nba", "nhl", "nfl", "cricket", "soccer", "esports", "news", "talk", "stream_chat",
                                 "meeting", "court_appellate", "court_trial", "assistant", "assistant_noisy",
                                 "short_clips", "unknown"};
  domain = DOMS[c->dom >= 0 && c->dom < 16 ? c->dom : 15];
  arc_nt_policy_cfg(c->e->cfg, domain, "builtin", &x.pol);
  if (l->nkb) {
    arc_nt_resc_params rp;
    arc_nt_resc_params_cfg(c->e->cfg, domain, &x.pol, &rp);
    x.resc = arc_nt_rescorer_new(l->kb, l->nkb, &rp);
  }
  /* WordStore.add per utterance: spans_of_words over each utterance's words (sorted by time) */
  int cap = 16, np = 0;
  pend* P = arc_malloc(sizeof(pend) * (size_t)cap);
  for (int u = 0; u < R->n; ++u) {
    /* this utterance's words in store order */
    int cur = -1;
    for (int k = 0; k <= W->n; ++k) {
      const arc_wsword* w = k < W->n ? &W->w[k] : NULL;
      if (w && w->utt != u) continue;
      const char* t = w && w->tag >= 0 ? arc_tag_name(w->tag) : NULL;
      if (w && t && cur >= 0 && !strcmp(t, P[cur].ti.dtag) && !w->tag_begin) {
        P[cur].keys[P[cur].nk++] = k;
        P[cur].end = w->end;
        continue;
      }
      cur = -1;
      if (!w) break;
      tag_info ti;
      if (t && tag_info_of(t, &ti)) {
        if (np == cap) {
          pend* nP = arc_malloc(sizeof(pend) * (size_t)cap * 2);
          memcpy(nP, P, sizeof(pend) * (size_t)np);
          arc_free(P);
          P = nP;
          cap *= 2;
        }
        pend* p = &P[np];
        memset(p, 0, sizeof *p);
        p->ti = ti;
        p->start = w->start;
        p->end = w->end;
        p->keys = arc_malloc(sizeof(int) * (size_t)(W->n + 1));
        p->keys[p->nk++] = k;
        cur = np++;
      }
    }
  }
  /* _dec_pd, then read each when the trunk has passed it (Arc.predict feeds 3000 frames at a time) */
  name_rec* names = arc_malloc(sizeof(name_rec) * (size_t)(np + 1));
  num_rec* nums = arc_malloc(sizeof(num_rec) * (size_t)(np + 1));
  int nn = 0, nu = 0;
  unsigned char* done = arc_calloc((size_t)np + 1);
  for (int k = 0; k < np; ++k) {
    pend* p = &P[k];
    frames_of(p->start, p->end, &p->a, &p->b);
    if (p->ti.is_name) {
      if (!strcmp(p->ti.dtag, "ENTITY_LEGAL")) done[k] = 1;
      else if ((p->i = arc_engine_tag_index(c->e, p->ti.tid)) < 0) done[k] = 1;
    }
  }
  for (int s0 = 0;; s0 += 3000) {
    int f_end = s0 + 3000 < c->N ? s0 + 3000 : c->N;
    int final = f_end >= c->N;
    for (int k = 0; k < np; ++k) {
      if (done[k] || !(final || f_end > P[k].b)) continue;
      done[k] = 1;
      if (P[k].ti.is_name) {
        if (!c->names) continue;
        if (read_name(&x, &P[k], &names[nn])) nn++;
      } else {
        if (!c->numbers) continue;
        read_number(&x, &P[k], &nums[nu]);
        nu++;
      }
    }
    if (final) break;
  }
  /* api._post: kbsnap link for names left unlinked, then capitalised phrases that match the list */
  int capn = nn + 16;
  name_rec* all = arc_malloc(sizeof(name_rec) * (size_t)capn);
  memcpy(all, names, sizeof(name_rec) * (size_t)nn);
  int na = nn;
  if (c->names && l->nkb && l->snap && !arc_nt_kbsnap_empty(l->snap)) {
    for (int k = 0; k < na; ++k) {
      name_rec* r = &all[k];
      if (r->n.entity || (r->n.state && !strcmp(r->n.state, "linked"))) continue;
      int row;
      double sc;
      if (arc_nt_kbsnap_link(l->snap, r->n.heard, &row, &sc)) {
        r->n.entity = l->kb[row].name;
        r->n.state = "linked";
        r->n.via = "snap";
        r->snap_score = sc;
      }
    }
    int added = 0;
    for (int u = 0; u < R->n; ++u) {
      const arc_utt_build* ub = &R->u[u];
      if (!ub->n_aw) continue;
      arc_nt_word* aw = arc_malloc(sizeof(arc_nt_word) * (size_t)ub->n_aw);
      for (int k = 0; k < ub->n_aw; ++k)
        aw[k] = (arc_nt_word){ub->aw[k].w, ub->aw[k].start, ub->aw[k].end, ub->aw[k].tag >= 0 ? arc_tag_name(ub->aw[k].tag) : NULL};
      double(*taken)[2] = arc_malloc(sizeof(double[2]) * (size_t)(na + 1));
      for (int k = 0; k < na; ++k) {
        taken[k][0] = all[k].start;
        taken[k][1] = all[k].end;
      }
      arc_nt_snap_span* ss = NULL;
      int ns = arc_nt_add_spans(l->snap, aw, ub->n_aw, (const double(*)[2])taken, na, &ss);
      for (int k = 0; k < ns; ++k) {
        if (na == capn) {
          name_rec* nA = arc_malloc(sizeof(name_rec) * (size_t)capn * 2);
          memcpy(nA, all, sizeof(name_rec) * (size_t)na);
          arc_free(all);
          all = nA;
          capn *= 2;
        }
        name_rec* r = &all[na++];
        memset(r, 0, sizeof *r);
        r->start = ss[k].start;
        r->end = ss[k].end;
        r->n.tag = ss[k].tag;
        r->n.heard = arc_strdup(ss[k].heard);
        r->n.src = "snap";
        r->n.entity = l->kb[ss[k].row].name;
        r->n.kb_index = -1;
        r->n.state = "linked";
        r->n.via = "snap";
        r->n.model_p = -1;
        r->n.show = 1;
        r->p = NAN;
        r->span_p = NAN;
        r->snap_score = ss[k].snap_score;
        r->nw = ss[k].n_words;
        r->w = arc_malloc(sizeof(arc_wordb) * (size_t)(r->nw + 1));
        for (int q = 0; q < r->nw; ++q) {
          const arc_wordb* src = &ub->aw[ss[k].w[q]];
          r->w[q] = (arc_wordb){src->w, src->start, src->end, src->tag, src->tag_begin};
        }
        added++;
      }
      arc_nt_snap_spans_free(ss, ns);
      arc_free(taken);
      arc_free(aw);
    }
    if (added)   /* sorted by (start, end), stable */
      for (int a = 1; a < na; ++a) {
        name_rec t = all[a];
        int b = a - 1;
        while (b >= 0 && (all[b].start > t.start || (all[b].start == t.start && all[b].end > t.end))) {
          all[b + 1] = all[b];
          b--;
        }
        all[b + 1] = t;
      }
  }
  /* tidy_out: names tidied and de-duplicated; numbers extended, tidied, kept when they have a value or a number */
  arc_tname* tn = arc_malloc(sizeof(arc_tname) * (size_t)(na + 1));
  arc_tword** tw = arc_malloc(sizeof(arc_tword*) * (size_t)(na + 1));
  for (int k = 0; k < na; ++k) {
    name_rec* r = &all[k];
    tw[k] = arc_malloc(sizeof(arc_tword) * (size_t)(r->nw + 1));
    for (int q = 0; q < r->nw; ++q) tw[k][q] = (arc_tword){r->w[q].w, r->w[q].start, r->w[q].end};
    double rp = r->has_p && !isnan(r->p) && r->p != 0 ? r->p : !isnan(r->span_p) && r->span_p != 0 ? r->span_p : 0.0;
    tn[k] = (arc_tname){(char*)r->n.heard, r->n.entity, r->n.state, r->n.via, (float)rp, r->start, r->end, tw[k], r->nw, k};
  }
  for (int k = 0; k < na; ++k) arc_tidy_name(&tn[k]);
  int nk = arc_dedupe_names(tn, na);
  arc_name* out = arc_arena_alloc(x.A, sizeof(arc_name) * (size_t)(nk + 1));
  for (int k = 0; k < nk; ++k) {
    name_rec* r = &all[tn[k].id];
    arc_name* o = &out[k];
    *o = r->n;
    o->start = (float)tn[k].start;
    o->end = (float)tn[k].end;
    o->heard = arc_arena_strdup(x.A, tn[k].heard);
    o->entity = arc_arena_strdup(x.A, r->n.entity);
    o->model_entity = arc_arena_strdup(x.A, r->n.model_entity);
    o->p = r->has_p && !isnan(r->p) ? (float)r->p : -1.0f;
    o->span_p = isnan(r->span_p) ? -1.0f : (float)r->span_p;
    o->snap_score = isnan(r->snap_score) ? -1.0f : (float)r->snap_score;
    /* the tidied words: those of r->w that tidy_name kept (a contiguous run) */
    int wn = tn[k].n_words, w0 = wn ? (int)(tn[k].words - tw[tn[k].id]) : 0;
    o->words = arena_words(x.A, r->w + w0, wn);
    o->n_words = wn;
  }
  R->names = out;
  R->n_names = nk;
  arc_number* on = arc_arena_alloc(x.A, sizeof(arc_number) * (size_t)(nu + 1));
  int nko = 0;
  for (int k = 0; k < nu; ++k) {
    num_rec* r = &nums[k];
    if (!arc_keep_number(r->n.has_value, r->n.heard)) continue;   /* keep_number on the record as read */
    extend_number(&x, r, R);
    char* h = arc_tidy_number(r->n.heard);
    arc_number* o = &on[nko++];
    *o = r->n;
    o->start = (float)r->start;
    o->end = (float)r->end;
    o->heard = arc_arena_strdup(x.A, h);
    arc_free(h);
  }
  R->numbers = on;
  R->n_numbers = nko;
  /* free */
  for (int k = 0; k < nu; ++k) arc_free((void*)nums[k].n.heard);
  for (int k = 0; k < na; ++k) {
    arc_free(all[k].w);
    arc_free(tw[k]);
  }
  for (int k = 0; k < nk; ++k) arc_free(tn[k].heard);   /* dedupe freed the dropped ones */
  arc_free(tn);
  arc_free(tw);
  arc_free(all);
  arc_free(names);
  arc_free(nums);
  for (int k = 0; k < np; ++k) arc_free(P[k].keys);
  arc_free(P);
  arc_free(done);
  arc_nt_rescorer_free(x.resc);
  return 0;
}
