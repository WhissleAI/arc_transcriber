/* Arc's trunk, heads and readout: features from the Moonshine encoder (mean+max pooled to 10 Hz), the device-directed
 * and sound-event heads on them, the trunk (5 rotating-state SSM layers with FiLM conditioning on the domain and the
 * read-out delay), the tag head and the engine's per-frame decoding into utterances (arc_audio api.py / model.py /
 * engine.py / events.py), in C. The word channel is left out: offline it does not change the utterance readout.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arc_engine.h"
#include "arc_store.h"
#include "arc_wordch.h"
#include "arc_link.h"

#define DM 512   /* trunk width */
#define DIN 640  /* pooled encoder features */
#define NLY 5
#define SH 8     /* SSM heads */
#define SN 64    /* state */
#define SP 64    /* head dim */
#define NDOM 16
#define NQ 9
#define NEG (-10000.0f)
#define MAXTAG 512

static const char* QUERIES[NQ] = {"frame", "span_in", "span_end", "utt", "speaker", "ev_at", "ev_imm", "ev_rec", "item"};
enum { Q_FRAME, Q_SPAN_IN, Q_SPAN_END, Q_UTT, Q_SPEAKER, Q_EV_AT, Q_EV_IMM, Q_EV_REC, Q_ITEM };
static const char* DOMAINS[NDOM] = {"nba", "nhl", "nfl", "cricket", "soccer", "esports", "news", "talk",
                                    "stream_chat", "meeting", "court_appellate", "court_trial", "assistant",
                                    "assistant_noisy", "short_clips", "unknown"};
static const char* EV_LABELS[] = {
    "scene.cheer", "scene.applause", "scene.crowd_ambience", "scene.boo_jeer", "scene.chant", "scene.music",
    "scene.laughter", "scene.horn", "scene.whistle", "scene.buzzer", "scene.gunshot_explosion", "scene.silence",
    "scene.background_speech", "scene.siren", "scene.phone_ring", "scene.typing", "scene.door_knock", "scene.beep",
    "scene.alarm", "scene.wind", "scene.traffic", "scene.ambience.cafe", "scene.ambience.office",
    "scene.ambience.street", "scene.noise_other", "bio_ev.cough", "bio_ev.throat_clear", "bio_ev.sneeze",
    "meta.degraded_audio", "meta.overlap", "meta.channel.telephone", "meta.channel.narrowband",
    "meta.channel.mulaw", "meta.channel.mp3", "meta.channel.opus", "meta.room.small", "meta.room.medium",
    "meta.room.large", "meta.clipping", "meta.snr.high", "meta.snr.mid", "meta.snr.low", "scene.scream",
    "scene.crying", "scene.glass_break", "scene.distress_call", "scene.distress", "meta.people_present",
    "meta.media_command", "meta.hidden_command"};
#define NEV 50

typedef struct {
  const float *norm, *in_b, *B_b, *C_b, *x_b, *dt_b, *th_b, *lam_b, *out_b, *Bn, *Cn, *Bbias, *Cbias, *A_log;
  arc_w in_w, B_w, C_w, x_w, dt_w, th_w, lam_w, out_w;
  float A[SH];   /* -exp(A_log) */
} ssm_layer;

struct arc_trunk {
  arc_store *st, *aux;
  arc_pool* pool;
  /* trunk */
  const float *in_norm_w, *in_norm_b, *inp_b, *dom_emb, *delay_emb, *delay_film, *adapt_down, *adapt_up, *out_norm;
  arc_w inp_w, film_w[NLY], mlp1_w[NLY], mlp2_w[NLY];
  const float* film_b[NLY];
  ssm_layer L[NLY];
  const float *mlp_n[NLY], *mlp1_b[NLY], *mlp2_b[NLY];
  int delays[8], n_delays;
  /* tag head */
  float* K;   /* keys [ntags, 512] = P(E) + r */
  const float *hb, *bq[NQ], *Qb[NQ];
  arc_w Qw[NQ];
  arc_w pool_w0, pool_w2;
  const float *pool_b0, *pool_b2;   /* utt pooling MLP */
  arc_w spool_w0, spool_w2, ipool_w0, ipool_w2;   /* speaker / item pooling MLPs (stages only) */
  const float *spool_b0, *spool_b2, *ipool_b0, *ipool_b2;
  arc_w slot_w, slotx_w;
  const float *slot_b, *slotx_b;
  int n_slot, n_slotx;
  /* aux heads */
  const float *dd_ln_w, *dd_ln_b, *dd_fc_w, *dd_fc_b;
  const float *ev_ln_w, *ev_ln_b, *ev_c1_w, *ev_c1_b, *ev_c2_w, *ev_c2_b;
  float dd_thr;
  float ev_thr[NEV];
  int ev_on[NEV];
  int ev_tap;
  /* ontology (tags.json) */
  int ntags;
  char* ids[MAXTAG];
  int trained[MAXTAG];
  int cols[NQ][MAXTAG], ncols[NQ];
  unsigned char valid[NDOM][MAXTAG];
  int group_of[MAXTAG];   /* index into groups, -1 independent */
  char* groups[128];
  int ngroups;
  float la[NDOM][MAXTAG];
  unsigned char emit[NDOM][MAXTAG];
  const arc_json* tag_status;   /* borrowed from cfg */
  arc_wordch* wc;   /* the word channel (NULL: weights without one) */
  arc_link* link;   /* names and numbers (arc_link.c) */
  /* FiLM scale / shift of every layer for a (domain, delay): computed once (they depend on nothing else) */
  float* film_gb[NDOM][8];
  /* memory: the SSM state carried between calls (arc_memory) */
  struct ssm_mem* mem;
  int mem_on;
  double mem_frames;
  uint64_t model_id;
};

static const float* W(arc_store* s, const char* name, int64_t n) { return arc_store_f32(s, name, n); }

/* a matrix: packed in place when it is stored so, else float32 */
static int WW(arc_store* s, const char* name, int64_t n, arc_w* w) {
  memset(w, 0, sizeof *w);
  if (arc_store_q4(s, name, &w->q) && (int64_t)w->q.R * w->q.C == n) {
    w->packed = 1;
    return 0;
  }
  w->f = arc_store_f32(s, name, n);
  return w->f ? 0 : -1;
}

/* ------------------------------------------------------------------ ontology */

static int load_tags(struct arc_trunk* t, const char* dir) {
  char path[1200], *text = NULL;
  size_t len;
  snprintf(path, sizeof path, "%s/tags.json", dir);
  if (arc_read_file(path, &text, &len)) return -1;
  arc_json* j = arc_json_parse(text, len);
  arc_free(text);
  if (!j) return -1;
  const arc_json* tags = arc_json_get(j, "tags");
  const arc_json* dg = arc_json_get(j, "domains");
  t->ntags = 0;
  for (int i = 0; i < tags->n; ++i) {
    int ix = (int)arc_json_num(arc_json_get(&tags->items[i], "index"), -1);
    if (ix + 1 > t->ntags) t->ntags = ix + 1;
  }
  if (t->ntags > MAXTAG) {
    arc_seterr("tags.json: %d tags", t->ntags);
    arc_json_free(j);
    return -1;
  }
  for (int i = 0; i < t->ntags; ++i) t->group_of[i] = -1;
  const arc_json* by_ix[MAXTAG] = {0};
  for (int i = 0; i < tags->n; ++i) {
    int ix = (int)arc_json_num(arc_json_get(&tags->items[i], "index"), -1);
    if (ix >= 0) by_ix[ix] = &tags->items[i];
  }
  /* groups in order of first member (Table.groups) */
  for (int i = 0; i < tags->n; ++i) {
    const arc_json* tg = &tags->items[i];
    (void)tg;
  }
  for (int ix = 0; ix < t->ntags; ++ix) {
    const arc_json* tg = by_ix[ix];
    char nm[32];
    snprintf(nm, sizeof nm, "_unused%d", ix);
    t->ids[ix] = arc_strdup(tg ? arc_json_str(arc_json_get(tg, "id"), nm) : nm);
    if (!tg) continue;
    const char* st = arc_json_str(arc_json_get(tg, "status"), "");
    t->trained[ix] = strcmp(st, "reserved") && strcmp(st, "derived");
    const char* ex = arc_json_str(arc_json_get(tg, "exclusivity"), NULL);
    if (ex && strcmp(ex, "independent")) {
      const char* g = arc_json_str(arc_json_get(tg, "group"), "");
      int gi = -1;
      for (int k = 0; k < t->ngroups; ++k)
        if (!strcmp(t->groups[k], g)) gi = k;
      if (gi < 0 && t->ngroups < 128) {
        gi = t->ngroups;
        t->groups[t->ngroups++] = arc_strdup(g);
      }
      t->group_of[ix] = gi;
    }
    if (!t->trained[ix]) continue;
    /* domains: '*', a fine domain, or a coarse one listing it */
    const arc_json* keys = arc_json_get(tg, "domains");
    for (int d = 0; d < NDOM; ++d) {
      int ok = 0;
      for (int k = 0; keys && k < keys->n && !ok; ++k) {
        const char* key = keys->items[k].str;
        if (!strcmp(key, "*") || !strcmp(key, DOMAINS[d])) ok = 1;
        const arc_json* fd = arc_json_get(dg, key);
        for (int m = 0; fd && m < fd->n && !ok; ++m)
          if (!strcmp(fd->items[m].str, DOMAINS[d])) ok = 1;
      }
      t->valid[d][ix] = (unsigned char)ok;
    }
    /* readouts by unit */
    const char* u = arc_json_str(arc_json_get(tg, "unit"), "");
    const char* id = t->ids[ix];
    size_t il = strlen(id);
    int qs[2], nqs = 0;
    if (!strcmp(u, "F") || !strcmp(u, "G")) qs[nqs++] = Q_FRAME;
    else if (!strcmp(u, "W")) {
      qs[nqs++] = Q_SPAN_IN;
      qs[nqs++] = Q_SPAN_END;
    } else if (!strcmp(u, "U") || !strcmp(u, "T")) qs[nqs++] = Q_UTT;
    else if (!strcmp(u, "S")) qs[nqs++] = Q_SPEAKER;
    else if (!strcmp(u, "D")) qs[nqs++] = Q_ITEM;
    else if (!strcmp(u, "E"))
      qs[nqs++] = il > 7 && !strcmp(id + il - 7, ".recent") ? Q_EV_REC
                  : il > 9 && !strcmp(id + il - 9, ".imminent") ? Q_EV_IMM
                                                                 : Q_EV_AT;
    for (int k = 0; k < nqs; ++k) t->cols[qs[k]][t->ncols[qs[k]]++] = ix;
  }
  /* 'unknown' is valid wherever any domain is */
  for (int ix = 0; ix < t->ntags; ++ix) {
    int any = 0;
    for (int d = 0; d < NDOM; ++d) any |= t->valid[d][ix];
    if (any) t->valid[15][ix] = 1;
  }
  arc_json_free(j);
  return 0;
}

/* ------------------------------------------------------------------ loading */

int arc_engine_load_trunk(arc_engine* e, const char* dir) {
  struct arc_trunk* t = arc_calloc(sizeof *t);
  if (!t) return -1;
  e->trunk = t;
  char path[1200];
  snprintf(path, sizeof path, "%s/model.q4.safetensors", dir);
  if (!(t->st = arc_store_open(path))) return -1;
  snprintf(path, sizeof path, "%s/heads.safetensors", dir);
  FILE* hf = fopen(path, "rb");
  if (hf) {
    fclose(hf);
    t->aux = arc_store_open(path);
  }
  t->pool = e->moon->pool;
  if (load_tags(t, dir)) return -1;
  arc_store* s = t->st;
#define G(field, nm, n) if (!(t->field = W(s, nm, n))) return -1
  G(in_norm_w, "trunk.in_norm.weight", DIN);
  G(in_norm_b, "trunk.in_norm.bias", DIN);
  if (WW(s, "trunk.inp.weight", DM * DIN, &t->inp_w)) return -1;
  G(inp_b, "trunk.inp.bias", DM);
  G(dom_emb, "trunk.dom_emb.weight", NDOM * DM);
  G(delay_emb, "trunk.delay_emb.weight", 6 * DM);
  G(delay_film, "trunk.delay_film.weight", 6 * DM);
  G(adapt_down, "trunk.adapt_down", NDOM * DM * 64);
  G(adapt_up, "trunk.adapt_up", NDOM * 64 * DM);
  G(out_norm, "trunk.out_norm.weight", DM);
  for (int l = 0; l < NLY; ++l) {
    ssm_layer* L = &t->L[l];
#define LW(field, nm, n) if (!(L->field = W(s, arc_namef("trunk.layers.%d." nm, l), n))) return -1
    LW(norm, "norm.weight", DM);
    if (WW(s, arc_namef("trunk.layers.%d.in_proj.weight", l), DM * DM, &L->in_w)) return -1;
    LW(in_b, "in_proj.bias", DM);
    if (WW(s, arc_namef("trunk.layers.%d.B_proj.weight", l), SH * SN * DM, &L->B_w)) return -1;
    LW(B_b, "B_proj.bias", SH * SN);
    if (WW(s, arc_namef("trunk.layers.%d.C_proj.weight", l), SH * SN * DM, &L->C_w)) return -1;
    LW(C_b, "C_proj.bias", SH * SN);
    if (WW(s, arc_namef("trunk.layers.%d.x_proj.weight", l), DM * DM, &L->x_w)) return -1;
    LW(x_b, "x_proj.bias", DM);
    if (WW(s, arc_namef("trunk.layers.%d.dt_proj.weight", l), SH * DM, &L->dt_w)) return -1;
    LW(dt_b, "dt_proj.bias", SH);
    if (WW(s, arc_namef("trunk.layers.%d.theta_proj.weight", l), SH * SN / 2 * DM, &L->th_w)) return -1;
    LW(th_b, "theta_proj.bias", SH * SN / 2);
    if (WW(s, arc_namef("trunk.layers.%d.lam_proj.weight", l), SH * DM, &L->lam_w)) return -1;
    LW(lam_b, "lam_proj.bias", SH);
    if (WW(s, arc_namef("trunk.layers.%d.out_proj.weight", l), DM * DM, &L->out_w)) return -1;
    LW(out_b, "out_proj.bias", DM);
    LW(Bn, "B_norm.weight", SN);
    LW(Cn, "C_norm.weight", SN);
    LW(Bbias, "B_bias", SH * SN);
    LW(Cbias, "C_bias", SH * SN);
    LW(A_log, "A_log", SH);
#undef LW
    for (int h = 0; h < SH; ++h) L->A[h] = -expf(L->A_log[h]);
    if (WW(s, arc_namef("trunk.film.%d.weight", l), 2 * DM * DM, &t->film_w[l])) return -1;
    if (!(t->film_b[l] = W(s, arc_namef("trunk.film.%d.bias", l), 2 * DM))) return -1;
    if (!(t->mlp_n[l] = W(s, arc_namef("trunk.mlps.%d.0.weight", l), DM))) return -1;
    if (WW(s, arc_namef("trunk.mlps.%d.1.weight", l), 2 * DM * DM, &t->mlp1_w[l])) return -1;
    if (!(t->mlp1_b[l] = W(s, arc_namef("trunk.mlps.%d.1.bias", l), 2 * DM))) return -1;
    if (WW(s, arc_namef("trunk.mlps.%d.4.weight", l), DM * 2 * DM, &t->mlp2_w[l])) return -1;
    if (!(t->mlp2_b[l] = W(s, arc_namef("trunk.mlps.%d.4.bias", l), DM))) return -1;
  }
  const arc_json* dl = arc_json_get(arc_json_get(arc_json_get(e->cfg, "model"), "trunk"), "delays");
  t->n_delays = dl ? dl->n : 0;
  for (int i = 0; i < t->n_delays && i < 8; ++i) t->delays[i] = (int)dl->items[i].num;
  /* tag head: keys = P(E) + r */
  int nt = t->ntags;
  const float *E, *P, *r;
  if (!(E = W(s, "head.E", (int64_t)nt * 384)) || !(P = W(s, "head.P.weight", DM * 384)) ||
      !(r = W(s, "head.r", (int64_t)nt * DM)))
    return -1;
  t->K = arc_malloc(sizeof(float) * (size_t)nt * DM);
  if (!t->K) return -1;
  arc_gemm_nt(t->pool, nt, DM, 384, E, 384, P, 384, t->K, DM, NULL, 0);
  for (size_t i = 0; i < (size_t)nt * DM; ++i) t->K[i] += r[i];
  G(hb, "head.b", nt);
  for (int q = 0; q < NQ; ++q) {
    if (!(t->bq[q] = W(s, arc_namef("head.bq.%s", QUERIES[q]), nt))) return -1;
    if (WW(s, arc_namef("head.Q.%s.weight", QUERIES[q]), DM * DM, &t->Qw[q])) return -1;
    if (!(t->Qb[q] = W(s, arc_namef("head.Q.%s.bias", QUERIES[q]), DM))) return -1;
  }
  if (WW(s, "head.pool_in.utt.0.weight", DM * DM, &t->pool_w0)) return -1;
  G(pool_b0, "head.pool_in.utt.0.bias", DM);
  if (WW(s, "head.pool_in.utt.2.weight", DM * DM, &t->pool_w2)) return -1;
  if (WW(s, "head.pool_in.speaker.0.weight", DM * DM, &t->spool_w0) || WW(s, "head.pool_in.speaker.2.weight", DM * DM, &t->spool_w2) ||
      WW(s, "head.pool_in.item.0.weight", DM * DM, &t->ipool_w0) || WW(s, "head.pool_in.item.2.weight", DM * DM, &t->ipool_w2))
    return -1;
  if (!(t->spool_b0 = W(s, "head.pool_in.speaker.0.bias", DM)) || !(t->spool_b2 = W(s, "head.pool_in.speaker.2.bias", DM)) ||
      !(t->ipool_b0 = W(s, "head.pool_in.item.0.bias", DM)) || !(t->ipool_b2 = W(s, "head.pool_in.item.2.bias", DM)))
    return -1;
  G(pool_b2, "head.pool_in.utt.2.bias", DM);
  int64_t sh[8];
  t->n_slot = arc_store_shape(s, "diar.slot.weight", sh) ? (int)sh[0] : 0;
  if (WW(s, "diar.slot.weight", (int64_t)t->n_slot * DM, &t->slot_w)) return -1;
  G(slot_b, "diar.slot.bias", t->n_slot);
  if (arc_store_has(s, "diar.slot_ext.weight")) {
    arc_store_shape(s, "diar.slot_ext.weight", sh);
    t->n_slotx = (int)sh[0];
    if (WW(s, "diar.slot_ext.weight", (int64_t)t->n_slotx * DM, &t->slotx_w)) return -1;
    G(slotx_b, "diar.slot_ext.bias", t->n_slotx);
  }
#undef G
  /* aux heads (heads.safetensors) load on first use: Arc's deployment notes, lazy optional heads */
  t->dd_thr = (float)arc_json_num(arc_json_get(arc_json_get(e->cfg, "device_directed"), "threshold"), 0.5);
  const arc_json* ae = arc_json_get(e->cfg, "audio_events");
  t->ev_tap = (int)arc_json_num(arc_json_get(ae, "tap"), 2);
  for (int i = 0; i < NEV; ++i) {
    t->ev_thr[i] = (float)arc_json_num(arc_json_get(arc_json_get(ae, "thresholds"), EV_LABELS[i]), 0.5);
    const char* st = arc_json_str(arc_json_get(arc_json_get(ae, "status"), EV_LABELS[i]), "experimental");
    t->ev_on[i] = !strcmp(st, "stable") || !strcmp(st, "beta");   /* events='default' */
  }
  /* readout calibration (config.json): priors, emit mask, statuses */
  const arc_json* la = arc_json_get(arc_json_get(e->cfg, "prior"), "la");
  const arc_json* em = arc_json_get(arc_json_get(e->cfg, "emit"), "mask");
  for (int d = 0; d < NDOM; ++d)
    for (int i = 0; i < nt; ++i) {
      t->la[d][i] = la ? (float)arc_json_num(arc_json_at(arc_json_at(la, d), i), 0) : 0.0f;
      t->emit[d][i] = em ? arc_json_num(arc_json_at(arc_json_at(em, d), i), 0) != 0 : 1;
    }
  t->tag_status = arc_json_get(e->cfg, "tag_status");
  t->wc = arc_wordch_load(s);
  t->link = arc_link_load(e, s);
  return 0;
}

void arc_engine_set_pool(arc_engine* e, arc_pool* p) {
  if (e->trunk) e->trunk->pool = p;
}

void arc_engine_free_trunk(arc_engine* e) {
  struct arc_trunk* t = e ? e->trunk : NULL;
  if (!t) return;
  for (int i = 0; i < t->ntags; ++i) arc_free(t->ids[i]);
  for (int i = 0; i < t->ngroups; ++i) arc_free(t->groups[i]);
  arc_free(t->K);
  for (int d = 0; d < NDOM; ++d)
    for (int k = 0; k < 8; ++k) arc_free(t->film_gb[d][k]);
  arc_free(t->mem);
  arc_wordch_free(t->wc);
  arc_link_free(t->link);
  arc_store_close(t->st);
  arc_store_close(t->aux);
  arc_free(t);
  e->trunk = NULL;
}

/* ------------------------------------------------------------------ small ops */

/* the device-directed and event heads, the first time they are asked for */
static void load_aux(struct arc_trunk* t) {
  if (!t->aux || t->dd_fc_w) return;
  t->dd_ln_w = W(t->aux, "device_directed.ln.weight", DIN);
  t->dd_ln_b = W(t->aux, "device_directed.ln.bias", DIN);
  t->dd_fc_w = W(t->aux, "device_directed.fc.weight", DIN);
  t->dd_fc_b = W(t->aux, "device_directed.fc.bias", 1);
  t->ev_ln_w = W(t->aux, "events.ln.weight", 2 * DIN);
  t->ev_ln_b = W(t->aux, "events.ln.bias", 2 * DIN);
  t->ev_c1_w = W(t->aux, "events.c1.weight", 256 * 2 * DIN * 5);
  t->ev_c1_b = W(t->aux, "events.c1.bias", 256);
  t->ev_c2_w = W(t->aux, "events.c2.weight", NEV * 256 * 3);
  t->ev_c2_b = W(t->aux, "events.c2.bias", NEV);
}

static float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
static float silu(float x) { return x / (1.0f + expf(-x)); }
static float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }
static float gelu1(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }

/* torch RMSNorm with eps=None: finfo(float32).eps */
static void rmsnorm(float* x, int n, const float* w) {
  float ms = 0;
  for (int i = 0; i < n; ++i) ms += x[i] * x[i];
  float inv = 1.0f / sqrtf(ms / n + FLT_EPSILON);
  for (int i = 0; i < n; ++i) x[i] = x[i] * inv * w[i];
}

static void lin(arc_pool* p, const float* x, int T, int K, const arc_w* Wt, int O, const float* b, float* y) {
  if (Wt->packed) arc_q4_linear(p, x, T, K, &Wt->q, b, y, O);
  else arc_gemm_nt(p, T, O, K, x, K, Wt->f, K, y, O, b, 0);
}

void arc_w_linear(arc_pool* p, const float* x, int T, int K, const arc_w* Wt, int O, const float* b, float* y) {
  lin(p, x, T, K, Wt, O, b, y);
}

int arc_w_load(arc_store* s, const char* name, int64_t n, arc_w* w) { return WW(s, name, n, w); }

/* ------------------------------------------------------------------ features */

/* pool10: encoder frame g (20 ms hop, 245 ms receptive field) -> 100 ms frame ceil((g*320 + 3920)/16000*10) - 1;
 * mean and max of the frames that land in each 100 ms frame (an empty one repeats the previous), via float16. */
static float* pool10(const float* enc, int n, int* N10) {
  if (n <= 0) {
    *N10 = 0;
    return arc_malloc(sizeof(float));
  }
  int* f = arc_malloc(sizeof(int) * (size_t)n);
  int mx = 0;
  for (int g = 0; g < n; ++g) {
    f[g] = (int)ceil(((double)g * 320 + 3920) / 16000.0 * 10 - 1e-9) - 1;
    if (f[g] > mx) mx = f[g];
  }
  int N = mx + 1;
  float* out = arc_calloc(sizeof(float) * (size_t)N * 2 * ARC_MOON_D);
  float* cnt = arc_calloc(sizeof(float) * (size_t)N);
  float* mxv = arc_malloc(sizeof(float) * (size_t)N * ARC_MOON_D);
  for (size_t i = 0; i < (size_t)N * ARC_MOON_D; ++i) mxv[i] = -INFINITY;
  for (int g = 0; g < n; ++g) {
    float* s = out + (size_t)f[g] * 2 * ARC_MOON_D;
    float* m = mxv + (size_t)f[g] * ARC_MOON_D;
    const float* e = enc + (size_t)g * ARC_MOON_D;
    for (int c = 0; c < ARC_MOON_D; ++c) {
      s[c] += e[c];
      if (e[c] > m[c]) m[c] = e[c];
    }
    cnt[f[g]] += 1;
  }
  for (int i = 0; i < N; ++i) {
    float* s = out + (size_t)i * 2 * ARC_MOON_D;
    float* m = mxv + (size_t)i * ARC_MOON_D;
    if (cnt[i] == 0) {
      if (i) memcpy(s, s - 2 * ARC_MOON_D, sizeof(float) * 2 * ARC_MOON_D);
      else memset(s, 0, sizeof(float) * 2 * ARC_MOON_D);
      continue;
    }
    for (int c = 0; c < ARC_MOON_D; ++c) {
      s[c] = arc_round_half(s[c] / cnt[i]);
      s[ARC_MOON_D + c] = arc_round_half(m[c]);
    }
  }
  arc_free(f);
  arc_free(cnt);
  arc_free(mxv);
  *N10 = N;
  return out;
}

/* ------------------------------------------------------------------ trunk */

typedef struct {
  float h[SH][SN][SP];
  float pbx[SH][SN][SP];
  int started;
} ssm_state;

struct ssm_mem {
  ssm_state L[NLY];
};

/* x [N, 640] (already delay-shifted) -> h [N, 512]. mem: the SSM state to go on from and update (NULL: from zero) */
static float* trunk_run_mem(struct arc_trunk* t, const float* x, int N, int dom, int di, struct ssm_mem* mem);
static float* trunk_run(struct arc_trunk* t, const float* x, int N, int dom, int di, int carry) {
  return trunk_run_mem(t, x, N, dom, di, carry && t->mem_on && t->mem ? t->mem : NULL);
}
static float* trunk_run_mem(struct arc_trunk* t, const float* x, int N, int dom, int di, struct ssm_mem* mem) {
  int carry = mem != NULL;
  float* out = arc_malloc(sizeof(float) * (size_t)N * DM);
  float* hn = arc_malloc(sizeof(float) * (size_t)N * DIN);
  float* h = arc_malloc(sizeof(float) * (size_t)N * DM);
  float* nu = arc_malloc(sizeof(float) * (size_t)N * DM);
  float* tmp = arc_malloc(sizeof(float) * (size_t)N * 2 * DM);
  float* Bm = arc_malloc(sizeof(float) * (size_t)N * SH * SN);
  float* Cm = arc_malloc(sizeof(float) * (size_t)N * SH * SN);
  float* xx = arc_malloc(sizeof(float) * (size_t)N * DM);
  float* dt = arc_malloc(sizeof(float) * (size_t)N * SH);
  float* lam = arc_malloc(sizeof(float) * (size_t)N * SH);
  float* th = arc_malloc(sizeof(float) * (size_t)N * SH * SN / 2);
  float* y = arc_malloc(sizeof(float) * (size_t)N * DM);
  float* gate = arc_malloc(sizeof(float) * (size_t)N * DM);
  ssm_state* S = arc_malloc(sizeof(ssm_state));
  float e[DM], gb[2 * DM];
  if (!out || !hn || !h || !nu || !tmp || !Bm || !Cm || !xx || !dt || !lam || !th || !y || !gate || !S) {
    arc_free(out);
    out = NULL;
    goto done;
  }
  /* _cond */
  memcpy(hn, x, sizeof(float) * (size_t)N * DIN);
  arc_layernorm(hn, N, DIN, DIN, t->in_norm_w, t->in_norm_b, 1e-5f);
  lin(t->pool, hn, N, DIN, &t->inp_w, DM, t->inp_b, h);
  for (int c = 0; c < DM; ++c) {
    float add = t->dom_emb[dom * DM + c] + t->delay_emb[di * DM + c];
    for (int i = 0; i < N; ++i) h[(size_t)i * DM + c] += add;
    e[c] = t->dom_emb[dom * DM + c] + t->delay_film[di * DM + c];
  }
  float** fc = di >= 0 && di < 8 && dom >= 0 && dom < NDOM ? &t->film_gb[dom][di] : NULL;
  if (fc && !*fc && (*fc = arc_malloc(sizeof(float) * NLY * 2 * DM)))
    for (int l = 0; l < NLY; ++l) lin(t->pool, e, 1, DM, &t->film_w[l], 2 * DM, t->film_b[l], *fc + (size_t)l * 2 * DM);
  for (int l = 0; l < NLY; ++l) {
    const ssm_layer* L = &t->L[l];
    if (fc && *fc) memcpy(gb, *fc + (size_t)l * 2 * DM, sizeof gb);
    else lin(t->pool, e, 1, DM, &t->film_w[l], 2 * DM, t->film_b[l], gb);
    for (int i = 0; i < N; ++i)
      for (int c = 0; c < DM; ++c) {
        float* v = &h[(size_t)i * DM + c];
        *v = *v * (1.0f + gb[c]) + gb[DM + c];
      }
    /* the SSM layer: coefficients for all frames, then the recurrence */
    memcpy(nu, h, sizeof(float) * (size_t)N * DM);
    for (int i = 0; i < N; ++i) rmsnorm(nu + (size_t)i * DM, DM, L->norm);
    lin(t->pool, nu, N, DM, &L->dt_w, SH, L->dt_b, dt);
    lin(t->pool, nu, N, DM, &L->lam_w, SH, L->lam_b, lam);
    lin(t->pool, nu, N, DM, &L->B_w, SH * SN, L->B_b, Bm);
    lin(t->pool, nu, N, DM, &L->C_w, SH * SN, L->C_b, Cm);
    lin(t->pool, nu, N, DM, &L->x_w, DM, L->x_b, xx);
    lin(t->pool, nu, N, DM, &L->th_w, SH * SN / 2, L->th_b, th);
    lin(t->pool, nu, N, DM, &L->in_w, DM, L->in_b, gate);
    if (carry) memcpy(S, &mem->L[l], sizeof *S);   /* go on from the last call's state */
    else memset(S, 0, sizeof *S);
    for (int i = 0; i < N; ++i) {
      for (int hh = 0; hh < SH; ++hh) {
        float d = softplus(dt[i * SH + hh]);
        float alpha = expf(d * L->A[hh]);
        float lm = sigm(lam[i * SH + hh]);
        float gamma = lm * d, beta = (1.0f - lm) * d * alpha;
        float* B = Bm + ((size_t)i * SH + hh) * SN;
        float* C = Cm + ((size_t)i * SH + hh) * SN;
        rmsnorm(B, SN, L->Bn);
        rmsnorm(C, SN, L->Cn);
        for (int n = 0; n < SN; ++n) {
          B[n] += L->Bbias[hh * SN + n];
          C[n] += L->Cbias[hh * SN + n];
        }
        const float* xv = xx + (size_t)i * DM + hh * SP;
        const float* tv = th + ((size_t)i * SH + hh) * (SN / 2);
        float(*hs)[SP] = S->h[hh];
        float(*pb)[SP] = S->pbx[hh];
        if (S->started) {
          for (int n2 = 0; n2 < SN / 2; ++n2) {
            float ang = tv[n2] * d, c = cosf(ang), sn = sinf(ang);
            for (int p = 0; p < SP; ++p) {
              float a = alpha * hs[2 * n2][p] + beta * pb[2 * n2][p];
              float b = alpha * hs[2 * n2 + 1][p] + beta * pb[2 * n2 + 1][p];
              hs[2 * n2][p] = c * a - sn * b;
              hs[2 * n2 + 1][p] = sn * a + c * b;
            }
          }
          for (int n = 0; n < SN; ++n)
            for (int p = 0; p < SP; ++p) hs[n][p] += gamma * (B[n] * xv[p]);
        } else {
          for (int n = 0; n < SN; ++n)
            for (int p = 0; p < SP; ++p) hs[n][p] = gamma * (B[n] * xv[p]);
        }
        for (int n = 0; n < SN; ++n)
          for (int p = 0; p < SP; ++p) pb[n][p] = B[n] * xv[p];
        float* yo = y + (size_t)i * DM + hh * SP;
        for (int p = 0; p < SP; ++p) {
          float sacc = 0;
          for (int n = 0; n < SN; ++n) sacc += C[n] * hs[n][p];
          yo[p] = sacc;
        }
      }
      S->started = 1;
    }
    if (carry) memcpy(&mem->L[l], S, sizeof *S);
    for (size_t k = 0; k < (size_t)N * DM; ++k) y[k] *= silu(gate[k]);
    lin(t->pool, y, N, DM, &L->out_w, DM, L->out_b, tmp);
    for (size_t k = 0; k < (size_t)N * DM; ++k) h[k] += tmp[k];
    /* MLP */
    memcpy(nu, h, sizeof(float) * (size_t)N * DM);
    for (int i = 0; i < N; ++i) rmsnorm(nu + (size_t)i * DM, DM, t->mlp_n[l]);
    lin(t->pool, nu, N, DM, &t->mlp1_w[l], 2 * DM, t->mlp1_b[l], tmp);
    arc_gelu(tmp, (size_t)N * 2 * DM);
    lin(t->pool, tmp, N, 2 * DM, &t->mlp2_w[l], DM, t->mlp2_b[l], nu);
    for (size_t k = 0; k < (size_t)N * DM; ++k) h[k] += nu[k];
  }
  /* domain adapter, out norm */
  {
    const float* Dn = t->adapt_down + (size_t)dom * DM * 64;   /* [512, 64] */
    const float* Up = t->adapt_up + (size_t)dom * 64 * DM;     /* [64, 512] */
    float r[64];
    for (int i = 0; i < N; ++i) {
      const float* hi = h + (size_t)i * DM;
      for (int k = 0; k < 64; ++k) {
        float s2 = 0;
        for (int c = 0; c < DM; ++c) s2 += hi[c] * Dn[(size_t)c * 64 + k];
        r[k] = gelu1(s2);
      }
      float* o = out + (size_t)i * DM;
      for (int c = 0; c < DM; ++c) {
        float s2 = 0;
        for (int k = 0; k < 64; ++k) s2 += r[k] * Up[(size_t)k * DM + c];
        o[c] = hi[c] + s2;
      }
      rmsnorm(o, DM, t->out_norm);
    }
  }
done:
  arc_free(hn);
  arc_free(h);
  arc_free(nu);
  arc_free(tmp);
  arc_free(Bm);
  arc_free(Cm);
  arc_free(xx);
  arc_free(dt);
  arc_free(lam);
  arc_free(th);
  arc_free(y);
  arc_free(gate);
  arc_free(S);
  return out;
}

/* ------------------------------------------------------------------ readout */

/* logits of query q for rows z [T, 512]: Q(z) K[c]^T + b + bq, invalid columns NEG */
static void logits_q(struct arc_trunk* t, int q, const float* z, int T, int dom, float* out /* [T, ncols] */) {
  int nc = t->ncols[q];
  float* qz = arc_malloc(sizeof(float) * (size_t)T * DM);
  lin(t->pool, z, T, DM, &t->Qw[q], DM, t->Qb[q], qz);
  for (int i = 0; i < T; ++i)
    for (int j = 0; j < nc; ++j) {
      int c = t->cols[q][j];
      float s = 0;
      const float* k = t->K + (size_t)c * DM;
      const float* a = qz + (size_t)i * DM;
      for (int d = 0; d < DM; ++d) s += a[d] * k[d];
      s += t->hb[c] + t->bq[q][c];
      out[(size_t)i * nc + j] = t->valid[dom][c] ? s : NEG;
    }
  arc_free(qz);
}

/* Readout.probs: sigmoid(z + prior), with a softmax inside each exclusive group */
static void probs_q(struct arc_trunk* t, int q, float* z, int T, int dom) {
  int nc = t->ncols[q];
  for (int i = 0; i < T; ++i) {
    float* r = z + (size_t)i * nc;
    float zz[MAXTAG], p[MAXTAG];
    int valid[MAXTAG];
    for (int j = 0; j < nc; ++j) {
      int c = t->cols[q][j];
      valid[j] = r[j] > NEG / 2;
      zz[j] = r[j] + t->la[dom][c];
    }
    for (int j = 0; j < nc; ++j) p[j] = (float)(1.0 / (1.0 + exp(-(double)(valid[j] ? zz[j] : -50.0f))));
    for (int g = 0; g < t->ngroups && q != Q_SPAN_END; ++g) {   /* span_end has no group softmax (Readout.grp_loc) */
      double mx = -INFINITY;
      int any = 0;
      for (int j = 0; j < nc; ++j)
        if (t->group_of[t->cols[q][j]] == g) {
          any = 1;
          if (valid[j] && zz[j] > mx) mx = zz[j];
        }
      if (!any) continue;
      double s = 0;
      for (int j = 0; j < nc; ++j)
        if (t->group_of[t->cols[q][j]] == g && valid[j]) s += exp(zz[j] - mx);
      for (int j = 0; j < nc; ++j)
        if (t->group_of[t->cols[q][j]] == g)
          p[j] = isfinite(mx) && valid[j] ? (float)(exp(zz[j] - mx) / (s > 1e-30 ? s : 1e-30)) : 0.0f;
    }
    memcpy(r, p, sizeof(float) * (size_t)nc);
  }
}

/* status of tag i in domain dom: 1 shipped (output), 0 not */
static int shipped(struct arc_trunk* t, int i, int dom) {
  if (!t->trained[i] || !t->valid[dom][i]) return 0;
  const char* st = arc_json_str(arc_json_get(t->tag_status, t->ids[i]), "shipped");
  if (!strncmp(st, "not_shipped", 11)) return 0;
  if (!t->emit[dom][i]) return 0;   /* experimental:not_supported_in_domain */
  return !strcmp(st, "shipped");
}

typedef struct {
  float start, end;
  float p[MAXTAG];   /* utt probs at the last speech frame */
} trunk_utt;

/* the engine's speaker / utterance state machine and utterance readout over the trunk output */
static int readout(struct arc_trunk* t, const float* Ha, const float* H, int N, int dom, trunk_utt** out_utts) {
  int ncf = t->ncols[Q_FRAME], ncu = t->ncols[Q_UTT];
  float* pf = arc_malloc(sizeof(float) * (size_t)N * ncf);
  logits_q(t, Q_FRAME, H, N, dom, pf);
  probs_q(t, Q_FRAME, pf, N, dom);
  int sp_col = 0;
  for (int j = 0; j < ncf; ++j)
    if (!strcmp(t->ids[t->cols[Q_FRAME][j]], "meta.speech")) sp_col = j;
  int ns = t->n_slot + t->n_slotx;
  float* slot = arc_malloc(sizeof(float) * (size_t)N * (ns ? ns : 1));
  float* sl1 = arc_malloc(sizeof(float) * (size_t)N * (t->n_slot ? t->n_slot : 1));
  lin(t->pool, Ha, N, DM, &t->slot_w, t->n_slot, t->slot_b, sl1);
  float* sl2 = NULL;
  if (t->n_slotx) {
    sl2 = arc_malloc(sizeof(float) * (size_t)N * t->n_slotx);
    lin(t->pool, Ha, N, DM, &t->slotx_w, t->n_slotx, t->slotx_b, sl2);
  }
  for (int i = 0; i < N; ++i) {
    memcpy(slot + (size_t)i * ns, sl1 + (size_t)i * t->n_slot, sizeof(float) * t->n_slot);
    if (sl2) memcpy(slot + (size_t)i * ns + t->n_slot, sl2 + (size_t)i * t->n_slotx, sizeof(float) * t->n_slotx);
  }
  arc_free(sl1);
  arc_free(sl2);
  /* _spk_utt */
  unsigned char* ust = arc_malloc((size_t)N + 1);
  unsigned char* ucl = arc_malloc((size_t)N + 1);
  int spk_cur = -1, spk_pend = -2, spk_n = 0, u_in = 0, u_sil = 0, u_spk = -1;
  for (int i = 0; i < N; ++i) {
    const float* z = slot + (size_t)i * ns;
    int am = 0;
    for (int k = 1; k < ns; ++k)
      if (z[k] > z[am]) am = k;
    int s = am - 1;
    if (s != spk_cur) {
      if (s == spk_pend) spk_n++;
      else {
        spk_pend = s;
        spk_n = 1;
      }
      if (spk_n >= 5 || (spk_cur == -1 && s >= 0 && spk_n >= 2)) {
        spk_cur = s;
        spk_pend = -2;
        spk_n = 0;
      }
    } else {
      spk_pend = -2;
      spk_n = 0;
    }
    int speech = pf[(size_t)i * ncf + sp_col] >= 0.5f;
    int sp = speech ? spk_cur : -1, start = 0, close = 0;
    if (speech) {
      if (!u_in || (sp >= 0 && u_spk >= 0 && sp != u_spk)) {
        close = u_in;
        start = 1;
        u_in = 1;
        u_spk = sp;
      } else if (sp >= 0 && u_spk < 0) u_spk = sp;
      u_sil = 0;
    } else if (u_in) {
      u_sil++;
      if (u_sil >= 5) {
        close = 1;
        u_in = 0;
        u_spk = -1;
      }
    }
    ust[i] = (unsigned char)start;
    ucl[i] = (unsigned char)close;
  }
  /* utt pooling: running mean of z = h + MLP(h) since the utterance start (double sums) */
  float* z = arc_malloc(sizeof(float) * (size_t)N * DM);
  float* z1 = arc_malloc(sizeof(float) * (size_t)N * DM);
  lin(t->pool, H, N, DM, &t->pool_w0, DM, t->pool_b0, z1);
  arc_gelu(z1, (size_t)N * DM);
  lin(t->pool, z1, N, DM, &t->pool_w2, DM, t->pool_b2, z);
  for (size_t k = 0; k < (size_t)N * DM; ++k) z[k] += H[k];
  double* acc = arc_calloc(sizeof(double) * DM);
  double cnt = 0;
  for (int i = 0; i < N; ++i) {
    float* zi = z + (size_t)i * DM;
    if (ust[i]) {
      for (int c = 0; c < DM; ++c) acc[c] = zi[c];
      cnt = 1;
    } else {
      for (int c = 0; c < DM; ++c) acc[c] += zi[c];
      cnt += 1;
    }
    for (int c = 0; c < DM; ++c) zi[c] = (float)(acc[c] / (cnt > 1 ? cnt : 1));
  }
  arc_free(acc);
  float* pu = arc_malloc(sizeof(float) * (size_t)N * (ncu ? ncu : 1));
  logits_q(t, Q_UTT, z, N, dom, pu);
  probs_q(t, Q_UTT, pu, N, dom);
  /* _decode_frame / _close_utt for utterances (and finish) */
  int cap = 16, nu = 0, in_utt = 0, utt0 = -1, utt_last = -1, have_p = 0;
  trunk_utt* U = arc_malloc(sizeof(trunk_utt) * (size_t)cap);
  float cur_p[MAXTAG];
  for (int i = 0; i <= N; ++i) {
    int close = i < N ? ucl[i] : in_utt;
    if (close) {
      if (utt0 >= 0 && have_p) {
        if (nu == cap) {
          trunk_utt* nU = arc_malloc(sizeof(trunk_utt) * (size_t)cap * 2);
          memcpy(nU, U, sizeof(trunk_utt) * (size_t)nu);
          arc_free(U);
          U = nU;
          cap *= 2;
        }
        U[nu].start = utt0 / 10.0f;
        U[nu].end = (utt_last + 1) / 10.0f;
        memcpy(U[nu].p, cur_p, sizeof(float) * (size_t)ncu);
        nu++;
      }
      in_utt = 0;
      utt0 = -1;
      have_p = 0;
    }
    if (i == N) break;
    if (ust[i]) {
      in_utt = 1;
      utt0 = i;
    }
    if (pf[(size_t)i * ncf + sp_col] >= 0.5f && in_utt) {
      utt_last = i;
      memcpy(cur_p, pu + (size_t)i * ncu, sizeof(float) * (size_t)ncu);
      have_p = 1;
    }
  }
  arc_free(pf);
  arc_free(slot);
  arc_free(ust);
  arc_free(ucl);
  arc_free(z);
  arc_free(z1);
  arc_free(pu);
  *out_utts = U;
  return nu;
}

/* ------------------------------------------------------------------ for arc_link.c */

int arc_engine_tag_index(arc_engine* e, const char* id) {
  struct arc_trunk* t = e->trunk;
  for (int i = 0; i < t->ntags; ++i)
    if (!strcmp(t->ids[i], id)) return i;
  return -1;
}

const char* arc_engine_tag_name(arc_engine* e, int i) { return e->trunk->ids[i]; }
const float* arc_engine_tag_key(arc_engine* e, int i) { return e->trunk->K + (size_t)i * DM; }

const char* arc_engine_tag_group(arc_engine* e, int i) {
  int g = e->trunk->group_of[i];
  return g >= 0 ? e->trunk->groups[g] : NULL;
}

float* arc_engine_span_probs(arc_engine* e, const float* H, int T, int dom, int* ncols, const int** cols) {
  struct arc_trunk* t = e->trunk;
  int nc = t->ncols[Q_SPAN_IN];
  float* p = arc_malloc(sizeof(float) * (size_t)(T > 0 ? T : 1) * (nc ? nc : 1));
  if (!p) return NULL;
  if (T > 0) {
    logits_q(t, Q_SPAN_IN, H, T, dom, p);
    probs_q(t, Q_SPAN_IN, p, T, dom);
  }
  *ncols = nc;
  *cols = t->cols[Q_SPAN_IN];
  return p;
}

int arc_engine_set_names(arc_engine* e, const arc_name_entry* names, int n) {
  if (!e->trunk || !e->trunk->link) {
    arc_seterr("these weights have no entity heads (names cannot be linked)");
    return n ? -1 : 0;
  }
  return arc_link_set_names(e->trunk->link, names, n);
}

/* ------------------------------------------------------------------ run */

static int dom_index(const char* d) {
  for (int i = 0; i < NDOM; ++i)
    if (d && !strcmp(DOMAINS[i], d)) return i;
  return 15;
}

int arc_engine_run(arc_engine* e, const float* pcm, size_t n, const arc_options* o, arc_result_build* R) {
  struct arc_trunk* t = e->trunk;
  int dom = dom_index(o->domain);
  int nn = (int)(n / 80 * 80);
  if (nn < 80) return 0;
  load_aux(t);
  int Te;
  float* tap = NULL;
  float* H = arc_moon_encode_ex(e->moon, pcm, nn, &Te, t->ev_tap, o->events ? &tap : NULL);
  if (!H) return -1;
  int N;
  float* X = pool10(H, Te, &N);
  arc_free(H);
  /* the device-directed head over each transcript utterance's frames */
  if (t->dd_fc_w)
    for (int i = 0; i < R->n; ++i) {
      int a = R->u[i].f0, b = R->u[i].f1 + 1;
      if (a < 0) a = 0;
      if (b > N) b = N;
      if (b <= a) continue;
      double s = 0;
      float row[DIN];
      for (int f = a; f < b; ++f) {
        memcpy(row, X + (size_t)f * DIN, sizeof row);
        arc_layernorm(row, 1, DIN, DIN, t->dd_ln_w, t->dd_ln_b, 1e-5f);
        float v = t->dd_fc_b[0];
        for (int c = 0; c < DIN; ++c) v += row[c] * t->dd_fc_w[c];
        s += v;
      }
      R->u[i].dir_head = sigm((float)(s / (b - a)));
    }
  /* sound events: LN -> conv k5 -> GELU -> conv k3 over [features, tapped features] */
  if (o->events && t->ev_c1_w && tap) {
    int Nt;
    float* XT = pool10(tap, Te, &Nt);
    int M = N < Nt ? N : Nt;
    float* xe = arc_malloc(sizeof(float) * (size_t)M * 2 * DIN);
    for (int f = 0; f < M; ++f) {
      memcpy(xe + (size_t)f * 2 * DIN, X + (size_t)f * DIN, sizeof(float) * DIN);
      memcpy(xe + (size_t)f * 2 * DIN + DIN, XT + (size_t)f * DIN, sizeof(float) * DIN);
    }
    arc_layernorm(xe, M, 2 * DIN, 2 * DIN, t->ev_ln_w, t->ev_ln_b, 1e-5f);
    float* h1 = arc_malloc(sizeof(float) * (size_t)M * 256);
    float* pr = arc_malloc(sizeof(float) * (size_t)M * NEV);
    arc_mat c1 = {.type = ARC_W_F32, .O = 256, .K = 2 * DIN * 5, .w = t->ev_c1_w};
    arc_mat c2 = {.type = ARC_W_F32, .O = NEV, .K = 256 * 3, .w = t->ev_c2_w};
    arc_conv1d(t->pool, xe, M, 2 * DIN, 2 * DIN, &c1, 5, 1, 1, 2, 2, t->ev_c1_b, h1, 256);
    arc_gelu(h1, (size_t)M * 256);
    arc_conv1d(t->pool, h1, M, 256, 256, &c2, 3, 1, 1, 1, 1, t->ev_c2_b, pr, NEV);
    /* EventTracker: open while p >= threshold, closed when it drops; at the end the open ones in opening order */
    float st[NEV], en[NEV], pm[NEV];
    int open[NEV], order[NEV], no = 0;
    memset(open, 0, sizeof open);
    for (int f = 0; f < M; ++f)
      for (int j = 0; j < NEV; ++j) {
        if (!t->ev_on[j]) continue;
        float p = sigm(pr[(size_t)f * NEV + j]);
        if (p >= t->ev_thr[j]) {
          if (!open[j]) {
            open[j] = 1;
            st[j] = f / 10.0f;
            en[j] = (f + 1) / 10.0f;
            pm[j] = p;
            order[no++] = j;
          } else {
            en[j] = (f + 1) / 10.0f;
            if (p > pm[j]) pm[j] = p;
          }
        } else if (open[j]) {
          arc_result_add_event(R, EV_LABELS[j], st[j], en[j], pm[j]);
          open[j] = 0;
          for (int k = 0; k < no; ++k)
            if (order[k] == j) {
              memmove(order + k, order + k + 1, sizeof(int) * (size_t)(no - k - 1));
              no--;
              break;
            }
        }
      }
    for (int k = 0; k < no; ++k) arc_result_add_event(R, EV_LABELS[order[k]], st[order[k]], en[order[k]], pm[order[k]]);
    arc_free(XT);
    arc_free(xe);
    arc_free(h1);
    arc_free(pr);
  }
  arc_free(tap);
  /* the trunk, read out into utterances (delay 0.3 s: frames shifted, the last one repeated) */
  if (o->trunk || o->names || o->numbers) {
    int d = 3, di = 0;
    for (int k = 0; k < t->n_delays; ++k)
      if (t->delays[k] == d) di = k;
    float* xs = arc_malloc(sizeof(float) * (size_t)N * DIN);
    for (int f = 0; f < N; ++f) {
      int src = f + d < N ? f + d : N - 1;
      memcpy(xs + (size_t)f * DIN, X + (size_t)src * DIN, sizeof(float) * DIN);
    }
    float* Hh = trunk_run(t, xs, N, dom, di, 1);
    if (Hh && t->mem_on) t->mem_frames += N;
    /* the transcript's words (Arc's WordStore) and, with them, the word channel added to the trunk output */
    arc_wstore ws;
    arc_wstore_init(&ws, R);
    float* Hf = Hh;
    if (Hh && t->wc && ws.n) {
      Hf = arc_malloc(sizeof(float) * (size_t)N * DM);
      arc_wc_word* wcw = arc_malloc(sizeof(arc_wc_word) * (size_t)ws.n);
      for (int k = 0; k < ws.n; ++k) wcw[k] = (arc_wc_word){ws.w[k].w, ws.w[k].f0, ws.w[k].f1, 1};
      arc_wordch_fuse(t->wc, t->pool, e->tok, wcw, ws.n, Hh, N, d, 3000, Hf);
      arc_free(wcw);
    }
    const char* dump = getenv("ARC_DEBUG_FUSE");
    if (dump && Hh) {
      char pth[1024];
      snprintf(pth, sizeof pth, "%s_ha.f32", dump);
      FILE* f = fopen(pth, "wb");
      if (f) { fwrite(Hh, sizeof(float), (size_t)N * DM, f); fclose(f); }
      snprintf(pth, sizeof pth, "%s_hf.f32", dump);
      f = fopen(pth, "wb");
      if (f) { fwrite(Hf, sizeof(float), (size_t)N * DM, f); fclose(f); }
    }
    if (Hh && (o->names || o->numbers) && t->link) {
      arc_link_ctx lc = {e, t->pool, Hf, N, dom, d, o->names, o->numbers};
      if (arc_link_run(t->link, &lc, &ws, R)) {
        arc_wstore_free(&ws);
        if (Hf != Hh) arc_free(Hf);
        arc_free(Hh);
        arc_free(xs);
        arc_free(X);
        return -1;
      }
    }
    arc_wstore_free(&ws);
    trunk_utt* U = NULL;
    int nu = Hh && o->trunk ? readout(t, Hh, Hf, N, dom, &U) : 0;
    /* each transcript utterance takes the trunk utterance it overlaps most */
    int ncu = t->ncols[Q_UTT];
    for (int i = 0; i < R->n; ++i) {
      int best = -1;
      float bo = 0;
      for (int k = 0; k < nu; ++k) {
        float ov = fminf(R->u[i].end, U[k].end) - fmaxf(R->u[i].start, U[k].start);
        if (ov > bo) {
          bo = ov;
          best = k;
        }
      }
      if (best < 0) continue;
      const float* p = U[best].p;
      for (int j = 0; j < ncu; ++j) {
        int c = t->cols[Q_UTT][j];
        if (!shipped(t, c, dom)) continue;
        const char* id = t->ids[c];
        int g = t->group_of[c];
        if (g < 0) {
          if (!strcmp(id, "discourse.assistant.device_directed")) R->u[i].dd = p[j];
          else if (!strcmp(id, "discourse.assistant.command_complete")) R->u[i].cc = p[j];
        } else if (!strcmp(t->groups[g], "discourse.intent") && p[j] > R->u[i].intent_p) {
          R->u[i].intent = id;
          R->u[i].intent_p = p[j];
        } else if (!strcmp(t->groups[g], "discourse.act") && p[j] > R->u[i].act_p) {
          R->u[i].act = id;
          R->u[i].act_p = p[j];
        }
      }
    }
    arc_free(U);
    if (Hf != Hh) arc_free(Hf);
    arc_free(Hh);
    arc_free(xs);
  }
  arc_free(X);
  return 0;
}

/* ------------------------------------------------------------------ stages (parity) */

/* z = h + MLP(h) for a pooled query */
static float* pool_in(struct arc_trunk* t, const float* H, int N, const arc_w* w0, const float* b0, const arc_w* w2,
                      const float* b2) {
  float* z = arc_malloc(sizeof(float) * (size_t)N * DM);
  float* z1 = arc_malloc(sizeof(float) * (size_t)N * DM);
  lin(t->pool, H, N, DM, w0, DM, b0, z1);
  arc_gelu(z1, (size_t)N * DM);
  lin(t->pool, z1, N, DM, w2, DM, b2, z);
  for (size_t k = 0; k < (size_t)N * DM; ++k) z[k] += H[k];
  arc_free(z1);
  return z;
}

void arc_stages_free(arc_stages* s) {
  arc_free(s->X);
  arc_free(s->XT);
  arc_free(s->ev);
  for (int i = 0; i < s->nq; ++i) {
    arc_free(s->fr[i].p);
    arc_free((void*)s->fr[i].ids);
  }
  memset(s, 0, sizeof *s);
}

int arc_engine_stages(arc_engine* e, const float* pcm, size_t n, const char* domain, arc_stages* out) {
  struct arc_trunk* t = e->trunk;
  memset(out, 0, sizeof *out);
  load_aux(t);
  int dom = dom_index(domain);
  int nn = (int)(n / 80 * 80);
  if (nn < 80) return -1;
  int Te;
  float* tap = NULL;
  float* H = arc_moon_encode_ex(e->moon, pcm, nn, &Te, t->ev_tap, &tap);
  if (!H) return -1;
  int N, Nt;
  out->X = pool10(H, Te, &N);
  out->XT = pool10(tap, Te, &Nt);
  out->N = N;
  arc_free(H);
  arc_free(tap);
  /* event probabilities */
  if (t->ev_c1_w) {
    float* xe = arc_malloc(sizeof(float) * (size_t)N * 2 * DIN);
    for (int f = 0; f < N; ++f) {
      memcpy(xe + (size_t)f * 2 * DIN, out->X + (size_t)f * DIN, sizeof(float) * DIN);
      memcpy(xe + (size_t)f * 2 * DIN + DIN, out->XT + (size_t)(f < Nt ? f : Nt - 1) * DIN, sizeof(float) * DIN);
    }
    arc_layernorm(xe, N, 2 * DIN, 2 * DIN, t->ev_ln_w, t->ev_ln_b, 1e-5f);
    float* h1 = arc_malloc(sizeof(float) * (size_t)N * 256);
    out->ev = arc_malloc(sizeof(float) * (size_t)N * NEV);
    arc_mat c1 = {.type = ARC_W_F32, .O = 256, .K = 2 * DIN * 5, .w = t->ev_c1_w};
    arc_mat c2 = {.type = ARC_W_F32, .O = NEV, .K = 256 * 3, .w = t->ev_c2_w};
    arc_conv1d(t->pool, xe, N, 2 * DIN, 2 * DIN, &c1, 5, 1, 1, 2, 2, t->ev_c1_b, h1, 256);
    arc_gelu(h1, (size_t)N * 256);
    arc_conv1d(t->pool, h1, N, 256, 256, &c2, 3, 1, 1, 1, 1, t->ev_c2_b, out->ev, NEV);
    for (size_t k = 0; k < (size_t)N * NEV; ++k) out->ev[k] = sigm(out->ev[k]);
    arc_free(xe);
    arc_free(h1);
  }
  /* trunk on the delay-shifted features */
  int d = 3, di = 0;
  for (int k = 0; k < t->n_delays; ++k)
    if (t->delays[k] == d) di = k;
  float* xs = arc_malloc(sizeof(float) * (size_t)N * DIN);
  for (int f = 0; f < N; ++f) memcpy(xs + (size_t)f * DIN, out->X + (size_t)(f + d < N ? f + d : N - 1) * DIN, sizeof(float) * DIN);
  float* Hh = trunk_run(t, xs, N, dom, di, 0);
  arc_free(xs);
  if (!Hh) return -1;
  /* frame probabilities, slots, the speaker / utterance state machine (as readout) */
  int ncf = t->ncols[Q_FRAME];
  float* pf = arc_malloc(sizeof(float) * (size_t)N * ncf);
  logits_q(t, Q_FRAME, Hh, N, dom, pf);
  probs_q(t, Q_FRAME, pf, N, dom);
  int sp_col = 0;
  for (int j = 0; j < ncf; ++j)
    if (!strcmp(t->ids[t->cols[Q_FRAME][j]], "meta.speech")) sp_col = j;
  int ns = t->n_slot + t->n_slotx;
  float* slot = arc_malloc(sizeof(float) * (size_t)N * ns);
  float* sl1 = arc_malloc(sizeof(float) * (size_t)N * t->n_slot);
  lin(t->pool, Hh, N, DM, &t->slot_w, t->n_slot, t->slot_b, sl1);
  float* sl2 = t->n_slotx ? arc_malloc(sizeof(float) * (size_t)N * t->n_slotx) : NULL;
  if (sl2) lin(t->pool, Hh, N, DM, &t->slotx_w, t->n_slotx, t->slotx_b, sl2);
  for (int i = 0; i < N; ++i) {
    memcpy(slot + (size_t)i * ns, sl1 + (size_t)i * t->n_slot, sizeof(float) * t->n_slot);
    if (sl2) memcpy(slot + (size_t)i * ns + t->n_slot, sl2 + (size_t)i * t->n_slotx, sizeof(float) * t->n_slotx);
  }
  arc_free(sl1);
  arc_free(sl2);
  int* spk = arc_malloc(sizeof(int) * (size_t)N);
  unsigned char* ust = arc_malloc((size_t)N + 1);
  int spk_cur = -1, spk_pend = -2, spk_n = 0, u_in = 0, u_sil = 0, u_spk = -1;
  for (int i = 0; i < N; ++i) {
    const float* z = slot + (size_t)i * ns;
    int am = 0;
    for (int k = 1; k < ns; ++k)
      if (z[k] > z[am]) am = k;
    int sv = am - 1;
    if (sv != spk_cur) {
      if (sv == spk_pend) spk_n++;
      else {
        spk_pend = sv;
        spk_n = 1;
      }
      if (spk_n >= 5 || (spk_cur == -1 && sv >= 0 && spk_n >= 2)) {
        spk_cur = sv;
        spk_pend = -2;
        spk_n = 0;
      }
    } else {
      spk_pend = -2;
      spk_n = 0;
    }
    int speech = pf[(size_t)i * ncf + sp_col] >= 0.5f;
    int sp = speech ? spk_cur : -1, start = 0;
    if (speech) {
      if (!u_in || (sp >= 0 && u_spk >= 0 && sp != u_spk)) {
        start = 1;
        u_in = 1;
        u_spk = sp;
      } else if (sp >= 0 && u_spk < 0) u_spk = sp;
      u_sil = 0;
    } else if (u_in) {
      if (++u_sil >= 5) {
        u_in = 0;
        u_spk = -1;
      }
    }
    spk[i] = sp;
    ust[i] = (unsigned char)start;
  }
  arc_free(slot);
  /* sources per query: the trunk output, or a pooled form of it */
  float* zu = pool_in(t, Hh, N, &t->pool_w0, t->pool_b0, &t->pool_w2, t->pool_b2);
  float* zs = pool_in(t, Hh, N, &t->spool_w0, t->spool_b0, &t->spool_w2, t->spool_b2);
  float* zi = pool_in(t, Hh, N, &t->ipool_w0, t->ipool_b0, &t->ipool_w2, t->ipool_b2);
  {
    double* a = arc_calloc(sizeof(double) * DM);
    double c = 0;
    double* ai = arc_calloc(sizeof(double) * DM);
    double (*as)[DM] = arc_calloc(sizeof(double) * 64 * DM);
    double cs[64] = {0};
    for (int i = 0; i < N; ++i) {
      float* u = zu + (size_t)i * DM;
      float* it = zi + (size_t)i * DM;
      float* sk = zs + (size_t)i * DM;
      if (ust[i]) {
        for (int k = 0; k < DM; ++k) a[k] = u[k];
        c = 1;
      } else {
        for (int k = 0; k < DM; ++k) a[k] += u[k];
        c += 1;
      }
      for (int k = 0; k < DM; ++k) u[k] = (float)(a[k] / (c > 1 ? c : 1));
      for (int k = 0; k < DM; ++k) ai[k] += it[k];
      for (int k = 0; k < DM; ++k) it[k] = (float)(ai[k] / (i + 1));
      int sv = spk[i];
      if (sv >= 0 && sv < 64) {
        for (int k = 0; k < DM; ++k) as[sv][k] += sk[k];
        cs[sv] += 1;
        for (int k = 0; k < DM; ++k) sk[k] = (float)(as[sv][k] / cs[sv]);
      } else
        for (int k = 0; k < DM; ++k) sk[k] = 0.0f;
    }
    arc_free(a);
    arc_free(ai);
    arc_free(as);
  }
  for (int q = 0; q < NQ; ++q) {
    int nc = t->ncols[q];
    int keep[MAXTAG], nk = 0;
    for (int j = 0; j < nc; ++j)
      if (shipped(t, t->cols[q][j], dom)) keep[nk++] = j;
    if (!nk) continue;
    const float* src = q == Q_UTT ? zu : q == Q_SPEAKER ? zs : q == Q_ITEM ? zi : Hh;
    float* p = arc_malloc(sizeof(float) * (size_t)N * nc);
    logits_q(t, q, src, N, dom, p);
    probs_q(t, q, p, N, dom);
    float* o = arc_malloc(sizeof(float) * (size_t)N * nk);
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < nk; ++j) o[(size_t)i * nk + j] = p[(size_t)i * nc + keep[j]];
    arc_free(p);
    const char** ids = arc_malloc(sizeof(char*) * (size_t)nk);
    for (int j = 0; j < nk; ++j) ids[j] = t->ids[t->cols[q][keep[j]]];
    out->fr[out->nq].q = QUERIES[q];
    out->fr[out->nq].ncols = nk;
    out->fr[out->nq].ids = ids;
    out->fr[out->nq++].p = o;
  }
  arc_free(pf);
  arc_free(spk);
  arc_free(ust);
  arc_free(zu);
  arc_free(zs);
  arc_free(zi);
  arc_free(Hh);
  return 0;
}

/* ------------------------------------------------------------------ frames (arc_frames_compute) */

/* The per-frame (10 Hz) probabilities an app reacts to: the frame tags (meta.speech, ...), the utterance readout as it
 * accumulates within each utterance (discourse.assistant.command_complete, device_directed, act.*, intent.*), and the
 * sound / channel events. Built on arc_engine_stages, the path scripts/check_parity.py matches against Arc's Python. */
arc_frames* arc_frames_compute(arc_engine* e, const float* pcm, size_t n, const char* domain) {
  if (!e || !e->trunk) { arc_seterr("no trunk"); return NULL; }
  arc_stages st;
  if (arc_engine_stages(e, pcm, n, domain ? domain : "assistant", &st)) {
    if (!arc_last_error()[0]) arc_seterr("clip too short");
    return NULL;
  }
  /* frame and utt columns under their tag ids; the instant queries (ev_at detects, ev_imm forecasts, ev_rec confirms:
   * speaker.change, discourse.assistant.command_complete) as "ev_at:<id>"; then the sound events */
  static const char* want[] = {"frame", "utt", "ev_at", "ev_imm", "ev_rec"};
  int nl = NEV;
  for (int i = 0; i < st.nq; ++i)
    for (int w = 0; w < 5; ++w)
      if (!strcmp(st.fr[i].q, want[w])) nl += st.fr[i].ncols;
  arc_frames* f = arc_calloc(sizeof *f);
  char** labels = arc_malloc(sizeof(char*) * (size_t)nl);
  float* p = arc_malloc(sizeof(float) * (size_t)st.N * (size_t)nl);
  int c = 0;
  for (int w = 0; w < 5; ++w)
    for (int i = 0; i < st.nq; ++i) {
      if (strcmp(st.fr[i].q, want[w])) continue;
      int k = st.fr[i].ncols, prefixed = w >= 2;
      for (int j = 0; j < k; ++j) {
        const char* id = st.fr[i].ids[j];
        size_t len = strlen(id) + (prefixed ? strlen(want[w]) + 1 : 0) + 1;
        labels[c + j] = arc_malloc(len);
        if (prefixed) snprintf(labels[c + j], len, "%s:%s", want[w], id);
        else memcpy(labels[c + j], id, len);
        for (int r = 0; r < st.N; ++r) p[(size_t)r * nl + c + j] = st.fr[i].p[(size_t)r * k + j];
      }
      c += k;
    }
  for (int j = 0; j < NEV; ++j) {
    size_t len = strlen(EV_LABELS[j]) + 1;
    labels[c + j] = arc_malloc(len);
    memcpy(labels[c + j], EV_LABELS[j], len);
    for (int r = 0; r < st.N; ++r) p[(size_t)r * nl + c + j] = st.ev ? st.ev[(size_t)r * NEV + j] : 0.0f;
  }
  f->n_frames = st.N;
  f->n_labels = nl;
  f->labels = (const char* const*)labels;
  f->p = p;
  arc_stages_free(&st);
  return f;
}

void arc_frames_free(arc_frames* f) {
  if (!f) return;
  for (int j = 0; j < f->n_labels; ++j) arc_free((void*)f->labels[j]);
  arc_free((void*)f->labels);
  arc_free((void*)f->p);
  arc_free(f);
}

/* ------------------------------------------------------------------ memory (arc_memory*) */

/* The model a saved state belongs to: FNV-1a over config.json and the trunk's shape. */
static uint64_t model_id(const char* dir) {
  char path[4096], *text = NULL;
  size_t len = 0;
  uint64_t h = 1469598103934665603ULL;
  snprintf(path, sizeof path, "%s/config.json", dir);
  if (arc_read_file(path, &text, &len) == 0) {
    for (size_t i = 0; i < len; ++i) h = (h ^ (unsigned char)text[i]) * 1099511628211ULL;
    arc_free(text);
  }
  const int shape[] = {DM, DIN, NLY, SH, SN, SP};
  for (int i = 0; i < 6; ++i) h = (h ^ (uint64_t)shape[i]) * 1099511628211ULL;
  return h;
}

int arc_memory(arc_engine* e, int on) {
  struct arc_trunk* t = e ? e->trunk : NULL;
  if (!t) { arc_seterr("no trunk"); return -1; }
  if (on && !t->mem) {
    if (!(t->mem = arc_malloc(sizeof *t->mem))) return -1;
    memset(t->mem, 0, sizeof *t->mem);
    t->mem_frames = 0;
  }
  if (!t->model_id) t->model_id = model_id(e->dir);
  t->mem_on = on != 0;
  return 0;
}

void arc_memory_reset(arc_engine* e) {
  struct arc_trunk* t = e ? e->trunk : NULL;
  if (!t || !t->mem) return;
  memset(t->mem, 0, sizeof *t->mem);
  t->mem_frames = 0;
}

double arc_memory_seconds(arc_engine* e) {
  struct arc_trunk* t = e ? e->trunk : NULL;
  return t && t->mem ? t->mem_frames / 10.0 : 0;
}

/* file: "ARCMEM01", model id (u64), frames (f64), then NLY ssm_state records */
static const char MEM_MAGIC[8] = {'A', 'R', 'C', 'M', 'E', 'M', '0', '1'};

int arc_memory_save(arc_engine* e, const char* path) {
  struct arc_trunk* t = e ? e->trunk : NULL;
  if (!t || !t->mem) { arc_seterr("no memory to save"); return -1; }
  if (!t->model_id) t->model_id = model_id(e->dir);
  char tmp[4096];
  snprintf(tmp, sizeof tmp, "%s.tmp", path);
  FILE* f = fopen(tmp, "wb");
  if (!f) { arc_seterr("cannot write %s", tmp); return -1; }
  int ok = fwrite(MEM_MAGIC, 1, 8, f) == 8 && fwrite(&t->model_id, sizeof t->model_id, 1, f) == 1 &&
           fwrite(&t->mem_frames, sizeof t->mem_frames, 1, f) == 1 && fwrite(t->mem, sizeof *t->mem, 1, f) == 1;
  ok = fclose(f) == 0 && ok;
  if (!ok || rename(tmp, path) != 0) { remove(tmp); arc_seterr("cannot write %s", path); return -1; }
  return 0;
}

int arc_memory_load(arc_engine* e, const char* path) {
  struct arc_trunk* t = e ? e->trunk : NULL;
  if (!t) { arc_seterr("no trunk"); return -1; }
  if (!t->model_id) t->model_id = model_id(e->dir);
  FILE* f = fopen(path, "rb");
  if (!f) { arc_seterr("cannot read %s", path); return -1; }
  char magic[8];
  uint64_t id = 0;
  double frames = 0;
  struct ssm_mem* m = arc_malloc(sizeof *m);
  int ok = m && fread(magic, 1, 8, f) == 8 && !memcmp(magic, MEM_MAGIC, 8) && fread(&id, sizeof id, 1, f) == 1 &&
           fread(&frames, sizeof frames, 1, f) == 1 && fread(m, sizeof *m, 1, f) == 1;
  fclose(f);
  if (!ok) { arc_free(m); arc_seterr("%s is not an Arc memory file", path); return -1; }
  if (id != t->model_id) { arc_free(m); arc_seterr("%s belongs to another model", path); return -1; }
  arc_free(t->mem);
  t->mem = m;
  t->mem_frames = frames;
  return 0;
}

/* ------------------------------------------------------------------ streaming read-out (arc_live)
 * The steps of arc_engine_run / readout as audio arrives: encoder frames pooled to 10 Hz as each 100 ms frame
 * completes, the device-directed head's per-frame logit, the event head once its 300 ms look-ahead has arrived, the
 * trunk 300 ms behind the features (its read-out delay) with its own SSM state, and the speaker / utterance state
 * machine with the running utterance pooling, frame by frame. Every step is the same per-frame computation as the
 * whole-clip path, so a stream that ends gives what arc_engine_run gives for the same samples. */

typedef struct {
  float* d;
  int w;
  long base, n, cap;
} lbuf;

static float* lb_at(lbuf* b, long t) { return b->d + (size_t)(t - b->base) * b->w; }
static int lb_push(lbuf* b, const float* rows, long k) {
  if (k <= 0) return 0;
  if (b->n + k > b->cap) {
    long nc = b->cap ? b->cap : 64;
    while (nc < b->n + k) nc *= 2;
    float* d = arc_malloc(sizeof(float) * (size_t)nc * b->w);
    if (!d) return -1;
    if (b->n) memcpy(d, b->d, sizeof(float) * (size_t)b->n * b->w);
    arc_free(b->d);
    b->d = d;
    b->cap = nc;
  }
  if (rows) memcpy(b->d + (size_t)b->n * b->w, rows, sizeof(float) * (size_t)k * b->w);
  else memset(b->d + (size_t)b->n * b->w, 0, sizeof(float) * (size_t)k * b->w);
  b->n += k;
  return 0;
}
static void lb_drop(lbuf* b, long t) {
  long k = t - b->base;
  if (k <= 0) return;
  if (k > b->n) k = b->n;
  memmove(b->d, b->d + (size_t)k * b->w, sizeof(float) * (size_t)(b->n - k) * b->w);
  b->n -= k;
  b->base += k;
}

/* pool10 over a stream of encoder frames: the open 100 ms frame's sum, max and count */
typedef struct {
  long g;          /* encoder frames seen */
  long f;          /* the open 10 Hz frame (-1: none yet) */
  float sum[ARC_MOON_D], mx[ARC_MOON_D], cnt;
  lbuf out;        /* finished frames [640] */
} pool_st;

static long pool_frame_of(long g) { return (long)ceil(((double)g * 320 + 3920) / 16000.0 * 10 - 1e-9) - 1; }

static int pool_close(pool_st* p, long upto) {   /* finish frames p->out.n .. upto - 1 */
  while (p->out.base + p->out.n < upto) {
    long i = p->out.base + p->out.n;
    float row[2 * ARC_MOON_D];
    if (i == p->f && p->cnt > 0) {
      for (int c = 0; c < ARC_MOON_D; ++c) {
        row[c] = arc_round_half(p->sum[c] / p->cnt);
        row[ARC_MOON_D + c] = arc_round_half(p->mx[c]);
      }
    } else if (p->out.n) memcpy(row, lb_at(&p->out, i - 1), sizeof row);   /* an empty frame repeats the last */
    else if (i > 0 && p->out.base > 0) return -1;                              /* (kept: never dropped to empty) */
    else memset(row, 0, sizeof row);
    if (lb_push(&p->out, row, 1)) return -1;
  }
  return 0;
}

static int pool_push(pool_st* p, const float* H, long n) {
  for (long i = 0; i < n; ++i, ++p->g) {
    long f = pool_frame_of(p->g);
    if (f != p->f) {
      if (pool_close(p, f)) return -1;   /* every frame before f is complete */
      p->f = f;
      p->cnt = 0;
      for (int c = 0; c < ARC_MOON_D; ++c) {
        p->sum[c] = 0;
        p->mx[c] = -INFINITY;
      }
    }
    const float* e = H + (size_t)i * ARC_MOON_D;
    for (int c = 0; c < ARC_MOON_D; ++c) {
      p->sum[c] += e[c];
      if (e[c] > p->mx[c]) p->mx[c] = e[c];
    }
    p->cnt += 1;
  }
  return 0;
}

struct arc_live {
  arc_engine* e;
  struct arc_trunk* t;
  int dom, di, d, trunk_on, events_on;
  pool_st X, XT;
  long nX;               /* 10 Hz frames complete */
  int finished;
  /* device-directed head: logit per frame */
  lbuf dd;               /* [1] */
  /* events */
  lbuf xe, h1;           /* LN([X, XT]) [1280], conv1 + GELU [256] */
  long n_h1, n_pr;
  float ev_st[NEV], ev_en[NEV], ev_pm[NEV];
  int ev_open[NEV], ev_order[NEV], ev_no;
  /* trunk */
  struct ssm_mem* mem;
  long nT;               /* trunk frames done */
  /* readout state (readout()) */
  int ncf, ncu, sp_col, dd_col, cc_col;
  int spk_cur, spk_pend, spk_n, u_in, u_sil, u_spk;
  double* acc;
  double cnt;
  int in_utt, have_p;
  long utt0, utt_last;
  float cur_p[MAXTAG];
  arc_live_utt* U;
  int nU, capU;
  long pause_seq;        /* trunk utterances that have paused (first non-speech frame after speech) */
  /* full frames (the columns of arc_frames_compute): trunk part of each frame read out, waiting for its events */
  int fullfr, nlab, nlt, qcols[NQ][MAXTAG], nq_cols[NQ];
  char** labels;
  lbuf fr_pend, ev_p;    /* [nlt + 5] (columns, then speech, in_utt, dd, cc, t), [NEV] */
  long n_emit;
  arc_live_fn fn;
  void* user;
};

static const int FRAME_QS[5] = {Q_FRAME, Q_UTT, Q_EV_AT, Q_EV_IMM, Q_EV_REC};

arc_live* arc_live_new(arc_engine* e, const char* domain, int trunk, int events, int delay, arc_live_fn fn, void* user) {
  struct arc_trunk* t = e ? e->trunk : NULL;
  if (!t) { arc_seterr("no trunk"); return NULL; }
  load_aux(t);
  arc_live* L = arc_calloc(sizeof *L);
  if (!L) return NULL;
  L->e = e;
  L->t = t;
  L->dom = dom_index(domain);
  L->d = delay < 0 ? 3 : delay;   /* the read-out delay in frames: one the trunk was trained with */
  L->di = -1;
  for (int k = 0; k < t->n_delays; ++k)
    if (t->delays[k] == L->d) L->di = k;
  if (L->di < 0) {
    arc_seterr("read-out delay %d frames: not one Arc was trained with", L->d);
    arc_free(L);
    return NULL;
  }
  L->trunk_on = trunk;
  L->fullfr = events < 0;   /* -1: full frames (all columns), and so the events too */
  L->events_on = events != 0 && t->ev_c1_w;
  L->X.f = L->XT.f = -1;
  L->X.out.w = L->XT.out.w = DIN;
  L->dd.w = 1;
  L->xe.w = 2 * DIN;
  L->h1.w = 256;
  L->fn = fn;
  L->user = user;
  L->ncf = t->ncols[Q_FRAME];
  L->ncu = t->ncols[Q_UTT];
  L->dd_col = L->cc_col = -1;
  for (int j = 0; j < L->ncf; ++j)
    if (!strcmp(t->ids[t->cols[Q_FRAME][j]], "meta.speech")) L->sp_col = j;
  for (int j = 0; j < L->ncu; ++j)
    if (!strcmp(t->ids[t->cols[Q_UTT][j]], "discourse.assistant.device_directed")) L->dd_col = j;
  /* command_complete is an instant query (ev_at), not an utterance column: as arc_audio's live view, the open
   * utterance's command_complete is this frame's ev_at probability */
  for (int j = 0; j < t->ncols[Q_EV_AT]; ++j)
    if (!strcmp(t->ids[t->cols[Q_EV_AT][j]], "discourse.assistant.command_complete")) L->cc_col = j;
  L->spk_cur = -1;
  L->spk_pend = -2;
  L->u_spk = -1;
  L->utt0 = L->utt_last = -1;
  if (L->fullfr) {   /* the columns, named as arc_frames_compute names them */
    static const char* pre[5] = {NULL, NULL, "ev_at", "ev_imm", "ev_rec"};
    L->labels = arc_calloc(sizeof(char*) * (MAXTAG * 5 + NEV));
    for (int w = 0; w < 5 && L->labels; ++w) {
      int q = FRAME_QS[w];
      for (int j = 0; j < t->ncols[q]; ++j) {
        if (!shipped(t, t->cols[q][j], L->dom)) continue;
        L->qcols[q][L->nq_cols[q]++] = j;
        const char* id = t->ids[t->cols[q][j]];
        size_t len = strlen(id) + (pre[w] ? strlen(pre[w]) + 1 : 0) + 1;
        char* lab = arc_malloc(len);
        if (pre[w]) snprintf(lab, len, "%s:%s", pre[w], id);
        else memcpy(lab, id, len);
        L->labels[L->nlab++] = lab;
      }
    }
    L->nlt = L->nlab;
    for (int j = 0; j < NEV && L->labels; ++j) L->labels[L->nlab++] = arc_strdup(EV_LABELS[j]);
    L->fr_pend.w = L->nlt + 5;
    L->ev_p.w = NEV;
  }
  L->mem = arc_calloc(sizeof *L->mem);
  L->acc = arc_calloc(sizeof(double) * DM);
  if (!L->mem || !L->acc) {
    arc_live_free(L);
    return NULL;
  }
  return L;
}

void arc_live_free(arc_live* L) {
  if (!L) return;
  arc_free(L->X.out.d);
  arc_free(L->XT.out.d);
  arc_free(L->dd.d);
  arc_free(L->xe.d);
  arc_free(L->h1.d);
  arc_free(L->mem);
  arc_free(L->acc);
  arc_free(L->U);
  for (int i = 0; i < L->nlab; ++i) arc_free(L->labels[i]);
  arc_free(L->labels);
  arc_free(L->fr_pend.d);
  arc_free(L->ev_p.d);
  arc_free(L);
}

/* full frames whose trunk columns and events are both there (all at the end) */
static void live_emit_frames(arc_live* L) {
  long to = L->nT;
  if (L->events_on && L->n_pr < to) to = L->n_pr;
  float* row = arc_malloc(sizeof(float) * (size_t)(L->nlab + 1));
  if (!row) return;
  for (; L->n_emit < to; ++L->n_emit) {
    const float* r = lb_at(&L->fr_pend, L->n_emit);
    memcpy(row, r, sizeof(float) * (size_t)L->nlt);
    for (int j = 0; j < NEV; ++j) row[L->nlt + j] = L->events_on ? sigm(lb_at(&L->ev_p, L->n_emit)[j]) : 0.0f;
    const float* x = r + L->nlt;
    arc_frame fr = {x[4], x[0], (int)x[1], x[2], x[3], L->nlab, (const char* const*)L->labels, row};
    if (L->fn) L->fn(L->user, &fr, NULL);
  }
  arc_free(row);
  lb_drop(&L->fr_pend, L->n_emit);
  lb_drop(&L->ev_p, L->n_emit);
}

int arc_live_tap(arc_live* L) { return L->t->ev_tap; }
long arc_live_pauses(const arc_live* L) { return L->pause_seq; }
int arc_live_open(const arc_live* L, long* f0, long* f1) {
  if (!L->in_utt || !L->have_p) return 0;
  *f0 = L->utt0;
  *f1 = L->utt_last;
  return 1;
}
long arc_live_frames(const arc_live* L) { return L->nX; }
long arc_live_trunk_frames(const arc_live* L) { return L->nT; }

float arc_live_directed(arc_live* L, long a, long b) {
  if (!L->t->dd_fc_w) return -1;
  if (a < 0) a = 0;
  if (b > L->nX) {
    if (!L->finished) return -2;   /* not there yet */
    b = L->nX;
  }
  if (b <= a || a < L->dd.base) return -1;
  double s = 0;
  for (long f = a; f < b; ++f) s += *lb_at(&L->dd, f);
  return sigm((float)(s / (b - a)));
}

/* the utterance read-out for a transcript utterance [start, end): the trunk utterance it overlaps most (as
 * arc_engine_run matches them), the open one included; *settled = 1 when that one has closed and no later one can
 * overlap more */
int arc_live_tags(arc_live* L, float start, float end, arc_utt_build* u, int* settled) {
  struct arc_trunk* t = L->t;
  const float* best = NULL;
  float bo = 0;
  int closed = 0;
  for (int k = 0; k < L->nU; ++k) {
    float ov = fminf(end, L->U[k].end) - fmaxf(start, L->U[k].start);
    if (ov > bo) {
      bo = ov;
      best = L->U[k].p;
      closed = 1;
    }
  }
  if (L->in_utt && L->have_p) {
    float ov = fminf(end, (L->utt_last + 1) / 10.0f) - fmaxf(start, L->utt0 / 10.0f);
    if (ov > bo) {
      bo = ov;
      best = L->cur_p;
      closed = 0;
    }
  }
  *settled = (L->finished || L->nT >= (long)(end * 10 + 0.5f)) && (closed || !best) && !(L->in_utt && L->have_p &&
              L->utt0 / 10.0f < end);
  u->dd = u->cc = -1;
  u->intent = u->act = NULL;
  u->intent_p = u->act_p = -1;
  if (!best) return 0;
  for (int j = 0; j < L->ncu; ++j) {
    int c = t->cols[Q_UTT][j];
    if (!shipped(t, c, L->dom)) continue;
    const char* id = t->ids[c];
    int g = t->group_of[c];
    if (g < 0) {
      if (!strcmp(id, "discourse.assistant.device_directed")) u->dd = best[j];
      else if (!strcmp(id, "discourse.assistant.command_complete")) u->cc = best[j];
    } else if (!strcmp(t->groups[g], "discourse.intent") && best[j] > u->intent_p) {
      u->intent = id;
      u->intent_p = best[j];
    } else if (!strcmp(t->groups[g], "discourse.act") && best[j] > u->act_p) {
      u->act = id;
      u->act_p = best[j];
    }
  }
  return 1;
}

/* rows [a, b) of h1 (conv k5, pad 2 each side, + GELU) from xe, zeros outside [0, M) */
static int live_conv(struct arc_trunk* t, lbuf* in, long M, long a, long b, int k, int Cin, const float* w,
                     const float* bias, int Cout, float* y) {
  int half = k / 2;
  long lo = a - half, n = (b - 1 + half) - lo + 1;
  float* win = arc_calloc(sizeof(float) * (size_t)n * Cin);
  if (!win) return -1;
  for (long f = lo; f < lo + n; ++f)
    if (f >= 0 && f < M && f >= in->base && f < in->base + in->n)
      memcpy(win + (size_t)(f - lo) * Cin, lb_at(in, f), sizeof(float) * Cin);
  arc_mat wm = {.type = ARC_W_F32, .O = Cout, .K = Cin * k, .w = w};
  int r = arc_conv1d(t->pool, win, (int)n, Cin, Cin, &wm, k, 1, 1, 0, 0, bias, y, Cout);
  arc_free(win);
  return r == b - a ? 0 : -1;
}

static void live_event_frame(arc_live* L, long f, const float* pr) {
  struct arc_trunk* t = L->t;
  for (int j = 0; j < NEV; ++j) {
    if (!t->ev_on[j]) continue;
    float p = sigm(pr[j]);
    if (p >= t->ev_thr[j]) {
      if (!L->ev_open[j]) {
        L->ev_open[j] = 1;
        L->ev_st[j] = f / 10.0f;
        L->ev_en[j] = (f + 1) / 10.0f;
        L->ev_pm[j] = p;
        L->ev_order[L->ev_no++] = j;
      } else {
        L->ev_en[j] = (f + 1) / 10.0f;
        if (p > L->ev_pm[j]) L->ev_pm[j] = p;
      }
    } else if (L->ev_open[j]) {
      if (L->fn) {
        arc_event ev = {EV_LABELS[j], L->ev_st[j], L->ev_en[j], L->ev_pm[j]};
        L->fn(L->user, NULL, &ev);
      }
      L->ev_open[j] = 0;
      for (int k = 0; k < L->ev_no; ++k)
        if (L->ev_order[k] == j) {
          memmove(L->ev_order + k, L->ev_order + k + 1, sizeof(int) * (size_t)(L->ev_no - k - 1));
          L->ev_no--;
          break;
        }
    }
  }
}

/* the events for the 10 Hz frames complete so far: conv1 rows need 2 frames ahead, conv2 rows 1 more */
static int live_events(arc_live* L) {
  struct arc_trunk* t = L->t;
  long M = L->nX;
  long h1_to = L->finished ? M : M - 2;
  if (h1_to > L->n_h1) {
    long k = h1_to - L->n_h1;
    float* h = arc_malloc(sizeof(float) * (size_t)k * 256);
    if (!h || live_conv(t, &L->xe, M, L->n_h1, h1_to, 5, 2 * DIN, t->ev_c1_w, t->ev_c1_b, 256, h)) {
      arc_free(h);
      return -1;
    }
    arc_gelu(h, (size_t)k * 256);
    int rc = lb_push(&L->h1, h, k);
    arc_free(h);
    if (rc) return -1;
    L->n_h1 = h1_to;
  }
  long pr_to = L->finished ? M : L->n_h1 - 1;
  if (pr_to > L->n_pr) {
    long k = pr_to - L->n_pr;
    float* pr = arc_malloc(sizeof(float) * (size_t)k * NEV);
    if (!pr || live_conv(t, &L->h1, L->n_h1, L->n_pr, pr_to, 3, 256, t->ev_c2_w, t->ev_c2_b, NEV, pr)) {
      arc_free(pr);
      return -1;
    }
    for (long i = 0; i < k; ++i) live_event_frame(L, L->n_pr + i, pr + (size_t)i * NEV);
    if (L->fullfr && lb_push(&L->ev_p, pr, k)) {
      arc_free(pr);
      return -1;
    }
    arc_free(pr);
    L->n_pr = pr_to;
  }
  lb_drop(&L->xe, L->n_h1 - 2);
  lb_drop(&L->h1, L->n_pr - 1);
  if (L->finished && L->fn)
    for (int k = 0; k < L->ev_no; ++k) {
      int j = L->ev_order[k];
      arc_event ev = {EV_LABELS[j], L->ev_st[j], L->ev_en[j], L->ev_pm[j]};
      L->fn(L->user, NULL, &ev);
    }
  if (L->finished) L->ev_no = 0;
  return 0;
}

static int live_close_utt(arc_live* L) {
  if (L->utt0 >= 0 && L->have_p) {
    if (L->nU == L->capU) {
      int nc = L->capU ? L->capU * 2 : 16;
      arc_live_utt* nu = arc_malloc(sizeof(arc_live_utt) * (size_t)nc);
      if (!nu) return -1;
      if (L->nU) memcpy(nu, L->U, sizeof(arc_live_utt) * (size_t)L->nU);
      arc_free(L->U);
      L->U = nu;
      L->capU = nc;
    }
    arc_live_utt* u = &L->U[L->nU++];
    u->start = L->utt0 / 10.0f;
    u->end = (L->utt_last + 1) / 10.0f;
    memcpy(u->p, L->cur_p, sizeof(float) * (size_t)L->ncu);
  }
  L->in_utt = 0;
  L->utt0 = -1;
  L->have_p = 0;
  return 0;
}

/* trunk frames [nT, to): the trunk, then readout()'s per-frame steps */
static int live_trunk(arc_live* L, long to) {
  struct arc_trunk* t = L->t;
  long n = to - L->nT;
  if (n <= 0) return 0;
  long N = L->nX;
  float* xs = arc_malloc(sizeof(float) * (size_t)n * DIN);
  if (!xs) return -1;
  for (long i = 0; i < n; ++i) {
    long f = L->nT + i, src = f + L->d < N ? f + L->d : N - 1;
    memcpy(xs + (size_t)i * DIN, lb_at(&L->X.out, src), sizeof(float) * DIN);
  }
  float* Hh = trunk_run_mem(t, xs, (int)n, L->dom, L->di, L->mem);
  arc_free(xs);
  if (!Hh) return -1;
  int ncf = L->ncf, ncu = L->ncu, ns = t->n_slot + t->n_slotx;
  float* pf = arc_malloc(sizeof(float) * (size_t)n * (ncf ? ncf : 1));
  float* slot = arc_malloc(sizeof(float) * (size_t)n * (ns ? ns : 1));
  float* sl1 = arc_malloc(sizeof(float) * (size_t)n * (t->n_slot ? t->n_slot : 1));
  float* sl2 = t->n_slotx ? arc_malloc(sizeof(float) * (size_t)n * t->n_slotx) : NULL;
  float* z = arc_malloc(sizeof(float) * (size_t)n * DM);
  float* z1 = arc_malloc(sizeof(float) * (size_t)n * DM);
  float* pu = arc_malloc(sizeof(float) * (ncu ? ncu : 1));
  float* pev[3] = {NULL, NULL, NULL};   /* ev_at / ev_imm / ev_rec on the trunk output (full frames) */
  int rc = -1;
  if (!pf || !slot || !sl1 || (t->n_slotx && !sl2) || !z || !z1 || !pu) goto out;
  logits_q(t, Q_FRAME, Hh, (int)n, L->dom, pf);
  probs_q(t, Q_FRAME, pf, (int)n, L->dom);
  for (int w = 0; w < 3 && (L->fullfr || (w == 0 && L->cc_col >= 0)); ++w) {
    int q = FRAME_QS[2 + w];
    if (!L->nq_cols[q] && !(w == 0 && L->cc_col >= 0)) continue;
    pev[w] = arc_malloc(sizeof(float) * (size_t)n * t->ncols[q]);
    if (!pev[w]) goto out;
    logits_q(t, q, Hh, (int)n, L->dom, pev[w]);
    probs_q(t, q, pev[w], (int)n, L->dom);
  }
  lin(t->pool, Hh, (int)n, DM, &t->slot_w, t->n_slot, t->slot_b, sl1);
  if (sl2) lin(t->pool, Hh, (int)n, DM, &t->slotx_w, t->n_slotx, t->slotx_b, sl2);
  for (long i = 0; i < n; ++i) {
    memcpy(slot + (size_t)i * ns, sl1 + (size_t)i * t->n_slot, sizeof(float) * t->n_slot);
    if (sl2) memcpy(slot + (size_t)i * ns + t->n_slot, sl2 + (size_t)i * t->n_slotx, sizeof(float) * t->n_slotx);
  }
  lin(t->pool, Hh, (int)n, DM, &t->pool_w0, DM, t->pool_b0, z1);
  arc_gelu(z1, (size_t)n * DM);
  lin(t->pool, z1, (int)n, DM, &t->pool_w2, DM, t->pool_b2, z);
  for (size_t k = 0; k < (size_t)n * DM; ++k) z[k] += Hh[k];
  for (long i = 0; i < n; ++i) {
    long fi = L->nT + i;
    /* _spk_utt */
    const float* zs = slot + (size_t)i * ns;
    int am = 0;
    for (int k = 1; k < ns; ++k)
      if (zs[k] > zs[am]) am = k;
    int sv = am - 1;
    if (sv != L->spk_cur) {
      if (sv == L->spk_pend) L->spk_n++;
      else {
        L->spk_pend = sv;
        L->spk_n = 1;
      }
      if (L->spk_n >= 5 || (L->spk_cur == -1 && sv >= 0 && L->spk_n >= 2)) {
        L->spk_cur = sv;
        L->spk_pend = -2;
        L->spk_n = 0;
      }
    } else {
      L->spk_pend = -2;
      L->spk_n = 0;
    }
    float psp = pf[(size_t)i * ncf + L->sp_col];
    int speech = psp >= 0.5f;
    int sp = speech ? L->spk_cur : -1, ust = 0, ucl = 0;
    if (speech) {
      if (!L->u_in || (sp >= 0 && L->u_spk >= 0 && sp != L->u_spk)) {
        ucl = L->u_in;
        ust = 1;
        L->u_in = 1;
        L->u_spk = sp;
      } else if (sp >= 0 && L->u_spk < 0) L->u_spk = sp;
      L->u_sil = 0;
    } else if (L->u_in) {
      L->u_sil++;
      if (L->u_sil >= 5) {
        ucl = 1;
        L->u_in = 0;
        L->u_spk = -1;
      }
    }
    /* utterance pooling: running mean since the utterance start (double sums) */
    float* zi = z + (size_t)i * DM;
    if (ust) {
      for (int c = 0; c < DM; ++c) L->acc[c] = zi[c];
      L->cnt = 1;
    } else {
      for (int c = 0; c < DM; ++c) L->acc[c] += zi[c];
      L->cnt += 1;
    }
    for (int c = 0; c < DM; ++c) zi[c] = (float)(L->acc[c] / (L->cnt > 1 ? L->cnt : 1));
    logits_q(t, Q_UTT, zi, 1, L->dom, pu);
    probs_q(t, Q_UTT, pu, 1, L->dom);
    /* _decode_frame / _close_utt */
    if (ucl && live_close_utt(L)) goto out;
    if (ust) {
      L->in_utt = 1;
      L->utt0 = fi;
    }
    if (speech && L->in_utt) {
      L->utt_last = fi;
      memcpy(L->cur_p, pu, sizeof(float) * (size_t)ncu);
      L->have_p = 1;
    } else if (!speech && L->in_utt && L->have_p && L->utt_last == fi - 1)
      L->pause_seq++;   /* the read-out now holds what the utterance ends with, unless speech resumes */
    float ddv = L->in_utt && L->dd_col >= 0 ? pu[L->dd_col] : -1.0f;
    float ccv = L->in_utt && L->cc_col >= 0 && pev[0] ? pev[0][(size_t)i * t->ncols[Q_EV_AT] + L->cc_col] : -1.0f;
    if (L->fullfr) {   /* this frame's columns, sent once its events are there */
      if (lb_push(&L->fr_pend, NULL, 1)) goto out;
      float* r = lb_at(&L->fr_pend, fi);
      int c = 0;
      for (int k = 0; k < L->nq_cols[Q_FRAME]; ++k) r[c++] = pf[(size_t)i * ncf + L->qcols[Q_FRAME][k]];
      for (int k = 0; k < L->nq_cols[Q_UTT]; ++k) r[c++] = pu[L->qcols[Q_UTT][k]];
      for (int w = 0; w < 3; ++w) {
        int q = FRAME_QS[2 + w];
        for (int k = 0; k < L->nq_cols[q]; ++k) r[c++] = pev[w][(size_t)i * t->ncols[q] + L->qcols[q][k]];
      }
      r[c] = psp;
      r[c + 1] = (float)L->in_utt;
      r[c + 2] = ddv;
      r[c + 3] = ccv;
      r[c + 4] = fi / 10.0f;
    } else if (L->fn) {
      arc_frame fr = {fi / 10.0f, psp, L->in_utt, ddv, ccv, 0, NULL, NULL};
      L->fn(L->user, &fr, NULL);
    }
  }
  L->nT = to;
  rc = 0;
out:
  arc_free(Hh);
  arc_free(pf);
  arc_free(slot);
  arc_free(sl1);
  arc_free(sl2);
  arc_free(z);
  arc_free(z1);
  arc_free(pu);
  for (int w = 0; w < 3; ++w) arc_free(pev[w]);
  return rc;
}

int arc_live_push(arc_live* L, const float* H, long n, const float* tap, long n_tap, int final) {
  struct arc_trunk* t = L->t;
  if (L->finished) return 0;
  if (pool_push(&L->X, H, n)) return -1;
  if (L->events_on && pool_push(&L->XT, tap, n_tap)) return -1;
  long N = L->X.f > 0 ? L->X.f : 0;   /* frames before the open one are complete */
  if (final) {
    L->finished = 1;
    N = L->X.g ? pool_frame_of(L->X.g - 1) + 1 : 0;
  }
  if (pool_close(&L->X, N)) return -1;
  if (L->events_on && pool_close(&L->XT, N)) return -1;
  for (long f = L->nX; f < N; ++f) {
    const float* x = lb_at(&L->X.out, f);
    if (t->dd_fc_w) {
      float row[DIN];
      memcpy(row, x, sizeof row);
      arc_layernorm(row, 1, DIN, DIN, t->dd_ln_w, t->dd_ln_b, 1e-5f);
      float v = t->dd_fc_b[0];
      for (int c = 0; c < DIN; ++c) v += row[c] * t->dd_fc_w[c];
      if (lb_push(&L->dd, &v, 1)) return -1;
    }
    if (L->events_on) {
      float xe[2 * DIN];
      memcpy(xe, x, sizeof(float) * DIN);
      memcpy(xe + DIN, lb_at(&L->XT.out, f), sizeof(float) * DIN);
      arc_layernorm(xe, 1, 2 * DIN, 2 * DIN, t->ev_ln_w, t->ev_ln_b, 1e-5f);
      if (lb_push(&L->xe, xe, 1)) return -1;
    }
  }
  L->nX = N;
  if (L->events_on && live_events(L)) return -1;
  if (L->trunk_on && live_trunk(L, L->finished ? N : N - L->d)) return -1;
  if (L->finished && L->trunk_on && L->in_utt && live_close_utt(L)) return -1;
  if (L->fullfr) live_emit_frames(L);
  /* keep 2 frames of features for the trunk's look-ahead and the pooling's repeat; the dd logits stay (4 bytes per
   * 100 ms) so any utterance can be scored */
  long keep = L->nT - 1 < N - 1 ? L->nT - 1 : N - 1;
  if (!L->trunk_on) keep = N - 1;
  if (keep > 0) lb_drop(&L->X.out, keep);
  if (L->events_on && N - 1 > 0) lb_drop(&L->XT.out, N - 1);
  return 0;
}
