/* Moonshine Streaming (tiny): encoder, decoder and decoding. See arc_moon.h.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include "arc_moon.h"

#if defined(_WIN32)
#include <malloc.h>
#else
#include <alloca.h>
#endif
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define D ARC_MOON_D
#define H ARC_MOON_HEADS
#define HD ARC_MOON_HD
#define FF ARC_MOON_FF
#define V ARC_MOON_VOCAB
#define NL ARC_MOON_LAYERS

/* ------------------------------------------------------------------ loading */

static const float* need(arc_moon* m, const char* name, int64_t n) {
  const float* p = arc_store_f32(m->st, name, n);
  return p;
}

/* a matrix: packed in place when it is, else float32 */
static int needw(arc_moon* m, const char* name, int64_t n, arc_w* w) {
  memset(w, 0, sizeof *w);
  if (arc_store_q4(m->st, name, &w->q)) {
    if ((int64_t)w->q.R * w->q.C != n) {
      arc_seterr("%s: %d x %d, expected %lld values", name, w->q.R, w->q.C, (long long)n);
      return -1;
    }
    w->packed = 1;
    return 0;
  }
  w->f = arc_store_f32(m->st, name, n);
  return w->f ? 0 : -1;
}

/* gamma + 1 (Moonshine's unit-offset LayerNorm), owned by the model (kept in a small arena) */
typedef struct { float* p[64]; int n; } arena;
static arena g_unused;

static float* plus_one(const float* g, int n) {
  float* o = arc_malloc(sizeof(float) * (size_t)n);
  if (!o) return NULL;
  for (int i = 0; i < n; ++i) o[i] = g[i] + 1.0f;
  return o;
}

typedef struct {
  arc_moon m;
  float* owned[64];
  int n_owned;
} moon_impl;

static const float* own(moon_impl* x, float* p) {
  if (p && x->n_owned < 64) x->owned[x->n_owned++] = p;
  return p;
}

arc_moon* arc_moon_load(const char* folder, int threads) { return arc_moon_load_ex(folder, threads, 0); }

arc_moon* arc_moon_load_ex(const char* folder, int threads, int encoder_only) {
  (void)g_unused;
  moon_impl* x = arc_calloc(sizeof *x);
  if (!x) return NULL;
  arc_moon* m = &x->m;
  char path[1024];
  snprintf(path, sizeof path, "%s/model.q4.safetensors", folder);
  FILE* f = fopen(path, "rb");
  if (f) fclose(f);
  else snprintf(path, sizeof path, "%s/model.safetensors", folder);
  if (!(m->st = arc_store_open(path))) goto fail;
  m->pool = arc_pool_create(threads);
  const float* lk = need(m, "model.encoder.embedder.comp.log_k", 1);
  if (!lk) goto fail;
  m->comp_k = expf(lk[0]);
  if (!(m->lin = need(m, "model.encoder.embedder.linear.weight", (int64_t)D * ARC_MOON_FRAME))) goto fail;
  if (!(m->conv1 = need(m, "model.encoder.embedder.conv1.weight", (int64_t)2 * D * D * 5))) goto fail;
  if (!(m->conv1b = need(m, "model.encoder.embedder.conv1.bias", 2 * D))) goto fail;
  if (!(m->conv2 = need(m, "model.encoder.embedder.conv2.weight", (int64_t)D * 2 * D * 5))) goto fail;
  if (!(m->conv2b = need(m, "model.encoder.embedder.conv2.bias", D))) goto fail;
  const float* fg = need(m, "model.encoder.final_norm.gamma", D);
  if (!fg || !(m->final_g = own(x, plus_one(fg, D)))) goto fail;
  static const int win[NL][2] = {{16, 4}, {16, 4}, {16, 0}, {16, 0}, {16, 4}, {16, 4}};
  for (int l = 0; l < NL; ++l) {
    arc_moon_enc_layer* L = &m->enc[l];
    m->win[l][0] = win[l][0];
    m->win[l][1] = win[l][1];
    const float* g1 = need(m, arc_namef("model.encoder.layers.%d.input_layernorm.gamma", l), D);
    if (!g1 || !(L->ln1 = own(x, plus_one(g1, D)))) goto fail;
    const float* g2 = need(m, arc_namef("model.encoder.layers.%d.post_attention_layernorm.gamma", l), D);
    if (!g2 || !(L->ln2 = own(x, plus_one(g2, D)))) goto fail;
    if (needw(m, arc_namef("model.encoder.layers.%d.self_attn.q_proj.weight", l), D * D, &L->q)) goto fail;
    if (needw(m, arc_namef("model.encoder.layers.%d.self_attn.k_proj.weight", l), D * D, &L->k)) goto fail;
    if (needw(m, arc_namef("model.encoder.layers.%d.self_attn.v_proj.weight", l), D * D, &L->v)) goto fail;
    if (needw(m, arc_namef("model.encoder.layers.%d.self_attn.o_proj.weight", l), D * D, &L->o)) goto fail;
    if (needw(m, arc_namef("model.encoder.layers.%d.mlp.fc1.weight", l), FF * D, &L->fc1)) goto fail;
    if (!(L->fc1b = need(m, arc_namef("model.encoder.layers.%d.mlp.fc1.bias", l), FF))) goto fail;
    if (needw(m, arc_namef("model.encoder.layers.%d.mlp.fc2.weight", l), D * FF, &L->fc2)) goto fail;
    if (!(L->fc2b = need(m, arc_namef("model.encoder.layers.%d.mlp.fc2.bias", l), D))) goto fail;
  }
  m->has_decoder = !encoder_only;
  if (encoder_only) return m;   /* the decoder's tensors stay unread (its float32 tables are never made) */
  if (needw(m, "model.decoder.embed_tokens.weight", (int64_t)V * D, &m->embed)) goto fail;
  if (!(m->pos_emb = need(m, "model.decoder.pos_emb.weight", (int64_t)4096 * D))) goto fail;
  if (!(m->norm = need(m, "model.decoder.norm.weight", D))) goto fail;
  for (int l = 0; l < NL; ++l) {
    arc_moon_dec_layer* L = &m->dec[l];
#define DW(field, nm, n) if (!(L->field = need(m, arc_namef("model.decoder.layers.%d." nm, l), n))) goto fail
#define DWW(field, nm, n) if (needw(m, arc_namef("model.decoder.layers.%d." nm, l), n, &L->field)) goto fail
    DW(ln_in, "input_layernorm.weight", D);
    DW(ln_post, "post_attention_layernorm.weight", D);
    DW(ln_final, "final_layernorm.weight", D);
    DWW(sq, "self_attn.q_proj.weight", D * D);
    DWW(sk, "self_attn.k_proj.weight", D * D);
    DWW(sv, "self_attn.v_proj.weight", D * D);
    DWW(so, "self_attn.o_proj.weight", D * D);
    DWW(cq, "encoder_attn.q_proj.weight", D * D);
    DWW(ck, "encoder_attn.k_proj.weight", D * D);
    DWW(cv, "encoder_attn.v_proj.weight", D * D);
    DWW(co, "encoder_attn.o_proj.weight", D * D);
    DWW(fc1, "mlp.fc1.weight", 2 * FF * D);
    DW(fc1b, "mlp.fc1.bias", 2 * FF);
    DWW(fc2, "mlp.fc2.weight", D * FF);
    DW(fc2b, "mlp.fc2.bias", D);
#undef DW
#undef DWW
  }
  /* partial RoPE: dim = int(40 * 0.8) = 32; inv_freq = 1 / 10000^(arange(0, 32, 2) / 32), in float32 as torch */
  m->n_rot = 32;
  for (int i = 0; i < 16; ++i) m->inv_freq[i] = 1.0f / powf(10000.0f, (float)(2 * i) / 32.0f);
  m->bos = 1;
  m->eos = 2;
  return m;
fail:
  arc_moon_free(m);
  return NULL;
}

void arc_moon_free(arc_moon* m) {
  if (!m) return;
  moon_impl* x = (moon_impl*)m;
  for (int i = 0; i < x->n_owned; ++i) arc_free(x->owned[i]);
  arc_pool_destroy(m->pool);
  arc_store_close(m->st);
  arc_free(x);
}

/* ------------------------------------------------------------------ helpers */

static void linear(arc_moon* m, const float* x, int T, int K, const float* W, int O, const float* b, float* y) {
  arc_gemm_nt(m->pool, T, O, K, x, K, W, K, y, O, b, 0);
}

static void linw(arc_moon* m, const float* x, int T, int K, const arc_w* W, int O, const float* b, float* y) {
  if (W->packed) arc_q4_linear(m->pool, x, T, K, &W->q, b, y, O);
  else arc_gemm_nt(m->pool, T, O, K, x, K, W->f, K, y, O, b, 0);
}

static float silu1(float v) { return v / (1.0f + expf(-v)); }

/* ------------------------------------------------------------------ encoder */

float* arc_moon_encode(arc_moon* m, const float* pcm, int n, int* T_out) { return arc_moon_encode_ex(m, pcm, n, T_out, -1, NULL); }

float* arc_moon_encode_ex(arc_moon* m, const float* pcm, int n, int* T_out, int tap, float** tap_out) {
  const int F = ARC_MOON_FRAME;
  int T5 = n / F;
  if (T5 < 1) {
    arc_seterr("encoder: need at least %d samples", F);
    return NULL;
  }
  float *fr = NULL, *e = NULL, *c1 = NULL, *h = NULL, *xn = NULL, *q = NULL, *k = NULL, *v = NULL, *att = NULL,
        *tmp = NULL, *ff = NULL;
  /* frames: CMVN per 5 ms frame, asinh(k x), linear (no bias), SiLU */
  fr = arc_malloc(sizeof(float) * (size_t)T5 * F);
  e = arc_malloc(sizeof(float) * (size_t)T5 * D);
  if (!fr || !e) goto fail;
  for (int t = 0; t < T5; ++t) {
    const float* s = pcm + (size_t)t * F;
    float* o = fr + (size_t)t * F;
    float mean = 0;
    for (int i = 0; i < F; ++i) mean += s[i];
    mean /= F;
    float ms = 0;
    for (int i = 0; i < F; ++i) {
      float c = s[i] - mean;
      ms += c * c;
    }
    float rms = sqrtf(ms / F + 1e-6f);
    for (int i = 0; i < F; ++i) o[i] = asinhf(m->comp_k * ((s[i] - mean) / rms));
  }
  linear(m, fr, T5, F, m->lin, D, NULL, e);
  for (size_t i = 0; i < (size_t)T5 * D; ++i) e[i] = silu1(e[i]);
  /* two causal convs, kernel 5, stride 2 (left padding 4) */
  int T1 = (T5 - 1) / 2 + 1, T2 = (T1 - 1) / 2 + 1;
  c1 = arc_malloc(sizeof(float) * (size_t)T1 * 2 * D);
  h = arc_malloc(sizeof(float) * (size_t)T2 * D);
  if (!c1 || !h) goto fail;
  arc_mat w1 = {.type = ARC_W_F32, .O = 2 * D, .K = D * 5, .w = m->conv1};
  if (arc_conv1d(m->pool, e, T5, D, D, &w1, 5, 2, 1, 4, 0, m->conv1b, c1, 2 * D) != T1) goto fail;
  for (size_t i = 0; i < (size_t)T1 * 2 * D; ++i) c1[i] = silu1(c1[i]);
  arc_mat w2 = {.type = ARC_W_F32, .O = D, .K = 2 * D * 5, .w = m->conv2};
  if (arc_conv1d(m->pool, c1, T1, 2 * D, 2 * D, &w2, 5, 2, 1, 4, 0, m->conv2b, h, D) != T2) goto fail;
  /* sliding-window attention layers */
  int T = T2;
  xn = arc_malloc(sizeof(float) * (size_t)T * D);
  q = arc_malloc(sizeof(float) * (size_t)T * D);
  k = arc_malloc(sizeof(float) * (size_t)T * D);
  v = arc_malloc(sizeof(float) * (size_t)T * D);
  att = arc_malloc(sizeof(float) * (size_t)T * D);
  tmp = arc_malloc(sizeof(float) * (size_t)T * D);
  ff = arc_malloc(sizeof(float) * (size_t)T * FF);
  if (!xn || !q || !k || !v || !att || !tmp || !ff) goto fail;
  const float scale = 1.0f / sqrtf((float)HD);
  if (tap_out) *tap_out = NULL;
  for (int l = 0; l < NL; ++l) {
    if (tap_out && l == tap) {   /* hidden state before layer `tap` (MST's H[tap]) */
      *tap_out = arc_malloc(sizeof(float) * (size_t)T * D);
      if (*tap_out) memcpy(*tap_out, h, sizeof(float) * (size_t)T * D);
    }
    const arc_moon_enc_layer* L = &m->enc[l];
    int left = m->win[l][0], right = m->win[l][1];
    memcpy(xn, h, sizeof(float) * (size_t)T * D);
    arc_layernorm(xn, T, D, D, L->ln1, NULL, 1e-5f);
    linw(m, xn, T, D, &L->q, D, NULL, q);
    linw(m, xn, T, D, &L->k, D, NULL, k);
    linw(m, xn, T, D, &L->v, D, NULL, v);
    float sc[64];
    for (int t = 0; t < T; ++t) {
      int k0 = t - left + 1, k1 = t + (right > 0 ? right - 1 : 0);   /* inclusive */
      if (k0 < 0) k0 = 0;
      if (k1 > T - 1) k1 = T - 1;
      for (int hh = 0; hh < H; ++hh) {
        const float* qv = q + (size_t)t * D + hh * HD;
        float mx = -FLT_MAX;
        for (int j = k0; j <= k1; ++j) {
          const float* kv = k + (size_t)j * D + hh * HD;
          float s = 0;
          for (int d = 0; d < HD; ++d) s += qv[d] * kv[d];
          s *= scale;
          sc[j - k0] = s;
          if (s > mx) mx = s;
        }
        float sum = 0;
        for (int j = k0; j <= k1; ++j) {
          sc[j - k0] = expf(sc[j - k0] - mx);
          sum += sc[j - k0];
        }
        float* o = att + (size_t)t * D + hh * HD;
        for (int d = 0; d < HD; ++d) o[d] = 0;
        for (int j = k0; j <= k1; ++j) {
          float w = sc[j - k0] / sum;
          const float* vv = v + (size_t)j * D + hh * HD;
          for (int d = 0; d < HD; ++d) o[d] += w * vv[d];
        }
      }
    }
    linw(m, att, T, D, &L->o, D, NULL, tmp);
    for (size_t i = 0; i < (size_t)T * D; ++i) h[i] += tmp[i];
    memcpy(xn, h, sizeof(float) * (size_t)T * D);
    arc_layernorm(xn, T, D, D, L->ln2, NULL, 1e-5f);
    linw(m, xn, T, D, &L->fc1, FF, L->fc1b, ff);
    arc_gelu(ff, (size_t)T * FF);
    linw(m, ff, T, FF, &L->fc2, D, L->fc2b, tmp);
    for (size_t i = 0; i < (size_t)T * D; ++i) h[i] += tmp[i];
  }
  arc_layernorm(h, T, D, D, m->final_g, NULL, 1e-5f);
  arc_free(fr);
  arc_free(e);
  arc_free(c1);
  arc_free(xn);
  arc_free(q);
  arc_free(k);
  arc_free(v);
  arc_free(att);
  arc_free(tmp);
  arc_free(ff);
  *T_out = T;
  return h;
fail:
  arc_free(fr);
  arc_free(e);
  arc_free(c1);
  arc_free(h);
  arc_free(xn);
  arc_free(q);
  arc_free(k);
  arc_free(v);
  arc_free(att);
  arc_free(tmp);
  arc_free(ff);
  return NULL;
}

/* ------------------------------------------------------------------ decoder */

typedef struct {
  arc_moon* m;
  int T;            /* encoder length */
  int cap;          /* max positions */
  int B;            /* beams */
  float* ck[NL];    /* cross keys / values [T, D] (encoder states + positions, projected) */
  float* cv[NL];
  float* sk;        /* self keys / values [B][NL][cap][D] */
  float* sv;
  float* sk2;       /* reorder scratch */
  float* sv2;
  int len;          /* positions filled (same for all beams) */
  float *x, *xn, *q, *kk, *vv, *att, *tmp, *ff, *logits;
  float* xatt;      /* NULL, or [NL][H][T]: the cross-attention weights of the last step (beam 0), for alignment */
  int no_logits;    /* 1: skip the output projection (alignment needs only the attention) */
} dec_state;

static float* kv_at(const dec_state* s, float* base, int b, int l) {
  return base + (((size_t)b * NL + l) * s->cap) * D;
}

static void dec_free(dec_state* s) {
  for (int l = 0; l < NL; ++l) {
    arc_free(s->ck[l]);
    arc_free(s->cv[l]);
  }
  arc_free(s->sk);
  arc_free(s->sv);
  arc_free(s->sk2);
  arc_free(s->sv2);
  arc_free(s->x);
  arc_free(s->xn);
  arc_free(s->q);
  arc_free(s->kk);
  arc_free(s->vv);
  arc_free(s->att);
  arc_free(s->tmp);
  arc_free(s->ff);
  arc_free(s->logits);
}

static int dec_init(dec_state* s, arc_moon* m, const float* enc, int T, int B, int cap) {
  memset(s, 0, sizeof *s);
  s->m = m;
  s->T = T;
  s->B = B;
  s->cap = cap;
  float* e = arc_malloc(sizeof(float) * (size_t)T * D);
  if (!e) return -1;
  for (int t = 0; t < T; ++t)
    for (int c = 0; c < D; ++c) e[(size_t)t * D + c] = enc[(size_t)t * D + c] + m->pos_emb[(size_t)t * D + c];
  for (int l = 0; l < NL; ++l) {
    s->ck[l] = arc_malloc(sizeof(float) * (size_t)T * D);
    s->cv[l] = arc_malloc(sizeof(float) * (size_t)T * D);
    if (!s->ck[l] || !s->cv[l]) {
      arc_free(e);
      return -1;
    }
    linw(m, e, T, D, &m->dec[l].ck, D, NULL, s->ck[l]);
    linw(m, e, T, D, &m->dec[l].cv, D, NULL, s->cv[l]);
  }
  arc_free(e);
  size_t kvn = (size_t)B * NL * cap * D;
  s->sk = arc_malloc(sizeof(float) * kvn);
  s->sv = arc_malloc(sizeof(float) * kvn);
  s->sk2 = arc_malloc(sizeof(float) * kvn);
  s->sv2 = arc_malloc(sizeof(float) * kvn);
  s->x = arc_malloc(sizeof(float) * (size_t)B * D);
  s->xn = arc_malloc(sizeof(float) * (size_t)B * D);
  s->q = arc_malloc(sizeof(float) * (size_t)B * D);
  s->kk = arc_malloc(sizeof(float) * (size_t)B * D);
  s->vv = arc_malloc(sizeof(float) * (size_t)B * D);
  s->att = arc_malloc(sizeof(float) * (size_t)B * D);
  s->tmp = arc_malloc(sizeof(float) * (size_t)B * D);
  s->ff = arc_malloc(sizeof(float) * (size_t)B * 2 * FF);
  s->logits = arc_malloc(sizeof(float) * (size_t)B * V);
  if (!s->sk || !s->sv || !s->sk2 || !s->sv2 || !s->x || !s->xn || !s->q || !s->kk || !s->vv || !s->att || !s->tmp ||
      !s->ff || !s->logits)
    return -1;
  return 0;
}

static void rope(const arc_moon* m, float* x, int pos) {
  /* interleaved pairs (2i, 2i+1) of the first 32 dims of each head, frequency inv_freq[i] */
  for (int hh = 0; hh < H; ++hh) {
    float* r = x + hh * HD;
    for (int i = 0; i < m->n_rot / 2; ++i) {
      float f = (float)pos * m->inv_freq[i];
      float c = cosf(f), sn = sinf(f);
      float a = r[2 * i], b = r[2 * i + 1];
      r[2 * i] = a * c - b * sn;
      r[2 * i + 1] = b * c + a * sn;
    }
  }
}

/* wout: NULL, or [H][n] receiving the attention weights */
static void attend(const float* q, const float* K, const float* Vv, int n, float* out, float* wout) {
  const float scale = 1.0f / sqrtf((float)HD);
  float* sc = (float*)alloca(sizeof(float) * (size_t)n);
  for (int hh = 0; hh < H; ++hh) {
    const float* qv = q + hh * HD;
    float mx = -FLT_MAX;
    for (int j = 0; j < n; ++j) {
      const float* kv = K + (size_t)j * D + hh * HD;
      float s = 0;
      for (int d = 0; d < HD; ++d) s += qv[d] * kv[d];
      s *= scale;
      sc[j] = s;
      if (s > mx) mx = s;
    }
    float sum = 0;
    for (int j = 0; j < n; ++j) {
      sc[j] = expf(sc[j] - mx);
      sum += sc[j];
    }
    float* o = out + hh * HD;
    for (int d = 0; d < HD; ++d) o[d] = 0;
    for (int j = 0; j < n; ++j) {
      float w = sc[j] / sum;
      if (wout) wout[(size_t)hh * n + j] = w;
      const float* vv = Vv + (size_t)j * D + hh * HD;
      for (int d = 0; d < HD; ++d) o[d] += w * vv[d];
    }
  }
}

/* One decoder position for every beam: tok[b] at position s->len; logits into s->logits [B, V]. */
static void dec_step(dec_state* s, const int* tok) {
  arc_moon* m = s->m;
  const int B = s->B, pos = s->len;
  for (int b = 0; b < B; ++b) {
    if (m->embed.packed) arc_q4_row(&m->embed.q, tok[b], s->x + (size_t)b * D);
    else memcpy(s->x + (size_t)b * D, m->embed.f + (size_t)tok[b] * D, sizeof(float) * D);
  }
  for (int l = 0; l < NL; ++l) {
    const arc_moon_dec_layer* L = &m->dec[l];
    /* causal self-attention with RoPE */
    memcpy(s->xn, s->x, sizeof(float) * (size_t)B * D);
    arc_layernorm(s->xn, B, D, D, L->ln_in, NULL, 1e-5f);
    linw(m, s->xn, B, D, &L->sq, D, NULL, s->q);
    linw(m, s->xn, B, D, &L->sk, D, NULL, s->kk);
    linw(m, s->xn, B, D, &L->sv, D, NULL, s->vv);
    for (int b = 0; b < B; ++b) {
      rope(m, s->q + (size_t)b * D, pos);
      rope(m, s->kk + (size_t)b * D, pos);
      memcpy(kv_at(s, s->sk, b, l) + (size_t)pos * D, s->kk + (size_t)b * D, sizeof(float) * D);
      memcpy(kv_at(s, s->sv, b, l) + (size_t)pos * D, s->vv + (size_t)b * D, sizeof(float) * D);
      attend(s->q + (size_t)b * D, kv_at(s, s->sk, b, l), kv_at(s, s->sv, b, l), pos + 1, s->att + (size_t)b * D, NULL);
    }
    linw(m, s->att, B, D, &L->so, D, NULL, s->tmp);
    for (size_t i = 0; i < (size_t)B * D; ++i) s->x[i] += s->tmp[i];
    /* cross-attention */
    memcpy(s->xn, s->x, sizeof(float) * (size_t)B * D);
    arc_layernorm(s->xn, B, D, D, L->ln_post, NULL, 1e-5f);
    linw(m, s->xn, B, D, &L->cq, D, NULL, s->q);
    for (int b = 0; b < B; ++b)
      attend(s->q + (size_t)b * D, s->ck[l], s->cv[l], s->T, s->att + (size_t)b * D,
             b == 0 && s->xatt ? s->xatt + (size_t)l * H * s->T : NULL);
    linw(m, s->att, B, D, &L->co, D, NULL, s->tmp);
    for (size_t i = 0; i < (size_t)B * D; ++i) s->x[i] += s->tmp[i];
    /* gated MLP: fc1 -> [h, gate], silu(gate) * h, fc2 */
    memcpy(s->xn, s->x, sizeof(float) * (size_t)B * D);
    arc_layernorm(s->xn, B, D, D, L->ln_final, NULL, 1e-5f);
    linw(m, s->xn, B, D, &L->fc1, 2 * FF, L->fc1b, s->ff);
    for (int b = 0; b < B; ++b) {
      float* r = s->ff + (size_t)b * 2 * FF;
      for (int i = 0; i < FF; ++i) r[i] = silu1(r[FF + i]) * r[i];
      /* compact into the first half for fc2 */
    }
    /* fc2 over the first halves (rows 2 FF apart), all beams in one pass */
    if (L->fc2.packed) arc_q4_linear(m->pool, s->ff, B, 2 * FF, &L->fc2.q, L->fc2b, s->tmp, D);
    else arc_gemm_nt(m->pool, B, D, FF, s->ff, 2 * FF, L->fc2.f, FF, s->tmp, D, L->fc2b, 0);
    for (size_t i = 0; i < (size_t)B * D; ++i) s->x[i] += s->tmp[i];
  }
  if (!s->no_logits) {
    memcpy(s->xn, s->x, sizeof(float) * (size_t)B * D);
    arc_layernorm(s->xn, B, D, D, m->norm, NULL, 1e-5f);
    linw(m, s->xn, B, D, &m->embed, V, NULL, s->logits);
  }
  s->len++;
}

/* Beam b's caches become those of beam src[b]. */
static void dec_reorder(dec_state* s, const int* src) {
  size_t per = (size_t)NL * s->cap * D;
  for (int b = 0; b < s->B; ++b) {
    memcpy(s->sk2 + (size_t)b * per, s->sk + (size_t)src[b] * per, sizeof(float) * per);
    memcpy(s->sv2 + (size_t)b * per, s->sv + (size_t)src[b] * per, sizeof(float) * per);
  }
  float* t = s->sk;
  s->sk = s->sk2;
  s->sk2 = t;
  t = s->sv;
  s->sv = s->sv2;
  s->sv2 = t;
}

float* arc_moon_logits(arc_moon* m, const float* enc, int T, const int* tokens, int n) {
  dec_state s;
  if (dec_init(&s, m, enc, T, 1, n + 1)) {
    dec_free(&s);
    return NULL;
  }
  float* out = arc_malloc(sizeof(float) * (size_t)n * V);
  if (!out) {
    dec_free(&s);
    return NULL;
  }
  for (int i = 0; i < n; ++i) {
    dec_step(&s, &tokens[i]);
    memcpy(out + (size_t)i * V, s.logits, sizeof(float) * V);
  }
  dec_free(&s);
  return out;
}

float* arc_moon_cross_attn(arc_moon* m, const float* enc, int T, const int* tokens, int n) {
  dec_state s;
  if (dec_init(&s, m, enc, T, 1, n + 1)) {
    dec_free(&s);
    return NULL;
  }
  float* out = arc_malloc(sizeof(float) * (size_t)n * NL * H * T);
  if (!out) {
    dec_free(&s);
    return NULL;
  }
  s.no_logits = 1;
  for (int i = 0; i < n; ++i) {
    s.xatt = out + (size_t)i * NL * H * T;
    dec_step(&s, &tokens[i]);
  }
  dec_free(&s);
  return out;
}

/* ------------------------------------------------------------------ decoding */

static void log_softmax(float* x, int n) {
  float mx = -INFINITY;
  for (int i = 0; i < n; ++i)
    if (x[i] > mx) mx = x[i];
  double sum = 0;
  for (int i = 0; i < n; ++i) sum += exp((double)(x[i] - mx));
  float lse = mx + (float)log(sum);
  for (int i = 0; i < n; ++i) x[i] -= lse;
}

static int looping(const int* seq, int n, const arc_moon_procs* p) {
  for (int u = 1; u <= p->rg_max_unit; ++u) {
    int k = p->rg_copies;
    int need = (p->rg_min_tokens + u - 1) / u;
    if (need > k) k = need;
    if (n < u * k) break;
    int ok = 1;
    for (int c = 1; c < k && ok; ++c)
      for (int j = 0; j < u; ++j)
        if (seq[n - u + j] != seq[n - (c + 1) * u + j]) {
          ok = 0;
          break;
        }
    if (ok) return 1;
  }
  return 0;
}

/* BiasProcessor on one row: the longest suffix of the generated tokens that is a name's prefix gives continuation
 * bonuses; scores are log-softmaxed, start bonuses added, then what continuations add beyond them. */
static void bias_row(const arc_moon_bias* b, float* row, const int* seq, int n) {
  const int* gen = seq + b->prompt_len;
  int ng = n - b->prompt_len;
  if (ng < 0) ng = 0;
  /* start bonus per token: max over the sequences starting with it */
  int any_start = b->n > 0;
  /* continuation: the longest k with gen[-k:] a proper prefix of some sequence */
  int nc = 0;
  int cont_tok[256];
  float cont_val[256];
  for (int k = ng < b->maxlen - 1 ? ng : b->maxlen - 1; k >= 1 && !nc; --k) {
    for (int s = 0; s < b->n; ++s) {
      int len = b->off[s + 1] - b->off[s];
      const int* ids = b->ids + b->off[s];
      if (len <= k) continue;
      int match = 1;
      for (int j = 0; j < k && match; ++j)
        if (ids[j] != gen[ng - k + j]) match = 0;
      if (!match) continue;
      int t = ids[k];
      float v = b->cont_bonus * b->w[s];
      int found = -1;
      for (int c = 0; c < nc; ++c)
        if (cont_tok[c] == t) found = c;
      if (found >= 0) {
        if (v > cont_val[found]) cont_val[found] = v;
      } else if (nc < 256) {
        cont_tok[nc] = t;
        cont_val[nc++] = v;
      }
    }
  }
  if (!any_start && !nc) return;
  log_softmax(row, V);
  /* start bonuses (one per distinct first token: the max weight among its sequences) */
  for (int s = 0; s < b->n; ++s) {
    int t = b->ids[b->off[s]];
    int first = 1;
    float w = b->w[s];
    for (int s2 = 0; s2 < b->n; ++s2)
      if (b->ids[b->off[s2]] == t) {
        if (s2 < s) first = 0;
        if (b->w[s2] > w) w = b->w[s2];
      }
    if (first) row[t] += b->start_bonus * w;
  }
  for (int c = 0; c < nc; ++c) {
    int t = cont_tok[c];
    float sb = 0;
    for (int s = 0; s < b->n; ++s)
      if (b->ids[b->off[s]] == t) {
        float v = b->start_bonus * b->w[s];
        if (v > sb) sb = v;
      }
    if (cont_val[c] > sb) row[t] += cont_val[c] - sb;
  }
}

/* Arc's processors on one row of scores (seq: the row's tokens so far, start: where the repeat guard looks from):
 * tag offsets, name bias, suppressed ids, repeat guard, in that order (transcribe.py _gen). */
static void process(const arc_moon_procs* p, float* row, const int* seq, int n, int start, int eos, int renorm_offsets) {
  (void)renorm_offsets;
  if (!p) return;
  if (p->n_off) {   /* TagOffsets: add, then log_softmax */
    for (int i = 0; i < p->n_off; ++i) row[p->off_id[i]] += p->off_val[i];
    log_softmax(row, V);
  }
  if (p->bias) bias_row(p->bias, row, seq, n);
  for (int i = 0; i < p->n_suppress; ++i) row[p->suppress[i]] = -INFINITY;
  if (p->repeat_guard && n - start > 0 && looping(seq + start, n - start, p)) {
    for (int i = 0; i < V; ++i) row[i] = -INFINITY;
    row[eos] = 0.0f;
  }
}

typedef struct {
  float v;
  int i;
} scored;

/* top k of x[n], descending (ties: lower index first) */
static void topk(const float* x, int n, int k, scored* out) {
  int have = 0;
  for (int i = 0; i < n; ++i) {
    float v = x[i];
    if (have == k && !(v > out[k - 1].v)) continue;
    int j = have < k ? have++ : k - 1;
    while (j > 0 && v > out[j - 1].v) {
      out[j] = out[j - 1];
      --j;
    }
    out[j].v = v;
    out[j].i = i;
  }
}

int arc_moon_decode(arc_moon* m, const float* enc, int T, const int* prompt, int n_prompt, int max_length, int beams,
                    const arc_moon_procs* procs, int* out, int cap) {
  return arc_moon_decode_nbest(m, enc, T, prompt, n_prompt, max_length, beams, procs, out, cap, 0, NULL, NULL, NULL, NULL);
}

int arc_moon_decode_nbest(arc_moon* m, const float* enc, int T, const int* prompt, int n_prompt, int max_length,
                          int beams, const arc_moon_procs* procs, int* out, int cap, int max_alts, int* alt_ids,
                          int* alt_len, float* alt_score, int* n_alts) {
  if (n_alts) *n_alts = 0;
  if (beams < 1) beams = 1;
  int start_len = n_prompt + 1;
  if (max_length < start_len + 1) max_length = start_len + 1;
  const int eos = m->eos, nb = beams;
  dec_state s;
  if (dec_init(&s, m, enc, T, nb, max_length + 1)) {
    dec_free(&s);
    return -1;
  }
  int* tok = arc_malloc(sizeof(int) * (size_t)nb);
  int* run = arc_malloc(sizeof(int) * (size_t)nb * max_length);   /* running sequences */
  if (!tok || !run) {
    arc_free(tok);
    arc_free(run);
    dec_free(&s);
    return -1;
  }
  for (int b = 0; b < nb; ++b) {
    for (int i = 0; i < n_prompt; ++i) run[(size_t)b * max_length + i] = prompt[i];
    run[(size_t)b * max_length + n_prompt] = m->bos;
  }
  /* prefill: the prompt and the start token, one position at a time, the same for every beam; only the last
   * position's logits are read, so the output projection (32768 x 320) is skipped before it */
  for (int i = 0; i < start_len; ++i) {
    for (int b = 0; b < nb; ++b) tok[b] = run[(size_t)b * max_length + i];
    s.no_logits = i < start_len - 1;
    dec_step(&s, tok);
  }
  s.no_logits = 0;
  int cur = start_len, n_out = -1;
  const int rg_start = start_len;
  if (nb == 1) {
    /* greedy (HF _sample without sampling): processors on the raw logits, argmax */
    while (1) {
      float* row = s.logits;
      process(procs, row, run, cur, rg_start, eos, 0);
      int best = 0;
      for (int i = 1; i < V; ++i)
        if (row[i] > row[best]) best = i;
      run[cur++] = best;
      if (best == eos || cur >= max_length) break;
      tok[0] = best;
      dec_step(&s, tok);
    }
    n_out = cur - n_prompt;
    if (n_out > cap) n_out = cap;
    memcpy(out, run + n_prompt, sizeof(int) * (size_t)n_out);
    if (max_alts > 0 && alt_ids && n_alts) {
      memcpy(alt_ids, out, sizeof(int) * (size_t)n_out);
      alt_len[0] = n_out;
      alt_score[0] = 0.0f;
      *n_alts = 1;
    }
  } else {
    /* beam search exactly as transformers' _beam_search (batch 1, one EOS, length_penalty 1, early_stopping False) */
    const int K = 2 * nb;
    const int dpl = start_len;
    float* rscore = arc_malloc(sizeof(float) * nb);
    float* bscore = arc_malloc(sizeof(float) * nb);
    int* fin = arc_malloc(sizeof(int) * nb);
    int* seq = arc_malloc(sizeof(int) * (size_t)nb * max_length);
    float* acc = arc_malloc(sizeof(float) * (size_t)nb * V);
    int* topseq = arc_malloc(sizeof(int) * (size_t)K * max_length);
    scored* tk = arc_malloc(sizeof(scored) * (size_t)K);
    scored* tk2 = arc_malloc(sizeof(scored) * (size_t)(nb + K));
    int* src = arc_malloc(sizeof(int) * nb);
    int* nrun = arc_malloc(sizeof(int) * (size_t)nb * max_length);
    int* nseq = arc_malloc(sizeof(int) * (size_t)nb * max_length);
    if (!rscore || !bscore || !fin || !seq || !acc || !topseq || !tk || !tk2 || !src || !nrun || !nseq) goto beam_done;
    for (int b = 0; b < nb; ++b) {
      rscore[b] = b == 0 ? 0.0f : -1e9f;
      bscore[b] = -1e9f;
      fin[b] = 0;
    }
    for (size_t i = 0; i < (size_t)nb * max_length; ++i) seq[i] = eos;
    for (int b = 0; b < nb; ++b) memcpy(seq + (size_t)b * max_length, run + (size_t)b * max_length, sizeof(int) * start_len);
    for (size_t i = (size_t)0; i < (size_t)nb; ++i)
      for (int j = start_len; j < max_length; ++j) run[i * max_length + j] = eos;
    int heur = 1;
    while (1) {
      for (int b = 0; b < nb; ++b) {
        float* row = acc + (size_t)b * V;
        memcpy(row, s.logits + (size_t)b * V, sizeof(float) * V);
        log_softmax(row, V);
        process(procs, row, run + (size_t)b * max_length, cur, rg_start, eos, 1);
        for (int i = 0; i < V; ++i) row[i] += rscore[b];
      }
      topk(acc, nb * V, K, tk);
      int hits[64];
      for (int k = 0; k < K; ++k) {
        int b = tk[k].i / V, t = tk[k].i % V;
        memcpy(topseq + (size_t)k * max_length, run + (size_t)b * max_length, sizeof(int) * max_length);
        topseq[(size_t)k * max_length + cur] = t;
        hits[k] = t == eos || cur + 1 >= max_length;
      }
      /* running beams for the next step: the best nb of the K that did not just finish */
      float trl[64];
      for (int k = 0; k < K; ++k) trl[k] = tk[k].v + (hits[k] ? -1e9f : 0.0f);
      scored nx[64];
      topk(trl, K, nb, nx);
      for (int b = 0; b < nb; ++b) {
        int k = nx[b].i;
        memcpy(nrun + (size_t)b * max_length, topseq + (size_t)k * max_length, sizeof(int) * max_length);
        rscore[b] = trl[k];
        src[b] = tk[k].i / V;
      }
      memcpy(run, nrun, sizeof(int) * (size_t)nb * max_length);
      /* finished beams: merge the previous ones with the top nb that just finished (length-penalised) */
      float merged[128];
      int mfin[128];
      for (int b = 0; b < nb; ++b) {
        merged[b] = bscore[b];
        mfin[b] = fin[b];
      }
      float lp = (float)(cur + 1 - dpl);
      for (int k = 0; k < K; ++k) {
        int did = hits[k] && k < nb;
        float v = tk[k].v / lp;
        if (!heur) v += -1e9f;
        if (!did) v += -1e9f;
        merged[nb + k] = v;
        mfin[nb + k] = did;
      }
      topk(merged, nb + K, nb, tk2);
      int nfin[64];
      float nbs[64];
      for (int b = 0; b < nb; ++b) {
        int j = tk2[b].i;
        const int* from = j < nb ? seq + (size_t)j * max_length : topseq + (size_t)(j - nb) * max_length;
        memcpy(nseq + (size_t)b * max_length, from, sizeof(int) * max_length);
        nbs[b] = merged[j];
        nfin[b] = mfin[j];
      }
      memcpy(seq, nseq, sizeof(int) * (size_t)nb * max_length);
      for (int b = 0; b < nb; ++b) {
        bscore[b] = nbs[b];
        fin[b] = nfin[b];
      }
      dec_reorder(&s, src);
      cur++;
      /* early-stop heuristic (early_stopping False): best running / (cur - dpl) vs the worst finished */
      float best_run = rscore[0] / (float)(cur - dpl);
      float worst = INFINITY;
      for (int b = 0; b < nb; ++b)
        if (bscore[b] < worst) worst = bscore[b];
      int any = 0;
      for (int b = 0; b < nb; ++b) {
        float w = fin[b] ? worst : -1e9f;
        if (best_run > w) any = 1;
      }
      heur = heur && any;
      int all_hit = 1;
      for (int k = 0; k < K; ++k) all_hit &= hits[k];
      if (!(heur && !all_hit)) break;
      if (cur >= max_length) break;
      for (int b = 0; b < nb; ++b) tok[b] = run[(size_t)b * max_length + cur - 1];
      dec_step(&s, tok);
    }
    /* the best finished sequence, from the start token to its EOS (HF pads the rest with EOS) */
    {
      const int* best = seq;
      int n = cur;
      for (int i = start_len; i < cur; ++i)
        if (best[i] == eos) {
          n = i + 1;
          break;
        }
      n_out = n - n_prompt;
      if (n_out > cap) n_out = cap;
      memcpy(out, best + n_prompt, sizeof(int) * (size_t)n_out);
    }
    /* the other finished beams, best first (seq / bscore are kept sorted by the merge's topk) */
    if (max_alts > 0 && alt_ids && n_alts) {
      for (int b = 0; b < nb && *n_alts < max_alts; ++b) {
        if (!fin[b] || bscore[b] < -1e8f) continue;
        const int* sq = seq + (size_t)b * max_length;
        int n = cur;
        for (int i = start_len; i < cur; ++i)
          if (sq[i] == eos) {
            n = i + 1;
            break;
          }
        int k = n - n_prompt;
        if (k > cap) k = cap;
        memcpy(alt_ids + (size_t)*n_alts * cap, sq + n_prompt, sizeof(int) * (size_t)k);
        alt_len[*n_alts] = k;
        alt_score[*n_alts] = bscore[b];
        ++*n_alts;
      }
    }
  beam_done:
    arc_free(rscore);
    arc_free(bscore);
    arc_free(fin);
    arc_free(seq);
    arc_free(acc);
    arc_free(topseq);
    arc_free(tk);
    arc_free(tk2);
    arc_free(src);
    arc_free(nrun);
    arc_free(nseq);
  }
  arc_free(tok);
  arc_free(run);
  dec_free(&s);
  return n_out;
}

/* ------------------------------------------------------------------ scoring */

/* a before b: lexicographic on the tokens, shorter first on a common prefix, then by index */
static int seq_before(const int* const* seqs, const int* lens, int i, int j) {
  const int *x = seqs[i], *y = seqs[j];
  int nx = lens[i], ny = lens[j];
  for (int k = 0; k < nx && k < ny; ++k)
    if (x[k] != y[k]) return x[k] < y[k];
  if (nx != ny) return nx < ny;
  return i < j;
}

static double row_lse(const float* row) {
  float mx = -INFINITY;
  for (int v = 0; v < V; ++v)
    if (row[v] > mx) mx = row[v];
  double s = 0;
  for (int v = 0; v < V; ++v) s += exp((double)(row[v] - mx));
  return (double)mx + log(s);
}

/* Child c takes the self-attention caches of node src[c] (positions [0, len) only). */
static void score_reorder(dec_state* s, const int* src, int nc) {
  size_t used = (size_t)s->len * D;
  for (int c = 0; c < nc; ++c)
    for (int l = 0; l < NL; ++l) {
      memcpy(kv_at(s, s->sk2, c, l), kv_at(s, s->sk, src[c], l), sizeof(float) * used);
      memcpy(kv_at(s, s->sv2, c, l), kv_at(s, s->sv, src[c], l), sizeof(float) * used);
    }
  float* t = s->sk;
  s->sk = s->sk2;
  s->sk2 = t;
  t = s->sv;
  s->sv = s->sv2;
  s->sv2 = t;
}

int arc_moon_score(arc_moon* m, const float* enc, int T, const int* prompt, int n_prompt, const int* const* seqs,
                   const int* lens, int n, double* out) {
  if (n <= 0) return 0;
  int maxlen = 0;
  for (int i = 0; i < n; ++i) {
    if (lens[i] < 0) return -1;
    if (lens[i] > maxlen) maxlen = lens[i];
  }
  const int G = ARC_MOON_SCORE_GROUP;
  int* idx = arc_malloc(sizeof(int) * (size_t)n);
  int* tok = arc_malloc(sizeof(int) * G);
  int* src = arc_malloc(sizeof(int) * G);
  int* na = arc_malloc(sizeof(int) * G);   /* node j covers group members [na[j], ne[j]) */
  int* ne = arc_malloc(sizeof(int) * G);
  int* ca = arc_malloc(sizeof(int) * G);
  int* ce = arc_malloc(sizeof(int) * G);
  float* root_logits = arc_malloc(sizeof(float) * V);
  dec_state s;
  int rc = dec_init(&s, m, enc, T, G, n_prompt + 1 + maxlen + 2);
  if (rc || !idx || !tok || !src || !na || !ne || !ca || !ce || !root_logits) rc = -1;
  if (!rc) {
    /* lexicographic order (insertion sort: lists are small), so sequences sharing a prefix are neighbours */
    for (int i = 0; i < n; ++i) {
      int v = idx[i] = i, k = i;
      while (k > 0 && seq_before(seqs, lens, v, idx[k - 1])) {
        idx[k] = idx[k - 1];
        --k;
      }
      idx[k] = v;
    }
    /* the prompt and the start token, once, in beam 0; each group starts from that state */
    s.B = 1;
    for (int i = 0; i <= n_prompt; ++i) {
      tok[0] = i < n_prompt ? prompt[i] : m->bos;
      dec_step(&s, tok);
    }
    memcpy(root_logits, s.logits, sizeof(float) * V);
  }
  const int root_len = s.len;
  for (int g0 = 0; !rc && g0 < n; g0 += G) {
    const int gn = n - g0 < G ? n - g0 : G;
    const int* mem = idx + g0;
    for (int k = 0; k < gn; ++k) out[mem[k]] = 0.0;
    /* beam 0 still holds the root's positions [0, root_len): every slot a reorder filled was copied, positions
     * [0, len) included, from a chain of nodes that starts at the root */
    s.len = root_len;
    s.B = 1;
    memcpy(s.logits, root_logits, sizeof(float) * V);
    int nn = 1;
    na[0] = 0;
    ne[0] = gn;
    for (int depth = 0;; ++depth) {
      int nc = 0;
      for (int j = 0; j < nn; ++j) {
        const float* row = s.logits + (size_t)j * V;
        const double lse = row_lse(row);
        for (int k = na[j]; k < ne[j]; ++k) {
          int i = mem[k];
          int t = depth < lens[i] ? seqs[i][depth] : m->eos;
          out[i] += (double)row[t] - lse;
        }
        /* children: the members that go on, split by their next token (contiguous: the group is sorted) */
        int k = na[j];
        while (k < ne[j] && lens[mem[k]] <= depth) ++k;
        while (k < ne[j]) {
          int t = seqs[mem[k]][depth], e = k;
          while (e < ne[j] && seqs[mem[e]][depth] == t) ++e;
          ca[nc] = k;
          ce[nc] = e;
          src[nc] = j;
          tok[nc] = t;
          ++nc;
          k = e;
        }
      }
      if (!nc) break;
      score_reorder(&s, src, nc);
      s.B = nc;
      dec_step(&s, tok);
      memcpy(na, ca, sizeof(int) * (size_t)nc);
      memcpy(ne, ce, sizeof(int) * (size_t)nc);
      nn = nc;
    }
  }
  s.B = G;
  arc_free(root_logits);
  arc_free(idx);
  arc_free(tok);
  arc_free(src);
  arc_free(na);
  arc_free(ne);
  arc_free(ca);
  arc_free(ce);
  dec_free(&s);
  return rc;
}

/* ------------------------------------------------------------------ streaming encoder
 * The encoder above, run as audio arrives: every operation is per 5 ms / 20 ms frame except the two causal convs
 * (which only look back) and the attention windows (16 frames back; layers 0, 1, 4 and 5 also 3 frames ahead, 60 ms
 * each, 240 ms in all), so each layer keeps its last inputs, keys and values and emits a frame once the frames it
 * looks ahead to have arrived. A frame's state is computed by the same per-row operations as in arc_moon_encode_ex,
 * so a streamed clip gives bit-identical states to the whole clip (the end is flushed the same way). */

typedef struct {
  float* d;
  int w;
  long base, n, cap;   /* frames base .. base + n - 1 */
} fbuf;

static float* fb_at(fbuf* b, long t) { return b->d + (size_t)(t - b->base) * b->w; }

static int fb_room(fbuf* b, long k) {
  if (b->n + k <= b->cap) return 0;
  long nc = b->cap ? b->cap : 64;
  while (nc < b->n + k) nc *= 2;
  float* d = arc_malloc(sizeof(float) * (size_t)nc * b->w);
  if (!d) return -1;
  if (b->n) memcpy(d, b->d, sizeof(float) * (size_t)b->n * b->w);
  arc_free(b->d);
  b->d = d;
  b->cap = nc;
  return 0;
}

static int fb_push(fbuf* b, const float* rows, long k) {
  if (k <= 0) return 0;
  if (fb_room(b, k)) return -1;
  memcpy(b->d + (size_t)b->n * b->w, rows, sizeof(float) * (size_t)k * b->w);
  b->n += k;
  return 0;
}

/* forget frames before t */
static void fb_drop(fbuf* b, long t) {
  long k = t - b->base;
  if (k <= 0) return;
  if (k > b->n) k = b->n;
  memmove(b->d, b->d + (size_t)k * b->w, sizeof(float) * (size_t)(b->n - k) * b->w);
  b->n -= k;
  b->base += k;
}

typedef struct {
  fbuf x, q, k, v;   /* the layer's inputs and their queries / keys / values */
  long done;         /* outputs emitted */
} ms_layer;

struct arc_moon_stream {
  arc_moon* m;
  int tap;
  float pend[ARC_MOON_FRAME];
  int n_pend;
  fbuf e;            /* 5 ms frames after linear + SiLU */
  fbuf c1;           /* conv1 output after SiLU */
  long T5, T1, T2;   /* frames so far at each rate */
  ms_layer L[ARC_MOON_LAYERS];
  fbuf out, tapo;    /* encoder output and tap frames not yet taken */
  int finished;
};

arc_moon_stream* arc_moon_stream_new(arc_moon* m, int tap) {
  arc_moon_stream* s = arc_calloc(sizeof *s);
  if (!s) return NULL;
  s->m = m;
  s->tap = tap;
  s->e.w = D;
  s->c1.w = 2 * D;
  for (int l = 0; l < NL; ++l) s->L[l].x.w = s->L[l].q.w = s->L[l].k.w = s->L[l].v.w = D;
  s->out.w = s->tapo.w = D;
  return s;
}

void arc_moon_stream_free(arc_moon_stream* s) {
  if (!s) return;
  arc_free(s->e.d);
  arc_free(s->c1.d);
  for (int l = 0; l < NL; ++l) {
    arc_free(s->L[l].x.d);
    arc_free(s->L[l].q.d);
    arc_free(s->L[l].k.d);
    arc_free(s->L[l].v.d);
  }
  arc_free(s->out.d);
  arc_free(s->tapo.d);
  arc_free(s);
}

long arc_moon_stream_frames(const arc_moon_stream* s) { return s->T2; }

/* outputs [lo, hi) of a causal stride-2 conv (kernel 5, 4 frames of left padding) over in (frames 0 .. nin - 1) */
static int ms_conv(arc_moon* m, fbuf* in, long lo, long hi, const float* w, const float* b, int Cin, int Cout,
                   float* y) {
  long a = 2 * lo - 4, n = 2 * (hi - 1) - a + 1;   /* input frames a .. 2 (hi - 1) */
  float* win = arc_calloc(sizeof(float) * (size_t)n * Cin);
  if (!win) return -1;
  for (long t = a; t < a + n; ++t)
    if (t >= in->base) memcpy(win + (size_t)(t - a) * Cin, fb_at(in, t), sizeof(float) * Cin);
  arc_mat wm = {.type = ARC_W_F32, .O = Cout, .K = Cin * 5, .w = w};
  int r = arc_conv1d(m->pool, win, (int)n, Cin, Cin, &wm, 5, 2, 1, 0, 0, b, y, Cout);
  arc_free(win);
  return r == hi - lo ? 0 : -1;
}

/* layer l: take k new input frames, emit the outputs that are ready (all of them when final) into the next layer
 * or the output */
static int ms_layer_step(arc_moon_stream* s, int l, const float* rows, long k, int final);

static int ms_emit(arc_moon_stream* s, int l, const float* rows, long k, int final) {
  if (l + 1 < NL) return ms_layer_step(s, l + 1, rows, k, final);
  if (k <= 0) return 0;
  float* h = arc_malloc(sizeof(float) * (size_t)k * D);
  if (!h) return -1;
  memcpy(h, rows, sizeof(float) * (size_t)k * D);
  arc_layernorm(h, (int)k, D, D, s->m->final_g, NULL, 1e-5f);
  int r = fb_push(&s->out, h, k);
  arc_free(h);
  return r;
}

static int ms_layer_step(arc_moon_stream* s, int l, const float* rows, long k, int final) {
  arc_moon* m = s->m;
  ms_layer* L = &s->L[l];
  const arc_moon_enc_layer* W = &m->enc[l];
  if (l == s->tap && k > 0 && fb_push(&s->tapo, rows, k)) return -1;
  if (k > 0) {
    float* xn = arc_malloc(sizeof(float) * (size_t)k * D);
    if (!xn || fb_push(&L->x, rows, k) || fb_room(&L->q, k) || fb_room(&L->k, k) || fb_room(&L->v, k)) {
      arc_free(xn);
      return -1;
    }
    memcpy(xn, rows, sizeof(float) * (size_t)k * D);
    arc_layernorm(xn, (int)k, D, D, W->ln1, NULL, 1e-5f);
    linw(m, xn, (int)k, D, &W->q, D, NULL, L->q.d + (size_t)L->q.n * D);
    linw(m, xn, (int)k, D, &W->k, D, NULL, L->k.d + (size_t)L->k.n * D);
    linw(m, xn, (int)k, D, &W->v, D, NULL, L->v.d + (size_t)L->v.n * D);
    L->q.n += k;
    L->k.n += k;
    L->v.n += k;
    arc_free(xn);
  }
  long T = L->x.base + L->x.n;   /* inputs so far */
  int left = m->win[l][0], right = m->win[l][1], ahead = right > 0 ? right - 1 : 0;
  long ready = final ? T : T - ahead;
  long n = ready - L->done;
  if (n <= 0) return final ? ms_emit(s, l, NULL, 0, final) : 0;
  float* att = arc_malloc(sizeof(float) * (size_t)n * D);
  float* tmp = arc_malloc(sizeof(float) * (size_t)n * D);
  float* h = arc_malloc(sizeof(float) * (size_t)n * D);
  float* xn = arc_malloc(sizeof(float) * (size_t)n * D);
  float* ff = arc_malloc(sizeof(float) * (size_t)n * FF);
  int rc = -1;
  if (!att || !tmp || !h || !xn || !ff) goto out;
  const float scale = 1.0f / sqrtf((float)HD);
  float sc[64];
  for (long i = 0; i < n; ++i) {
    long t = L->done + i;
    long k0 = t - left + 1, k1 = t + ahead;   /* inclusive, as the whole-clip loop */
    if (k0 < 0) k0 = 0;
    if (k1 > T - 1) k1 = T - 1;
    for (int hh = 0; hh < H; ++hh) {
      const float* qv = fb_at(&L->q, t) + hh * HD;
      float mx = -FLT_MAX;
      for (long j = k0; j <= k1; ++j) {
        const float* kv = fb_at(&L->k, j) + hh * HD;
        float sv = 0;
        for (int d = 0; d < HD; ++d) sv += qv[d] * kv[d];
        sv *= scale;
        sc[j - k0] = sv;
        if (sv > mx) mx = sv;
      }
      float sum = 0;
      for (long j = k0; j <= k1; ++j) {
        sc[j - k0] = expf(sc[j - k0] - mx);
        sum += sc[j - k0];
      }
      float* o = att + (size_t)i * D + hh * HD;
      for (int d = 0; d < HD; ++d) o[d] = 0;
      for (long j = k0; j <= k1; ++j) {
        float wgt = sc[j - k0] / sum;
        const float* vv = fb_at(&L->v, j) + hh * HD;
        for (int d = 0; d < HD; ++d) o[d] += wgt * vv[d];
      }
    }
  }
  linw(m, att, (int)n, D, &W->o, D, NULL, tmp);
  memcpy(h, fb_at(&L->x, L->done), sizeof(float) * (size_t)n * D);
  for (size_t i = 0; i < (size_t)n * D; ++i) h[i] += tmp[i];
  memcpy(xn, h, sizeof(float) * (size_t)n * D);
  arc_layernorm(xn, (int)n, D, D, W->ln2, NULL, 1e-5f);
  linw(m, xn, (int)n, D, &W->fc1, FF, W->fc1b, ff);
  arc_gelu(ff, (size_t)n * FF);
  linw(m, ff, (int)n, FF, &W->fc2, D, W->fc2b, tmp);
  for (size_t i = 0; i < (size_t)n * D; ++i) h[i] += tmp[i];
  L->done = ready;
  /* keep what later outputs still read: inputs from the next output on, keys / values 15 frames back */
  fb_drop(&L->x, L->done);
  fb_drop(&L->q, L->done);
  fb_drop(&L->k, L->done - left + 1);
  fb_drop(&L->v, L->done - left + 1);
  rc = ms_emit(s, l, h, n, final);
out:
  arc_free(att);
  arc_free(tmp);
  arc_free(h);
  arc_free(xn);
  arc_free(ff);
  return rc;
}

int arc_moon_stream_push(arc_moon_stream* s, const float* pcm, size_t n, int final) {
  arc_moon* m = s->m;
  const int F = ARC_MOON_FRAME;
  if (s->finished) return n || !final ? -1 : 0;
  /* 5 ms frames: CMVN, asinh, linear, SiLU */
  size_t avail = (size_t)s->n_pend + n;
  long nf = (long)(avail / F);
  if (nf > 0) {
    float* fr = arc_malloc(sizeof(float) * (size_t)nf * F);
    float* e = arc_malloc(sizeof(float) * (size_t)nf * D);
    if (!fr || !e) {
      arc_free(fr);
      arc_free(e);
      return -1;
    }
    size_t used = 0;
    for (long t = 0; t < nf; ++t) {
      float sbuf[ARC_MOON_FRAME];
      for (int i = 0; i < F; ++i) {
        size_t idx = (size_t)t * F + i;
        sbuf[i] = idx < (size_t)s->n_pend ? s->pend[idx] : pcm[idx - s->n_pend];
      }
      used = (size_t)(t + 1) * F;
      float* o = fr + (size_t)t * F;
      float mean = 0;
      for (int i = 0; i < F; ++i) mean += sbuf[i];
      mean /= F;
      float ms = 0;
      for (int i = 0; i < F; ++i) {
        float c = sbuf[i] - mean;
        ms += c * c;
      }
      float rms = sqrtf(ms / F + 1e-6f);
      for (int i = 0; i < F; ++i) o[i] = asinhf(m->comp_k * ((sbuf[i] - mean) / rms));
    }
    linear(m, fr, (int)nf, F, m->lin, D, NULL, e);
    for (size_t i = 0; i < (size_t)nf * D; ++i) e[i] = silu1(e[i]);
    int rc = fb_push(&s->e, e, nf);
    arc_free(fr);
    arc_free(e);
    if (rc) return -1;
    s->T5 += nf;
    /* the rest of this push becomes the pending samples */
    size_t rest = avail - used;
    float keep[ARC_MOON_FRAME];
    for (size_t i = 0; i < rest; ++i) {
      size_t idx = used + i;
      keep[i] = idx < (size_t)s->n_pend ? s->pend[idx] : pcm[idx - s->n_pend];
    }
    memcpy(s->pend, keep, sizeof(float) * rest);
    s->n_pend = (int)rest;
  } else {
    memcpy(s->pend + s->n_pend, pcm, sizeof(float) * n);
    s->n_pend += (int)n;
  }
  if (final) {
    s->finished = 1;
    s->n_pend = 0;   /* the whole-clip encoder drops the last n % 80 samples too */
  }
  /* conv1 (+ SiLU), conv2: causal, so an output exists as soon as its last input does */
  long T1 = s->T5 > 0 ? (s->T5 - 1) / 2 + 1 : 0;
  if (T1 > s->T1) {
    long k = T1 - s->T1;
    float* c = arc_malloc(sizeof(float) * (size_t)k * 2 * D);
    if (!c || ms_conv(m, &s->e, s->T1, T1, m->conv1, m->conv1b, D, 2 * D, c)) {
      arc_free(c);
      return -1;
    }
    for (size_t i = 0; i < (size_t)k * 2 * D; ++i) c[i] = silu1(c[i]);
    int rc = fb_push(&s->c1, c, k);
    arc_free(c);
    if (rc) return -1;
    s->T1 = T1;
    fb_drop(&s->e, 2 * s->T1 - 4);
  }
  long T2 = s->T1 > 0 ? (s->T1 - 1) / 2 + 1 : 0;
  float* h = NULL;
  long k = T2 - s->T2;
  if (k > 0) {
    h = arc_malloc(sizeof(float) * (size_t)k * D);
    if (!h || ms_conv(m, &s->c1, s->T2, T2, m->conv2, m->conv2b, 2 * D, D, h)) {
      arc_free(h);
      return -1;
    }
    s->T2 = T2;
    fb_drop(&s->c1, 2 * s->T2 - 4);
  }
  int rc = (k > 0 || final) ? ms_layer_step(s, 0, h, k > 0 ? k : 0, final) : 0;
  arc_free(h);
  return rc;
}

long arc_moon_stream_take(arc_moon_stream* s, float** out, float** tap_out, long* n_tap) {
  long n = s->out.n;
  *out = NULL;
  if (n) {
    *out = arc_malloc(sizeof(float) * (size_t)n * D);
    if (*out) memcpy(*out, s->out.d, sizeof(float) * (size_t)n * D);
    s->out.base += n;
    s->out.n = 0;
  }
  if (tap_out) {
    *tap_out = NULL;
    *n_tap = s->tapo.n;
    if (s->tapo.n) {
      *tap_out = arc_malloc(sizeof(float) * (size_t)s->tapo.n * D);
      if (*tap_out) memcpy(*tap_out, s->tapo.d, sizeof(float) * (size_t)s->tapo.n * D);
      s->tapo.base += s->tapo.n;
      s->tapo.n = 0;
    }
  }
  return n;
}
