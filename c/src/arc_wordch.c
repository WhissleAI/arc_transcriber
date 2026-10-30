/* The word channel: a C port of arc_audio/wordch.py (WordChannel.embed_in / run_enc / frame_ctx, WordFeed.sync /
 * encode_upto / fuse, rule_arrival) as Engine._fuse runs it offline. Part of arc_transcriber (Whissle Inc.),
 * CC BY-NC 4.0; see LICENSE. */
#include "arc_wordch.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arc_engine.h"

#define DM 512
#define DE 320    /* piece embedding */
#define DW 256    /* word width */
#define DC 128    /* frame context width */
#define HC 4      /* frame context heads */
#define HE 4      /* word encoder heads */
#define FFW 512
#define P_MAX 8
#define K_CTX 8
#define BACK 6
#define ENC_WIN 32
#define N_ENC 2
#define RF ((ENC_WIN - 1) * N_ENC)
#define OFF_CLIP 32
#define NTOK 32768

typedef struct {
  const float *n1w, *n1b, *n2w, *n2b, *qkv_b, *o_b, *ff0_b, *ff3_b;
  arc_w qkv, o, ff0, ff3;
} enc_layer;

struct arc_wordch {
  const uint16_t* tokE;   /* [NTOK, DE] float16, in the mapping */
  const float *ppos, *W1b, *lnw, *lnb, *fl[5], *src, *Wab, *hav, *encw, *encb, *fqb, *fkb, *fvb, *R0, *R1, *fob;
  int fl_n[5];
  arc_w W1, W2, Wa, fq, fk, fv, fo;
  enc_layer L[N_ENC];
};

static const float* F(arc_store* s, const char* n, int64_t c, int* bad) {
  const float* p = arc_store_f32(s, n, c);
  if (!p) *bad = 1;
  return p;
}

arc_wordch* arc_wordch_load(arc_store* s) {
  if (!arc_store_has(s, "word.tok_E")) return NULL;
  const arc_tensor* te = arc_store_raw(s, "word.tok_E");
  if (!te || te->dtype != ARC_F16 || arc_numel(te) != (int64_t)NTOK * DE) return NULL;
  arc_wordch* w = arc_calloc(sizeof *w);
  if (!w) return NULL;
  int bad = 0;
  w->tokE = te->data;
  w->ppos = F(s, "word.ppos", P_MAX * DE, &bad);
  bad |= arc_w_load(s, "word.W1.weight", DW * DE, &w->W1);
  w->W1b = F(s, "word.W1.bias", DW, &bad);
  bad |= arc_w_load(s, "word.W2.weight", DW * DE, &w->W2);
  w->lnw = F(s, "word.ln.weight", DW, &bad);
  w->lnb = F(s, "word.ln.bias", DW, &bad);
  const int fln[5] = {2, 2, P_MAX + 1, 21, 21};
  for (int i = 0; i < 5; ++i) {
    w->fl_n[i] = fln[i];
    w->fl[i] = F(s, arc_namef("word.fl.%d.weight", i), (int64_t)fln[i] * DW, &bad);
  }
  w->src = F(s, "word.src.weight", 5 * DW, &bad);
  bad |= arc_w_load(s, "word.Wa.weight", DW * 2 * DM, &w->Wa);
  w->Wab = F(s, "word.Wa.bias", DW, &bad);
  w->hav = F(s, "word.hav.weight", 2 * DW, &bad);
  for (int l = 0; l < N_ENC; ++l) {
    enc_layer* L = &w->L[l];
    L->n1w = F(s, arc_namef("word.enc.%d.n1.weight", l), DW, &bad);
    L->n1b = F(s, arc_namef("word.enc.%d.n1.bias", l), DW, &bad);
    L->n2w = F(s, arc_namef("word.enc.%d.n2.weight", l), DW, &bad);
    L->n2b = F(s, arc_namef("word.enc.%d.n2.bias", l), DW, &bad);
    bad |= arc_w_load(s, arc_namef("word.enc.%d.qkv.weight", l), 3 * DW * DW, &L->qkv);
    L->qkv_b = F(s, arc_namef("word.enc.%d.qkv.bias", l), 3 * DW, &bad);
    bad |= arc_w_load(s, arc_namef("word.enc.%d.o.weight", l), DW * DW, &L->o);
    L->o_b = F(s, arc_namef("word.enc.%d.o.bias", l), DW, &bad);
    bad |= arc_w_load(s, arc_namef("word.enc.%d.ff.0.weight", l), FFW * DW, &L->ff0);
    L->ff0_b = F(s, arc_namef("word.enc.%d.ff.0.bias", l), FFW, &bad);
    bad |= arc_w_load(s, arc_namef("word.enc.%d.ff.3.weight", l), DW * FFW, &L->ff3);
    L->ff3_b = F(s, arc_namef("word.enc.%d.ff.3.bias", l), DW, &bad);
  }
  w->encw = F(s, "word.enc_norm.weight", DW, &bad);
  w->encb = F(s, "word.enc_norm.bias", DW, &bad);
  bad |= arc_w_load(s, "word.fq.weight", DC * DM, &w->fq);
  w->fqb = F(s, "word.fq.bias", DC, &bad);
  bad |= arc_w_load(s, "word.fk.weight", DC * DW, &w->fk);
  w->fkb = F(s, "word.fk.bias", DC, &bad);
  bad |= arc_w_load(s, "word.fv.weight", DC * DW, &w->fv);
  w->fvb = F(s, "word.fv.bias", DC, &bad);
  w->R0 = F(s, "word.R0.weight", (2 * OFF_CLIP + 1) * DC, &bad);
  w->R1 = F(s, "word.R1.weight", (2 * OFF_CLIP + 1) * DC, &bad);
  bad |= arc_w_load(s, "word.fo.weight", DM * DC, &w->fo);
  w->fob = F(s, "word.fo.bias", DM, &bad);
  if (bad) {
    arc_free(w);
    return NULL;
  }
  return w;
}

void arc_wordch_free(arc_wordch* w) { arc_free(w); }

/* norm_word: lower case, spaces collapsed, non-word characters off both ends (UTF-8 letters count as word
 * characters; the General Punctuation block and Latin-1 punctuation do not) */
static int wordch(const unsigned char* p, int* len) {
  if (p[0] < 0x80) {
    *len = 1;
    return (p[0] >= '0' && p[0] <= '9') || (p[0] >= 'a' && p[0] <= 'z') || (p[0] >= 'A' && p[0] <= 'Z') || p[0] == '_';
  }
  int l = p[0] >= 0xF0 ? 4 : p[0] >= 0xE0 ? 3 : p[0] >= 0xC0 ? 2 : 1;
  for (int k = 1; k < l; ++k)
    if (!p[k]) l = k;
  *len = l;
  if (l == 3 && p[0] == 0xE2 && (p[1] == 0x80 || p[1] == 0x81)) return 0;   /* U+2000..U+207F */
  if (l == 2 && p[0] == 0xC2 && p[1] >= 0xA0 && p[1] <= 0xBF && p[1] != 0xAA && p[1] != 0xB5 && p[1] != 0xBA &&
      p[1] != 0xB2 && p[1] != 0xB3 && p[1] != 0xB9 && p[1] != 0xBC && p[1] != 0xBD && p[1] != 0xBE)
    return 0;
  if (l == 2 && p[0] == 0xC3 && (p[1] == 0x97 || p[1] == 0xB7)) return 0;   /* × ÷ */
  return 1;
}

static char* norm_word(const char* w) {
  size_t n = strlen(w);
  char* t = arc_malloc(n + 1);
  size_t k = 0;
  int sp = 0;
  for (const char* p = w; *p; ++p) {   /* ' '.join(lower().split()) */
    if (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == '\f' || *p == '\v') {
      sp = k > 0;
      continue;
    }
    if (sp) t[k++] = ' ';
    sp = 0;
    t[k++] = (*p >= 'A' && *p <= 'Z') ? (char)(*p + 32) : *p;
  }
  t[k] = 0;
  /* strip leading non-word characters */
  size_t a = 0;
  while (a < k) {
    int l;
    if (wordch((const unsigned char*)t + a, &l)) break;
    a += (size_t)l;
  }
  /* strip trailing: find the end of the last word character */
  size_t end = a, i = a;
  while (i < k) {
    int l;
    int wc = wordch((const unsigned char*)t + i, &l);
    i += (size_t)l;
    if (wc) end = i;
  }
  memmove(t, t + a, end - a);
  t[end - a] = 0;
  return t;
}

static void rope64(float* x, int pos) {   /* _rope on one head of 64: interleaved pairs */
  for (int i = 0; i < 32; ++i) {
    double inv = 1.0 / pow(10000.0, (2.0 * i) / 64.0);
    float ang = (float)pos * (float)inv, c = cosf(ang), s = sinf(ang);
    float x1 = x[2 * i], x2 = x[2 * i + 1];
    x[2 * i] = x1 * c - x2 * s;
    x[2 * i + 1] = x1 * s + x2 * c;
  }
}

/* run_enc over rows X [n, DW] with absolute positions pos0..; out [n, DW] */
static void run_enc(const arc_wordch* w, arc_pool* pool, const float* X, int n, int pos0, float* out) {
  float* x = arc_malloc(sizeof(float) * (size_t)n * DW);
  float* xn = arc_malloc(sizeof(float) * (size_t)n * DW);
  float* qkv = arc_malloc(sizeof(float) * (size_t)n * 3 * DW);
  float* att = arc_malloc(sizeof(float) * (size_t)n * DW);
  float* tmp = arc_malloc(sizeof(float) * (size_t)n * FFW);
  float* sc = arc_malloc(sizeof(float) * ENC_WIN);
  memcpy(x, X, sizeof(float) * (size_t)n * DW);
  for (int l = 0; l < N_ENC; ++l) {
    const enc_layer* L = &w->L[l];
    memcpy(xn, x, sizeof(float) * (size_t)n * DW);
    arc_layernorm(xn, n, DW, DW, L->n1w, L->n1b, 1e-5f);
    arc_w_linear(pool, xn, n, DW, &L->qkv, 3 * DW, L->qkv_b, qkv);
    for (int i = 0; i < n; ++i)
      for (int h = 0; h < HE; ++h) {
        rope64(qkv + (size_t)i * 3 * DW + h * 64, pos0 + i);
        rope64(qkv + (size_t)i * 3 * DW + DW + h * 64, pos0 + i);
      }
    const float scale = 1.0f / 8.0f;
    for (int i = 0; i < n; ++i)
      for (int h = 0; h < HE; ++h) {
        const float* q = qkv + (size_t)i * 3 * DW + h * 64;
        int j0 = i - ENC_WIN + 1 > 0 ? i - ENC_WIN + 1 : 0;
        float mx = -INFINITY;
        for (int j = j0; j <= i; ++j) {
          const float* k = qkv + (size_t)j * 3 * DW + DW + h * 64;
          float s = 0;
          for (int d = 0; d < 64; ++d) s += q[d] * k[d];
          s *= scale;
          sc[j - j0] = s;
          if (s > mx) mx = s;
        }
        float sum = 0;
        for (int j = j0; j <= i; ++j) {
          sc[j - j0] = expf(sc[j - j0] - mx);
          sum += sc[j - j0];
        }
        float* o = att + (size_t)i * DW + h * 64;
        for (int d = 0; d < 64; ++d) o[d] = 0;
        for (int j = j0; j <= i; ++j) {
          const float* v = qkv + (size_t)j * 3 * DW + 2 * DW + h * 64;
          float a = sc[j - j0] / sum;
          for (int d = 0; d < 64; ++d) o[d] += a * v[d];
        }
      }
    arc_w_linear(pool, att, n, DW, &L->o, DW, L->o_b, xn);
    for (size_t k = 0; k < (size_t)n * DW; ++k) x[k] += xn[k];
    memcpy(xn, x, sizeof(float) * (size_t)n * DW);
    arc_layernorm(xn, n, DW, DW, L->n2w, L->n2b, 1e-5f);
    arc_w_linear(pool, xn, n, DW, &L->ff0, FFW, L->ff0_b, tmp);
    arc_gelu(tmp, (size_t)n * FFW);
    arc_w_linear(pool, tmp, n, FFW, &L->ff3, DW, L->ff3_b, xn);
    for (size_t k = 0; k < (size_t)n * DW; ++k) x[k] += xn[k];
  }
  arc_layernorm(x, n, DW, DW, w->encw, w->encb, 1e-5f);
  memcpy(out, x, sizeof(float) * (size_t)n * DW);
  arc_free(x);
  arc_free(xn);
  arc_free(qkv);
  arc_free(att);
  arc_free(tmp);
  arc_free(sc);
}

static int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

int arc_wordch_fuse(arc_wordch* w, arc_pool* pool, const arc_tok* tok, const arc_wc_word* words, int nw0, const float* Ha,
                    int N, int d, int chunk, float* Hf) {
  if (Hf != Ha) memcpy(Hf, Ha, sizeof(float) * (size_t)N * DM);
  if (!w || nw0 <= 0 || N <= 0) return 0;
  /* sync: the words with a non-empty normalised form */
  char** txt = arc_malloc(sizeof(char*) * (size_t)nw0);
  int *f0 = arc_malloc(sizeof(int) * (size_t)nw0), *f1 = arc_malloc(sizeof(int) * (size_t)nw0);
  int *oe = arc_malloc(sizeof(int) * (size_t)nw0), *av = arc_malloc(sizeof(int) * (size_t)nw0);
  int *src = arc_malloc(sizeof(int) * (size_t)nw0), *ue = arc_malloc(sizeof(int) * (size_t)nw0);
  int nw = 0;
  for (int k = 0; k < nw0; ++k) {
    char* t = norm_word(words[k].w);
    if (!t[0]) {
      arc_free(t);
      continue;
    }
    txt[nw] = t;
    f0[nw] = words[k].f0;
    f1[nw] = words[k].f1;
    src[nw] = words[k].src;
    nw++;
  }
  if (!nw) goto done;
  /* own_ends; rule_arrival (mode final: the utterance's end + 15, at least oe + d, non-decreasing, minus d) */
  for (int k = 0; k < nw; ++k) {
    oe[k] = f1[k];
    if (k + 1 < nw && f0[k] < f0[k + 1] && f0[k + 1] <= f1[k]) oe[k] = f0[k + 1] - 1;
    if (oe[k] < f0[k]) oe[k] = f0[k];
  }
  for (int k = nw - 1, m = 0x7fffffff; k >= 0; --k) {
    int brk = k == nw - 1 || f0[k + 1] - oe[k] > 4;
    if (brk && oe[k] < m) m = oe[k];
    ue[k] = m;
  }
  for (int k = 0, mx = -0x7fffffff; k < nw; ++k) {
    int a = ue[k] + 15;
    if (a < oe[k] + d) a = oe[k] + d;
    if (a > mx) mx = a;
    av[k] = mx - d;
  }
  float* X = arc_malloc(sizeof(float) * (size_t)nw * DW);
  float* U = arc_malloc(sizeof(float) * (size_t)nw * DW);
  float* E = arc_malloc(sizeof(float) * (size_t)nw * DW);   /* fk(U) */
  float* Vv = arc_malloc(sizeof(float) * (size_t)nw * DC);
  float* Kk = arc_malloc(sizeof(float) * (size_t)nw * DC);
  int n_x = 0, n_u = 0;
  float mean[DE], last[DE], row[DE], a1[DW], a2[DW], hcat[2 * DM], fq[DC], ctx[DC], sc[K_CTX];
  for (int s0 = 0; s0 < N; s0 += chunk) {
    int f_done = s0 + chunk < N ? s0 + chunk : N;
    /* encode_upto(f_done) */
    int n_new = 0;
    while (n_x + n_new < nw && oe[n_x + n_new] < f_done) {
      int k = n_x + n_new;
      if (k + 1 >= nw && f_done <= f1[k] + 4) break;
      n_new++;
    }
    if (n_new) {
      for (int k = n_x; k < n_x + n_new; ++k) {
        int ids[64];
        int np = arc_tok_encode(tok, txt[k], ids, 64);
        if (np < 0) np = 0;
        if (np > 64) np = 64;
        if (np > P_MAX) {
          ids[P_MAX - 1] = ids[np - 1];
          np = P_MAX;
        }
        /* embed_in: mean of (piece + position) over the pieces, the last piece, two linears, LN, flags, source, the
         * trunk states under the word (mean, last) and "have" */
        for (int c = 0; c < DE; ++c) mean[c] = 0;
        for (int p = 0; p < np; ++p) {
          const uint16_t* e = w->tokE + (size_t)ids[p] * DE;
          for (int c = 0; c < DE; ++c) mean[c] += arc_half_to_float(e[c]) + w->ppos[p * DE + c];
        }
        for (int c = 0; c < DE; ++c) mean[c] /= (float)(np > 1 ? np : 1);
        int lp = np > 0 ? ids[np - 1] : ids[0];
        if (np == 0) lp = 0;
        for (int c = 0; c < DE; ++c) last[c] = arc_half_to_float(w->tokE[(size_t)lp * DE + c]);
        (void)row;
        arc_w_linear(NULL, mean, 1, DE, &w->W1, DW, w->W1b, a1);
        arc_w_linear(NULL, last, 1, DE, &w->W2, DW, NULL, a2);
        float* x = X + (size_t)k * DW;
        for (int c = 0; c < DW; ++c) x[c] = a1[c] + a2[c];
        arc_layernorm(x, 1, DW, DW, w->lnw, w->lnb, 1e-5f);
        int hd = 0, alld = 1;
        for (const char* p = txt[k]; *p; ++p) {
          if (*p >= '0' && *p <= '9') hd = 1;
          else alld = 0;
        }
        int prev_oe = k > 0 ? oe[k - 1] : -1;
        int fl[5] = {hd, hd && alld, clampi(np, 0, P_MAX), clampi(oe[k] - f0[k] + 1, 1, 20),
                     k == 0 ? 20 : clampi(f0[k] - prev_oe - 1, 0, 20)};
        for (int i = 0; i < 5; ++i) {
          const float* e = w->fl[i] + (size_t)clampi(fl[i], 0, w->fl_n[i] - 1) * DW;
          for (int c = 0; c < DW; ++c) x[c] += e[c];
        }
        const float* se = w->src + (size_t)clampi(src[k], 0, 4) * DW;
        for (int c = 0; c < DW; ++c) x[c] += se[c];
        /* trunk states over f0..oe (clamped to what has been fed) */
        for (int c = 0; c < 2 * DM; ++c) hcat[c] = 0;
        int cnt = 0;
        for (int f = f0[k]; f <= oe[k]; ++f) {
          const float* h = Ha + (size_t)clampi(f, 0, f_done - 1) * DM;
          for (int c = 0; c < DM; ++c) hcat[c] += h[c];
          cnt++;
        }
        for (int c = 0; c < DM; ++c) hcat[c] /= (float)(cnt ? cnt : 1);
        memcpy(hcat + DM, Ha + (size_t)clampi(oe[k], 0, f_done - 1) * DM, sizeof(float) * DM);
        arc_w_linear(NULL, hcat, 1, 2 * DM, &w->Wa, DW, w->Wab, a1);
        for (int c = 0; c < DW; ++c) x[c] += a1[c] + w->hav[DW + c];
      }
      n_x += n_new;
      int lo = n_u - RF > 0 ? n_u - RF : 0;
      float* uo = arc_malloc(sizeof(float) * (size_t)(n_x - lo) * DW);
      run_enc(w, pool, X + (size_t)lo * DW, n_x - lo, lo, uo);
      memcpy(U + (size_t)n_u * DW, uo + (size_t)(n_u - lo) * DW, sizeof(float) * (size_t)(n_x - n_u) * DW);
      arc_free(uo);
      arc_w_linear(pool, U + (size_t)n_u * DW, n_x - n_u, DW, &w->fk, DC, w->fkb, Kk + (size_t)n_u * DC);
      arc_w_linear(pool, U + (size_t)n_u * DW, n_x - n_u, DW, &w->fv, DC, w->fvb, Vv + (size_t)n_u * DC);
      n_u = n_x;
    }
    if (!n_u) continue;
    /* fuse frames [s0, f_done): frame_window over the arrived words, then frame_ctx */
    for (int t = s0; t < f_done; ++t) {
      int na = 0, np = 0;
      while (na < n_u && av[na] <= t) na++;
      while (np < n_u && f0[np] <= t) np++;
      int q = np < na ? np : na;
      int idx[K_CTX], any = 0;
      for (int k = 0; k < K_CTX; ++k) {
        int j = q - BACK + k;
        idx[k] = j >= 0 && j < na ? j : -1;
        any |= idx[k] >= 0;
      }
      if (!any) continue;
      const float* h = Hf + (size_t)t * DM;
      arc_w_linear(NULL, h, 1, DM, &w->fq, DC, w->fqb, fq);
      for (int c = 0; c < DC; ++c) ctx[c] = 0;
      for (int hh = 0; hh < HC; ++hh) {
        float mx = -INFINITY;
        for (int k = 0; k < K_CTX; ++k) {
          if (idx[k] < 0) {
            sc[k] = -INFINITY;
            continue;
          }
          int j = idx[k];
          const float* r0 = w->R0 + (size_t)(clampi(t - f0[j], -OFF_CLIP, OFF_CLIP) + OFF_CLIP) * DC;
          const float* r1 = w->R1 + (size_t)(clampi(t - oe[j], -OFF_CLIP, OFF_CLIP) + OFF_CLIP) * DC;
          const float* kk = Kk + (size_t)j * DC;
          float s = 0;
          for (int c = hh * 32; c < hh * 32 + 32; ++c) s += fq[c] * (kk[c] + r0[c] + r1[c]);
          s /= sqrtf(32.0f);
          sc[k] = s;
          if (s > mx) mx = s;
        }
        float sum = 0;
        for (int k = 0; k < K_CTX; ++k) {
          sc[k] = idx[k] >= 0 ? expf(sc[k] - mx) : 0.0f;
          sum += sc[k];
        }
        for (int k = 0; k < K_CTX; ++k) {
          if (idx[k] < 0) continue;
          const float* v = Vv + (size_t)idx[k] * DC;
          float a = sc[k] / sum;
          for (int c = hh * 32; c < hh * 32 + 32; ++c) ctx[c] += a * v[c];
        }
      }
      float o[DM];
      arc_w_linear(NULL, ctx, 1, DC, &w->fo, DM, w->fob, o);
      float* hf = Hf + (size_t)t * DM;
      for (int c = 0; c < DM; ++c) hf[c] += o[c];
    }
  }
  arc_free(X);
  arc_free(U);
  arc_free(E);
  arc_free(Vv);
  arc_free(Kk);
done:
  for (int k = 0; k < nw; ++k) arc_free(txt[k]);
  arc_free(txt);
  arc_free(f0);
  arc_free(f1);
  arc_free(oe);
  arc_free(av);
  arc_free(src);
  arc_free(ue);
  return 0;
}
