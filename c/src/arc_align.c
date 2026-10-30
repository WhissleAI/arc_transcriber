/* Word timestamps by cross-attention DTW: a C port of arc_audio/align.py (align_eo, token_frames, dtw,
 * words_from_tokens). Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include "arc_align.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arc_base.h"

#define NL ARC_MOON_LAYERS
#define NH ARC_MOON_HEADS

double arc_pyround(double x, int nd) {
  char b[64];
  snprintf(b, sizeof b, "%.*f", nd, x);   /* correctly rounded from the exact binary value, as CPython's round */
  return arc_strtod(b, NULL);
}

void arc_align_cfg_load(arc_align_cfg* c, const arc_json* a) {
  memset(c, 0, sizeof *c);
  c->medfilt = (int)arc_json_num(arc_json_get(a, "medfilt"), 7);
  c->offset_s = arc_json_num(arc_json_get(a, "offset_s"), 0.0);
  const arc_json* h = arc_json_get(a, "heads");
  for (int i = 0; h && h->type == ARC_JARR && i < h->n && c->n_heads < 16; ++i) {
    c->heads[c->n_heads][0] = (int)arc_json_num(arc_json_at(&h->items[i], 0), 0);
    c->heads[c->n_heads][1] = (int)arc_json_num(arc_json_at(&h->items[i], 1), 0);
    c->n_heads++;
  }
}

void arc_align_free(arc_aword* w, int n) {
  for (int i = 0; i < n; ++i) arc_free(w[i].w);
  arc_free(w);
}

static int cmpf(const void* a, const void* b) {
  float x = *(const float*)a, y = *(const float*)b;
  return x < y ? -1 : x > y;
}

/* dtw(cost): the path as (token, frame) pairs, ties: diagonal, then up, then left. Returns its length. */
static int dtw(const float* cost, int N, int F, int* ti, int* fj) {
  size_t W = (size_t)F + 1;
  double* D = arc_malloc(sizeof(double) * (size_t)(N + 1) * W);
  signed char* tr = arc_calloc((size_t)(N + 1) * W);
  if (!D || !tr) {
    arc_free(D);
    arc_free(tr);
    return -1;
  }
  for (size_t k = 0; k < (size_t)(N + 1) * W; ++k) D[k] = INFINITY;
  D[0] = 0;
  for (int i = 1; i <= N; ++i) {
    const float* ci = cost + (size_t)(i - 1) * F;
    for (int j = 1; j <= F; ++j) {
      double a = D[(i - 1) * W + j - 1], b = D[(i - 1) * W + j], c = D[i * W + j - 1];
      if (a <= b && a <= c) {
        D[i * W + j] = ci[j - 1] + a;
        tr[i * W + j] = 0;
      } else if (b <= c) {
        D[i * W + j] = ci[j - 1] + b;
        tr[i * W + j] = 1;
      } else {
        D[i * W + j] = ci[j - 1] + c;
        tr[i * W + j] = 2;
      }
    }
  }
  int i = N, j = F, n = 0;
  while (i > 0 && j > 0) {
    ti[n] = i - 1;
    fj[n++] = j - 1;
    int t = tr[i * W + j];
    if (t == 0) {
      i--;
      j--;
    } else if (t == 1) i--;
    else j--;
  }
  for (int k = 0; k < n / 2; ++k) {   /* reverse */
    int a = ti[k], b = fj[k];
    ti[k] = ti[n - 1 - k];
    fj[k] = fj[n - 1 - k];
    ti[n - 1 - k] = a;
    fj[n - 1 - k] = b;
  }
  arc_free(D);
  arc_free(tr);
  return n;
}

int arc_align_words(arc_moon* m, const arc_tok* tok, const arc_align_cfg* c, const float* enc, int T, int n_samples,
                    const char* text, double t0, arc_aword** out) {
  *out = NULL;
  const char* p = text;
  while (p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
  if (!p || !*p || n_samples < 800) return 0;
  int cap = (int)strlen(text) + 8;
  int* ids = arc_malloc(sizeof(int) * (size_t)(cap + 1));
  if (!ids) return -1;
  ids[0] = m->bos;   /* text_ids: the decoder start token, then tok(text) (no BOS / EOS of its own) */
  int ne = arc_tok_encode(tok, text, ids + 1, cap);
  if (ne < 0) {
    arc_free(ids);
    return -1;
  }
  if (ne > cap) ne = cap;
  int n = 1;
  for (int k = 0; k < ne; ++k)
    if (ids[1 + k] != m->eos) ids[n++] = ids[1 + k];
  if (n < 2 || T < 2) {
    arc_free(ids);
    return 0;
  }
  float* att = arc_moon_cross_attn(m, enc, T, ids, n);   /* [n][NL][NH][T] */
  if (!att) {
    arc_free(ids);
    return -1;
  }
  /* token_frames: rows of the chosen heads without the last position, z-scored over tokens, median-filtered over
   * frames, averaged over heads */
  int hs[NL * NH][2], nh = 0;
  if (c->n_heads)
    for (int k = 0; k < c->n_heads; ++k) {
      hs[nh][0] = c->heads[k][0];
      hs[nh++][1] = c->heads[k][1];
    }
  else
    for (int l = NL / 2; l < NL; ++l)
      for (int h = 0; h < NH; ++h) {
        hs[nh][0] = l;
        hs[nh++][1] = h;
      }
  int N = n - 1, w = c->medfilt;
  float* A = arc_calloc(sizeof(float) * (size_t)N * T);
  float* row = arc_malloc(sizeof(float) * (size_t)T);
  float* win = arc_malloc(sizeof(float) * (size_t)(w > 0 ? w : 1));
  float* zs = arc_malloc(sizeof(float) * (size_t)N * T);
  int* ti = arc_malloc(sizeof(int) * (size_t)(N + T + 2));
  int* fj = arc_malloc(sizeof(int) * (size_t)(N + T + 2));
  if (!A || !row || !win || !zs || !ti || !fj) goto fail;
  for (int k = 0; k < nh; ++k) {
    int l = hs[k][0], h = hs[k][1];
    for (int i = 0; i < N; ++i)
      memcpy(zs + (size_t)i * T, att + (((size_t)i * NL + l) * NH + h) * T, sizeof(float) * (size_t)T);
    for (int f = 0; f < T; ++f) {
      double s = 0, ss = 0;
      for (int i = 0; i < N; ++i) s += zs[(size_t)i * T + f];
      double mu = s / N;
      for (int i = 0; i < N; ++i) {
        double d = zs[(size_t)i * T + f] - mu;
        ss += d * d;
      }
      float mean = (float)mu, sd = (float)sqrt(ss / N);
      for (int i = 0; i < N; ++i) zs[(size_t)i * T + f] = (zs[(size_t)i * T + f] - mean) / (sd + 1e-6f);
    }
    for (int i = 0; i < N; ++i) {
      float* z = zs + (size_t)i * T;
      if (w > 1 && T >= w) {
        int pd = w / 2;
        for (int f = 0; f < T; ++f) {
          for (int q = 0; q < w; ++q) {
            int src = f - pd + q;
            win[q] = z[src < 0 ? 0 : src >= T ? T - 1 : src];
          }
          qsort(win, (size_t)w, sizeof(float), cmpf);
          row[f] = w % 2 ? win[w / 2] : 0.5f * (win[w / 2 - 1] + win[w / 2]);
        }
      } else memcpy(row, z, sizeof(float) * (size_t)T);
      for (int f = 0; f < T; ++f) A[(size_t)i * T + f] += row[f];
    }
  }
  for (size_t k = 0; k < (size_t)N * T; ++k) A[k] = -(A[k] / nh);
  int np = dtw(A, N, T, ti, fj);
  if (np < 0) goto fail;
  /* words_from_tokens: a word starts at a piece with the SentencePiece space mark */
  int* first = arc_malloc(sizeof(int) * (size_t)N);
  int* last = arc_malloc(sizeof(int) * (size_t)N);
  arc_aword* W = arc_calloc(sizeof(arc_aword) * (size_t)(N + 1));
  int* wf0 = arc_malloc(sizeof(int) * (size_t)(N + 1));
  int* wf1 = arc_malloc(sizeof(int) * (size_t)(N + 1));
  size_t* wl = arc_calloc(sizeof(size_t) * (size_t)(N + 1));
  if (!first || !last || !W || !wf0 || !wf1 || !wl) {
    arc_free(first);
    arc_free(last);
    arc_free(W);
    arc_free(wf0);
    arc_free(wf1);
    arc_free(wl);
    goto fail;
  }
  for (int k = 0; k < N; ++k) first[k] = last[k] = -1;
  for (int k = 0; k < np; ++k) {
    if (first[ti[k]] < 0) first[ti[k]] = fj[k];
    last[ti[k]] = fj[k];
  }
  int nw = 0;
  for (int k = 0; k < N; ++k) {
    const char* pc = arc_tok_piece(tok, ids[1 + k]);
    if (!pc) pc = "";
    int sp = !strncmp(pc, "\xe2\x96\x81", 3);
    /* txt = piece without any space marks */
    char buf[256];
    size_t bl = 0;
    for (const char* q = pc; *q && bl + 4 < sizeof buf;) {
      if (!strncmp(q, "\xe2\x96\x81", 3)) {
        q += 3;
        continue;
      }
      buf[bl++] = *q++;
    }
    buf[bl] = 0;
    int f0 = first[k] < 0 ? 0 : first[k], f1 = last[k] < 0 ? 0 : last[k];
    if (sp || !nw) {
      W[nw].w = arc_malloc(bl + 1);
      memcpy(W[nw].w, buf, bl + 1);
      wl[nw] = bl;
      wf0[nw] = f0;
      wf1[nw] = f1;
      nw++;
    } else {
      char* g = arc_malloc(wl[nw - 1] + bl + 1);
      memcpy(g, W[nw - 1].w, wl[nw - 1]);
      memcpy(g + wl[nw - 1], buf, bl + 1);
      arc_free(W[nw - 1].w);
      W[nw - 1].w = g;
      wl[nw - 1] += bl;
      if (f1 > wf1[nw - 1]) wf1[nw - 1] = f1;
    }
  }
  const double hop = 320.0 / 16000.0, lo = t0, hi = t0 + (double)n_samples / 16000.0;
  int no = 0;
  for (int k = 0; k < nw; ++k) {
    if (!W[k].w[0]) {
      arc_free(W[k].w);
      continue;
    }
    double s = t0 + wf0[k] * hop + c->offset_s, e = t0 + (wf1[k] + 1) * hop + c->offset_s;
    s = fmin(fmax(s, lo), hi);
    e = fmin(fmax(e, s + 0.04), hi);
    W[no].w = W[k].w;
    W[no].start = arc_pyround(s, 3);
    W[no].end = arc_pyround(e, 3);
    no++;
  }
  arc_free(first);
  arc_free(last);
  arc_free(wf0);
  arc_free(wf1);
  arc_free(wl);
  arc_free(A);
  arc_free(row);
  arc_free(win);
  arc_free(zs);
  arc_free(ti);
  arc_free(fj);
  arc_free(att);
  arc_free(ids);
  *out = W;
  return no;
fail:
  arc_free(A);
  arc_free(row);
  arc_free(win);
  arc_free(zs);
  arc_free(ti);
  arc_free(fj);
  arc_free(att);
  arc_free(ids);
  return -1;
}
