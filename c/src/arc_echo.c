/* arc_echo: see arc_echo.h (arc_audio/echo.py). Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include "arc_echo.h"

#include <math.h>
#include <string.h>

#include "arc_align.h"
#include "arc_base.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define ESR 16000

struct arc_echo {
  arc_echo_cfg c;
  int K, N;
  double *W, *X;          /* [parts][K] complex interleaved; X newest first */
  float *ref, *mic, *res;
  size_t nref, capref, nmic, capmic, nres, capres;
  long nb, n_on, n_dt;
  double *e_mic, *e_res, *e_ref;
  long cap_e;
  int any_ref;
  arc_fft* fft;
  float *fx, *fy;
  double *S, *G, *Pw, *E, *work;
};

void arc_echo_defaults(arc_echo_cfg* c) {
  c->block = 256;
  c->parts = 32;
  c->mu = 0.03;
  c->ref_db = -55.0;
  c->eps = 1e-6;
  c->threshold = 0.7;
  c->dt_ratio = 2.0;
  c->dt_after_s = 1.0;
}

static int grow_f(float** p, size_t* cap, size_t need) {
  if (need <= *cap) return 0;
  size_t nc = need * 2 + ESR;
  float* q = arc_calloc(sizeof(float) * nc);
  if (!q) return -1;
  if (*p) memcpy(q, *p, sizeof(float) * *cap);
  arc_free(*p);
  *p = q;
  *cap = nc;
  return 0;
}

arc_echo* arc_echo_new(const arc_echo_cfg* c) {
  arc_echo* e = arc_calloc(sizeof *e);
  if (!e) return NULL;
  if (c) e->c = *c;
  else
    arc_echo_defaults(&e->c);
  int B = e->c.block, P = e->c.parts;
  e->N = 2 * B;
  e->K = B + 1;
  e->W = arc_calloc(sizeof(double) * 2 * (size_t)P * e->K);
  e->X = arc_calloc(sizeof(double) * 2 * (size_t)P * e->K);
  e->fft = arc_fft_create(e->N);
  e->fx = arc_malloc(sizeof(float) * (size_t)e->N);
  e->fy = arc_malloc(sizeof(float) * (size_t)e->N);
  e->S = arc_malloc(sizeof(double) * 2 * (size_t)e->K);
  e->G = arc_malloc(sizeof(double) * 2 * (size_t)e->K);
  e->Pw = arc_malloc(sizeof(double) * (size_t)e->K);
  e->E = arc_malloc(sizeof(double) * 2 * (size_t)e->K);
  e->work = arc_malloc(sizeof(double) * 4 * (size_t)e->N);
  return e;
}

void arc_echo_free(arc_echo* e) {
  if (!e) return;
  arc_free(e->W);
  arc_free(e->X);
  arc_free(e->ref);
  arc_free(e->mic);
  arc_free(e->res);
  arc_free(e->e_mic);
  arc_free(e->e_res);
  arc_free(e->e_ref);
  arc_fft_destroy(e->fft);
  arc_free(e->fx);
  arc_free(e->fy);
  arc_free(e->S);
  arc_free(e->G);
  arc_free(e->Pw);
  arc_free(e->E);
  arc_free(e->work);
  arc_free(e);
}

int arc_echo_feed_ref(arc_echo* e, const float* pcm, size_t n, double t) {
  size_t a = t < 0 ? e->nref : (size_t)llround(t * ESR);
  if (grow_f(&e->ref, &e->capref, a + n)) return -1;
  if (a + n > e->nref) e->nref = a + n;
  for (size_t i = 0; i < n; ++i) {
    e->ref[a + i] += pcm[i];
    if (pcm[i] != 0) e->any_ref = 1;
  }
  return 0;
}

/* one block (EchoCanceller._block) */
static int block(arc_echo* e) {
  int B = e->c.block, P = e->c.parts, K = e->K, N = e->N;
  long a = e->nb * (long)B;
  double xr2 = 0;
  for (int i = 0; i < N; ++i) {
    long j = a - B + i;
    double r = j >= 0 && (size_t)j < e->nref ? e->ref[j] : 0.0;
    e->fx[i] = (float)r;
    if (i >= B) xr2 += r * r;
  }
  memmove(e->X + 2 * (size_t)K, e->X, sizeof(double) * 2 * (size_t)(P - 1) * K);
  arc_rfft(e->fft, e->fx, e->X, e->work);
  /* y = irfft(sum W X)[B:] */
  for (int k = 0; k < K; ++k) {
    double sr = 0, si = 0;
    for (int p = 0; p < P; ++p) {
      const double* x = e->X + 2 * ((size_t)p * K + k);
      const double* w = e->W + 2 * ((size_t)p * K + k);
      sr += w[0] * x[0] - w[1] * x[1];
      si += w[0] * x[1] + w[1] * x[0];
    }
    e->S[2 * k] = sr;
    e->S[2 * k + 1] = si;
  }
  arc_irfft(e->fft, e->S, e->fy, e->work);
  if (grow_f(&e->res, &e->capres, e->nres + B)) return -1;
  double em = 0, er = 0, ey = 0;
  float* res = e->res + e->nres;
  for (int i = 0; i < B; ++i) {
    double d = e->mic[a + i], y = e->fy[B + i], r = d - y;
    res[i] = (float)r;
    em += d * d;
    er += r * r;
    ey += y * y;
  }
  e->nres += B;
  double thr = pow(10.0, e->c.ref_db / 10.0);
  if ((double)e->n_on * B / ESR >= e->c.dt_after_s && er > e->c.dt_ratio * ey)
    e->n_dt++;
  else {
    for (int i = 0; i < B; ++i) e->fx[i] = 0.0f;
    for (int i = 0; i < B; ++i) e->fx[B + i] = res[i];
    double* E = e->E;
    arc_rfft(e->fft, e->fx, E, e->work);
    for (int k = 0; k < K; ++k) {
      double s = 0;
      for (int p = 0; p < P; ++p) {
        const double* x = e->X + 2 * ((size_t)p * K + k);
        s += x[0] * x[0] + x[1] * x[1];
      }
      e->Pw[k] = s / P + e->c.eps * N;
    }
    const double* Ec = E;
    for (int p = 0; p < P; ++p) {
      for (int k = 0; k < K; ++k) {   /* G = mu conj(X) E / Pw */
        const double* x = e->X + 2 * ((size_t)p * K + k);
        double g = e->c.mu / e->Pw[k];
        e->G[2 * k] = g * (x[0] * Ec[2 * k] + x[1] * Ec[2 * k + 1]);
        e->G[2 * k + 1] = g * (x[0] * Ec[2 * k + 1] - x[1] * Ec[2 * k]);
      }
      arc_irfft(e->fft, e->G, e->fy, e->work);
      for (int i = B; i < N; ++i) e->fy[i] = 0.0f;   /* the gradient constraint */
      arc_rfft(e->fft, e->fy, e->G, e->work);
      double* w = e->W + 2 * (size_t)p * K;
      for (int k = 0; k < 2 * K; ++k) w[k] += e->G[k];
    }
  }
  if (e->nb + 1 > e->cap_e) {
    long nc = (e->nb + 1) * 2 + 64;
    double** arr[3] = {&e->e_mic, &e->e_res, &e->e_ref};
    for (int q = 0; q < 3; ++q) {
      double* p = arc_malloc(sizeof(double) * (size_t)nc);
      if (*arr[q]) memcpy(p, *arr[q], sizeof(double) * (size_t)e->nb);
      arc_free(*arr[q]);
      *arr[q] = p;
    }
    e->cap_e = nc;
  }
  e->e_mic[e->nb] = em;
  e->e_res[e->nb] = er;
  e->e_ref[e->nb] = xr2 / B;
  e->n_on += xr2 / B > thr;
  e->nb++;
  return 0;
}

int arc_echo_feed_mic(arc_echo* e, const float* pcm, size_t n) {
  if (grow_f(&e->mic, &e->capmic, e->nmic + n)) return -1;
  memcpy(e->mic + e->nmic, pcm, sizeof(float) * n);
  e->nmic += n;
  while ((size_t)(e->nb + 1) * e->c.block <= e->nmic)
    if (block(e)) return -1;
  return 0;
}

int arc_echo_flush(arc_echo* e) {
  size_t m = e->nmic, B = (size_t)e->c.block;
  if (m > (size_t)e->nb * B) {
    size_t need = (size_t)(e->nb + 1) * B;
    if (grow_f(&e->mic, &e->capmic, need)) return -1;
    for (size_t i = m; i < need; ++i) e->mic[i] = 0;
    e->nmic = need;
    if (block(e)) return -1;
    e->nmic = m;
    e->nres = m;
  }
  return 0;
}

const float* arc_echo_residual(arc_echo* e, size_t* n) {
  *n = e->nres;
  return e->res;
}

double arc_echo_leak(const arc_echo* e, double t0, double t1) {
  if (!e->any_ref) return -1.0;
  int B = e->c.block;
  long j0 = (long)floor(t0 * ESR / B), j1 = (long)ceil(t1 * ESR / B);
  if (j0 < 0) j0 = 0;
  if (j1 > e->nb) j1 = e->nb;
  if (j1 <= j0) return 0.0;
  double thr = pow(10.0, e->c.ref_db / 10.0), tot = 0, ex = 0;
  int on = 0;
  for (long j = j0; j < j1; ++j) {
    tot += e->e_mic[j];
    if (e->e_ref[j] > thr) {
      on = 1;
      double d = e->e_mic[j] - e->e_res[j];
      if (d > 0) ex += d;
    }
  }
  if (tot <= 0 || !on) return 0.0;
  double v = ex / tot;
  v = v < 0 ? 0 : v > 1 ? 1 : v;
  return arc_pyround(v, 4);
}

int arc_echo_playing(const arc_echo* e, double t, double span) {
  long a = (long)((t - span) * ESR), b = (long)(t * ESR);
  if (a < 0) a = 0;
  if (b < 0) b = 0;
  if ((size_t)b > e->nref) b = (long)e->nref;
  if (b <= a) return 0;
  double s = 0;
  for (long i = a; i < b; ++i) s += (double)e->ref[i] * e->ref[i];
  return s / (double)(b - a) > pow(10.0, e->c.ref_db / 10.0);
}

int arc_echo_barge_in(const arc_echo* e, double t_open, double t_now, double min_s, double max_leak) {
  const double win = 0.3;
  double span = t_now - t_open < win ? t_now - t_open : win;
  if (span < min_s) span = min_s;
  if (t_now - t_open < min_s || arc_echo_trained_s(e) < 1.0 || !arc_echo_playing(e, t_now, span)) return 0;
  double lk = arc_echo_leak(e, t_now - win > t_open ? t_now - win : t_open, t_now);
  return lk >= 0 && lk < max_leak;
}

double arc_echo_threshold(const arc_echo* e) { return e->c.threshold; }

double arc_echo_trained_s(const arc_echo* e) { return (double)e->n_on * e->c.block / ESR; }
