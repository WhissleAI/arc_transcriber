/* Speakers: a port of arc_audio engine.py (_decode_frame's diarization turns, _bank_step / _bank_segments, the speaker
 * index the speaker read-out pools over) and trackbank.py (TrackBank). Part of arc_transcriber (Whissle Inc.),
 * CC BY-NC 4.0; see LICENSE. */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "arc_base.h"
#include "arc_speaker.h"

/* a growable array of fixed-size items */
static int grow(void** p, long* cap, long need, size_t item) {
  if (need <= *cap) return 0;
  long nc = *cap ? *cap : 16;
  while (nc < need) nc *= 2;
  void* d = arc_malloc(item * (size_t)nc);
  if (!d) return -1;
  if (*p && *cap) memcpy(d, *p, item * (size_t)*cap);
  arc_free(*p);
  *p = d;
  *cap = nc;
  return 0;
}

struct arc_spk {
  int mode, dm, ns, de;
  /* the speaker read-out's pooling (model.py pool_speaker): double sums per index */
  double* sum;   /* [ARC_SPK_MAX, dm] */
  double cnt[ARC_SPK_MAX];
  long last[ARC_SPK_MAX];   /* engine.py speakers[k]["last"], -1: never */
  long f;                   /* frames seen */
  /* slots: closed turns and the open one (engine.py diar_seg) */
  arc_spk_turn* turns;
  long n_turns, cap_turns;
  int has_seg, seg_k;
  long seg_f0;
  /* bank (TrackBank) */
  arc_bank_cfg bc;
  int M;                    /* measure width: de + ns - 1 */
  float* S;                 /* [ARC_BANK_TRACKS, M] */
  int nt, cur, pend;        /* pend 0: None */
  float* ring;              /* [smooth, M] */
  int rn;
  long* pend_fr;
  long n_pend, cap_pend;
  float* newbuf;            /* [k, M] */
  long* new_fr;
  long n_new, cap_new, cap_newfr;
  int* lab;                 /* engine.py bank_labels per frame */
  long cap_lab;
};

arc_spk* arc_spk_new(int mode, const arc_bank_cfg* bc, int dm, int ns, int de) {
  arc_spk* s = arc_calloc(sizeof *s);
  if (!s) return NULL;
  s->mode = mode;
  s->dm = dm;
  s->ns = ns;
  s->de = de;
  s->sum = arc_calloc(sizeof(double) * ARC_SPK_MAX * (size_t)dm);
  for (int k = 0; k < ARC_SPK_MAX; ++k) s->last[k] = -1;
  if (!s->sum) {
    arc_spk_free(s);
    return NULL;
  }
  if (mode != ARC_DIAR_SLOTS) {
    s->bc = *bc;
    if (s->bc.smooth < 1) s->bc.smooth = 1;
    s->M = de + ns - 1;
    s->S = arc_calloc(sizeof(float) * ARC_BANK_TRACKS * (size_t)s->M);
    s->ring = arc_calloc(sizeof(float) * (size_t)s->bc.smooth * (size_t)s->M);
    if (!s->S || !s->ring) {
      arc_spk_free(s);
      return NULL;
    }
  }
  return s;
}

void arc_spk_free(arc_spk* s) {
  if (!s) return;
  arc_free(s->sum);
  arc_free(s->turns);
  arc_free(s->S);
  arc_free(s->ring);
  arc_free(s->pend_fr);
  arc_free(s->newbuf);
  arc_free(s->new_fr);
  arc_free(s->lab);
  arc_free(s);
}

int arc_spk_mode(const arc_spk* s) { return s->mode; }

static int add_turn(arc_spk* s, int k, long f0, long f1) {
  if (grow((void**)&s->turns, &s->cap_turns, s->n_turns + 1, sizeof(arc_spk_turn))) return -1;
  s->turns[s->n_turns++] = (arc_spk_turn){k, f0, f1};
  return 0;
}

static double norm(const float* v, int n) {
  double a = 0;
  for (int i = 0; i < n; ++i) a += (double)v[i] * v[i];
  return sqrt(a);
}

/* TrackBank.measure: [e, beta * P(slots | not silence)], L2-normalised */
static void measure(const arc_spk* s, const float* e, const float* z, float* m) {
  float mx = z[0];
  for (int k = 1; k < s->ns; ++k)
    if (z[k] > mx) mx = z[k];
  float P[256];
  double tot = 0, rest = 0;
  for (int k = 0; k < s->ns; ++k) tot += P[k] = expf(z[k] - mx);
  for (int k = 0; k < s->ns; ++k) P[k] = (float)(P[k] / tot);
  for (int k = 1; k < s->ns; ++k) rest += P[k];
  if (rest < 1e-6) rest = 1e-6;
  memcpy(m, e, sizeof(float) * (size_t)s->de);
  for (int k = 1; k < s->ns; ++k) m[s->de + k - 1] = s->bc.beta * (float)(P[k] / rest);
  double n = norm(m, s->M);
  if (n < 1e-6) n = 1e-6;
  for (int i = 0; i < s->M; ++i) m[i] = (float)(m[i] / n);
}

/* TrackBank.step for a speech frame: the track label and the earlier frames it relabels (into rev, *nrev) */
static int bank_step(arc_spk* s, long f, const float* e, const float* z, int* cur, long** rev, long* nrev) {
  int M = s->M;
  float* E = arc_malloc(sizeof(float) * (size_t)M);
  float* eb = arc_malloc(sizeof(float) * (size_t)M);
  if (!E || !eb) {
    arc_free(E);
    arc_free(eb);
    return -1;
  }
  measure(s, e, z, E);
  /* ring = (ring + [E])[-smooth:] */
  if (s->rn == s->bc.smooth) {
    memmove(s->ring, s->ring + M, sizeof(float) * (size_t)(s->rn - 1) * M);
    s->rn--;
  }
  memcpy(s->ring + (size_t)s->rn * M, E, sizeof(float) * (size_t)M);
  s->rn++;
  for (int i = 0; i < M; ++i) {
    double a = 0;
    for (int r = 0; r < s->rn; ++r) a += s->ring[(size_t)r * M + i];
    eb[i] = (float)(a / s->rn);
  }
  double ne = norm(eb, M);
  if (ne < 1e-6) ne = 1e-6;
  for (int i = 0; i < M; ++i) eb[i] = (float)(eb[i] / ne);
  int nt = s->nt;
  double sb = -1.0;
  int arg = 0;
  for (int k = 0; k < nt; ++k) {
    const float* Sk = s->S + (size_t)k * M;
    double n = norm(Sk, M), d = 0;
    if (n < 1e-6) n = 1e-6;
    for (int i = 0; i < M; ++i) d += (double)(float)(Sk[i] / n) * eb[i];
    if (k == 0 || d > sb) {
      sb = d;
      arg = k;
    }
  }
  int cand = nt && sb >= s->bc.tau_new ? arg + 1 : -1;
  *nrev = 0;
  if (cand == -1) {
    if (grow((void**)&s->newbuf, &s->cap_new, s->n_new + 1, sizeof(float) * (size_t)M) ||
        grow((void**)&s->new_fr, &s->cap_newfr, s->n_new + 1, sizeof(long)))
      goto fail;
    memcpy(s->newbuf + (size_t)s->n_new * M, E, sizeof(float) * (size_t)M);
    s->new_fr[s->n_new++] = f;
    s->pend = 0;
    s->n_pend = 0;
    if (s->n_new >= s->bc.confirm && nt < ARC_BANK_TRACKS) {
      float* Sn = s->S + (size_t)nt * M;
      for (int i = 0; i < M; ++i) {
        double a = 0;
        for (long r = 0; r < s->n_new; ++r) a += s->newbuf[(size_t)r * M + i];
        Sn[i] = (float)a;
      }
      s->nt++;
      s->cur = s->nt;
      *rev = s->new_fr;   /* handed over; a fresh buffer next time */
      *nrev = s->n_new;
      s->new_fr = NULL;
      s->cap_newfr = 0;
      s->n_new = 0;
      if (s->rn > 1) memmove(s->ring, s->ring + (size_t)(s->rn - 1) * M, sizeof(float) * (size_t)M);
      s->rn = 1;
    }
  } else {
    s->n_new = 0;
    if (cand != s->cur) {
      if (s->pend != cand) {
        s->pend = cand;
        s->n_pend = 0;
      }
      if (grow((void**)&s->pend_fr, &s->cap_pend, s->n_pend + 1, sizeof(long))) goto fail;
      s->pend_fr[s->n_pend++] = f;
      if (s->n_pend >= s->bc.sw || s->cur == 0) {
        s->cur = cand;
        *rev = s->pend_fr;
        *nrev = s->n_pend;
        s->pend_fr = NULL;
        s->cap_pend = 0;
        s->pend = 0;
        s->n_pend = 0;
        if (s->rn > 1) memmove(s->ring, s->ring + (size_t)(s->rn - 1) * M, sizeof(float) * (size_t)M);
        s->rn = 1;
      }
    } else {
      s->pend = 0;
      s->n_pend = 0;
    }
  }
  if (s->cur > 0) {
    float* Sc = s->S + (size_t)(s->cur - 1) * M;
    double nc = norm(Sc, M), nE = norm(E, M), d = 0;
    if (nc < 1e-6) nc = 1e-6;
    if (nE < 1e-6) nE = 1e-6;
    for (int i = 0; i < M; ++i) d += (double)(float)(Sc[i] / nc) * (float)(E[i] / nE);
    if (d >= s->bc.tau_upd)
      for (int i = 0; i < M; ++i) Sc[i] += E[i];
  }
  *cur = s->cur;
  arc_free(E);
  arc_free(eb);
  return 0;
fail:
  arc_free(E);
  arc_free(eb);
  return -1;
}

int arc_spk_frame(arc_spk* s, long f, int spk, int speech, const float* slot, const float* emb, const float* zs) {
  /* the speaker read-out's pooled sums (engine.py speakers[spk] = p at the speaker's last frame) */
  if (spk >= 0 && spk < ARC_SPK_MAX) {
    double* a = s->sum + (size_t)spk * s->dm;
    for (int c = 0; c < s->dm; ++c) a[c] += zs[c];
    s->cnt[spk] += 1;
    s->last[spk] = f;
  }
  if (s->mode == ARC_DIAR_SLOTS) {
    if (!s->has_seg || s->seg_k != spk) {
      if (s->has_seg && s->seg_k >= 0 && add_turn(s, s->seg_k, s->seg_f0, f)) return -1;
      s->has_seg = 1;
      s->seg_k = spk;
      s->seg_f0 = f;
    }
  } else {
    if (grow((void**)&s->lab, &s->cap_lab, f + 1, sizeof(int))) return -1;
    int lab = 0;
    long* rev = NULL;
    long nrev = 0;
    if (speech && bank_step(s, f, emb, slot, &lab, &rev, &nrev)) return -1;
    s->lab[f] = lab;
    for (long r = 0; r < nrev; ++r)
      if (rev[r] >= 0 && rev[r] <= f) s->lab[rev[r]] = lab;
    arc_free(rev);
  }
  s->f = f + 1;
  return 0;
}

void arc_spk_finish(arc_spk* s, long f_end) {
  if (s->mode == ARC_DIAR_SLOTS && s->has_seg && s->seg_k >= 0) add_turn(s, s->seg_k, s->seg_f0, f_end);
  s->has_seg = 0;
}

int arc_spk_count(const arc_spk* s) {
  int n = 0;
  for (int k = 0; k < ARC_SPK_MAX; ++k) n += s->cnt[k] > 0;
  return n;
}

int arc_spk_info(const arc_spk* s, int i, int* index, long* last, float* mean) {
  for (int k = 0; k < ARC_SPK_MAX; ++k) {
    if (s->cnt[k] <= 0) continue;
    if (i--) continue;
    *index = k;
    *last = s->last[k];
    if (mean)
      for (int c = 0; c < s->dm; ++c) mean[c] = (float)(s->sum[(size_t)k * s->dm + c] / s->cnt[k]);
    return 0;
  }
  return -1;
}

int arc_spk_turns(const arc_spk* s, int open, arc_spk_turn** out) {
  long cap = 0, n = 0;
  arc_spk_turn* t = NULL;
  if (s->mode == ARC_DIAR_SLOTS) {
    if (grow((void**)&t, &cap, s->n_turns + 2, sizeof *t)) return -1;
    if (s->n_turns) memcpy(t, s->turns, sizeof *t * (size_t)s->n_turns);
    n = s->n_turns;
    if (open && s->has_seg && s->seg_k >= 0 && s->f > s->seg_f0) t[n++] = (arc_spk_turn){s->seg_k, s->seg_f0, s->f};
  } else {   /* _bank_segments */
    if (grow((void**)&t, &cap, 2, sizeof *t)) return -1;
    int ck = -1;
    long c0 = 0;
    for (long f = 0; f <= s->f; ++f) {
      int k = f < s->f ? s->lab[f] : -1;
      if (k == ck) continue;
      if (ck > 0) {
        if (grow((void**)&t, &cap, n + 1, sizeof *t)) {
          arc_free(t);
          return -1;
        }
        t[n++] = (arc_spk_turn){ck, c0, f};
      }
      ck = k;
      c0 = f;
    }
  }
  *out = t;
  return (int)n;
}

const char* arc_spk_label(const arc_spk* s, const arc_spk_turn* t, char* buf) {
  if (s->mode == ARC_DIAR_SLOTS) snprintf(buf, 16, "S%d", t->k + 1);
  else snprintf(buf, 16, "T%d", t->k);
  return buf;
}
