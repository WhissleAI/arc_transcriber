/* Token n-gram LM for shallow fusion: see arc_lm.h.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#define _GNU_SOURCE
#include "arc_lm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

enum { HDR = 24, MAX_ORDER = 8, MAX_VOCAB = 1 << 24 };
#define MAX_CAP ((int64_t)1 << 30)

typedef struct {
  uint64_t key;
  float logp, bow;
} lm_entry;

struct arc_lm {
  int order, vocab;
  float unscored;
  uint32_t mask;
  long used;
  const uint8_t* scored;
  const lm_entry* tab;
  void* map;   /* the whole file */
  size_t map_len;
  int mapped;  /* 1: mmap, 0: malloc */
};

uint64_t arc_lm_key(const int* ids, int n) {
  uint64_t h = 1469598103934665603ULL;
  for (int i = 0; i < n; ++i) {
    uint32_t v = (uint32_t)ids[i];
    for (int b = 0; b < 4; ++b) {
      h ^= (v >> (8 * b)) & 0xFF;
      h *= 1099511628211ULL;
    }
  }
  h ^= (uint64_t)n * 0x9E3779B97F4A7C15ULL;
  return h | 1;
}

/* the entry of an n-gram, or NULL; the table always has an empty slot (checked at load), so the probe ends */
static const lm_entry* find(const arc_lm* lm, const int* ids, int n) {
  uint64_t k = arc_lm_key(ids, n);
  uint32_t i = (uint32_t)(k ^ (k >> 32)) & lm->mask;
  for (;;) {
    const lm_entry* e = lm->tab + i;
    if (e->key == k) return e;
    if (!e->key) return NULL;
    i = (i + 1) & lm->mask;
  }
}

static void seterr(char* err, size_t n, const char* path, const char* msg) {
  if (err && n) snprintf(err, n, "lm %s: %s", path, msg);
}

static int read_all(const char* path, void** data, size_t* len, int* mapped) {
#ifdef _WIN32
  FILE* f = fopen(path, "rb");
  if (!f) return -1;
  __int64 n = -1;
  if (_fseeki64(f, 0, SEEK_END) == 0) n = _ftelli64(f);
  if (n < 0 || _fseeki64(f, 0, SEEK_SET) != 0 || (uint64_t)n >= (uint64_t)SIZE_MAX) {
    fclose(f);
    return -1;
  }
  void* d = malloc(n ? (size_t)n : 1);
  if (!d || fread(d, 1, (size_t)n, f) != (size_t)n) {
    free(d);
    fclose(f);
    return -1;
  }
  fclose(f);
  *data = d, *len = (size_t)n, *mapped = 0;
  return 0;
#else
  int fd = open(path, O_RDONLY);
  struct stat sb;
  if (fd < 0) return -1;
  if (fstat(fd, &sb) != 0) {
    close(fd);
    return -1;
  }
  if (sb.st_size < HDR) {
    close(fd);
    return -2;
  }
  void* m = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);   /* engines loading one file share its pages */
  close(fd);
  if (m == MAP_FAILED) return -1;
  *data = m, *len = (size_t)sb.st_size, *mapped = 1;
  return 0;
#endif
}

static int little_endian(void) {
  const uint16_t one = 1;
  return *(const uint8_t*)&one == 1;
}

arc_lm* arc_lm_load(const char* path, char* err, size_t err_len) {
  if (!path) {
    seterr(err, err_len, "(null)", "no path");
    return NULL;
  }
  if (!little_endian()) {
    seterr(err, err_len, path, "big-endian hosts are not supported");
    return NULL;
  }
  arc_lm* lm = calloc(1, sizeof *lm);
  if (!lm) {
    seterr(err, err_len, path, "out of memory");
    return NULL;
  }
  int rc = read_all(path, &lm->map, &lm->map_len, &lm->mapped);
  if (rc) {
    seterr(err, err_len, path, rc == -2 ? "too short for an ARCLM1 header" : "cannot open or read the file");
    free(lm);
    return NULL;
  }
  const uint8_t* p = (const uint8_t*)lm->map;
  int32_t order, vocab, cap;
  float uns;
  memcpy(&order, p + 8, 4);
  memcpy(&vocab, p + 12, 4);
  memcpy(&uns, p + 16, 4);
  memcpy(&cap, p + 20, 4);
  const char* bad = NULL;
  if (memcmp(p, "ARCLM1\0\0", 8)) bad = memcmp(p, "ARCLM", 5) ? "not an ARCLM file (bad magic)" : "unsupported ARCLM version";
  else if (order < 1 || order > MAX_ORDER) bad = "order must be 1-8";
  else if (vocab < 1 || vocab > MAX_VOCAB) bad = "bad vocabulary size";
  else if (cap < 2 || (int64_t)cap > MAX_CAP || (cap & (cap - 1))) bad = "table size must be a power of 2";
  else if (!isfinite(uns) || uns > 0) bad = "bad unscored log-probability";
  size_t sc_len = ((size_t)vocab + 7) & ~(size_t)7;
  if (!bad && lm->map_len != HDR + sc_len + (size_t)cap * sizeof(lm_entry)) bad = "file size does not match its header";
  if (bad) {
    seterr(err, err_len, path, bad);
    arc_lm_free(lm);
    return NULL;
  }
  lm->order = order;
  lm->vocab = vocab;
  lm->unscored = uns;
  lm->mask = (uint32_t)cap - 1;
  lm->scored = p + HDR;
  lm->tab = (const lm_entry*)(const void*)(p + HDR + sc_len);   /* 8-byte aligned: HDR and sc_len are multiples of 8 */
  long empty = 0;
  for (int64_t i = 0; i < cap; ++i) {
    const lm_entry* e = lm->tab + i;
    if (!e->key) {
      empty++;
      continue;
    }
    if (!(e->key & 1) || !isfinite(e->logp) || !isfinite(e->bow)) {
      bad = "bad table entry";
      break;
    }
  }
  if (!bad && !empty) bad = "table has no empty slot";
  if (bad) {
    seterr(err, err_len, path, bad);
    arc_lm_free(lm);
    return NULL;
  }
  lm->used = (long)(cap - empty);
  return lm;
}

void arc_lm_free(arc_lm* lm) {
  if (!lm) return;
#ifndef _WIN32
  if (lm->mapped) munmap(lm->map, lm->map_len);
  else
#endif
    free(lm->map);
  free(lm);
}

int arc_lm_order(const arc_lm* lm) { return lm ? lm->order : 0; }
int arc_lm_vocab(const arc_lm* lm) { return lm ? lm->vocab : 0; }
long arc_lm_entries(const arc_lm* lm) { return lm ? lm->used : 0; }

int arc_lm_scored(const arc_lm* lm, int tok) { return lm && tok >= 0 && tok < lm->vocab && lm->scored[tok]; }

float arc_lm_logp(const arc_lm* lm, const int* hist, int nh, int tok) {
  int buf[MAX_ORDER];
  int k = nh < lm->order - 1 ? nh : lm->order - 1;
  float bo = 0.0f;
  for (; k >= 0; --k) {
    /* (h_k, tok): the last k history tokens then tok */
    for (int j = 0; j < k; ++j) buf[j] = hist[nh - k + j];
    buf[k] = tok;
    const lm_entry* e = find(lm, buf, k + 1);
    if (e && e->logp <= 0.0f) return bo + e->logp;
    if (k > 0) {
      const lm_entry* c = find(lm, buf, k);   /* the context h_k: its backoff weight */
      if (c) bo += c->bow;
    }
  }
  return bo + 2.0f * lm->unscored;   /* not even a unigram (the trainer gives every scored token one) */
}

/* ------------------------------------------------------------------ fusion */

static void log_softmax(float* x, int n) {
  float mx = -INFINITY;
  for (int i = 0; i < n; ++i)
    if (x[i] > mx) mx = x[i];
  double sum = 0;
  for (int i = 0; i < n; ++i) sum += exp((double)(x[i] - mx));
  float lse = mx + (float)log(sum);
  for (int i = 0; i < n; ++i) x[i] -= lse;
}

/* indices of the k largest of x[n], descending (ties: lower index first) */
static void topk_idx(const float* x, int n, int k, int* idx) {
  float v[64];
  int have = 0;
  for (int i = 0; i < n; ++i) {
    float a = x[i];
    if (have == k && !(a > v[k - 1])) continue;
    int j = have < k ? have++ : k - 1;
    while (j > 0 && a > v[j - 1]) {
      v[j] = v[j - 1];
      idx[j] = idx[j - 1];
      --j;
    }
    v[j] = a;
    idx[j] = i;
  }
}

void arc_lm_fuse(const arc_lm* lm, float weight, int topk, float* row, int V, const int* gen, int ng, int eos) {
  if (!lm || !(weight > 0) || V < 1) return;
  /* the last order-1 scored tokens (the start token <s> is scored: the sentence start) */
  int hist[MAX_ORDER], tmp[MAX_ORDER], nh = 0, nt = 0, need = lm->order - 1;
  for (int i = ng - 1; i >= 0 && nt < need; --i)
    if (arc_lm_scored(lm, gen[i])) tmp[nt++] = gen[i];
  for (int i = nt - 1; i >= 0; --i) hist[nh++] = tmp[i];
  log_softmax(row, V);
  int k = topk > 0 ? topk : 16;
  if (k > 64) k = 64;
  if (k > V) k = V;
  int idx[64];
  topk_idx(row, V, k, idx);
  const float best = row[idx[0]], w = weight;
  float lp[64], top = -INFINITY;
  int has[64];   /* 1: scored by the LM (a text piece within the margin); 0: a tag, the end token, or implausible */
  for (int j = 0; j < k; ++j) {
    int t = idx[j];
    has[j] = row[t] >= best - ARC_LM_MARGIN && t != eos && arc_lm_scored(lm, t);
    lp[j] = has[j] ? arc_lm_logp(lm, hist, nh, t) : 0.0f;
    if (has[j] && lp[j] > top) top = lp[j];
  }
  float keep[64], adj_best = 0.0f;
  for (int j = 0; j < k; ++j) {
    int t = idx[j];
    float a = 0.0f;
    if (has[j]) {
      a = lp[j] - top;
      if (a < -ARC_LM_CAP) a = -ARC_LM_CAP;
    } else if (row[t] < best - ARC_LM_MARGIN) a = -ARC_LM_CAP;
    a *= w;
    if (j == 0) adj_best = a;
    keep[j] = row[t] + a;
  }
  float end = eos >= 0 && eos < V ? row[eos] + adj_best : 0.0f;
  const float floor_v = -w * ARC_LM_CAP;
  for (int i = 0; i < V; ++i) row[i] += floor_v;
  for (int j = 0; j < k; ++j) row[idx[j]] = keep[j];
  if (eos >= 0 && eos < V) row[eos] = end;
}
