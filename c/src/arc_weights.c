/* safetensors files (mmapped) and the release weight layouts (plain float, packed int8 / int4).
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#define _GNU_SOURCE
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arc_base.h"

/* safetensors data (and the quantized payloads) are little-endian and are read in place */
#if (defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__) || \
    defined(__BIG_ENDIAN__) || defined(__ARMEB__) || defined(__MIPSEB__)
#error "libarc reads little-endian safetensors in place: big-endian targets are not supported"
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

float arc_half_to_float(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits;
  if (exp == 0) {
    if (man == 0) bits = sign;
    else {
      exp = 127 - 15 + 1;
      while (!(man & 0x400)) {
        man <<= 1;
        --exp;
      }
      man &= 0x3FF;
      bits = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 31) bits = sign | 0x7F800000 | (man << 13);
  else bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

float arc_bf16_to_float(uint16_t h) {
  uint32_t bits = (uint32_t)h << 16;
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

/* ------------------------------------------------------------------ safetensors */
struct arc_file {
  char* path;
  void* map;
  size_t map_len;
  int mapped;
  arc_tensor* t;   /* sorted by name */
  int n;
  arc_json* meta;  /* __metadata__ object or NULL */
};

static int dtype_of(const char* s, arc_dtype* d, int* size) {
  static const struct { const char* n; arc_dtype d; int sz; } tab[] = {
      {"F32", ARC_F32, 4}, {"F16", ARC_F16, 2}, {"BF16", ARC_BF16, 2}, {"I8", ARC_I8, 1},  {"U8", ARC_U8, 1},
      {"I16", ARC_I16, 2}, {"I32", ARC_I32, 4}, {"I64", ARC_I64, 8},   {"BOOL", ARC_BOOL, 1}, {"F64", ARC_F64, 8}};
  for (size_t i = 0; i < sizeof tab / sizeof *tab; ++i)
    if (!strcmp(s, tab[i].n)) {
      *d = tab[i].d;
      *size = tab[i].sz;
      return 0;
    }
  return -1;
}

static int cmp_tensor(const void* a, const void* b) {
  return strcmp(((const arc_tensor*)a)->name, ((const arc_tensor*)b)->name);
}

typedef struct {
  arc_tensor t;
  int order;   /* position in the header: a repeated name keeps the last entry, as Python's json does */
} tensor_ord;

static int cmp_tensor_ord(const void* a, const void* b) {
  const tensor_ord *x = (const tensor_ord*)a, *y = (const tensor_ord*)b;
  int c = strcmp(x->t.name, y->t.name);
  return c ? c : (x->order > y->order) - (x->order < y->order);
}

/* a JSON integer (no fraction / exponent) in [0, max] (max <= 2^53, so the double is exact) */
static int json_uint(const arc_json* j, double max, uint64_t* out) {
  if (!j || j->type != ARC_JNUM || !j->is_int || !(j->num >= 0) || !(j->num <= max)) return -1;
  *out = (uint64_t)j->num;
  return 0;
}

#define ARC_MAX_SAFE_INT 9007199254740992.0   /* 2^53 */

arc_file* arc_file_open(const char* path) {
  arc_file* f = (arc_file*)arc_calloc(sizeof *f);
  arc_json* h = NULL;
  tensor_ord* tmp = NULL;
  if (!f) return NULL;
  f->path = arc_strdup(path);
  if (!f->path) goto fail;
#ifdef _WIN32
  {
    char* d;
    size_t n;
    if (arc_read_file(path, &d, &n)) goto fail;
    f->map = d;
    f->map_len = n;
  }
#else
  {
    int fd = open(path, O_RDONLY);
    struct stat sb;
    if (fd < 0 || fstat(fd, &sb) != 0) {
      if (fd >= 0) close(fd);
      arc_seterr("cannot open %s", path);
      goto fail;
    }
    if (sb.st_size < 8) {
      close(fd);
      arc_seterr("%s: not a safetensors file", path);
      goto fail;
    }
    f->map_len = (size_t)sb.st_size;
    f->map = mmap(NULL, f->map_len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (f->map == MAP_FAILED) {
      f->map = NULL;
      arc_seterr("cannot map %s", path);
      goto fail;
    }
    f->mapped = 1;
  }
#endif
  if (f->map_len < 8) {
    arc_seterr("%s: not a safetensors file", path);
    goto fail;
  }
  uint64_t hl;
  memcpy(&hl, f->map, 8);
  if (hl > f->map_len - 8) {
    arc_seterr("%s: bad safetensors header", path);
    goto fail;
  }
  h = arc_json_parse((const char*)f->map + 8, (size_t)hl);
  if (!h || h->type != ARC_JOBJ) {
    arc_seterr("%s: bad safetensors header (%s)", path, h ? "not an object" : arc_geterr());
    goto fail;
  }
  const char* base = (const char*)f->map + 8 + hl;
  size_t data_len = f->map_len - 8 - (size_t)hl;
  double max_off = (double)data_len < ARC_MAX_SAFE_INT ? (double)data_len : ARC_MAX_SAFE_INT;
  tmp = (tensor_ord*)arc_calloc(sizeof(tensor_ord) * (size_t)(h->n + 1));
  if (!tmp) goto fail;
  int nt = 0;
  for (int i = 0; i < h->n; ++i) {
    const arc_json* v = &h->items[i];
    if (!strcmp(h->keys[i], "__metadata__")) {
      /* keep a private copy: move the node out of the header tree (a repeated key: the last one wins) */
      arc_json* m = (arc_json*)calloc(1, sizeof(arc_json));
      if (!m) {
        arc_seterr("out of memory");
        goto fail;
      }
      *m = *v;
      h->items[i].type = ARC_JNULL;
      arc_json_free(f->meta);
      f->meta = m;
      continue;
    }
    arc_tensor* t = &tmp[nt].t;
    int sz;
    const arc_json* dt = arc_json_get(v, "dtype");
    const arc_json* sh = arc_json_get(v, "shape");
    const arc_json* off = arc_json_get(v, "data_offsets");
    if (!dt || !sh || !off || dtype_of(arc_json_str(dt, ""), &t->dtype, &sz) || sh->type != ARC_JARR || sh->n > 8 ||
        off->type != ARC_JARR || off->n != 2) {
      arc_seterr("%s: bad entry %s", path, h->keys[i]);
      goto fail;
    }
    t->ndim = sh->n;
    uint64_t numel = 1, a, b;
    for (int d = 0; d < sh->n; ++d) {
      uint64_t dim;
      if (json_uint(&sh->items[d], ARC_MAX_SAFE_INT, &dim)) {
        arc_seterr("%s: bad shape for %s", path, h->keys[i]);
        goto fail;
      }
      t->shape[d] = (int64_t)dim;
      if (dim && numel > UINT64_MAX / dim) numel = UINT64_MAX;   /* saturate: cannot match b - a below */
      else numel *= dim;
    }
    if (json_uint(&off->items[0], max_off, &a) || json_uint(&off->items[1], max_off, &b) || b < a ||
        b > data_len || numel > UINT64_MAX / (uint64_t)sz || numel * (uint64_t)sz != b - a) {
      arc_seterr("%s: bad offsets for %s", path, h->keys[i]);
      goto fail;
    }
    t->data = base + a;
    t->nbytes = (size_t)(b - a);
    t->name = arc_strdup(h->keys[i]);
    if (!t->name) goto fail;
    tmp[nt].order = i;
    nt++;
  }
  arc_json_free(h);
  h = NULL;
  qsort(tmp, (size_t)nt, sizeof(tensor_ord), cmp_tensor_ord);
  f->t = (arc_tensor*)arc_calloc(sizeof(arc_tensor) * (size_t)(nt + 1));
  if (!f->t) goto fail;
  for (int i = 0; i < nt; ++i) {
    if (i + 1 < nt && !strcmp(tmp[i].t.name, tmp[i + 1].t.name)) {   /* superseded by a later entry */
      arc_free(tmp[i].t.name);
      continue;
    }
    f->t[f->n++] = tmp[i].t;
  }
  arc_free(tmp);
  return f;
fail:
  if (tmp)
    for (int i = 0; tmp[i].t.name; ++i) arc_free(tmp[i].t.name);
  arc_free(tmp);
  arc_json_free(h);
  arc_file_close(f);
  return NULL;
}

void arc_file_close(arc_file* f) {
  if (!f) return;
  for (int i = 0; i < f->n; ++i) arc_free(f->t[i].name);   /* f->t is NULL when f->n == 0 */
  arc_free(f->t);
  arc_json_free(f->meta);
#ifndef _WIN32
  if (f->mapped && f->map) munmap(f->map, f->map_len);
  else
#endif
    arc_free(f->map);
  arc_free(f->path);
  arc_free(f);
}

const arc_tensor* arc_file_find(const arc_file* f, const char* name) {
  arc_tensor key;
  key.name = (char*)name;
  return (const arc_tensor*)bsearch(&key, f->t, (size_t)f->n, sizeof(arc_tensor), cmp_tensor);
}

int arc_file_count(const arc_file* f) { return f->n; }
const arc_tensor* arc_file_tensor(const arc_file* f, int i) { return &f->t[i]; }
const char* arc_file_metadata(const arc_file* f, const char* key) { return arc_json_str(arc_json_get(f->meta, key), NULL); }

int64_t arc_numel(const arc_tensor* t) {
  int64_t n = 1;
  for (int d = 0; d < t->ndim; ++d) n *= t->shape[d];
  return n;
}

/* ------------------------------------------------------------------ weights */
typedef struct {
  char* name;
  float* f32;      /* owned converted copy (or NULL) */
  arc_mat* mat;     /* owned (or NULL) */
  float* scale;    /* owned f32 scales for quantized mats */
  void* qcopy;     /* owned aligned copy of quantized data (only if needed) */
} wcache;

struct arc_weights {
  arc_file* f;
  arc_json* layout;   /* "layout" object of steertts_quant, or NULL (plain file) */
  arc_json* quant;    /* whole steertts_quant metadata */
  wcache* c;
  int nc, capc;
};

arc_weights* arc_weights_open(const char* path) {
  arc_weights* w = (arc_weights*)arc_calloc(sizeof *w);
  if (!w) return NULL;
  w->f = arc_file_open(path);
  if (!w->f) {
    arc_free(w);
    return NULL;
  }
  const char* q = arc_file_metadata(w->f, "steertts_quant");
  if (q) {
    w->quant = arc_json_parse(q, strlen(q));
    if (!w->quant) {
      arc_seterr("%s: bad steertts_quant metadata", path);
      arc_weights_close(w);
      return NULL;
    }
    w->layout = (arc_json*)arc_json_get(w->quant, "layout");
  }
  return w;
}

void arc_weights_close(arc_weights* w) {
  if (!w) return;
  for (int i = 0; i < w->nc; ++i) {
    arc_free(w->c[i].name);
    arc_free(w->c[i].f32);
    arc_free(w->c[i].mat);
    arc_free(w->c[i].scale);
    arc_free(w->c[i].qcopy);
  }
  free(w->c);
  arc_json_free(w->quant);
  arc_file_close(w->f);
  arc_free(w);
}

const char* arc_weights_path(const arc_weights* w) { return w->f->path; }

static wcache* cache_get(arc_weights* w, const char* name, int create) {
  for (int i = 0; i < w->nc; ++i)
    if (!strcmp(w->c[i].name, name)) return &w->c[i];
  if (!create) return NULL;
  if (w->nc == w->capc) {
    int nc = w->capc ? w->capc * 2 : 64;
    wcache* c = (wcache*)realloc(w->c, sizeof(wcache) * (size_t)nc);
    if (!c) {
      arc_seterr("out of memory");
      return NULL;
    }
    w->c = c;
    w->capc = nc;
  }
  wcache* e = &w->c[w->nc];
  memset(e, 0, sizeof *e);
  e->name = arc_strdup(name);
  if (!e->name) return NULL;
  w->nc++;
  return e;
}

static const arc_tensor* need(const arc_weights* w, const char* name) {
  const arc_tensor* t = arc_file_find(w->f, name);
  if (!t) arc_seterr("%s: missing tensor %s", w->f->path, name);
  return t;
}

/* any float tensor -> owned f32 copy (or the mapped data when it is aligned F32) */
static const float* to_f32(const arc_tensor* t, float** owned) {
  int64_t n = arc_numel(t);
  *owned = NULL;
  if (t->dtype == ARC_F32 && ((uintptr_t)t->data & 3) == 0) return (const float*)t->data;
  float* o = (float*)arc_malloc(sizeof(float) * (size_t)(n ? n : 1));
  if (!o) return NULL;
  const unsigned char* d = (const unsigned char*)t->data;
  for (int64_t i = 0; i < n; ++i) {
    uint16_t h;
    switch (t->dtype) {
      case ARC_F32: memcpy(&o[i], d + 4 * i, 4); break;
      case ARC_F16: memcpy(&h, d + 2 * i, 2); o[i] = arc_half_to_float(h); break;
      case ARC_BF16: memcpy(&h, d + 2 * i, 2); o[i] = arc_bf16_to_float(h); break;
      case ARC_F64: { double v; memcpy(&v, d + 8 * i, 8); o[i] = (float)v; break; }
      case ARC_I64: { int64_t v; memcpy(&v, d + 8 * i, 8); o[i] = (float)v; break; }
      case ARC_I32: { int32_t v; memcpy(&v, d + 4 * i, 4); o[i] = (float)v; break; }
      case ARC_I8: o[i] = (float)((const int8_t*)d)[i]; break;
      case ARC_U8: case ARC_BOOL: o[i] = (float)d[i]; break;
      default: arc_free(o); arc_seterr("unsupported dtype"); return NULL;
    }
  }
  *owned = o;
  return o;
}

static const arc_json* layout_of(const arc_weights* w, const char* name) {
  return w->layout ? arc_json_get(w->layout, name) : NULL;
}

int arc_weights_has(const arc_weights* w, const char* name) {
  if (layout_of(w, name)) return 1;
  return arc_file_find(w->f, name) != NULL;
}

/* *bad = 1 (and arc_seterr) when a layout shape exists but is invalid */
static int shape_of(const arc_weights* w, const char* name, int64_t shape[8], int* bad) {
  const arc_json* L = layout_of(w, name);
  *bad = 0;
  if (L) {
    const arc_json* sh = arc_json_get(L, "shape");
    if (sh && sh->type == ARC_JARR) {
      if (sh->n > 8) {
        arc_seterr("%s: layout shape of %s has %d dims", w->f->path, name, sh->n);
        *bad = 1;
        return 0;
      }
      for (int i = 0; i < sh->n; ++i) {
        uint64_t d;
        if (json_uint(&sh->items[i], ARC_MAX_SAFE_INT, &d)) {
          arc_seterr("%s: bad layout shape for %s", w->f->path, name);
          *bad = 1;
          return 0;
        }
        shape[i] = (int64_t)d;
      }
      return sh->n;
    }
  }
  const arc_tensor* t = arc_file_find(w->f, name);
  if (!t) return 0;
  for (int i = 0; i < t->ndim; ++i) shape[i] = t->shape[i];
  return t->ndim;
}

int arc_weights_shape(const arc_weights* w, const char* name, int64_t shape[8]) {
  int bad;
  return shape_of(w, name, shape, &bad);
}

static const float* dequant_full(arc_weights* w, const char* name, const arc_json* L, int64_t* numel, float** owned);

const float* arc_weights_f32(arc_weights* w, const char* name, int64_t n_expect) {
  wcache* e = cache_get(w, name, 0);
  int64_t n = 0;
  const float* r = NULL;
  if (e && e->f32) return e->f32;
  const arc_json* L = layout_of(w, name);
  float* owned = NULL;
  if (L && strcmp(arc_json_str(arc_json_get(L, "s"), "raw"), "raw") && strcmp(arc_json_str(arc_json_get(L, "s"), ""), "fp16")) {
    r = dequant_full(w, name, L, &n, &owned);
  } else {
    const arc_tensor* t = need(w, name);
    if (!t) return NULL;
    n = arc_numel(t);
    r = to_f32(t, &owned);
  }
  if (!r) return NULL;
  if (n_expect > 0 && n != n_expect) {
    arc_free(owned);
    arc_seterr("%s: tensor %s has %lld elements, expected %lld", w->f->path, name, (long long)n, (long long)n_expect);
    return NULL;
  }
  if (owned) {
    e = cache_get(w, name, 1);
    if (!e) {
      arc_free(owned);
      return NULL;
    }
    e->f32 = owned;
  }
  return r;
}

/* shape [O, ...] of a layout entry or tensor -> O, K (both must fit in an int) */
static int ok_shape(const int64_t* sh, int nd, int* O, int* K) {
  if (nd < 1 || nd > 8 || sh[0] < 0 || sh[0] > INT_MAX) return -1;
  int64_t k = 1;
  for (int i = 1; i < nd; ++i) {
    if (sh[i] < 0) return -1;
    if (sh[i] && k > INT_MAX / sh[i]) return -1;
    k *= sh[i];
  }
  *O = (int)sh[0];
  *K = (int)k;
  return 0;
}

/* an optional integer layout field in [lo, hi]: dflt when absent, -1 when present but not such an integer */
static int64_t layout_int(const arc_json* L, const char* key, int64_t dflt, int64_t lo, int64_t hi) {
  const arc_json* j = arc_json_get(L, key);
  if (!j) return dflt;
  if (j->type != ARC_JNUM || !j->is_int || !(j->num >= (double)lo) || !(j->num <= (double)hi)) return -1;
  return (int64_t)j->num;
}

const arc_mat* arc_weights_mat(arc_weights* w, const char* name, int O_expect, int K_expect) {
  wcache* e = cache_get(w, name, 0);
  if (e && e->mat) return e->mat;
  int64_t sh[8];
  int bad, nd = shape_of(w, name, sh, &bad), O = 0, K = 0;
  if (nd == 0) {
    if (!bad) arc_seterr("%s: missing tensor %s", w->f->path, name);
    return NULL;
  }
  if (ok_shape(sh, nd, &O, &K)) {
    arc_seterr("%s: %s has an unsupported shape for a matrix", w->f->path, name);
    return NULL;
  }
  if ((O_expect > 0 && O != O_expect) || (K_expect > 0 && K != K_expect)) {
    arc_seterr("%s: %s is [%d, %d], expected [%d, %d]", w->f->path, name, O, K, O_expect, K_expect);
    return NULL;
  }
  const arc_json* L = layout_of(w, name);
  const char* s = L ? arc_json_str(arc_json_get(L, "s"), "raw") : "raw";
  if (strcmp(s, "raw") && strcmp(s, "fp16") && strcmp(s, "int8") && strcmp(s, "int4")) {
    arc_seterr("%s: unknown storage '%s' for %s", w->f->path, s, name);
    return NULL;
  }
  arc_mat* m = (arc_mat*)arc_calloc(sizeof *m);
  if (!m) return NULL;
  m->O = O;
  m->K = K;
  float* scale = NULL;
  void* qcopy = NULL;
  if (!strcmp(s, "int8") || !strcmp(s, "int4")) {
    const arc_tensor* q = need(w, arc_namef("%s::q", name));
    const arc_tensor* st = q ? need(w, arc_namef("%s::s", name)) : NULL;
    if (!q || !st) goto fail;
    float* so;
    const float* sf = to_f32(st, &so);
    if (!sf) goto fail;
    if (!so) {   /* aligned F32 scales in the file: copy anyway, they are tiny */
      so = (float*)arc_malloc(st->nbytes);
      if (!so) goto fail;
      memcpy(so, sf, st->nbytes);
    }
    scale = so;
    m->s = scale;
    if (!strcmp(s, "int8")) {
      if (q->dtype != ARC_I8 || (int64_t)q->nbytes != (int64_t)O * K || arc_numel(st) != O) {
        arc_seterr("%s: bad int8 layout for %s", w->f->path, name);
        goto fail;
      }
      m->type = ARC_W_I8;
      m->q8 = (const int8_t*)q->data;
    } else {
      /* validate before any division or narrowing: 2 <= g <= 65536, K <= Kp <= INT_MAX, g | Kp, Kp even */
      int64_t g = layout_int(L, "g", 32, 2, 65536), Kp = layout_int(L, "Kp", K, K, INT_MAX);
      if (g < 0 || Kp < 0 || Kp % g || Kp % 2) {
        arc_seterr("%s: bad int4 group / padding for %s", w->f->path, name);
        goto fail;
      }
      m->type = ARC_W_I4;
      m->g = (int)g;
      m->Kp = (int)Kp;
      if ((q->dtype != ARC_U8 && q->dtype != ARC_I8) || (int64_t)q->nbytes < (int64_t)O * Kp / 2 ||
          arc_numel(st) != (int64_t)O * (Kp / g)) {
        arc_seterr("%s: bad int4 layout for %s", w->f->path, name);
        goto fail;
      }
      m->q4 = (const uint8_t*)q->data;
    }
  } else {
    float* owned;
    const arc_tensor* t = need(w, name);
    if (!t) goto fail;
    if (arc_numel(t) != (int64_t)O * K) {   /* the layout shape must describe the stored tensor */
      arc_seterr("%s: %s has %lld elements, layout shape says [%d, %d]", w->f->path, name, (long long)arc_numel(t), O, K);
      goto fail;
    }
    const float* f = to_f32(t, &owned);
    if (!f) goto fail;
    m->type = ARC_W_F32;
    m->w = f;
    qcopy = owned;
  }
  e = cache_get(w, name, 1);
  if (!e) goto fail;
  e->mat = m;
  e->scale = scale;
  e->qcopy = qcopy;
  return m;
fail:
  arc_free(scale);
  arc_free(qcopy);
  arc_free(m);
  return NULL;
}

static const float* dequant_full(arc_weights* w, const char* name, const arc_json* L, int64_t* numel, float** owned) {
  const arc_mat* m = arc_weights_mat(w, name, 0, 0);
  if (!m) return NULL;
  float* o = (float*)arc_malloc(sizeof(float) * (size_t)m->O * (size_t)m->K);
  if (!o) return NULL;
  for (int r = 0; r < m->O; ++r)
    for (int k = 0; k < m->K; ++k) {
      float v;
      if (m->type == ARC_W_I8) v = m->q8[(size_t)r * m->K + k] * m->s[r];
      else if (m->type == ARC_W_I4) {
        size_t idx = (size_t)r * m->Kp + k;
        int nib = (m->q4[idx >> 1] >> ((idx & 1) * 4)) & 15;
        v = (float)(nib - 8) * m->s[(size_t)r * (m->Kp / m->g) + k / m->g];
      } else v = m->w[(size_t)r * m->K + k];
      o[(size_t)r * m->K + k] = v;
    }
  (void)L;
  *numel = (int64_t)m->O * m->K;
  *owned = o;
  return o;
}

const char* arc_namef(const char* fmt, ...) {
  static _Thread_local char buf[4][512];
  static _Thread_local int k;
  k = (k + 1) & 3;
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf[k], sizeof buf[k], fmt, ap);
  va_end(ap);
  return buf[k];
}
