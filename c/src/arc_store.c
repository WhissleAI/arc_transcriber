/* Arc weights in their safetensors file: plain tensors and Arc's packed 4-bit tensors, as float32.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include "arc_store.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "arc_base.h"
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

typedef struct {
  char* name;
  int packed;                 /* 1: W.q / W.s / W.z with a layout entry */
  int64_t shape[8];
  int ndim;
  int axis;                   /* -1 or -2 */
  int64_t R, C, pad;
  int half;                   /* layout dtype float16: round through half */
  const arc_tensor* plain;    /* plain: the tensor itself */
  const arc_tensor *q, *sc, *zp;
  float* f32;                 /* dequantized / converted, owned; NULL until asked for */
  float* scales;              /* aligned copy of the scales for in-place use */
} entry;

struct arc_store {
  arc_file* f;
  arc_json* layout;
  int group;
  entry* e;
  int n;
};

/* float32 -> float16 bits, round to nearest even (no _Float16 needed) */
static uint16_t to_half(float x) {
  uint32_t b;
  memcpy(&b, &x, 4);
  uint32_t sign = (b >> 16) & 0x8000, m = b & 0x7FFFFF;
  int32_t ex = (int32_t)((b >> 23) & 0xFF);
  if (ex == 0xFF) return (uint16_t)(sign | 0x7C00 | (m ? 0x200 : 0));
  int32_t e = ex - 127 + 15;
  if (e >= 31) return (uint16_t)(sign | 0x7C00);
  if (e <= 0) {   /* half subnormal: units of 2^-24 */
    if (e < -10) return (uint16_t)sign;
    m |= 0x800000;
    int sh = 14 - e;
    uint32_t r = m >> sh, rem = m & ((1u << sh) - 1), halfway = 1u << (sh - 1);
    if (rem > halfway || (rem == halfway && (r & 1))) r++;
    return (uint16_t)(sign | r);
  }
  uint32_t h = sign | ((uint32_t)e << 10) | (m >> 13), rem = m & 0x1FFF;
  if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;   /* a carry rolls into the exponent, as it should */
  return (uint16_t)h;
}

float arc_round_half(float x) {
#if defined(__FLT16_MAX__)
  return (float)(_Float16)x;
#else
  return arc_half_to_float(to_half(x));
#endif
}

static entry* find(const arc_store* s, const char* name) {
  for (int i = 0; i < s->n; ++i)
    if (!strcmp(s->e[i].name, name)) return &s->e[i];
  return NULL;
}

const arc_tensor* arc_store_raw(const arc_store* s, const char* name) {
  const entry* e = find(s, name);
  return e && !e->packed ? e->plain : NULL;
}

static int suffixed(const char* name, const char* suf) {
  size_t n = strlen(name), m = strlen(suf);
  return n > m && !strcmp(name + n - m, suf);
}

arc_store* arc_store_open(const char* path) {
  arc_store* s = arc_calloc(sizeof *s);
  if (!s) return NULL;
  if (!(s->f = arc_file_open(path))) goto fail;
  s->group = 32;
  const char* g = arc_file_metadata(s->f, "group_size");
  if (g) s->group = atoi(g);
  const char* lay = arc_file_metadata(s->f, "int4_layout");
  if (lay && !(s->layout = arc_json_parse(lay, strlen(lay)))) goto fail;
  int nt = arc_file_count(s->f), cap = nt + 1;
  s->e = arc_calloc(sizeof(entry) * (size_t)cap);
  if (!s->e) goto fail;
  /* packed tensors first, from the layout */
  if (s->layout && s->layout->type == ARC_JOBJ) {
    for (int i = 0; i < s->layout->n; ++i) {
      const char* name = s->layout->keys[i];
      const arc_json* m = &s->layout->items[i];
      entry* e = &s->e[s->n];
      e->name = arc_strdup(name);
      e->packed = 1;
      const arc_json* sh = arc_json_get(m, "shape");
      e->ndim = sh ? sh->n : 0;
      for (int k = 0; k < e->ndim && k < 8; ++k) e->shape[k] = (int64_t)arc_json_num(arc_json_at(sh, k), 0);
      e->axis = (int)arc_json_num(arc_json_get(m, "axis"), -1);
      const arc_json* rc = arc_json_get(m, "rc");
      e->R = (int64_t)arc_json_num(arc_json_at(rc, 0), 0);
      e->C = (int64_t)arc_json_num(arc_json_at(rc, 1), 0);
      e->pad = (int64_t)arc_json_num(arc_json_at(rc, 2), 0);
      e->half = !strcmp(arc_json_str(arc_json_get(m, "dtype"), "float32"), "float16");
      e->q = arc_file_find(s->f, arc_namef("%s.q", name));
      e->sc = arc_file_find(s->f, arc_namef("%s.s", name));
      e->zp = arc_file_find(s->f, arc_namef("%s.z", name));
      if (!e->name || !e->q || !e->sc || !e->zp) {
        arc_seterr("%s: packed tensor %s is incomplete", path, name);
        s->n++;
        goto fail;
      }
      s->n++;
    }
  }
  /* then plain tensors that are not parts of a packed one */
  for (int i = 0; i < nt; ++i) {
    const arc_tensor* t = arc_file_tensor(s->f, i);
    if (suffixed(t->name, ".q") || suffixed(t->name, ".s") || suffixed(t->name, ".z")) {
      char base[512];
      size_t l = strlen(t->name) - 2;
      if (l < sizeof base) {
        memcpy(base, t->name, l);
        base[l] = 0;
        if (find(s, base) && find(s, base)->packed) continue;
      }
    }
    if (s->n >= cap) break;
    entry* e = &s->e[s->n++];
    e->name = arc_strdup(t->name);
    e->plain = t;
    e->ndim = t->ndim;
    for (int k = 0; k < t->ndim; ++k) e->shape[k] = t->shape[k];
  }
  return s;
fail:
  arc_store_close(s);
  return NULL;
}

void arc_store_close(arc_store* s) {
  if (!s) return;
  for (int i = 0; i < s->n; ++i) {
    arc_free(s->e[i].name);
    arc_free(s->e[i].f32);
    arc_free(s->e[i].scales);
  }
  arc_free(s->e);
  arc_json_free(s->layout);
  arc_file_close(s->f);
  arc_free(s);
}

int arc_store_has(const arc_store* s, const char* name) { return find(s, name) != NULL; }

int arc_store_shape(const arc_store* s, const char* name, int64_t shape[8]) {
  entry* e = find(s, name);
  if (!e) return 0;
  for (int k = 0; k < e->ndim; ++k) shape[k] = e->shape[k];
  return e->ndim;
}

int arc_store_count(const arc_store* s) { return s->n; }
const char* arc_store_name(const arc_store* s, int i) { return i >= 0 && i < s->n ? s->e[i].name : NULL; }

static int64_t numel(const entry* e) {
  int64_t n = 1;
  for (int k = 0; k < e->ndim; ++k) n *= e->shape[k];
  return n;
}

static int dequant(arc_store* s, entry* e) {
  const int g = s->group;
  const int64_t R = e->R, C = e->C, Cp = e->C + e->pad, n = numel(e);
  if (R * C != n || Cp % g) {
    arc_seterr("%s: layout rc [%lld, %lld, %lld] does not match its shape", e->name, (long long)R, (long long)C,
               (long long)e->pad);
    return -1;
  }
  const int64_t groups = R * Cp / g;
  if ((int64_t)e->q->nbytes < groups * g / 2 || (int64_t)e->sc->nbytes < groups * 4 ||
      (int64_t)e->zp->nbytes < (groups + 1) / 2) {
    arc_seterr("%s: packed data is too short", e->name);
    return -1;
  }
  float* out = arc_malloc(sizeof(float) * (size_t)n);
  if (!out) return -1;
  const uint8_t* q = e->q->data;
  const uint8_t* z = e->zp->data;
  const unsigned char* sb = e->sc->data;
  /* the matrix the codes describe: [R, C] (axis -1) or, for axis -2, the shape with that axis moved last */
  for (int64_t r = 0; r < R; ++r) {
    for (int64_t c = 0; c < C; ++c) {
      int64_t k = r * Cp + c, gi = k / g;
      int code = (q[k >> 1] >> ((k & 1) * 4)) & 15;
      int zero = (z[gi >> 1] >> ((gi & 1) * 4)) & 15;
      float sc;
      memcpy(&sc, sb + 4 * gi, 4);
      float v = ((float)code - (float)zero) * sc;
      if (e->half) v = arc_round_half(v);
      int64_t dst;
      if (e->axis == -1) dst = r * C + c;
      else {
        /* moved: [d, b, a] with R = d * b, C = a; logical [d, a, b] */
        int64_t d0 = e->shape[0], a = e->shape[1], b = e->shape[2];
        (void)d0;
        int64_t di = r / b, bi = r % b, ai = c;
        dst = (di * a + ai) * b + bi;
      }
      out[dst] = v;
    }
  }
  e->f32 = out;
  return 0;
}

static int convert(entry* e) {
  const arc_tensor* t = e->plain;
  int64_t n = arc_numel(t);
  float* out = arc_malloc(sizeof(float) * (size_t)(n ? n : 1));
  if (!out) return -1;
  const unsigned char* p = t->data;
  for (int64_t i = 0; i < n; ++i) {
    switch (t->dtype) {
    case ARC_F32: memcpy(&out[i], p + 4 * i, 4); break;
    case ARC_F16: {
      uint16_t h;
      memcpy(&h, p + 2 * i, 2);
      out[i] = arc_half_to_float(h);
      break;
    }
    case ARC_BF16: {
      uint16_t h;
      memcpy(&h, p + 2 * i, 2);
      out[i] = arc_bf16_to_float(h);
      break;
    }
    case ARC_F64: {
      double d;
      memcpy(&d, p + 8 * i, 8);
      out[i] = (float)d;
      break;
    }
    case ARC_I64: {
      int64_t v;
      memcpy(&v, p + 8 * i, 8);
      out[i] = (float)v;
      break;
    }
    default:
      arc_free(out);
      arc_seterr("%s: unsupported dtype", e->name);
      return -1;
    }
  }
  e->f32 = out;
  return 0;
}

const float* arc_store_f32(arc_store* s, const char* name, int64_t n_expect) {
  entry* e = find(s, name);
  if (!e) {
    arc_seterr("missing tensor %s", name);
    return NULL;
  }
  if (n_expect > 0 && numel(e) != n_expect) {
    arc_seterr("%s: %lld elements, expected %lld", name, (long long)numel(e), (long long)n_expect);
    return NULL;
  }
  if (!e->f32 && (e->packed ? dequant(s, e) : convert(e))) return NULL;
  return e->f32;
}

/* ------------------------------------------------------------------ packed use */

int arc_store_q4(arc_store* s, const char* name, arc_q4* out) {
  entry* e = find(s, name);
  if (!e || !e->packed || e->axis != -1 || e->half) return 0;
  const int64_t Cp = e->C + e->pad, groups = e->R * Cp / s->group;
  if (s->group != 32 || Cp % 32) return 0;
  if (!e->scales) {
    e->scales = arc_malloc(sizeof(float) * (size_t)groups);
    if (!e->scales) return 0;
    memcpy(e->scales, e->sc->data, sizeof(float) * (size_t)groups);
  }
  out->R = (int)e->R;
  out->C = (int)e->C;
  out->Cp = (int)Cp;
  out->q = e->q->data;
  out->z = e->zp->data;
  out->s = e->scales;
  return 1;
}

void arc_q4_row(const arc_q4* W, int r, float* out) {
  const int64_t base = (int64_t)r * W->Cp;
#if defined(__ARM_NEON)
  /* whole groups of 32: 16 code bytes -> 32 values in order (low nibble first), (code - zero) * scale */
  if (!(base & 31)) {
    int c = 0;
    for (; c + 32 <= W->C; c += 32) {
      int64_t k = base + c, g = k >> 5;
      uint8x16_t b = vld1q_u8(W->q + (k >> 1));
      uint8x16_t lo = vandq_u8(b, vdupq_n_u8(15)), hi = vshrq_n_u8(b, 4);
      uint8x16_t v0 = vzip1q_u8(lo, hi), v1 = vzip2q_u8(lo, hi);
      int zero = (W->z[g >> 1] >> ((g & 1) * 4)) & 15;
      float32x4_t zf = vdupq_n_f32((float)zero), sf = vdupq_n_f32(W->s[g]);
      uint8x16_t vs[2] = {v0, v1};
      for (int h = 0; h < 2; ++h) {
        uint16x8_t w0 = vmovl_u8(vget_low_u8(vs[h])), w1 = vmovl_u8(vget_high_u8(vs[h]));
        uint16x8_t ws[2] = {w0, w1};
        for (int q2 = 0; q2 < 2; ++q2) {
          float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(ws[q2])));
          float32x4_t f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(ws[q2])));
          float* o = out + c + h * 16 + q2 * 8;
          vst1q_f32(o, vmulq_f32(vsubq_f32(f0, zf), sf));
          vst1q_f32(o + 4, vmulq_f32(vsubq_f32(f1, zf), sf));
        }
      }
    }
    for (; c < W->C; ++c) {
      int64_t k = base + c, g = k >> 5;
      int code = (W->q[k >> 1] >> ((k & 1) * 4)) & 15;
      int zero = (W->z[g >> 1] >> ((g & 1) * 4)) & 15;
      out[c] = ((float)code - (float)zero) * W->s[g];
    }
    return;
  }
#endif
  for (int c = 0; c < W->C; ++c) {
    int64_t k = base + c, g = k >> 5;
    int code = (W->q[k >> 1] >> ((k & 1) * 4)) & 15;
    int zero = (W->z[g >> 1] >> ((g & 1) * 4)) & 15;
    out[c] = ((float)code - (float)zero) * W->s[g];
  }
}

typedef struct {
  const float* x;
  int M, ldx;
  const arc_q4* W;
  const float* bias;
  float* y;
  int ldy;
} q4job;

static void q4_task(void* ctx, int task, int n_tasks) {
  const q4job* j = ctx;
  const arc_q4* W = j->W;
  int r0 = (int)((int64_t)W->R * task / n_tasks), r1 = (int)((int64_t)W->R * (task + 1) / n_tasks);
  float* row = arc_malloc(sizeof(float) * (size_t)W->C + 64);
  if (!row) return;
  for (int r = r0; r < r1; ++r) {
    arc_q4_row(W, r, row);
    float b = j->bias ? j->bias[r] : 0.0f;
    for (int m = 0; m < j->M; ++m) {
      const float* xr = j->x + (size_t)m * j->ldx;
#if defined(__ARM_NEON)
      float32x4_t v0 = vdupq_n_f32(0), v1 = v0, v2 = v0, v3 = v0;
      int c = 0;
      for (; c + 16 <= W->C; c += 16) {
        v0 = vfmaq_f32(v0, vld1q_f32(xr + c), vld1q_f32(row + c));
        v1 = vfmaq_f32(v1, vld1q_f32(xr + c + 4), vld1q_f32(row + c + 4));
        v2 = vfmaq_f32(v2, vld1q_f32(xr + c + 8), vld1q_f32(row + c + 8));
        v3 = vfmaq_f32(v3, vld1q_f32(xr + c + 12), vld1q_f32(row + c + 12));
      }
      float acc = vaddvq_f32(vaddq_f32(vaddq_f32(v0, v1), vaddq_f32(v2, v3)));
      for (; c < W->C; ++c) acc += xr[c] * row[c];
      j->y[(size_t)m * j->ldy + r] = b + acc;
#else
      float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
      int c = 0;
      for (; c + 4 <= W->C; c += 4) {
        a0 += xr[c] * row[c];
        a1 += xr[c + 1] * row[c + 1];
        a2 += xr[c + 2] * row[c + 2];
        a3 += xr[c + 3] * row[c + 3];
      }
      for (; c < W->C; ++c) a0 += xr[c] * row[c];
      j->y[(size_t)m * j->ldy + r] = b + ((a0 + a1) + (a2 + a3));
#endif
    }
  }
  arc_free(row);
}

void arc_q4_linear(void* pool, const float* x, int M, int ldx, const arc_q4* W, const float* bias, float* y, int ldy) {
  q4job j = {x, M, ldx, W, bias, y, ldy};
  arc_pool* p = pool;
  int n = arc_pool_size(p) * 4;
  if (n > W->R) n = W->R;
  if (n < 1) n = 1;
  arc_parallel_for(p, n, q4_task, &j);
}
