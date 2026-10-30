/* Arc weights: one safetensors file, plain (F32 / F16 tensors) or packed in Arc's 4-bit format (docs/WEIGHTS_FORMAT.md
 * in arc_transcriber: metadata "int4_layout", tensors W.q / W.s / W.z). Every tensor is handed out as float32,
 * dequantized bit for bit as arc_audio/weights.py does (a float16 layout entry is rounded to float16 and back), and
 * kept until the store is closed.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#ifndef ARC_STORE_H
#define ARC_STORE_H

#include <stdint.h>

#include "arc_base.h"

typedef struct arc_store arc_store;

arc_store* arc_store_open(const char* path);   /* NULL + arc_seterr */
void arc_store_close(arc_store* s);
/* 1 when the tensor exists (packed or plain), under its logical name ("model.encoder.layers.0.fc1.weight"). */
int arc_store_has(const arc_store* s, const char* name);
/* The tensor as float32 [prod(shape)], with its logical shape. n_expect > 0 checks the element count. NULL +
 * arc_seterr when absent or the count disagrees. */
const float* arc_store_f32(arc_store* s, const char* name, int64_t n_expect);
/* A plain (not packed) tensor as stored (its dtype and data in the mapping), or NULL. */
const arc_tensor* arc_store_raw(const arc_store* s, const char* name);
/* Its logical shape: returns ndim (0 if absent). */
int arc_store_shape(const arc_store* s, const char* name, int64_t shape[8]);
/* Number of logical tensors and the i-th name (for checks). */
int arc_store_count(const arc_store* s);
const char* arc_store_name(const arc_store* s, int i);
/* A packed 4-bit matrix used in place: [R, C] (axis -1, float32 layout), groups of 32 along each row padded to Cp.
 * w[r, c] = (code - zero) * scale, exactly as the dequantised tensor. */
typedef struct {
  int R, C, Cp;
  const uint8_t* q;    /* codes, two per byte, low nibble first, row-major over [R, Cp] */
  const uint8_t* z;    /* zero points, two per byte */
  const float* s;      /* scales, one per group (aligned copy, owned by the store) */
} arc_q4;
/* The packed view of a tensor, when it is one (axis -1, float32): 1, else 0 (use arc_store_f32). */
int arc_store_q4(arc_store* s, const char* name, arc_q4* out);
/* y[M, R] (+)= x[M, C] W^T + bias, W packed. Rows of W are dequantised once each, in place of a float copy. */
void arc_q4_linear(void* pool, const float* x, int M, int ldx, const arc_q4* W, const float* bias, float* y, int ldy);
/* One row of W as float32 (an embedding lookup). */
void arc_q4_row(const arc_q4* W, int r, float* out);

/* float32 -> float16 -> float32, round to nearest even (torch's .half()). */
float arc_round_half(float x);

#endif
