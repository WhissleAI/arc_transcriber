/* Minimal .npy reader for tests: little-endian float32 ('<f4'), C order. Returns data (free()) and the shape.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#ifndef ARC_TEST_NPY_H
#define ARC_TEST_NPY_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static float* npy_load(const char* path, int* ndim, long shape[4]) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  unsigned char hdr[10];
  if (fread(hdr, 1, 10, f) != 10) { fclose(f); return NULL; }
  int maj = hdr[6];
  size_t hl = maj == 1 ? (size_t)(hdr[8] | hdr[9] << 8) : 0;
  if (maj != 1) { unsigned char b[2]; if (fread(b, 1, 2, f) != 2) { fclose(f); return NULL; } hl = (size_t)(hdr[8] | hdr[9] << 8 | b[0] << 16 | b[1] << 24); }
  char* h = malloc(hl + 1);
  if (fread(h, 1, hl, f) != hl) { fclose(f); free(h); return NULL; }
  h[hl] = 0;
  if (!strstr(h, "'<f4'")) { fclose(f); free(h); return NULL; }
  char* p = strstr(h, "'shape': (");
  *ndim = 0;
  long n = 1;
  if (p) {
    p += 10;
    while (*p && *p != ')') {
      while (*p == ' ' || *p == ',') p++;
      if (*p == ')') break;
      shape[(*ndim)++] = strtol(p, &p, 10);
      n *= shape[*ndim - 1];
    }
  }
  free(h);
  float* d = malloc(sizeof(float) * (size_t)(n ? n : 1));
  if (fread(d, 4, (size_t)n, f) != (size_t)n) { fclose(f); free(d); return NULL; }
  fclose(f);
  return d;
}
#endif
