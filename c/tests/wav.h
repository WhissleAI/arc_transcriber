/* 16-bit PCM / float WAV reader for tests and tools (mono, or the first channel).
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#ifndef ARC_WAV_H
#define ARC_WAV_H
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static float* wav_read(const char* path, size_t* n, int* rate) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  unsigned char h[12];
  if (fread(h, 1, 12, f) != 12 || memcmp(h, "RIFF", 4) || memcmp(h + 8, "WAVE", 4)) { fclose(f); return NULL; }
  int fmt = 1, ch = 1, bits = 16;
  *rate = 16000;
  float* out = NULL;
  for (;;) {
    unsigned char c[8];
    if (fread(c, 1, 8, f) != 8) break;
    uint32_t sz = c[4] | c[5] << 8 | c[6] << 16 | (uint32_t)c[7] << 24;
    if (!memcmp(c, "fmt ", 4)) {
      unsigned char b[40];
      size_t r = fread(b, 1, sz < 40 ? sz : 40, f);
      (void)r;
      if (sz > 40) fseek(f, (long)(sz - 40), SEEK_CUR);
      fmt = b[0] | b[1] << 8;
      ch = b[2] | b[3] << 8;
      *rate = (int)(b[4] | b[5] << 8 | b[6] << 16 | (uint32_t)b[7] << 24);
      bits = b[14] | b[15] << 8;
    } else if (!memcmp(c, "data", 4)) {
      size_t bps = (size_t)bits / 8, frames = sz / (bps * (size_t)ch);
      unsigned char* d = malloc(sz);
      if (fread(d, 1, sz, f) != sz) { free(d); break; }
      out = malloc(sizeof(float) * (frames ? frames : 1));
      for (size_t i = 0; i < frames; ++i) {
        const unsigned char* p = d + i * bps * (size_t)ch;
        if (fmt == 3 && bits == 32) memcpy(&out[i], p, 4);
        else if (bits == 16) out[i] = (int16_t)(p[0] | p[1] << 8) / 32768.0f;
        else out[i] = 0;
      }
      free(d);
      *n = frames;
      break;
    } else fseek(f, (long)(sz + (sz & 1)), SEEK_CUR);
  }
  fclose(f);
  return out;
}
#endif
