/* audio.c: see audio.h. Part of the Arc deploy service (Whissle Inc.); CC BY-NC 4.0 with the Whissle terms. */
#define _GNU_SOURCE   /* strcasestr, M_PI, strdup, strtok_r, clock_gettime on glibc with -std=c11 */
#include "audio.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TARGET_RATE 16000

static uint32_t rd32(const unsigned char* p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t rd16(const unsigned char* p) { return (uint16_t)(p[0] | p[1] << 8); }

int audio_is_wav(const unsigned char* d, size_t n) {
  return n >= 12 && (!memcmp(d, "RIFF", 4) || !memcmp(d, "RF64", 4)) && !memcmp(d + 8, "WAVE", 4);
}

static long gcd(long a, long b) {
  while (b) { long t = a % b; a = b; b = t; }
  return a;
}

/* Polyphase windowed-sinc (Blackman, 32 zero crossings at the lower of the two Nyquist rates). */
float* audio_resample(const float* x, size_t n, int rin, int rout, size_t* out_n) {
  if (rin == rout || n == 0) {
    float* y = malloc(sizeof(float) * (n ? n : 1));
    if (n) memcpy(y, x, sizeof(float) * n);
    *out_n = n;
    return y;
  }
  long g = gcd(rin, rout), up = rout / g, down = rin / g;
  size_t m = (size_t)((double)n * up / down);
  float* y = malloc(sizeof(float) * (m ? m : 1));
  double cutoff = (rin < rout ? rin : rout) * 0.5 * 0.97;     /* Hz, a little under the lower Nyquist */
  double fc = cutoff / rin;                                    /* in input-sample units */
  const int zc = 32;
  int half = (int)ceil(zc / (2.0 * fc));                      /* taps each side, in input samples */
  for (size_t i = 0; i < m; ++i) {
    double t = (double)i * down / up;                          /* position in input samples */
    long c = (long)floor(t);
    double acc = 0, wsum = 0;
    for (long k = c - half + 1; k <= c + half; ++k) {
      double dx = t - (double)k;
      double s = dx == 0 ? 2 * fc : sin(2 * M_PI * fc * dx) / (M_PI * dx);
      double u = dx / (half + 1);                               /* -1..1 */
      double w = fabs(u) >= 1 ? 0 : 0.42 + 0.5 * cos(M_PI * u) + 0.08 * cos(2 * M_PI * u);
      double h = s * w;
      wsum += h;
      if (k >= 0 && (size_t)k < n) acc += h * x[k];
    }
    y[i] = (float)(wsum != 0 ? acc / wsum : 0);
  }
  *out_n = m;
  return y;
}

float* audio_decode_wav(const unsigned char* d, size_t n, size_t* out_n, int* rate_in, char* err, size_t el) {
  if (!audio_is_wav(d, n)) { snprintf(err, el, "not a RIFF/WAVE file"); return NULL; }
  int fmt = 0, ch = 0, bits = 0, rate = 0;
  const unsigned char* data = NULL;
  size_t dlen = 0;
  size_t p = 12;
  while (p + 8 <= n) {
    uint32_t sz = rd32(d + p + 4);
    const unsigned char* body = d + p + 8;
    size_t avail = n - (p + 8);
    if (!memcmp(d + p, "fmt ", 4) && sz >= 16 && avail >= 16) {
      fmt = rd16(body);
      ch = rd16(body + 2);
      rate = (int)rd32(body + 4);
      bits = rd16(body + 14);
      if (fmt == 0xFFFE && sz >= 40 && avail >= 40) fmt = rd16(body + 24);   /* WAVE_FORMAT_EXTENSIBLE: sub-format GUID */
    } else if (!memcmp(d + p, "data", 4)) {
      data = body;
      dlen = sz > avail || sz == 0xFFFFFFFFu ? avail : sz;   /* streamed WAVs write 0 / -1 / a too-large size */
      if (sz == 0) dlen = avail;
      break;
    }
    p += 8 + (size_t)sz + (sz & 1);
  }
  if (!fmt || !data) { snprintf(err, el, "WAV without fmt / data chunk"); return NULL; }
  if (ch < 1 || ch > 64 || rate < 1000 || rate > 768000) { snprintf(err, el, "unsupported WAV: %d channels at %d Hz", ch, rate); return NULL; }
  int is_float = fmt == 3;
  if (!(fmt == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) && !(is_float && (bits == 32 || bits == 64))) {
    snprintf(err, el, "unsupported WAV encoding (format %d, %d bits): send PCM 8/16/24/32-bit or float 32/64", fmt, bits);
    return NULL;
  }
  size_t bps = (size_t)bits / 8, frames = dlen / (bps * (size_t)ch);
  float* mono = malloc(sizeof(float) * (frames ? frames : 1));
  for (size_t i = 0; i < frames; ++i) {
    double s = 0;
    for (int c = 0; c < ch; ++c) {
      const unsigned char* q = data + (i * (size_t)ch + (size_t)c) * bps;
      double v;
      if (is_float && bits == 32) { float f; memcpy(&f, q, 4); v = f; }
      else if (is_float) { double f; memcpy(&f, q, 8); v = f; }
      else if (bits == 8) v = (q[0] - 128) / 128.0;
      else if (bits == 16) v = (int16_t)rd16(q) / 32768.0;
      else if (bits == 24) v = (double)((int32_t)((uint32_t)q[0] << 8 | (uint32_t)q[1] << 16 | (uint32_t)q[2] << 24) >> 8) / 8388608.0;
      else v = (int32_t)rd32(q) / 2147483648.0;
      s += v;
    }
    mono[i] = (float)(s / ch);
  }
  *rate_in = rate;
  float* y = audio_resample(mono, frames, rate, TARGET_RATE, out_n);
  free(mono);
  return y;
}

float* audio_decode_pcm16(const unsigned char* d, size_t n, int rate, int ch, size_t* out_n, char* err, size_t el) {
  if (ch < 1 || ch > 64 || rate < 1000 || rate > 768000) { snprintf(err, el, "bad PCM parameters (%d ch, %d Hz)", ch, rate); return NULL; }
  size_t frames = n / (2 * (size_t)ch);
  float* mono = malloc(sizeof(float) * (frames ? frames : 1));
  for (size_t i = 0; i < frames; ++i) {
    int s = 0;
    for (int c = 0; c < ch; ++c) s += (int16_t)rd16(d + 2 * (i * (size_t)ch + (size_t)c));
    mono[i] = (float)s / (32768.0f * ch);
  }
  float* y = audio_resample(mono, frames, rate, TARGET_RATE, out_n);
  free(mono);
  return y;
}
