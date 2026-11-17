/* arc_echo against arc_audio/echo.py (tools/dump_echo.py -> tests/data/echo.json): examples/example_synth.wav as the
 * playback, its echo through a synthetic room response, with and without another voice over it; playback_leak per window
 * within 0.02 of Python's and the same barge-in decisions.
 *   test_echo tests/data/echo.json ../examples/example_synth.wav
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arc_base.h"
#include "arc_echo.h"
#include "wav.h"

#define SR 16000

static void lcg(unsigned long long seed, double* out, int n) {
  unsigned long long x = seed;
  for (int i = 0; i < n; ++i) {
    x = (1103515245ULL * x + 12345ULL) % 2147483648ULL;
    out[i] = (double)x / 2147483648.0 - 0.5;
  }
}

/* tools/dump_echo.signals */
static void signals(const float* y, int talk, float* mic, float* ref) {
  int n = 8 * SR;
  double rir[1600];
  lcg(7, rir + 1, 1599);
  rir[0] = 1.0;
  double e2 = 1.0;
  for (int i = 1; i < 1600; ++i) {
    rir[i] *= exp(-(double)i / 400);
    e2 += rir[i] * rir[i];
  }
  e2 = 0;
  for (int i = 0; i < 1600; ++i) e2 += rir[i] * rir[i];
  for (int i = 0; i < 1600; ++i) rir[i] /= sqrt(e2);
  double* nz = malloc(sizeof(double) * (size_t)n);
  lcg(11, nz, n);
  for (int i = 0; i < n; ++i) ref[i] = y[i];
  for (int i = 0; i < n; ++i) {
    double ec = 0;
    if (i >= 960) {
      int k = i - 960;
      for (int j = 0; j < 1600 && j <= k; ++j) ec += (double)ref[k - j] * rir[j];
      ec *= 0.5;
    }
    double v = ec + nz[i] * 3e-4;
    if (talk && i >= (int)(5.5 * SR) && i < 7 * SR) v += y[(int)(21.5 * SR) - 1 - (i - (int)(5.5 * SR))];
    mic[i] = (float)v;
  }
  free(nz);
}

int main(int argc, char** argv) {
  char* text = NULL;
  size_t len = 0;
  if (argc < 3 || arc_read_file(argv[1], &text, &len)) {
    fprintf(stderr, "usage: test_echo tests/data/echo.json examples/example_synth.wav\n");
    return 1;
  }
  size_t ny;
  int rate;
  float* y = wav_read(argv[2], &ny, &rate);
  if (!y || ny < (size_t)22 * SR) {
    fprintf(stderr, "test_echo: cannot read %s\n", argv[2]);
    return 1;
  }
  arc_json* j = arc_json_parse(text, len);
  free(text);
  int n = 0, ok = 0, N = 8 * SR;
  float* mic = malloc(sizeof(float) * (size_t)N);
  float* ref = malloc(sizeof(float) * (size_t)N);
  const arc_json* cases = arc_json_get(j, "cases");
  for (int c = 0; c < cases->n; ++c) {
    const arc_json* k = &cases->items[c];
    const char* id = arc_json_str(arc_json_get(k, "id"), "?");
    signals(y, !strcmp(id, "echo_talk"), mic, ref);
    arc_echo* e = arc_echo_new(NULL);
    arc_echo_feed_ref(e, ref, (size_t)N, 0.0);
    for (int a = 0; a < N; a += 1600) arc_echo_feed_mic(e, mic + a, (size_t)(N - a < 1600 ? N - a : 1600));
    arc_echo_flush(e);
    const arc_json* w = arc_json_get(k, "windows");
    for (int i = 0; i < w->n; ++i) {
      const arc_json* x = &w->items[i];
      double t0 = arc_json_num(arc_json_at(x, 0), 0), t1 = arc_json_num(arc_json_at(x, 1), 0);
      double want = arc_json_num(arc_json_at(x, 2), -1);
      int bwant = (int)arc_json_num(arc_json_at(x, 3), 0);
      double got = arc_echo_leak(e, t0, t1);
      int bgot = arc_echo_barge_in(e, t0, t1, 0.2, 0.5);
      int good = fabs(got - want) <= 0.02 && bgot == bwant;
      n++;
      ok += good;
      printf("%s %s [%.1f, %.1f): leak %.4f (python %.4f), barge-in %d (%d)\n", good ? "ok  " : "FAIL", id, t0, t1, got,
             want, bgot, bwant);
    }
    arc_echo_free(e);
  }
  free(mic);
  free(ref);
  free(y);
  arc_json_free(j);
  printf("test_echo: %d / %d\n", ok, n);
  return ok == n ? 0 : 1;
}
