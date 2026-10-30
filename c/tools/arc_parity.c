/* arc-parity: libarc's candidate for arc_transcriber's golden check (scripts/check_parity.py): the bundled clip with
 * the example names (linked, and biasing the decoder, as predict() with names), domain nba, events on; writes the same
 * JSON keys ("output" transcript / names / numbers / audio_events, "stages" features / event_features / audio_events
 * / frames / transcript).
 *   arc-parity MINI_DIR examples/example_synth.wav examples/example_names.json [domain] > candidate.json
 *   python scripts/check_parity.py --backbone mini --candidate candidate.json
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arc.h"
#include "arc_engine.h"
#include "arc_store.h"
#include "../tests/wav.h"

/* ---- sha256 (for the float16 fingerprint) */
static const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
#define ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
static void sha256(const unsigned char* d, size_t n, char hex[65]) {
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  size_t tot = ((n + 9 + 63) / 64) * 64;
  unsigned char* m = calloc(tot, 1);
  memcpy(m, d, n);
  m[n] = 0x80;
  uint64_t bits = (uint64_t)n * 8;
  for (int i = 0; i < 8; ++i) m[tot - 1 - i] = (unsigned char)(bits >> (8 * i));
  for (size_t off = 0; off < tot; off += 64) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) w[i] = (uint32_t)m[off + 4 * i] << 24 | m[off + 4 * i + 1] << 16 | m[off + 4 * i + 2] << 8 | m[off + 4 * i + 3];
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = ROR(w[i - 15], 7) ^ ROR(w[i - 15], 18) ^ (w[i - 15] >> 3), s1 = ROR(w[i - 2], 17) ^ ROR(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], dd = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t S1 = ROR(e, 6) ^ ROR(e, 11) ^ ROR(e, 25), ch = (e & f) ^ (~e & g), t1 = hh + S1 + ch + K256[i] + w[i];
      uint32_t S0 = ROR(a, 2) ^ ROR(a, 13) ^ ROR(a, 22), mj = (a & b) ^ (a & c) ^ (b & c), t2 = S0 + mj;
      hh = g; g = f; f = e; e = dd + t1; dd = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += dd; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  free(m);
  for (int i = 0; i < 8; ++i) sprintf(hex + 8 * i, "%08x", h[i]);
}

static uint16_t f2h(float x) {   /* the float16 bits numpy writes (round to nearest even) */
  float r = arc_round_half(x);
  uint32_t b;
  memcpy(&b, &r, 4);
  uint32_t sign = (b >> 16) & 0x8000, ex = (b >> 23) & 0xFF, man = b & 0x7FFFFF;
  if (ex == 0xFF) return (uint16_t)(sign | 0x7C00 | (man ? 0x200 : 0));
  int e = (int)ex - 127 + 15;
  if (e >= 31) return (uint16_t)(sign | 0x7C00);
  if (e <= 0) return (uint16_t)(sign | ((man | 0x800000) >> (14 - e)));
  return (uint16_t)(sign | (uint32_t)e << 10 | man >> 13);
}

static void summary(const float* x, int rows, int cols) {
  size_t n = (size_t)rows * cols;
  double s = 0, ss = 0;
  for (size_t i = 0; i < n; ++i) s += x[i];
  double mean = n ? s / n : 0;
  for (size_t i = 0; i < n; ++i) ss += (x[i] - mean) * (x[i] - mean);
  uint16_t* h = malloc(sizeof(uint16_t) * (n ? n : 1));
  for (size_t i = 0; i < n; ++i) h[i] = f2h(x[i]);
  char hex[65];
  sha256((const unsigned char*)h, n * 2, hex);
  free(h);
  printf("{\"shape\": [%d, %d], \"mean\": %.9g, \"std\": %.9g, \"head\": [", rows, cols, mean, n ? sqrt(ss / n) : 0);
  for (size_t i = 0; i < 8 && i < n; ++i) printf("%s%.5f", i ? ", " : "", x[i]);
  printf("], \"sha256_fp16\": \"%s\"}", hex);
}

#include "json_out.h"

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: arc-parity MINI_DIR clip.wav names.json [domain]\n"); return 2; }
  char err[512];
  arc_engine* e = arc_create(argv[1], 4, err, sizeof err);
  if (!e) { fprintf(stderr, "%s\n", err); return 1; }
  const char* domain = argc > 4 ? argv[4] : "nba";
  arc_name_entry* names;
  int nn;
  arc_json* nj = load_names(argv[3], &names, &nn);
  if (!nj) { fprintf(stderr, "%s: %s\n", argv[3], arc_last_error()); return 1; }
  arc_set_context(e, NULL, NULL, 0, NULL, 0, NULL, NULL, 0);   /* no prompt: predict() passes no context */
  if (arc_set_names(e, names, nn)) { fprintf(stderr, "%s\n", arc_last_error()); return 1; }   /* list + bias */
  size_t n; int rate;
  float* pcm = wav_read(argv[2], &n, &rate);
  arc_options o = {.domain = domain, .trunk = 1, .events = 1, .bias_names = 1, .names = 1, .numbers = 1};
  arc_result* r = arc_transcribe(e, pcm, n, &o);
  arc_stages st;
  if (!r || arc_engine_stages(e, pcm, n, domain, &st)) { fprintf(stderr, "%s\n", arc_last_error()); return 1; }
  printf("{\"backbone\": \"mini\", \"input\": \"example_synth.wav\", \"device\": \"libarc\",\n \"output\": {\"transcript\": [");
  for (int i = 0; i < r->n_utts; ++i) {
    printf("%s{\"start\": %.1f, \"end\": %.1f, \"text\": ", i ? ", " : "", r->utts[i].start, r->utts[i].end);
    jstr(r->utts[i].text);
    printf(", \"words\": ");
    print_words(r->utts[i].words, r->utts[i].n_words);
    printf("}");
  }
  printf("],\n  ");
  print_names_numbers(r);
  printf(",\n  \"audio_events\": [");
  for (int i = 0; i < r->n_events; ++i)
    printf("%s{\"label\": \"%s\", \"start\": %.1f, \"end\": %.1f, \"p\": %.4f}", i ? ", " : "", r->events[i].label, r->events[i].start, r->events[i].end, r->events[i].p);
  printf("]},\n \"stages\": {\"features\": ");
  summary(st.X, st.N, 640);
  printf(",\n  \"event_features\": ");
  summary(st.XT, st.N, 640);
  printf(",\n  \"audio_events\": ");
  summary(st.ev, st.N, 50);
  printf(",\n  \"frames\": {");
  for (int q = 0; q < st.nq; ++q) {
    printf("%s\"%s\": ", q ? ", " : "", st.fr[q].q);
    summary(st.fr[q].p, st.N, st.fr[q].ncols);
  }
  printf("},\n  \"transcript\": [");
  for (int i = 0; i < r->n_utts; ++i) {
    printf("%s{\"text\": ", i ? ", " : "");
    jstr(r->utts[i].text);
    printf(", \"tags\": [");
    for (int k = 0; k < r->utts[i].n_tags; ++k) {
      printf("%s[\"%s\", ", k ? ", " : "", r->utts[i].tags[k].tag);
      jstr(r->utts[i].tags[k].text);
      printf("]");
    }
    printf("]}");
  }
  printf("]}}\n");
  arc_stages_free(&st);
  arc_result_free(r);
  free(pcm);
  for (int i = 0; i < nn; ++i) free((void*)names[i].aliases);
  free(names);
  arc_json_free(nj);
  arc_destroy(e);
  return 0;
}
