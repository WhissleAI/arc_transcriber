/* arc_spk against arc_audio's diarization on multi-speaker clips (tools/dump_speakers.py): the per-frame inputs arc_audio's
 * Engine saw (speaker index, speech, slot logits, diar embedding) replayed through the slot turns and the TrackBank
 * (fused: beta of the domain, voice: beta 0); every turn's speaker, start and end must match. Also the turn a span
 * overlaps most (arc_transcriber output.py lines_of).
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arc_base.h"
#include "arc_speaker.h"

static int n = 0, ok = 0;

static int b64v(int c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  return c == '+' ? 62 : c == '/' ? 63 : -1;
}

/* base64 of little-endian float32 -> floats (free()); *count floats */
static float* b64_floats(const char* s, long* count) {
  size_t L = strlen(s);
  unsigned char* b = malloc(L / 4 * 3 + 4);
  size_t k = 0;
  unsigned v = 0;
  int bits = 0;
  for (size_t i = 0; i < L; ++i) {
    int d = b64v((unsigned char)s[i]);
    if (d < 0) continue;
    v = v << 6 | (unsigned)d;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      b[k++] = (unsigned char)(v >> bits);
    }
  }
  *count = (long)(k / 4);
  float* f = malloc(sizeof(float) * (size_t)(*count + 1));
  for (long i = 0; i < *count; ++i) {
    unsigned u = b[4 * i] | b[4 * i + 1] << 8 | b[4 * i + 2] << 16 | (unsigned)b[4 * i + 3] << 24;
    memcpy(&f[i], &u, 4);
  }
  free(b);
  return f;
}

static void one(const arc_json* j) {
  const char* clip = arc_json_str(arc_json_get(j, "clip"), "?");
  int N = (int)arc_json_num(arc_json_get(j, "n_frames"), 0), ns = (int)arc_json_num(arc_json_get(j, "n_slot"), 0);
  int de = (int)arc_json_num(arc_json_get(j, "d_emb"), 0);
  const arc_json *bk = arc_json_get(j, "bank"), *spk = arc_json_get(j, "spk"), *sp = arc_json_get(j, "speech");
  long nsl, nem;
  float* slot = b64_floats(arc_json_str(arc_json_get(j, "slot"), ""), &nsl);
  float* emb = b64_floats(arc_json_str(arc_json_get(j, "emb"), ""), &nem);
  if (nsl != (long)N * ns || nem != (long)N * de || spk->n != N || sp->n != N) {
    printf("%s: bad fixture\n", clip);
    n++;
    free(slot);
    free(emb);
    return;
  }
  static const char* modes[3] = {"slots", "fused", "voice"};
  float* zs = calloc(8, sizeof(float));   /* the pooled read-out is not part of this test: dm 8 of zeros */
  for (int m = 0; m < 3; ++m) {
    arc_bank_cfg bc = {(float)arc_json_num(arc_json_get(bk, "beta"), 2), (float)arc_json_num(arc_json_get(bk, "tau_new"), 0.5),
                       (float)arc_json_num(arc_json_get(bk, "tau_upd"), 0.5), (int)arc_json_num(arc_json_get(bk, "confirm"), 20),
                       (int)arc_json_num(arc_json_get(bk, "switch"), 10), (int)arc_json_num(arc_json_get(bk, "smooth"), 5)};
    if (m == ARC_DIAR_VOICE) bc.beta = 0;
    arc_spk* s = arc_spk_new(m, &bc, 8, ns, de);
    for (int f = 0; f < N; ++f)
      arc_spk_frame(s, f, (int)arc_json_num(arc_json_at(spk, f), -1), (int)arc_json_num(arc_json_at(sp, f), 0),
                    slot + (size_t)f * ns, emb + (size_t)f * de, zs);
    arc_spk_finish(s, N);
    arc_spk_turn* t = NULL;
    int nt = arc_spk_turns(s, 0, &t);
    const arc_json* want = arc_json_get(arc_json_get(j, "diarization"), modes[m]);
    int good = want && want->n == nt;
    for (int i = 0; good && i < nt; ++i) {
      char lb[16];
      const arc_json* w = arc_json_at(want, i);
      good = !strcmp(arc_spk_label(s, &t[i], lb), arc_json_str(arc_json_get(w, "speaker"), "")) &&
             t[i].f0 == lround(arc_json_num(arc_json_get(w, "start"), -1) * 10) &&
             t[i].f1 == lround(arc_json_num(arc_json_get(w, "end"), -1) * 10);
    }
    n++;
    ok += good;
    if (!good) {
      printf("%s %s: got", clip, modes[m]);
      for (int i = 0; i < nt; ++i) {
        char lb[16];
        printf(" %s %.1f-%.1f", arc_spk_label(s, &t[i], lb), t[i].f0 / 10.0, t[i].f1 / 10.0);
      }
      printf("; want %d turns\n", want ? want->n : -1);
    }
    /* the speakers the slot machine saw (the read-out's) are the indices with frames, last frame each */
    if (m == ARC_DIAR_SLOTS) {
      long last[ARC_SPK_MAX];
      for (int k = 0; k < ARC_SPK_MAX; ++k) last[k] = -1;
      for (int f = 0; f < N; ++f) {
        int k = (int)arc_json_num(arc_json_at(spk, f), -1);
        if (k >= 0 && k < ARC_SPK_MAX) last[k] = f;
      }
      int c = 0, good2 = 1;
      for (int k = 0; k < ARC_SPK_MAX; ++k) {
        if (last[k] < 0) continue;
        int idx;
        long l;
        good2 &= !arc_spk_info(s, c++, &idx, &l, NULL) && idx == k && l == last[k];
      }
      good2 &= c == arc_spk_count(s);
      n++;
      ok += good2;
      if (!good2) printf("%s: speaker indices / last frames differ\n", clip);
    }
    arc_free(t);
    arc_spk_free(s);
  }
  free(zs);
  free(slot);
  free(emb);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: test_speaker tests/data/speakers.jsonl\n");
    return 2;
  }
  char* text = NULL;
  size_t len = 0;
  if (arc_read_file(argv[1], &text, &len)) {
    perror(argv[1]);
    return 2;
  }
  int clips = 0;
  for (char *p = text, *e; *p; p = e + (*e != 0)) {
    e = strchr(p, '\n');
    if (!e) e = p + strlen(p);
    if (e > p) {
      arc_json* j = arc_json_parse(p, (size_t)(e - p));
      if (j) {
        one(j);
        clips++;
      }
      arc_json_free(j);
    }
  }
  arc_free(text);
  printf("test_speaker: %d / %d checks on %d clips\n", ok, n, clips);
  return ok == n && n > 0 ? 0 : 1;
}
