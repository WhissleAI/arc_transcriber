/* Transcribe a WAV (16 kHz mono, 16-bit) with Arc Mini: the transcript with inline tags, the utterance tags and
 * the sound events.
 *   cc hello.c -I../../c/include ../../c/build/libarc.a -lm -pthread -o hello
 *   ./hello arc-v1.1/mini clip.wav
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arc.h"

/* the samples of a 16-bit PCM WAV (first channel) as floats */
static float* read_wav(const char* path, size_t* n) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  unsigned char h[12], c[8];
  int ch = 1;
  float* out = NULL;
  if (fread(h, 1, 12, f) == 12 && !memcmp(h, "RIFF", 4))
    while (fread(c, 1, 8, f) == 8) {
      uint32_t sz = c[4] | c[5] << 8 | c[6] << 16 | (uint32_t)c[7] << 24;
      if (!memcmp(c, "fmt ", 4)) {
        unsigned char b[16];
        if (fread(b, 1, 16, f) != 16) break;
        ch = b[2] | b[3] << 8;
        fseek(f, (long)sz - 16, SEEK_CUR);
      } else if (!memcmp(c, "data", 4)) {
        int16_t* d = malloc(sz);
        *n = fread(d, 2, sz / 2, f) / (size_t)ch;
        out = malloc(sizeof(float) * (*n + 1));
        for (size_t i = 0; i < *n; ++i) out[i] = d[i * ch] / 32768.0f;
        free(d);
        break;
      } else fseek(f, (long)sz, SEEK_CUR);
    }
  fclose(f);
  return out;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: hello MINI_DIR clip.wav\n");
    return 2;
  }
  char err[512];
  arc_engine* e = arc_create(argv[1], 0, err, sizeof err);   /* 0 threads: all cores */
  if (!e) {
    fprintf(stderr, "arc: %s\n", err);
    return 1;
  }
  size_t n = 0;
  float* pcm = read_wav(argv[2], &n);
  if (!pcm) {
    fprintf(stderr, "cannot read %s\n", argv[2]);
    return 1;
  }
  arc_options o = {.domain = "assistant", .trunk = 1, .events = 1};
  arc_result* r = arc_transcribe(e, pcm, n, &o);
  if (!r) {
    fprintf(stderr, "arc: %s\n", arc_last_error());
    return 1;
  }
  for (int i = 0; i < r->n_utts; ++i) {
    const arc_utt* u = &r->utts[i];
    printf("%5.2f-%5.2f  %s\n", u->start, u->end, u->text);
    for (int k = 0; k < u->n_tags; ++k) printf("             %s: %s\n", u->tags[k].tag, u->tags[k].text);
    printf("             device-directed %.2f, %s %.2f, %s %.2f\n", u->device_directed,
           u->intent.label ? u->intent.label : "-", u->intent.p, u->act.label ? u->act.label : "-", u->act.p);
  }
  for (int i = 0; i < r->n_events; ++i)
    printf("event %s %.1f-%.1f s (p %.2f)\n", r->events[i].label, r->events[i].start, r->events[i].end, r->events[i].p);
  printf("%.1f s of audio in %.0f ms\n", r->audio_s, r->wall_ms);
  arc_result_free(r);
  free(pcm);
  arc_destroy(e);
  return 0;
}
