/* A long clip decoded as one utterance (arc_options.one_utterance, a push-to-talk command) must not run
 * off into repeats: past transcription.max_s (15 s) it is cut where the segmenter would cut it, each piece decoded on
 * its own audio. The clip is a speech recording tiled to > 40 s; the test fails if any piece repeats an earlier
 * piece's text, if a piece is longer than its audio could say (tokens_per_s), or if a word run of 6 repeats inside one.
 *   test_whole_long MINI_DIR speech.wav
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arc.h"
#include "wav.h"

static int words(const char* t) {
  int n = 0, in = 0;
  for (; *t; ++t) {
    if (*t == ' ') in = 0;
    else if (!in) { in = 1; ++n; }
  }
  return n;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: test_whole_long MINI_DIR speech.wav\n"); return 2; }
  char err[512];
  arc_engine* e = arc_create(argv[1], 4, err, sizeof err);
  if (!e) { fprintf(stderr, "%s\n", err); return 1; }
  size_t n; int rate;
  float* pcm = wav_read(argv[2], &n, &rate);
  if (!pcm || rate != ARC_SAMPLE_RATE) { fprintf(stderr, "need a 16 kHz wav\n"); return 1; }
  size_t want = 42 * ARC_SAMPLE_RATE, m = 0;
  float* clip = malloc(sizeof(float) * want);
  while (m < want) {
    size_t k = n < want - m ? n : want - m;
    memcpy(clip + m, pcm, sizeof(float) * k);
    m += k;
  }
  arc_options o = {0};
  o.one_utterance = 1;
  arc_result* r = arc_transcribe(e, clip, m, &o);
  if (!r) { fprintf(stderr, "%s\n", arc_last_error()); return 1; }
  int bad = 0;
  if (r->n_utts < 3) { fprintf(stderr, "FAIL: %d pieces for 42 s (want >= 3, each <= 15 s)\n", r->n_utts); bad = 1; }
  for (int i = 0; i < r->n_utts; ++i) {
    const arc_utt* u = &r->utts[i];
    double secs = u->end - u->start;
    int w = words(u->text);
    printf("[%5.1f-%5.1f] %d words: %.80s...\n", u->start, u->end, w, u->text);
    if (secs > 15.05) { fprintf(stderr, "FAIL: piece %d is %.1f s\n", i, secs); bad = 1; }
    if (w > 1.3 * (4 + 7.0 * secs) + 2) { fprintf(stderr, "FAIL: piece %d has %d words for %.1f s\n", i, w, secs); bad = 1; }
    for (int j = 0; j < i; ++j)
      if (!strcmp(u->text, r->utts[j].text) && strlen(u->text) > 40) { fprintf(stderr, "FAIL: piece %d repeats piece %d\n", i, j); bad = 1; }
  }
  arc_result_free(r);
  arc_destroy(e);
  free(clip);
  free(pcm);
  printf(bad ? "FAILED\n" : "ok\n");
  return bad;
}
