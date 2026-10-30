/* arc-score: how well texts match a clip (arc_score), and the clip's n-best (arc_transcribe's alts).
 *   arc-score MINI_DIR clip.wav [--tags] [--topic T] [--threads N] [--nbest] "text 1" "text 2" ...
 * prints "logp<TAB>n_tokens<TAB>text" per text (texts may mark a name span with braces: "read {Treasure Island.}").
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arc.h"
#include "arc_base.h"
#include "../tests/wav.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: arc-score MINI_DIR clip.wav [--tags] [--topic T] [--threads N] [--nbest] text...\n");
    return 2;
  }
  int flags = 0, threads = 2, nbest = 0;
  const char* topic = NULL;
  const char** texts = malloc(sizeof(char*) * (size_t)argc);
  int nt = 0;
  for (int i = 3; i < argc; ++i) {
    if (!strcmp(argv[i], "--tags")) flags |= ARC_SCORE_TAGS;
    else if (!strcmp(argv[i], "--nbest")) nbest = 1;
    else if (!strcmp(argv[i], "--topic") && i + 1 < argc) topic = argv[++i];
    else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    else texts[nt++] = argv[i];
  }
  char err[512];
  arc_engine* e = arc_create(argv[1], threads, err, sizeof err);
  if (!e) { fprintf(stderr, "arc: %s\n", err); return 1; }
  if (topic) arc_set_context(e, NULL, NULL, 0, NULL, 0, topic, NULL, 0);
  size_t n;
  int rate;
  float* pcm = wav_read(argv[2], &n, &rate);
  if (!pcm) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
  if (nbest) {
    arc_options o = {0};
    o.one_utterance = 1;
    double t0 = arc_time_ms();
    arc_result* r = arc_transcribe(e, pcm, n, &o);
    printf("decode %.1f ms\n", arc_time_ms() - t0);
    for (int u = 0; r && u < r->n_utts; ++u)
      for (int a = 0; a < r->utts[u].n_alts; ++a) printf("alt\t%.4f\t%s\n", r->utts[u].alt_scores[a], r->utts[u].alts[a]);
    arc_result_free(r);
  }
  float* lp = malloc(sizeof(float) * (size_t)(nt + 1));
  int* nk = malloc(sizeof(int) * (size_t)(nt + 1));
  double t0 = arc_time_ms();
  if (arc_score(e, pcm, n, texts, nt, flags, lp, nk)) { fprintf(stderr, "arc_score: %s\n", arc_last_error()); return 1; }
  double ms = arc_time_ms() - t0;
  for (int i = 0; i < nt; ++i) printf("%.4f\t%d\t%s\n", lp[i], nk[i], texts[i]);
  printf("score %.1f ms for %d texts\n", ms, nt);
  free(lp);
  free(nk);
  free(texts);
  free(pcm);
  arc_destroy(e);
  return 0;
}
