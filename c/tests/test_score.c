/* arc_score and the n-best against Arc Mini in Python (tools/dump_score_ref.py): teacher-forced log P(text | audio)
 * for each text and tag variant within 2e-2 nats (float32 vs float32, different summation order), and the final
 * decode's distinct beams (texts identical, scores within 2e-3).
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arc.h"
#include "arc_base.h"
#include "wav.h"

static int close_to(double a, double b, double tol) { return fabs(a - b) <= tol; }

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: test_score MINI_DIR WAV_DIR REF.jsonl\n"); return 2; }
  char err[512];
  arc_engine* e = arc_create(argv[1], 4, err, sizeof err);
  if (!e) { fprintf(stderr, "%s\n", err); return 1; }
  arc_set_context(e, NULL, NULL, 0, NULL, 0, "assistant", NULL, 0);
  FILE* f = fopen(argv[3], "r");
  if (!f) { fprintf(stderr, "cannot open %s\n", argv[3]); return 1; }
  static char line[1 << 18];
  int n_sc = 0, ok_sc = 0, n_nb = 0, ok_nb = 0;
  double worst = 0;
  while (fgets(line, sizeof line, f)) {
    arc_json* j = arc_json_parse(line, strlen(line));
    const char* clip = arc_json_str(arc_json_get(j, "clip"), "");
    /* "name" or "name@start-end" (seconds): a stretch of a file */
    char name[1024], path[2048];
    double a = -1, b = -1;
    snprintf(name, sizeof name, "%s", clip);
    char* at = strchr(name, '@');
    if (at) {
      *at = 0;
      sscanf(at + 1, "%lf-%lf", &a, &b);
    }
    snprintf(path, sizeof path, "%s/%s.wav", argv[2], name);
    size_t nw; int rate;
    float* pcm = wav_read(path, &nw, &rate);
    if (!pcm) { fprintf(stderr, "no %s\n", path); return 1; }
    if (a >= 0) {
      size_t ia = (size_t)(a * 16000 + 0.5), ib = (size_t)(b * 16000 + 0.5);
      if (ib > nw) ib = nw;
      if (ia > ib) ia = ib;
      memmove(pcm, pcm + ia, sizeof(float) * (ib - ia));
      nw = ib - ia;
    }
    const arc_json* T = arc_json_get(j, "texts");
    int nt = T->n;
    /* each variant on its own: plain (flags 0), person only (flags 2); work = all3 best when it is the best */
    const char** texts = malloc(sizeof(char*) * (size_t)nt);
    for (int i = 0; i < nt; ++i) texts[i] = arc_json_str(arc_json_at(T, i), "");
    float* lp = malloc(sizeof(float) * (size_t)nt);
    int* nk = malloc(sizeof(int) * (size_t)nt);
    const char* keys[2] = {"plain", "person"};
    for (int v = 0; v < 2; ++v) {
      if (arc_score(e, pcm, nw, texts, nt, v == 0 ? 0 : ARC_SCORE_PERSON, lp, nk)) { fprintf(stderr, "%s\n", arc_last_error()); return 1; }
      const arc_json* R = arc_json_get(j, keys[v]);
      for (int i = 0; i < nt; ++i) {
        const arc_json* r = arc_json_at(R, i);
        if (!r || r->type == ARC_JNULL) continue;
        double d = fabs(lp[i] - r->num);
        if (d > worst) worst = d;
        n_sc++;
        if (close_to(lp[i], r->num, 2e-2)) ok_sc++;
        else printf("%s %s [%s]: C %.4f, Python %.4f\n", clip, keys[v], texts[i], lp[i], r->num);
        if (v == 0 && nk[i] != (int)arc_json_num(arc_json_at(arc_json_get(j, "ntok"), i), -1)) printf("%s ntok differs: %s\n", clip, texts[i]);
      }
    }
    /* all three: the best of plain / person / work */
    if (arc_score(e, pcm, nw, texts, nt, ARC_SCORE_TAGS, lp, nk)) return 1;
    for (int i = 0; i < nt; ++i) {
      double best = -INFINITY;
      const char* k3[3] = {"plain", "person", "work"};
      for (int v = 0; v < 3; ++v) {
        const arc_json* r = arc_json_at(arc_json_get(j, k3[v]), i);
        if (r && r->type != ARC_JNULL && r->num > best) best = r->num;
      }
      n_sc++;
      if (close_to(lp[i], best, 2e-2)) ok_sc++;
      else printf("%s all3 [%s]: C %.4f, Python %.4f\n", clip, texts[i], lp[i], best);
    }
    /* n-best: the whole clip as one utterance */
    arc_options o = {0};
    o.one_utterance = 1;
    arc_result* res = arc_transcribe(e, pcm, nw, &o);
    const arc_json* A = arc_json_get(j, "alts");
    const arc_json* AS = arc_json_get(j, "alt_scores");
    n_nb++;
    int good = res && res->n_utts == 1 && res->utts[0].n_alts == A->n;
    for (int a = 0; good && a < A->n; ++a) {
      good = !strcmp(res->utts[0].alts[a], arc_json_str(arc_json_at(A, a), "")) &&
             close_to(res->utts[0].alt_scores[a], arc_json_num(arc_json_at(AS, a), 0), 2e-3);
    }
    if (good) ok_nb++;
    else {
      printf("%s n-best differs:", clip);
      for (int a = 0; res && res->n_utts && a < res->utts[0].n_alts; ++a) printf(" C[%s %.4f]", res->utts[0].alts[a], res->utts[0].alt_scores[a]);
      for (int a = 0; a < A->n; ++a) printf(" Py[%s %.4f]", arc_json_str(arc_json_at(A, a), ""), arc_json_num(arc_json_at(AS, a), 0));
      printf("\n");
    }
    arc_result_free(res);
    free(texts);
    free(lp);
    free(nk);
    free(pcm);
    arc_json_free(j);
  }
  fclose(f);
  printf("scores %d/%d within 2e-2 (worst %.2e); n-best %d/%d identical\n", ok_sc, n_sc, worst, ok_nb, n_nb);
  arc_destroy(e);
  return ok_sc == n_sc && ok_nb == n_nb ? 0 : 1;
}
