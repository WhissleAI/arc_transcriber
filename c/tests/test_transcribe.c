/* arc_transcribe against Arc.transcribe (tools/dump_transcribe.py): utterance times, text and tag spans.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arc.h"
#include "arc_base.h"
#include "wav.h"

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: test_transcribe EDGE_DIR WAV_DIR REF.jsonl\n"); return 2; }
  char err[512];
  arc_engine* e = arc_create(argv[1], 4, err, sizeof err);
  if (!e) { fprintf(stderr, "%s\n", err); return 1; }
  static const char* names[] = {"Priya Raman", "Doctor Okafor", "Whissle", "Elizabeth Bennet", "Mr. Darcy"};
  static const char* kinds[] = {"person", "person", "organization", "person", "person"};
  static const char* terms[] = {"Arc", "Pride and Prejudice"};
  /* ctx 2 (dump_bias.py): app and person names as context and bias */
  static const char* bnames[] = {"WhatsApp", "Slack", "Messages", "Telegram", "Mail", "Safari", "Priya Raman", "Doctor Okafor", "Mr. Darcy"};
  static const char* bkinds[] = {"product", "product", "product", "product", "product", "product", "person", "person", "person"};
  static const char* bterms[] = {"Arc"};
  FILE* f = fopen(argv[3], "r");
  static char line[1 << 16];
  int n = 0, ok = 0;
  double wall = 0, audio = 0;
  while (fgets(line, sizeof line, f)) {
    arc_json* j = arc_json_parse(line, strlen(line));
    const char* clip = arc_json_str(arc_json_get(j, "clip"), "");
    int ctx = (int)arc_json_num(arc_json_get(j, "ctx"), 0);
    const arc_json* U = arc_json_get(j, "utts");
    char path[2048];
    snprintf(path, sizeof path, "%s/%s.wav", argv[2], clip);
    size_t nw; int rate;
    float* pcm = wav_read(path, &nw, &rate);
    arc_options o = {0};
    if (ctx == 2) {
      arc_set_context(e, bnames, bkinds, 9, bterms, 1, "assistant", NULL, 0);
      o.bias_names = 1;
    } else if (ctx) arc_set_context(e, names, kinds, 5, terms, 2, "assistant", NULL, 0);
    else arc_set_context(e, NULL, NULL, 0, NULL, 0, NULL, NULL, 0);
    arc_result* r = arc_transcribe(e, pcm, nw, &o);
    wall += r->wall_ms; audio += r->audio_s;
    int same = r->n_utts == U->n;
    for (int i = 0; same && i < U->n; ++i) {
      const arc_json* u = arc_json_at(U, i);
      const arc_utt* c = &r->utts[i];
      if (fabs(c->start - arc_json_num(arc_json_at(u, 0), 0)) > 1e-3 || fabs(c->end - arc_json_num(arc_json_at(u, 1), 0)) > 1e-3 ||
          strcmp(c->text, arc_json_str(arc_json_at(u, 2), ""))) same = 0;
      const arc_json* sp = arc_json_at(u, 3);
      if (same && c->n_tags != sp->n) same = 0;
      for (int s = 0; same && s < sp->n; ++s) {
        const arc_json* t = arc_json_at(sp, s);
        if (strcmp(c->tags[s].tag, arc_json_str(arc_json_at(t, 0), "")) || c->tags[s].word_begin != (int)arc_json_num(arc_json_at(t, 1), -1) ||
            c->tags[s].word_end != (int)arc_json_num(arc_json_at(t, 2), -1)) same = 0;
      }
    }
    n++; ok += same;
    if (!same || argc > 4) {
      printf("%s ctx=%d %s\n", clip, ctx, same ? "same" : "DIFF");
      for (int i = 0; i < U->n; ++i) { const arc_json* u = arc_json_at(U, i); printf("  ref %.1f-%.1f %s\n", arc_json_num(arc_json_at(u,0),0), arc_json_num(arc_json_at(u,1),0), arc_json_str(arc_json_at(u, 2), "")); }
      for (int i = 0; i < r->n_utts; ++i) {
        printf("  c   %.1f-%.1f %s", r->utts[i].start, r->utts[i].end, r->utts[i].text);
        for (int s = 0; s < r->utts[i].n_tags; ++s) printf(" [%s:%s]", r->utts[i].tags[s].tag, r->utts[i].tags[s].text);
        printf("\n");
      }
    }
    arc_result_free(r); free(pcm); arc_json_free(j);
  }
  printf("%d/%d transcripts identical (times, text, tags); %.0f ms for %.1f s of audio (RTF %.3f)\n", ok, n, wall, audio, wall / 1000 / audio);
  arc_destroy(e);
  return ok == n ? 0 : 1;
}
