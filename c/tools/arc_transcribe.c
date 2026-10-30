/* arc-transcribe: WAV files (16 kHz mono; other rates are not resampled here) through libarc, printed as text: each
 * utterance with its tags and readout, the linked names and numbers, the sound events, timing and peak memory; or
 * with --json one JSON line per file (transcript with word times, names, numbers, events).
 *   arc-transcribe MINI_DIR file.wav [file.wav ...] [--domain D] [--names list.json] [--context "A,B"] [--whole]
 *                  [--no-trunk] [--words] [--json] [--threads N]
 * --names: a list as examples/example_names.json (strings or {"name", "kind", "group", "aliases"}); names are linked
 * to it and the decoder is biased towards it, as Arc.predict(names=...). --context: names for the decoder's context
 * prompt (and bias) only. Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arc.h"
#include "arc_base.h"
#include "json_out.h"
#include "../tests/wav.h"

static const char* base(const char* p) {
  const char* s = strrchr(p, '/');
  return s ? s + 1 : p;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: arc-transcribe MINI_DIR file.wav [file.wav ...] [--domain D] [--names list.json] "
                    "[--context A,B] [--whole] [--no-trunk] [--words] [--json] [--threads N]\n");
    return 2;
  }
  const char* domain = "unknown";
  const char* names_path = NULL;
  char* ctx = NULL;
  int whole = 0, trunk = 1, threads = 4, json = 0, words = 0;
  const char* files[4096];
  int nf = 0;
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--domain") && i + 1 < argc) domain = argv[++i];
    else if (!strcmp(argv[i], "--names") && i + 1 < argc) names_path = argv[++i];
    else if (!strcmp(argv[i], "--context") && i + 1 < argc) ctx = argv[++i];
    else if (!strcmp(argv[i], "--whole")) whole = 1;
    else if (!strcmp(argv[i], "--no-trunk")) trunk = 0;
    else if (!strcmp(argv[i], "--words")) words = 1;
    else if (!strcmp(argv[i], "--json")) json = 1;
    else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    else if (nf < 4096) files[nf++] = argv[i];
  }
  double t0 = arc_time_ms();
  char err[512];
  arc_engine* e = arc_create(argv[1], threads, err, sizeof err);
  if (!e) { fprintf(stderr, "arc: %s\n", err); return 1; }
  double t_load = arc_time_ms() - t0;
  const char* nm[256];
  int nn = 0;
  if (ctx) {
    for (char* p = strtok(ctx, ","); p && nn < 256; p = strtok(NULL, ",")) nm[nn++] = p;
    arc_set_context(e, nm, NULL, nn, NULL, 0, NULL, NULL, 0);
  }
  arc_name_entry* kb = NULL;
  int nkb = 0;
  arc_json* kbj = NULL;
  if (names_path) {
    kbj = load_names(names_path, &kb, &nkb);
    if (!kbj) { fprintf(stderr, "arc: %s: %s\n", names_path, arc_last_error()); return 1; }
    if (arc_set_names(e, kb, nkb)) { fprintf(stderr, "arc: %s\n", arc_last_error()); return 1; }
  }
  arc_options o = {.domain = domain, .one_utterance = whole, .trunk = trunk, .events = trunk, .bias_names = nn > 0 || nkb > 0,
                   .words = words || json, .names = trunk, .numbers = trunk};
  for (int fi = 0; fi < nf; ++fi) {
    size_t n; int rate;
    float* pcm = wav_read(files[fi], &n, &rate);
    if (!pcm) { fprintf(stderr, "cannot read %s\n", files[fi]); return 1; }
    if (rate != ARC_SAMPLE_RATE) fprintf(stderr, "warning: %s: %d Hz (Arc expects 16000)\n", files[fi], rate);
    arc_result* r = arc_transcribe(e, pcm, n, &o);
    if (!r) { fprintf(stderr, "arc: %s\n", arc_last_error()); return 1; }
    if (json) {
      printf("{\"clip\": ");
      jstr(base(files[fi]));
      printf(", \"domain\": ");
      jstr(domain);
      printf(", \"transcript\": [");
      for (int i = 0; i < r->n_utts; ++i) {
        const arc_utt* u = &r->utts[i];
        printf("%s{\"start\": %.1f, \"end\": %.1f, \"text\": ", i ? ", " : "", u->start, u->end);
        jstr(u->text);
        printf(", \"words\": [");
        for (int k = 0; k < u->n_words; ++k) {
          printf("%s{\"text\": ", k ? ", " : "");
          jstr(u->words[k].text);
          printf(", \"start\": %.3f, \"end\": %.3f", u->words[k].start, u->words[k].end);
          if (u->words[k].tag) printf(", \"tag\": \"%s\", \"tag_begin\": %s", u->words[k].tag, u->words[k].tag_begin ? "true" : "false");
          printf("}");
        }
        printf("], \"tags\": [");
        for (int k = 0; k < u->n_tags; ++k) {
          printf("%s[\"%s\", ", k ? ", " : "", u->tags[k].tag);
          jstr(u->tags[k].text);
          printf("]");
        }
        printf("]");
        if (u->intent.label) printf(", \"intent\": [\"%s\", %.4f]", u->intent.label, u->intent.p);
        if (u->act.label) printf(", \"act\": [\"%s\", %.4f]", u->act.label, u->act.p);
        if (u->device_directed >= 0) printf(", \"device_directed\": %.4f", u->device_directed);
        printf("}");
      }
      printf("], ");
      print_names_numbers(r);
      printf(", \"audio_events\": [");
      for (int i = 0; i < r->n_events; ++i)
        printf("%s{\"label\": \"%s\", \"start\": %.1f, \"end\": %.1f, \"p\": %.4f}", i ? ", " : "", r->events[i].label,
               r->events[i].start, r->events[i].end, r->events[i].p);
      printf("], \"wall_ms\": %.0f}\n", r->wall_ms);
      fflush(stdout);
    } else {
      if (nf > 1) printf("== %s\n", files[fi]);
      for (int i = 0; i < r->n_utts; ++i) {
        const arc_utt* u = &r->utts[i];
        printf("[%6.2f-%6.2f] %s\n", u->start, u->end, u->text);
        for (int k = 0; k < u->n_tags; ++k) printf("    %s: %s\n", u->tags[k].tag, u->tags[k].text);
        int any = 0;
        if (u->intent.label) any += printf("    intent %s %.3f", u->intent.label + 17, u->intent.p);
        if (u->act.label) any += printf("%sact %s %.3f", any ? "  " : "    ", u->act.label + 14, u->act.p);
        if (u->device_directed >= 0) any += printf("%sdirected %.3f", any ? "  " : "    ", u->device_directed);
        if (u->directed_head >= 0) any += printf("%sdirected head %.3f", any ? "  " : "    ", u->directed_head);
        if (any) printf("\n");
        if (words && u->n_words) {
          printf("   ");
          for (int k = 0; k < u->n_words; ++k) printf(" %s[%.2f-%.2f]", u->words[k].text, u->words[k].start, u->words[k].end);
          printf("\n");
        }
      }
      for (int i = 0; i < r->n_names; ++i) {
        const arc_name* m = &r->names[i];
        printf("name [%6.2f-%6.2f] %s -> %s (%s", m->start, m->end, m->heard, m->entity ? m->entity : "-", m->state);
        if (m->p >= 0) printf(", p=%.3f", m->p);
        if (m->via) printf(", via %s", m->via);
        printf(")\n");
      }
      for (int i = 0; i < r->n_numbers; ++i) {
        const arc_number* m = &r->numbers[i];
        printf("number [%6.2f-%6.2f] %s: %s", m->start, m->end, m->kind ? m->kind : "-", m->heard);
        if (m->has_value) printf(" = %.10g", m->value);
        if (m->has_value2) printf(", %.10g", m->value2);
        if (m->unit) printf(" (%s)", m->unit);
        printf("\n");
      }
      for (int i = 0; i < r->n_events; ++i)
        printf("event %s %.1f-%.1f p=%.3f\n", r->events[i].label, r->events[i].start, r->events[i].end, r->events[i].p);
      fprintf(stderr, "load %.0f ms; %.2f s of audio in %.0f ms (RTF %.3f); peak RSS %zu MB\n", t_load, r->audio_s,
              r->wall_ms, r->wall_ms / 1000 / (r->audio_s > 0 ? r->audio_s : 1), arc_peak_rss_kb() / 1024);
    }
    arc_result_free(r);
    free(pcm);
  }
  for (int i = 0; i < nkb; ++i) free((void*)kb[i].aliases);
  free(kb);
  arc_json_free(kbj);
  arc_destroy(e);
  return 0;
}
