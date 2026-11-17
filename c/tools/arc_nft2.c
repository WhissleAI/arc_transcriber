/* arc-nft2: WAV files (16 kHz mono) through Arc Transcriber Space (NFT2), printed as text or with --json one line per
 * file: the transcript (RNNT words with the decoder's tags and casing), the decoder's own text, names linked to the
 * list, numbers, sound events, the device-directed head, timing and peak memory.
 *   arc-nft2 NFT2_DIR file.wav [file.wav ...] [--names list.json] [--dom D] [--prev TEXT] [--domain D] [--no-bias]
 *            [--no-events] [--stream] [--json] [--threads N]
 *   arc-nft2 NFT2_DIR --manifest list.json [--dump DIR] [--json] [--threads N]
 * --names: a list as examples/example_names.json (strings or {"name", "kind", "aliases"}). --dom: the context's domain
 * (default: the model's, "space"; "" for none). --stream: through arc_nft2_stream_* in 100 ms pieces.
 * --manifest: [{"id", "wav" (relative to the manifest), "domain", "names": [...], "context": {"dom", "prev"}}].
 * --dump DIR: each stage's arrays as DIR/<id>.<stage>.<idx>.{f32,i32} (raw little-endian, shape in DIR/<id>.shapes)
 * for tools/compare_nft2.py. Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arc.h"
#include "arc_base.h"
#include "arc_nft2.h"
#include "json_out.h"
#include "../tests/wav.h"

typedef struct {
  const char* dir;
  const char* id;
  FILE* shapes;
} dumper;

static void dump_cb(void* user, const char* name, int idx, const float* f, const int* ints, int rows, int cols) {
  dumper* d = user;
  if (!d->dir) return;
  char p[2048];
  snprintf(p, sizeof p, "%s/%s.%s.%d.%s", d->dir, d->id, name, idx, f ? "f32" : "i32");
  FILE* o = fopen(p, "wb");
  if (!o) return;
  if (rows > 0) fwrite(f ? (const void*)f : (const void*)ints, 4, (size_t)rows * (size_t)cols, o);
  fclose(o);
  if (d->shapes) fprintf(d->shapes, "%s %d %s %d %d\n", name, idx, f ? "f32" : "i32", rows, cols);
}

/* names from a JSON array (strings or objects); the strings stay in j */
static arc_name_entry* names_from(const arc_json* j, int* n) {
  *n = 0;
  if (!j || j->type != ARC_JARR) return NULL;
  arc_name_entry* e = calloc((size_t)j->n + 1, sizeof *e);
  for (int i = 0; i < j->n; ++i) {
    const arc_json* c = &j->items[i];
    if (c->type == ARC_JSTR) {
      e[i].name = c->str;
      continue;
    }
    e[i].name = arc_json_str(arc_json_get(c, "name"), "");
    e[i].kind = arc_json_str(arc_json_get(c, "kind"), NULL);
    e[i].group = arc_json_str(arc_json_get(c, "group"), NULL);
    const arc_json* al = arc_json_get(c, "aliases");
    if (al && al->type == ARC_JARR && al->n) {
      const char** a = calloc((size_t)al->n, sizeof(char*));
      for (int k = 0; k < al->n; ++k) a[k] = al->items[k].str ? al->items[k].str : "";
      e[i].aliases = a;
      e[i].n_aliases = al->n;
    }
  }
  *n = j->n;
  return e;
}
static void names_free(arc_name_entry* e, int n) {
  for (int i = 0; i < n; ++i) free((void*)e[i].aliases);
  free(e);
}

static void print_json(const char* id, const arc_result* r, double load_ms) {
  printf("{\"id\": ");
  jstr(id);
  printf(", \"audio_s\": %.3f, \"wall_ms\": %.1f, \"transcript\": [", r->audio_s, r->wall_ms);
  for (int i = 0; i < r->n_utts; ++i) {
    const arc_utt* u = &r->utts[i];
    printf("%s{\"start\": %.1f, \"end\": %.1f, \"text\": ", i ? ", " : "", u->start, u->end);
    jstr(u->text);
    printf(", \"decoder_text\": ");
    jstr_or_null(u->decoder_text);
    printf(", \"words\": [");
    for (int k = 0; k < u->n_words; ++k) {
      const arc_word* w = &u->words[k];
      printf("%s{\"text\": ", k ? ", " : "");
      jstr(w->text);
      printf(", \"start\": %.3f, \"end\": %.3f", w->start, w->end);
      if (w->tag) printf(", \"tag\": \"%s\", \"tag_begin\": %s", w->tag, w->tag_begin ? "true" : "false");
      if (w->conf >= 0) printf(", \"conf\": %.4f", w->conf);
      printf(", \"disfl\": ");
      jstr_or_null(w->disfl);
      printf(", \"alts\": [");
      for (int q = 0; q < w->n_alts; ++q) {
        printf("%s{\"text\": ", q ? ", " : "");
        jstr(w->alts[q]);
        printf(", \"p\": %.4f}", w->alt_p[q]);
      }
      printf("]}");
    }
    printf("], \"tags\": [");
    for (int k = 0; k < u->n_tags; ++k) {
      printf("%s{\"tag\": \"%s\", \"text\": ", k ? ", " : "", u->tags[k].tag);
      jstr(u->tags[k].text);
      printf("}");
    }
    printf("]");
    if (u->directed >= 0)
      printf(", \"device_directed\": {\"p\": %.4f, \"directed\": %s}", u->directed_head, u->directed ? "true" : "false");
    if (u->conf >= 0) printf(", \"conf\": %.4f", u->conf);
    printf(", \"clean_text\": ");
    jstr_or_null(u->clean_text);
    printf(", \"n_dropped\": %d, \"quantities\": [", u->n_dropped);
    for (int k = 0; k < u->n_quantities; ++k) {
      const arc_quantity* q = &u->quantities[k];
      printf("%s{\"type\": \"%s\", \"value\": ", k ? ", " : "", q->type);
      jnum(q->value, q->is_int);
      printf(", \"unit\": ");
      jstr_or_null(q->unit);
      printf(", \"relative\": %d, \"word_begin\": %d, \"word_end\": %d, \"start\": %.3f, \"end\": %.3f, \"text\": ",
             q->relative, q->word_begin, q->word_end, q->start, q->end);
      jstr(q->text);
      printf(", \"written\": ");
      jstr(q->written);
      printf("}");
    }
    printf("]");
    if (u->n_alts > 0) {
      printf(", \"alts\": [");
      for (int k = 0; k < u->n_alts; ++k) {
        printf("%s{\"text\": ", k ? ", " : "");
        jstr(u->alts[k]);
        printf(", \"p\": %.4f}", u->alt_scores[k]);
      }
      printf("]");
    }
    if (u->playback_leak >= 0) printf(", \"playback_leak\": %.4f, \"playback\": %s", u->playback_leak, u->playback ? "true" : "false");
    if (u->tagger_json) printf(", \"tagger\": %s", u->tagger_json);
    printf("}");
  }
  printf("], ");
  print_names_numbers(r);
  printf(", \"events\": [");
  for (int i = 0; i < r->n_events; ++i)
    printf("%s{\"label\": \"%s\", \"start\": %.1f, \"end\": %.1f, \"p\": %.4f}", i ? ", " : "", r->events[i].label,
           r->events[i].start, r->events[i].end, r->events[i].p);
  printf("]");
  if (load_ms >= 0) printf(", \"load_ms\": %.1f", load_ms);
  printf(", \"peak_rss_mb\": %.1f}\n", (double)arc_peak_rss_kb() / 1024.0);
  fflush(stdout);
}

static void print_text(const char* id, const arc_result* r) {
  printf("== %s (%.2f s audio, %.0f ms)\n", id, r->audio_s, r->wall_ms);
  for (int i = 0; i < r->n_utts; ++i) {
    const arc_utt* u = &r->utts[i];
    printf("[%5.1f-%5.1f] %s\n", u->start, u->end, u->text);
    if (u->decoder_text && strcmp(u->decoder_text, u->text)) printf("              decoder: %s\n", u->decoder_text);
    for (int k = 0; k < u->n_quantities; ++k)
      printf("              quantity %s %s (%s)\n", u->quantities[k].type, u->quantities[k].written, u->quantities[k].text);
    if (u->tagger_json) printf("              tagger: %.300s\n", u->tagger_json);
    for (int k = 0; k < u->n_tags; ++k) printf("              %s: %s\n", u->tags[k].tag, u->tags[k].text);
    if (u->directed >= 0) printf("              device_directed %.3f\n", u->directed_head);
  }
  for (int i = 0; i < r->n_names; ++i)
    printf("  name  %-28s %-12s -> %s (%s)\n", r->names[i].heard, r->names[i].state,
           r->names[i].entity ? r->names[i].entity : "-", r->names[i].tag);
  for (int i = 0; i < r->n_numbers; ++i)
    printf("  number %s (%s)\n", r->numbers[i].heard, r->numbers[i].kind ? r->numbers[i].kind : "-");
  for (int i = 0; i < r->n_events; ++i)
    printf("  event %s %.1f-%.1f p %.2f\n", r->events[i].label, r->events[i].start, r->events[i].end, r->events[i].p);
}

static const float* g_ref;   /* --ref: the playback, fed as reference at --ref-at */
static size_t g_nref;
static double g_ref_at;
static int g_events;

static void print_events(arc_nft2_stream* s) {
  static const char* T[] = {"?", "PARTIAL", "ENTITY", "EOT", "BARGE_IN", "FINAL"};
  int n;
  const arc_nft2_event* e = arc_nft2_stream_events(s, &n);
  for (int i = 0; g_events && i < n; ++i) {
    fprintf(stderr, "  %6.2f %-8s utt %d", e[i].t, T[e[i].type < 6 ? e[i].type : 0], e[i].utt);
    if (e[i].text) fprintf(stderr, " \"%s\"", e[i].text);
    if (e[i].type == ARC_EV_PARTIAL) fprintf(stderr, " stable %d", e[i].n_stable);
    if (e[i].kind) fprintf(stderr, " %s", e[i].kind);
    if (e[i].entity) fprintf(stderr, " -> %s", e[i].entity);
    if (e[i].p >= 0) fprintf(stderr, " p %.3f", e[i].p);
    fprintf(stderr, "\n");
  }
}

static arc_result* run_stream(arc_nft2* m, const float* pcm, size_t n, const arc_nft2_options* o) {
  arc_nft2_stream* s = arc_nft2_stream_open(m, o);
  if (g_ref && arc_nft2_stream_feed_ref(s, g_ref, g_nref, g_ref_at)) {
    arc_nft2_stream_close(s);
    return NULL;
  }
  for (size_t a = 0; a < n; a += 1600) {
    arc_result* r = arc_nft2_stream_feed(s, pcm + a, n - a < 1600 ? n - a : 1600);
    if (!r) {
      arc_nft2_stream_close(s);
      return NULL;
    }
    print_events(s);
    for (int i = 0; i < r->n_utts; ++i)
      fprintf(stderr, "  [stream %.1f s] %s\n", (double)(a + 1600) / 16000, r->utts[i].text);
    arc_result_free(r);
  }
  arc_result* r = arc_nft2_stream_flush(s);
  print_events(s);
  arc_nft2_stream_close(s);
  return r;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: arc-nft2 NFT2_DIR file.wav [...] [--names list.json] [--dom D] [--prev TEXT] [--domain D] "
                    "[--no-bias] [--no-events] [--no-guard] [--almost] [--nbest N] [--stream] [--live] [--events] [--ref playback.wav] [--ref-at S] "
                    "[--echo-decode] [--playback-drop] [--tagger MODEL_DIR] [--tagger-domain D] [--json] [--threads N]\n"
                    "       arc-nft2 NFT2_DIR --manifest list.json [--dump DIR] [--json] [--threads N]\n");
    return 2;
  }
  const char *names_path = NULL, *manifest = NULL, *dump_dir = NULL, *dom = NULL, *prev = NULL, *ref_path = NULL;
  const char *tagger = NULL, *tagger_dom = NULL;
  arc_nft2_options opt = {0};
  int threads = 4, json = 0, stream = 0;
  const char* files[4096];
  int nf = 0;
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--names") && i + 1 < argc) names_path = argv[++i];
    else if (!strcmp(argv[i], "--manifest") && i + 1 < argc) manifest = argv[++i];
    else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump_dir = argv[++i];
    else if (!strcmp(argv[i], "--dom") && i + 1 < argc) dom = argv[++i];
    else if (!strcmp(argv[i], "--prev") && i + 1 < argc) prev = argv[++i];
    else if (!strcmp(argv[i], "--domain") && i + 1 < argc) opt.domain = argv[++i];
    else if (!strcmp(argv[i], "--no-bias")) opt.no_bias = 1;
    else if (!strcmp(argv[i], "--no-events")) opt.no_events = 1;
    else if (!strcmp(argv[i], "--no-guard")) opt.no_guard = 1;
    else if (!strcmp(argv[i], "--almost")) opt.almost = 1;
    else if (!strcmp(argv[i], "--live")) opt.live = 1, stream = 1;
    else if (!strcmp(argv[i], "--events")) g_events = 1;
    else if (!strcmp(argv[i], "--ref") && i + 1 < argc) ref_path = argv[++i], stream = 1;
    else if (!strcmp(argv[i], "--ref-at") && i + 1 < argc) g_ref_at = atof(argv[++i]);
    else if (!strcmp(argv[i], "--echo-decode")) opt.echo_decode = 1;
    else if (!strcmp(argv[i], "--tagger") && i + 1 < argc) tagger = argv[++i];
    else if (!strcmp(argv[i], "--tagger-domain") && i + 1 < argc) tagger_dom = argv[++i];
    else if (!strcmp(argv[i], "--playback-drop")) opt.playback_drop = 1;
    else if (!strcmp(argv[i], "--nbest") && i + 1 < argc) opt.nbest = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--stream")) stream = 1;
    else if (!strcmp(argv[i], "--json")) json = 1;
    else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    else if (nf < 4096) files[nf++] = argv[i];
  }
  double t0 = arc_time_ms();
  char err[512];
  float* refpcm = NULL;
  if (ref_path) {
    int rr;
    refpcm = wav_read(ref_path, &g_nref, &rr);
    if (!refpcm || rr != 16000) {
      fprintf(stderr, "arc-nft2: %s: not a 16 kHz WAV\n", ref_path);
      return 1;
    }
    g_ref = refpcm;
  }
  arc_nft2* m = arc_nft2_create(argv[1], threads, err, sizeof err);
  if (!m) {
    fprintf(stderr, "arc-nft2: %s\n", err);
    return 1;
  }
  double load_ms = arc_time_ms() - t0;
  if (!json) fprintf(stderr, "loaded %s in %.0f ms (backend %s, %d threads)\n", argv[1], load_ms, arc_nft2_backend(), threads);
  if (tagger && arc_nft2_set_tagger(m, tagger, tagger_dom, 1, err, sizeof err)) {
    fprintf(stderr, "arc-nft2: %s\n", err);
    arc_nft2_destroy(m);
    return 1;
  }
  dumper dd = {dump_dir, "", NULL};
  if (dump_dir) arc_nft2_set_dump(m, dump_cb, &dd);
  int rc = 0;
  if (manifest) {
    char* text = NULL;
    size_t len = 0;
    if (arc_read_file(manifest, &text, &len)) {
      fprintf(stderr, "arc-nft2: cannot read %s\n", manifest);
      return 1;
    }
    arc_json* j = arc_json_parse(text, len);
    free(text);
    if (!j || j->type != ARC_JARR) {
      fprintf(stderr, "arc-nft2: %s is not a JSON list\n", manifest);
      return 1;
    }
    char base[2048];
    snprintf(base, sizeof base, "%s", manifest);
    char* sl = strrchr(base, '/');
    if (sl) sl[1] = 0;
    else base[0] = 0;
    for (int i = 0; i < j->n; ++i) {
      const arc_json* c = &j->items[i];
      const char* id = arc_json_str(arc_json_get(c, "id"), "clip");
      const char* wav = arc_json_str(arc_json_get(c, "wav"), "");
      char path[4096];
      snprintf(path, sizeof path, "%s%s", wav[0] == '/' ? "" : base, wav);
      int nk;
      arc_name_entry* kb = names_from(arc_json_get(c, "names"), &nk);
      arc_nft2_set_names(m, kb, nk);
      const arc_json* cx = arc_json_get(c, "context");
      arc_nft2_context ctx = {0};
      ctx.dom = arc_json_str(arc_json_get(cx, "dom"), NULL);
      ctx.prev = arc_json_str(arc_json_get(cx, "prev"), NULL);
      arc_nft2_set_context(m, cx ? &ctx : NULL);
      arc_nft2_options o = opt;
      o.domain = arc_json_str(arc_json_get(c, "domain"), opt.domain);
      size_t n;
      int rate;
      float* pcm = wav_read(path, &n, &rate);
      if (!pcm || rate != 16000) {
        fprintf(stderr, "arc-nft2: %s: not a 16 kHz WAV\n", path);
        rc = 1;
        free(pcm);
        names_free(kb, nk);
        continue;
      }
      char sp[2300];
      if (dump_dir) {
        snprintf(sp, sizeof sp, "%s/%s.shapes", dump_dir, id);
        dd.shapes = fopen(sp, "w");
      }
      dd.id = id;
      arc_result* r = stream ? run_stream(m, pcm, n, &o) : arc_nft2_transcribe(m, pcm, n, &o);
      if (dd.shapes) fclose(dd.shapes);
      dd.shapes = NULL;
      if (!r) {
        fprintf(stderr, "arc-nft2: %s: %s\n", id, arc_last_error());
        rc = 1;
      } else {
        if (json) print_json(id, r, i == 0 ? load_ms : -1);
        else print_text(id, r);
        arc_result_free(r);
      }
      free(pcm);
      names_free(kb, nk);
    }
    arc_json_free(j);
  } else {
    arc_name_entry* kb = NULL;
    int nk = 0;
    arc_json* kj = NULL;
    if (names_path) {
      char* text = NULL;
      size_t len = 0;
      if (arc_read_file(names_path, &text, &len)) {
        fprintf(stderr, "arc-nft2: cannot read %s\n", names_path);
        return 1;
      }
      kj = arc_json_parse(text, len);
      free(text);
      kb = names_from(kj, &nk);
      arc_nft2_set_names(m, kb, nk);
    }
    if (dom || prev) {
      arc_nft2_context ctx = {0};
      ctx.dom = dom;
      ctx.prev = prev;
      arc_nft2_set_context(m, &ctx);
    }
    for (int i = 0; i < nf; ++i) {
      size_t n;
      int rate;
      float* pcm = wav_read(files[i], &n, &rate);
      if (!pcm || rate != 16000) {
        fprintf(stderr, "arc-nft2: %s: not a 16 kHz WAV\n", files[i]);
        rc = 1;
        free(pcm);
        continue;
      }
      arc_result* r = stream ? run_stream(m, pcm, n, &opt) : arc_nft2_transcribe(m, pcm, n, &opt);
      const char* id = strrchr(files[i], '/') ? strrchr(files[i], '/') + 1 : files[i];
      if (!r) {
        fprintf(stderr, "arc-nft2: %s: %s\n", id, arc_last_error());
        rc = 1;
      } else {
        if (json) print_json(id, r, i == 0 ? load_ms : -1);
        else print_text(id, r);
        arc_result_free(r);
      }
      free(pcm);
    }
    names_free(kb, nk);
    arc_json_free(kj);
  }
  if (!json) fprintf(stderr, "peak RSS %.0f MB\n", (double)arc_peak_rss_kb() / 1024.0);
  arc_nft2_destroy(m);
  free(refpcm);
  return rc;
}
