/* arc-stream: WAV files through an arc_stream as if they were live audio, in chunks, with the records it sends and
 * how long each took to come (time to the first words, to the final text, to settled tags, per-frame read-out).
 *
 *   arc-stream MINI_DIR file.wav [more.wav ...] [--mode full|heads|external] [--threads N] [--chunk-ms 100]
 *              [--partial-ms 400] [--beams N] [--domain D] [--events] [--no-decoder] [--tail-s 1.5]
 *              [--words-dir DIR] [--delay-ms 300] [--records] [--realtime] [--quiet]
 *
 * --delay-ms: the read-out delay (0, 100, 200, 300 = default, 500, 800: the ones Arc was trained with).
 * Every file is its own stream, followed by --tail-s of silence (so utterances close on Arc's pause rule as they do
 * live) and arc_stream_finish. Records are printed as text, or with --records as JSON lines; a JSON summary line per
 * file and one for the run follow (RTF, CPU, peak RSS, latencies in ms).
 *
 * Latency: each record's wait in audio (audio pushed when it came, less the time it refers to) plus the wall time
 * from the start of the push that produced it: what a live caller pushing chunks as they are captured would see,
 * as long as each push takes less than a chunk (the summary's max_push_ms says). --realtime paces the pushes.
 * The reference times come from the file's own levels: speech start / end = first / last 100 ms frame within 35 dB
 * of the loudest frame (and above -55 dBFS).
 *
 * --mode external replays another recogniser's output: DIR/<file basename>.jsonl, lines
 *   {"at": 1.23, "final": 0, "words": [{"text": "hello", "start": 0.4, "end": 0.7}, ...], "upto": 0.9}  or
 *   {"at":..., "text": "..."}   ("upto": a final result's end, arc_stream_covered)
 * each pushed (arc_stream_words / arc_stream_text) once the audio reaches "at" seconds.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE 1
#else
#define _POSIX_C_SOURCE 200809L
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#include "../tests/wav.h"
#include "arc.h"
#include "arc_base.h"
#include "json_out.h"

typedef struct {
  double push_t0;              /* wall ms at the start of the current push */
  float speech_start, speech_end;
  int records, quiet;
  double first_partial, final_lat, tags_lat, max_push;
  double frame_sum;
  long frame_n;
  double prov_lat;             /* first provisional tags of the utterance holding the end of speech */
  /* per utterance id: the first provisional tags, to compare with the settled ones */
  struct { int have; float dd; char intent[96], act[96]; } prov[512];
  int agree, compared;
  int n_final;
  char text[8192];             /* the finals, joined */
  float last_end;
} run_st;

static double now_ms(void) { return arc_time_ms(); }

static void on_out(const arc_stream_out* o, void* user) {
  run_st* r = user;
  double proc = now_ms() - r->push_t0;
  double lat = -1;
  const arc_utt* u = o->utt;
  switch (o->kind) {
    case ARC_OUT_PARTIAL:
      lat = (o->audio_s - r->speech_start) * 1000 + proc;
      if (r->first_partial < 0 && u && u->text && u->text[0]) r->first_partial = lat;
      break;
    case ARC_OUT_FINAL:
    case ARC_OUT_TAGS:
      lat = (o->audio_s - r->speech_end) * 1000 + proc;
      if (o->kind == ARC_OUT_FINAL) {
        if (u->text && u->text[0]) {
          size_t k = strlen(r->text);
          snprintf(r->text + k, sizeof r->text - k, "%s%s", k ? " " : "", u->text);
        }
        r->n_final++;
        /* the utterance holding the end of speech decides the final latency (a later one is silence noise) */
        if (u->end >= r->speech_end - 0.3f || r->final_lat < 0) r->final_lat = lat;
      }
      if (o->kind == ARC_OUT_TAGS && !o->tags_final) {   /* provisional */
        if (u->end >= r->speech_end - 0.3f || r->prov_lat < 0) r->prov_lat = lat;
        if (o->id >= 0 && o->id < 512 && !r->prov[o->id].have) {
          r->prov[o->id].have = 1;
          r->prov[o->id].dd = u->device_directed;
          snprintf(r->prov[o->id].intent, 96, "%s", u->intent.label ? u->intent.label : "");
          snprintf(r->prov[o->id].act, 96, "%s", u->act.label ? u->act.label : "");
        }
      }
      if (o->tags_final) {
        if (u->end >= r->speech_end - 0.3f || r->tags_lat < 0) r->tags_lat = lat;
        if (o->id >= 0 && o->id < 512 && r->prov[o->id].have) {   /* did the provisional tags hold? */
          r->compared++;
          r->agree += fabsf(r->prov[o->id].dd - u->device_directed) < 0.05f &&
                      !strcmp(r->prov[o->id].intent, u->intent.label ? u->intent.label : "") &&
                      !strcmp(r->prov[o->id].act, u->act.label ? u->act.label : "");
        }
      }
      break;
    case ARC_OUT_FRAME:
      lat = (o->audio_s - (o->frame->t + 0.1)) * 1000 + proc;
      r->frame_sum += lat;
      r->frame_n++;
      break;
    default:
      break;
  }
  if (r->quiet) return;
  static const char* kinds[] = {"", "partial", "final", "tags", "event", "frame", "endpoint"};
  if (r->records) {
    if (o->kind == ARC_OUT_FRAME) return;
    printf("{\"kind\": \"%s\", \"id\": %d, \"audio_s\": %.3f, \"latency_ms\": %.1f", kinds[o->kind], o->id, o->audio_s,
           lat);
    if (u) {
      printf(", \"start\": %.2f, \"end\": %.2f, \"text\": ", u->start, u->end);
      jstr(u->text ? u->text : "");
      printf(", \"device_directed\": %.4f, \"directed_head\": %.4f, \"intent\": ", u->device_directed,
             u->directed_head);
      jstr_or_null(u->intent.label);
      printf(", \"act\": ");
      jstr_or_null(u->act.label);
      printf(", \"tags_final\": %d, \"tags\": [", o->tags_final);
      for (int i = 0; i < u->n_tags; ++i) {
        printf("%s{\"tag\": ", i ? ", " : "");
        jstr(u->tags[i].tag);
        printf(", \"text\": ");
        jstr(u->tags[i].text);
        printf("}");
      }
      printf("], \"words\": [");
      for (int i = 0; i < u->n_words; ++i) {
        printf("%s{\"text\": ", i ? ", " : "");
        jstr(u->words[i].text);
        printf(", \"start\": %.3f, \"end\": %.3f}", u->words[i].start, u->words[i].end);
      }
      printf("]");
    }
    if (o->event) {
      printf(", \"label\": ");
      jstr(o->event->label);
      printf(", \"start\": %.2f, \"end\": %.2f, \"p\": %.3f", o->event->start, o->event->end, o->event->p);
    }
    printf("}\n");
    return;
  }
  if (o->kind == ARC_OUT_FRAME) return;
  if (o->kind == ARC_OUT_ENDPOINT) {
    printf("  endpoint #%d %5.2f-%5.2f\n", o->id, u->start, u->end);
    return;
  }
  if (o->kind == ARC_OUT_EVENT) {
    printf("  event %s %.1f-%.1f p=%.2f\n", o->event->label, o->event->start, o->event->end, o->event->p);
    return;
  }
  printf("  %-7s #%d %5.2f-%5.2f %6.0f ms  %s", kinds[o->kind], o->id, u->start, u->end, lat,
         u->text && u->text[0] ? u->text : "(no text)");
  if (o->kind != ARC_OUT_PARTIAL)
    printf("  [dd %.2f, head %.2f, %s, %s%s]", u->device_directed, u->directed_head,
           u->intent.label ? u->intent.label : "-", u->act.label ? u->act.label : "-",
           o->kind == ARC_OUT_FINAL && !o->tags_final ? ", tags pending" : "");
  printf("\n");
}

static void speech_bounds(const float* pcm, size_t n, float* a, float* b) {
  int nf = (int)((n + 1599) / 1600);
  double mx = -200;
  double* db = malloc(sizeof(double) * (size_t)(nf + 1));
  for (int f = 0; f < nf; ++f) {
    size_t lo = (size_t)f * 1600, hi = lo + 1600 < n ? lo + 1600 : n;
    double s = 0;
    for (size_t i = lo; i < hi; ++i) s += (double)pcm[i] * pcm[i];
    db[f] = 20 * log10(sqrt(s / (double)(hi - lo)) + 1e-6);
    if (db[f] > mx) mx = db[f];
  }
  int f0 = 0, f1 = nf - 1;
  double thr = mx - 35 > -55 ? mx - 35 : -55;
  while (f0 < nf && db[f0] < thr) f0++;
  while (f1 > f0 && db[f1] < thr) f1--;
  *a = f0 / 10.0f;
  *b = (f1 + 1) / 10.0f;
  free(db);
}

typedef struct {
  double at;
  int final;
  arc_ext_word* w;
  int n;
  char* text;
  double upto;   /* "upto": the recogniser has finalised up to here (-1: not given) */
} ext_ev;

static ext_ev* load_ext(const char* dir, const char* wav, int* n_out, arc_json*** keep, int* n_keep) {
  const char* base = strrchr(wav, '/');
  base = base ? base + 1 : wav;
  char path[4096];
  snprintf(path, sizeof path, "%s/%.*s.jsonl", dir, (int)(strlen(base) > 4 ? strlen(base) - 4 : strlen(base)), base);
  char* data = NULL;
  size_t len = 0;
  *n_out = 0;
  if (arc_read_file(path, &data, &len)) return NULL;
  int cap = 64, n = 0;
  ext_ev* ev = malloc(sizeof(ext_ev) * (size_t)cap);
  for (char* line = strtok(data, "\n"); line; line = strtok(NULL, "\n")) {
    arc_json* j = arc_json_parse(line, strlen(line));
    if (!j) continue;
    *keep = realloc(*keep, sizeof(arc_json*) * (size_t)(*n_keep + 1));
    (*keep)[(*n_keep)++] = j;
    if (n == cap) ev = realloc(ev, sizeof(ext_ev) * (size_t)(cap *= 2));
    ext_ev* e = &ev[n++];
    memset(e, 0, sizeof *e);
    e->at = arc_json_num(arc_json_get(j, "at"), 0);
    e->final = (int)arc_json_num(arc_json_get(j, "final"), 1);
    e->upto = arc_json_num(arc_json_get(j, "upto"), -1);
    const arc_json* ws = arc_json_get(j, "words");
    if (ws && ws->type == ARC_JARR) {
      e->w = malloc(sizeof(arc_ext_word) * (size_t)(ws->n + 1));
      for (int i = 0; i < ws->n; ++i) {
        const arc_json* w = arc_json_at(ws, i);
        e->w[e->n++] = (arc_ext_word){arc_json_str(arc_json_get(w, "text"), ""),
                                      (float)arc_json_num(arc_json_get(w, "start"), -1),
                                      (float)arc_json_num(arc_json_get(w, "end"), -1)};
      }
    } else e->text = (char*)arc_json_str(arc_json_get(j, "text"), "");
  }
  arc_free(data);
  *n_out = n;
  return ev;
}

static int cmp_d(const void* a, const void* b) {
  double x = *(const double*)a, y = *(const double*)b;
  return x < y ? -1 : x > y;
}

static void sleep_ms(double ms) {
  if (ms <= 0) return;
  struct timespec ts = {(time_t)(ms / 1000), (long)(fmod(ms, 1000) * 1e6)};
  nanosleep(&ts, NULL);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr,
            "usage: arc-stream MINI_DIR file.wav [...] [--mode full|heads|external] [--threads N] [--chunk-ms 100]\n"
            "       [--partial-ms 400] [--beams N] [--domain D] [--events] [--no-decoder] [--tail-s 1.5]\n"
            "       [--words-dir DIR] [--records] [--realtime] [--quiet]\n");
    return 2;
  }
  const char* files[4096];
  int delay_ms = 0;
  int nf = 0, threads = 4, chunk_ms = 100, partial_ms = 0, beams = 0, events = 0, no_dec = 0, records = 0,
      realtime = 0, quiet = 0;
  float tail_s = 1.5f;
  const char *domain = "unknown", *words_dir = NULL;
  arc_stream_mode mode = ARC_STREAM_FULL;
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
      const char* m = argv[++i];
      mode = !strcmp(m, "heads") ? ARC_STREAM_HEADS : !strcmp(m, "external") ? ARC_STREAM_EXTERNAL : ARC_STREAM_FULL;
    } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--chunk-ms") && i + 1 < argc) chunk_ms = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--partial-ms") && i + 1 < argc) partial_ms = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--beams") && i + 1 < argc) beams = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--delay-ms") && i + 1 < argc) {
      delay_ms = atoi(argv[++i]);
      if (delay_ms == 0) delay_ms = ARC_DELAY_NONE;
    }
    else if (!strcmp(argv[i], "--domain") && i + 1 < argc) domain = argv[++i];
    else if (!strcmp(argv[i], "--tail-s") && i + 1 < argc) tail_s = (float)atof(argv[++i]);
    else if (!strcmp(argv[i], "--words-dir") && i + 1 < argc) words_dir = argv[++i];
    else if (!strcmp(argv[i], "--events")) events = 1;
    else if (!strcmp(argv[i], "--no-decoder")) no_dec = 1;
    else if (!strcmp(argv[i], "--records")) records = 1;
    else if (!strcmp(argv[i], "--realtime")) realtime = 1;
    else if (!strcmp(argv[i], "--quiet")) quiet = 1;
    else if (nf < 4096) files[nf++] = argv[i];
  }
  if (mode != ARC_STREAM_FULL) no_dec = 1;   /* heads / external never need the decoder */
  double t0 = now_ms();
  char err[512];
  arc_engine* e = arc_create_ex(argv[1], threads, no_dec ? ARC_NO_DECODER : 0, err, sizeof err);
  if (!e) {
    fprintf(stderr, "arc: %s\n", err);
    return 1;
  }
  double load_ms = now_ms() - t0;
  struct rusage ru0;
  getrusage(RUSAGE_SELF, &ru0);
  double audio_all = 0, wall_all = 0;
  double* fp = malloc(sizeof(double) * (size_t)(nf + 1));
  double* fl = malloc(sizeof(double) * (size_t)(nf + 1));
  double* tl = malloc(sizeof(double) * (size_t)(nf + 1));
  double frame_sum = 0, max_push_all = 0;
  double* pl = malloc(sizeof(double) * (size_t)(nf + 1));
  int npl = 0, agree_all = 0, compared_all = 0;
  long frame_n = 0;
  int nfp = 0, nfl = 0, ntl = 0;
  for (int fi = 0; fi < nf; ++fi) {
    size_t n;
    int rate;
    float* pcm = wav_read(files[fi], &n, &rate);
    if (!pcm) {
      fprintf(stderr, "cannot read %s\n", files[fi]);
      return 1;
    }
    run_st r;
    memset(&r, 0, sizeof r);
    r.first_partial = r.final_lat = r.tags_lat = r.prov_lat = -1;
    r.records = records;
    r.quiet = quiet;
    speech_bounds(pcm, n, &r.speech_start, &r.speech_end);
    arc_stream_config c = {mode, domain, on_out, &r, 0, events, partial_ms, 0, beams, 0, 0, delay_ms};
    c.frames = 1;
    arc_stream* s = arc_stream_open(e, &c);
    if (!s) {
      fprintf(stderr, "arc: %s\n", arc_last_error());
      return 1;
    }
    arc_json** keep = NULL;
    int n_keep = 0, n_ev = 0, next_ev = 0;
    ext_ev* ev = mode == ARC_STREAM_EXTERNAL && words_dir ? load_ext(words_dir, files[fi], &n_ev, &keep, &n_keep) : NULL;
    if (!quiet && !records) printf("%s\n", files[fi]);
    size_t tail = (size_t)(tail_s * ARC_SAMPLE_RATE), total = n + tail;
    float* all = calloc(total + 1, sizeof(float));
    memcpy(all, pcm, sizeof(float) * n);
    size_t step = (size_t)chunk_ms * ARC_SAMPLE_RATE / 1000;
    if (step < 1) step = 1;
    double wall = 0, start = now_ms();
    for (size_t pos = 0; pos < total; pos += step) {
      size_t k = pos + step < total ? step : total - pos;
      if (realtime) sleep_ms(start + (pos + k) * 1000.0 / ARC_SAMPLE_RATE - now_ms());
      r.push_t0 = now_ms();
      if (arc_stream_push(s, all + pos, k)) {
        fprintf(stderr, "arc: %s\n", arc_last_error());
        return 1;
      }
      /* the recogniser's output that has arrived by now */
      double at = (double)(pos + k) / ARC_SAMPLE_RATE;
      while (next_ev < n_ev && ev[next_ev].at <= at) {
        ext_ev* x = &ev[next_ev++];
        if (x->w) arc_stream_words(s, x->w, x->n, x->final);
        else arc_stream_text(s, x->text, x->final);
        if (x->final && x->upto >= 0) arc_stream_covered(s, (float)x->upto);
      }
      double d = now_ms() - r.push_t0;
      wall += d;
      if (d > r.max_push) r.max_push = d;
    }
    while (next_ev < n_ev) {   /* the rest (after the audio) */
      ext_ev* x = &ev[next_ev++];
      r.push_t0 = now_ms();
      if (x->w) arc_stream_words(s, x->w, x->n, x->final);
      else arc_stream_text(s, x->text, x->final);
      if (x->final && x->upto >= 0) arc_stream_covered(s, (float)x->upto);
    }
    r.push_t0 = now_ms();
    if (arc_stream_finish(s)) {
      fprintf(stderr, "arc: %s\n", arc_last_error());
      return 1;
    }
    wall += now_ms() - r.push_t0;
    arc_stream_close(s);
    double audio = (double)n / ARC_SAMPLE_RATE;
    audio_all += audio;
    wall_all += wall;
    if (r.max_push > max_push_all) max_push_all = r.max_push;
    if (r.first_partial >= 0) fp[nfp++] = r.first_partial;
    if (r.final_lat >= 0) fl[nfl++] = r.final_lat;
    if (r.tags_lat >= 0) tl[ntl++] = r.tags_lat;
    if (r.prov_lat >= 0) pl[npl++] = r.prov_lat;
    agree_all += r.agree;
    compared_all += r.compared;
    frame_sum += r.frame_sum;
    frame_n += r.frame_n;
    printf("{\"file\": ");
    jstr(files[fi]);
    printf(", \"audio_s\": %.2f, \"wall_ms\": %.1f, \"rtf\": %.4f, \"max_push_ms\": %.1f, \"speech\": [%.1f, %.1f], "
           "\"first_partial_ms\": %.0f, \"final_ms\": %.0f, \"tags_ms\": %.0f, \"prov_tags_ms\": %.0f, \"frame_ms\": %.0f, \"text\": ",
           audio, wall, wall / (audio * 1000), r.max_push, r.speech_start, r.speech_end, r.first_partial, r.final_lat,
           r.tags_lat, r.prov_lat, r.frame_n ? r.frame_sum / r.frame_n : -1);
    jstr(r.text);
    printf("}\n");
    fflush(stdout);
    for (int i = 0; i < n_ev; ++i) free(ev[i].w);
    free(ev);
    for (int i = 0; i < n_keep; ++i) arc_json_free(keep[i]);
    free(keep);
    free(all);
    free(pcm);
  }
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  double cpu = (ru.ru_utime.tv_sec - ru0.ru_utime.tv_sec + ru.ru_stime.tv_sec - ru0.ru_stime.tv_sec) * 1000.0 +
               (ru.ru_utime.tv_usec - ru0.ru_utime.tv_usec + ru.ru_stime.tv_usec - ru0.ru_stime.tv_usec) / 1000.0;
#ifdef __APPLE__
  double rss_mb = ru.ru_maxrss / 1048576.0;
#else
  double rss_mb = ru.ru_maxrss / 1024.0;
#endif
  qsort(fp, (size_t)nfp, sizeof(double), cmp_d);
  qsort(fl, (size_t)nfl, sizeof(double), cmp_d);
  qsort(tl, (size_t)ntl, sizeof(double), cmp_d);
  qsort(pl, (size_t)npl, sizeof(double), cmp_d);
#define P50(a, n) ((n) ? (a)[(n) / 2] : -1)
#define P90(a, n) ((n) ? (a)[(int)((n) * 0.9) < (n) ? (int)((n) * 0.9) : (n) - 1] : -1)
  static const char* mn[] = {"full", "heads", "external"};
  printf("{\"summary\": true, \"mode\": \"%s\", \"threads\": %d, \"chunk_ms\": %d, \"files\": %d, \"audio_s\": %.1f, "
         "\"rtf\": %.4f, \"cpu_per_audio_s\": %.4f, \"load_ms\": %.0f, \"peak_rss_mb\": %.1f, \"max_push_ms\": %.1f, "
         "\"first_partial_ms_p50\": %.0f, \"first_partial_ms_p90\": %.0f, \"final_ms_p50\": %.0f, \"final_ms_p90\": %.0f, "
         "\"tags_ms_p50\": %.0f, \"tags_ms_p90\": %.0f, \"prov_tags_ms_p50\": %.0f, \"prov_tags_ms_p90\": %.0f, "
         "\"prov_agree\": %d, \"prov_compared\": %d, \"delay_ms\": %d, \"frame_ms_mean\": %.0f}\n",
         mn[mode], threads, chunk_ms, nf, audio_all, wall_all / (audio_all * 1000), cpu / (audio_all * 1000), load_ms,
         rss_mb, max_push_all, P50(fp, nfp), P90(fp, nfp), P50(fl, nfl), P90(fl, nfl), P50(tl, ntl), P90(tl, ntl),
         P50(pl, npl), P90(pl, npl), agree_all, compared_all, delay_ms == ARC_DELAY_NONE ? 0 : delay_ms ? delay_ms : 300,
         frame_n ? frame_sum / frame_n : -1);
  free(pl);
  free(fp);
  free(fl);
  free(tl);
  arc_destroy(e);
  return 0;
}
