/* A stream against arc_transcribe on the same audio (needs the weights):
 *   build/test_stream MINI_DIR file.wav [more.wav ...]
 * Each file is pushed in chunks of random sizes (1 sample to 0.4 s). Checked:
 *   - ARC_STREAM_FULL: the finals' text, times and inline tags identical to arc_transcribe's utterances;
 *   - its settled tags (device_directed, command_complete, intent, act) and directed_head identical to
 *     arc_transcribe with the trunk on (no word times, names or numbers: those change the read-out's input);
 *   - the events identical;
 *   - ARC_STREAM_HEADS (an engine without the decoder): the same utterances' times and tags;
 *   - FRAME records: every column (arc_frame.labels / p) identical to arc_frames_compute's;
 *   - ARC_STREAM_EXTERNAL fed Arc's own aligned words (each 0.3 s after it ends): the same utterances' words.
 * Records live only during the callback: keep copies. Bit-identical on aarch64 (all steps are the same per-frame computations); x86 may differ in the last bits
 * through the AVX2 GELU, so there the tolerance is 1e-5.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arc.h"
#include "wav.h"

#if defined(__aarch64__) || defined(_M_ARM64)
#define TOL 0.0
#else
#define TOL 1e-5
#endif

typedef struct {
  int n;
  struct {
    float start, end, dd, cc, head, ip, ap;
    char text[2048], tags[1024];
    char intent[128], act[128];
    int tags_final;
  } u[256];
  int ne;
  arc_event ev[256];
} got_t;

static void tags_str(const arc_utt* u, char* out, size_t cap) {
  out[0] = 0;
  for (int i = 0; i < u->n_tags; ++i) {
    size_t k = strlen(out);
    snprintf(out + k, cap - k, "%s%s:%d-%d", i ? "," : "", u->tags[i].tag, u->tags[i].word_begin, u->tags[i].word_end);
  }
}

static void on_out(const arc_stream_out* o, void* user) {
  got_t* g = user;
  if (o->kind == ARC_OUT_EVENT && g->ne < 256) g->ev[g->ne++] = *o->event;
  if (o->kind != ARC_OUT_FINAL && o->kind != ARC_OUT_TAGS) return;
  if (o->id >= 256) return;
  if (o->id >= g->n) g->n = o->id + 1;
  const arc_utt* u = o->utt;
  if (o->kind == ARC_OUT_FINAL) {
    g->u[o->id].start = u->start;
    g->u[o->id].end = u->end;
    snprintf(g->u[o->id].text, sizeof g->u[o->id].text, "%s", u->text);
    tags_str(u, g->u[o->id].tags, sizeof g->u[o->id].tags);
  }
  if (o->tags_final) {   /* settled (a TAGS record with tags_final 0 is provisional) */
    g->u[o->id].dd = u->device_directed;
    g->u[o->id].cc = u->command_complete;
    g->u[o->id].head = u->directed_head;
    snprintf(g->u[o->id].intent, 128, "%s", u->intent.label ? u->intent.label : "");
    g->u[o->id].ip = u->intent.p;
    snprintf(g->u[o->id].act, 128, "%s", u->act.label ? u->act.label : "");
    g->u[o->id].ap = u->act.p;
    g->u[o->id].tags_final = 1;
  }
}

static unsigned g_seed = 12345;
static size_t rnd_chunk(void) {
  g_seed = g_seed * 1103515245u + 12345u;
  unsigned r = g_seed >> 8;
  return r % 7 == 0 ? 1 + r % 50 : 1 + r % 6400;
}

static int run(arc_engine* e, arc_stream_mode mode, const float* pcm, size_t n, got_t* g) {
  memset(g, 0, sizeof *g);
  arc_stream_config c = {mode, "assistant", on_out, g, 0, 1, -1, 0, 0, 0, 0, 0};
  arc_stream* s = arc_stream_open(e, &c);
  if (!s) return -1;
  for (size_t pos = 0; pos < n;) {
    size_t k = rnd_chunk();
    if (pos + k > n) k = n - pos;
    if (arc_stream_push(s, pcm + pos, k)) return -1;
    pos += k;
  }
  int r = arc_stream_finish(s);
  arc_stream_close(s);
  return r;
}

/* EXTERNAL: the words of r (Arc's own, aligned) sent as another recogniser would, each final 0.3 s after it ends */
static int run_ext(arc_engine* e, const float* pcm, size_t n, const arc_result* r, got_t* g) {
  memset(g, 0, sizeof *g);
  arc_stream_config c = {ARC_STREAM_EXTERNAL, "assistant", on_out, g, 0, 0, -1, 0, 0, 0, 0, 0};
  arc_stream* s = arc_stream_open(e, &c);
  if (!s) return -1;
  int ui = 0, wi = 0;
  for (size_t pos = 0; pos < n;) {
    size_t k = rnd_chunk();
    if (pos + k > n) k = n - pos;
    if (arc_stream_push(s, pcm + pos, k)) return -1;
    pos += k;
    double t = (double)pos / ARC_SAMPLE_RATE;
    while (ui < r->n_utts) {
      const arc_utt* u = &r->utts[ui];
      if (wi >= u->n_words) {
        ui++;
        wi = 0;
        continue;
      }
      const arc_word* w = &u->words[wi];
      if (w->end + 0.3 > t) break;
      arc_ext_word x = {w->text, w->start, w->end};
      if (arc_stream_words(s, &x, 1, 1)) return -1;
      wi++;
    }
  }
  int rc = arc_stream_finish(s);
  arc_stream_close(s);
  return rc;
}

/* full FRAME records (every column) against arc_frames_compute: same labels, same values */
typedef struct {
  int n, nl, bad_label;
  float* p;    /* [cap, nl] by frame index */
  int cap;
} frames_t;
static void on_frame(const arc_stream_out* o, void* user) {
  frames_t* f = user;
  if (o->kind != ARC_OUT_FRAME || !o->frame->labels) return;
  int i = (int)lrintf(o->frame->t * 10);
  if (!f->nl) f->nl = o->frame->n_labels;
  if (i >= f->cap) {
    int nc = (i + 1) * 2;
    f->p = realloc(f->p, sizeof(float) * (size_t)nc * f->nl);
    f->cap = nc;
  }
  memcpy(f->p + (size_t)i * f->nl, o->frame->p, sizeof(float) * f->nl);
  if (i + 1 > f->n) f->n = i + 1;
}
static int check_frames(arc_engine* e, const float* pcm, size_t n, const char* name) {
  arc_frames* ref = arc_frames_compute(e, pcm, n, "assistant");
  frames_t f = {0};
  arc_stream_config c = {ARC_STREAM_HEADS, "assistant", on_frame, &f, 1, 0, -1, 0, 0, 0, 0, 0};
  arc_stream* s = arc_stream_open(e, &c);
  const char* const* labels = NULL;
  for (size_t pos = 0; pos < n;) {
    size_t k = rnd_chunk();
    if (pos + k > n) k = n - pos;
    arc_stream_push(s, pcm + pos, k);
    pos += k;
  }
  arc_stream_finish(s);
  (void)labels;
  int ok = ref && f.n == ref->n_frames && f.nl == ref->n_labels;
  double md = 0;
  for (int i = 0; ok && i < f.n; ++i)
    for (int j = 0; j < f.nl; ++j) md = fmax(md, fabs((double)f.p[(size_t)i * f.nl + j] - ref->p[(size_t)i * f.nl + j]));
  ok = ok && md <= TOL;
  if (!ok)
    printf("%s: frames %d x %d, arc_frames_compute %d x %d, max |diff| %g\n", name, f.n, f.nl, ref ? ref->n_frames : -1,
           ref ? ref->n_labels : -1, md);
  arc_stream_close(s);
  arc_frames_free(ref);
  free(f.p);
  return ok;
}

/* an utterance's words joined, as EXTERNAL joins them */
static void join_words(const arc_utt* u, char* out, size_t cap) {
  out[0] = 0;
  for (int i = 0; i < u->n_words; ++i) {
    size_t k = strlen(out);
    snprintf(out + k, cap - k, "%s%s", i ? " " : "", u->words[i].text);
  }
}

static int same(float a, float b) { return fabs((double)a - b) <= TOL; }
static int same_s(const char* a, const char* b) { return !strcmp(a ? a : "", b ? b : ""); }

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: test_stream MINI_DIR file.wav [...]\n");
    return 2;
  }
  char err[512];
  arc_engine* e = arc_create(argv[1], 4, err, sizeof err);
  arc_engine* eh = arc_create_ex(argv[1], 4, ARC_NO_DECODER, err, sizeof err);
  if (!e || !eh) {
    fprintf(stderr, "%s\n", err);
    return 1;
  }
  int bad = 0, nutt = 0;
  static got_t g, gh, gx;
  int ext_ok = 0, ext_n = 0, frames_ok = 0;
  for (int fi = 2; fi < argc; ++fi) {
    size_t n;
    int rate;
    float* pcm = wav_read(argv[fi], &n, &rate);
    if (!pcm) {
      fprintf(stderr, "cannot read %s\n", argv[fi]);
      return 1;
    }
    arc_options o = {.domain = "assistant", .trunk = 1, .events = 1};
    arc_result* r = arc_transcribe(e, pcm, n, &o);
    if (!r || run(e, ARC_STREAM_FULL, pcm, n, &g) || run(eh, ARC_STREAM_HEADS, pcm, n, &gh)) {
      fprintf(stderr, "%s: %s\n", argv[fi], arc_last_error());
      return 1;
    }
    int fb = 0;
    if (g.n != r->n_utts || gh.n != r->n_utts) {
      printf("%s: %d utterances, stream %d, heads %d\n", argv[fi], r->n_utts, g.n, gh.n);
      fb++;
    }
    for (int i = 0; i < r->n_utts && i < g.n && i < gh.n; ++i) {
      const arc_utt* u = &r->utts[i];
      char tg[1024];
      tags_str(u, tg, sizeof tg);
      int ok = !strcmp(u->text, g.u[i].text) && !strcmp(tg, g.u[i].tags) && same(u->start, g.u[i].start) &&
               same(u->end, g.u[i].end);
      int tok = g.u[i].tags_final && same(u->device_directed, g.u[i].dd) && same(u->command_complete, g.u[i].cc) &&
                same(u->directed_head, g.u[i].head) && same_s(u->intent.label, g.u[i].intent) &&
                same(u->intent.p, g.u[i].ip) && same_s(u->act.label, g.u[i].act) && same(u->act.p, g.u[i].ap);
      int hok = same(u->start, gh.u[i].start) && same(u->end, gh.u[i].end) && same(u->device_directed, gh.u[i].dd) &&
                same(u->directed_head, gh.u[i].head) && same_s(u->intent.label, gh.u[i].intent) &&
                same_s(u->act.label, gh.u[i].act);
      if (!ok || !tok || !hok) {
        printf("%s utt %d:%s%s%s\n  offline %.2f-%.2f \"%s\" [%s] dd %.6f cc %.6f head %.6f %s %s\n"
               "  stream  %.2f-%.2f \"%s\" [%s] dd %.6f cc %.6f head %.6f %s %s\n  heads   %.2f-%.2f dd %.6f head %.6f\n",
               argv[fi], i, ok ? "" : " text", tok ? "" : " tags", hok ? "" : " heads", u->start, u->end, u->text, tg,
               u->device_directed, u->command_complete, u->directed_head, u->intent.label, u->act.label, g.u[i].start,
               g.u[i].end, g.u[i].text, g.u[i].tags, g.u[i].dd, g.u[i].cc, g.u[i].head, g.u[i].intent, g.u[i].act,
               gh.u[i].start, gh.u[i].end, gh.u[i].dd, gh.u[i].head);
        fb++;
      }
      nutt++;
    }
    int evok = g.ne == r->n_events;
    for (int i = 0; evok && i < r->n_events; ++i)
      evok = !strcmp(r->events[i].label, g.ev[i].label) && same(r->events[i].start, g.ev[i].start) &&
             same(r->events[i].end, g.ev[i].end) && same(r->events[i].p, g.ev[i].p);
    if (!evok) {
      printf("%s: events differ (%d offline, %d streamed)\n", argv[fi], r->n_events, g.ne);
      fb++;
    }
    bad += fb != 0;
    /* EXTERNAL with Arc's own aligned words: the same utterances get the same words */
    arc_options ow = {.domain = "assistant", .words = 1};
    arc_result* rw = arc_transcribe(e, pcm, n, &ow);
    if (!rw || run_ext(eh, pcm, n, rw, &gx)) {
      fprintf(stderr, "%s: external: %s\n", argv[fi], arc_last_error());
      return 1;
    }
    for (int i = 0; i < rw->n_utts; ++i) {
      char want[2048];
      join_words(&rw->utts[i], want, sizeof want);
      if (!want[0]) continue;
      int hit = 0;
      for (int k = 0; k < gx.n && !hit; ++k) hit = !strcmp(gx.u[k].text, want);
      ext_n++;
      ext_ok += hit;
      if (!hit) printf("%s: external utterance %d \"%s\" not found\n", argv[fi], i, want);
    }
    arc_result_free(rw);
    frames_ok += check_frames(eh, pcm, n, argv[fi]);
    arc_result_free(r);
    free(pcm);
  }
  printf("external words: %d/%d utterances grouped as Arc's own\n", ext_ok, ext_n);
  bad += ext_ok != ext_n;
  printf("frames: %d/%d files: every column identical to arc_frames_compute\n", frames_ok, argc - 2);
  bad += frames_ok != argc - 2;
  printf("%d/%d files identical (%d utterances: text, tags, read-out, directed head, events; heads-only too)\n",
         argc - 2 - bad, argc - 2, nutt);
  arc_destroy(e);
  arc_destroy(eh);
  return bad != 0;
}
