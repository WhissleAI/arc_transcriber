/* arc_transcribe with the trunk and events against Arc.predict(domain='assistant') (ref_assistant.json): the
 * device-directed head, the trunk's utterance intent / act / device_directed, and sound events.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arc.h"
#include "arc_base.h"
#include "wav.h"

static double maxd = 0;
static int close_p(double a, double b, double tol) { double d = fabs(a - b); if (d > maxd) maxd = d; return d <= tol; }

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: test_full MINI_DIR WAV_DIR REF.json [verbose|words]\n"); return 2; }
  char err[512];
  arc_engine* e = arc_create(argv[1], 4, err, sizeof err);
  if (!e) { fprintf(stderr, "%s\n", err); return 1; }
  char* text; size_t len;
  if (arc_read_file(argv[3], &text, &len)) return 1;
  arc_json* ref = arc_json_parse(text, len);
  arc_options o = {.domain = "assistant", .trunk = 1, .events = 1};
  if (argc > 4 && !strcmp(argv[4], "words")) o.words = 1;   /* aligned words: the word channel joins the trunk, as in predict */
  int n_dir = 0, ok_dir = 0, n_utt = 0, ok_int = 0, ok_act = 0, ok_dd = 0, n_ev = 0, ok_ev = 0, n_clip = 0;
  double wall = 0, audio = 0;
  for (int c = 0; c < ref->n; ++c) {
    const char* clip = ref->keys[c];
    const arc_json* R = &ref->items[c];
    char path[2048];
    snprintf(path, sizeof path, "%s/%s.wav", argv[2], clip);
    size_t nw; int rate;
    float* pcm = wav_read(path, &nw, &rate);
    arc_result* r = arc_transcribe(e, pcm, nw, &o);
    wall += r->wall_ms; audio += r->audio_s; n_clip++;
    const arc_json* T = arc_json_get(R, "transcript");
    const arc_json* U = arc_json_get(R, "utterances");
    for (int i = 0; i < r->n_utts && T && i < T->n; ++i) {
      const arc_utt* u = &r->utts[i];
      const arc_json* dd = arc_json_get(arc_json_at(T, i), "device_directed");
      if (dd) { n_dir++; ok_dir += close_p(u->directed_head, arc_json_num(arc_json_get(dd, "p"), -1), 2e-3); }
      /* the reference trunk utterance overlapping most */
      int best = -1; double bo = 0;
      for (int k = 0; U && k < U->n; ++k) {
        const arc_json* x = arc_json_at(U, k);
        double ov = fmin(u->end, arc_json_num(arc_json_get(x, "end"), 0)) - fmax(u->start, arc_json_num(arc_json_get(x, "start"), 0));
        if (ov > bo) { bo = ov; best = k; }
      }
      if (best < 0) continue;
      const arc_json* x = arc_json_at(U, best);
      const arc_json* gi = arc_json_get(arc_json_get(x, "groups"), "discourse.intent");
      const arc_json* ga = arc_json_get(arc_json_get(x, "groups"), "discourse.act");
      double rdd = arc_json_num(arc_json_get(arc_json_get(x, "tags"), "discourse.assistant.device_directed"), -1);
      n_utt++;
      int ii = gi && u->intent.label && !strcmp(u->intent.label, arc_json_str(arc_json_get(gi, "label"), "")) && close_p(u->intent.p, arc_json_num(arc_json_get(gi, "p"), -1), 3e-3);
      int aa = ga && u->act.label && !strcmp(u->act.label, arc_json_str(arc_json_get(ga, "label"), "")) && close_p(u->act.p, arc_json_num(arc_json_get(ga, "p"), -1), 3e-3);
      int dd2 = close_p(u->device_directed, rdd, 3e-3);
      ok_int += ii; ok_act += aa; ok_dd += dd2;
      if (!(ii && aa && dd2) || argc > 4)
        printf("%s utt %d: intent %s %.4f / ref %s %.4f | act %s %.4f / ref %s %.4f | dd %.4f / ref %.4f | head %.4f\n", clip, i,
               u->intent.label, u->intent.p, arc_json_str(arc_json_get(gi, "label"), "-"), arc_json_num(arc_json_get(gi, "p"), -1),
               u->act.label, u->act.p, arc_json_str(arc_json_get(ga, "label"), "-"), arc_json_num(arc_json_get(ga, "p"), -1),
               u->device_directed, rdd, u->directed_head);
    }
    const arc_json* E = arc_json_get(R, "audio_events");
    int ne = E ? E->n : 0;
    n_ev += ne;
    for (int k = 0; k < ne; ++k) {
      const arc_json* ev = arc_json_at(E, k);
      int found = 0;
      for (int j = 0; j < r->n_events; ++j)
        if (!strcmp(r->events[j].label, arc_json_str(arc_json_get(ev, "label"), "")) && fabs(r->events[j].start - arc_json_num(arc_json_get(ev, "start"), 0)) < 0.05 &&
            fabs(r->events[j].end - arc_json_num(arc_json_get(ev, "end"), 0)) < 0.05 && close_p(r->events[j].p, arc_json_num(arc_json_get(ev, "p"), 0), 3e-3)) found = 1;
      ok_ev += found;
    }
    if (r->n_events != ne) printf("%s: %d events, ref %d\n", clip, r->n_events, ne);
    arc_result_free(r); free(pcm);
  }
  printf("directed head %d/%d; trunk utterances: intent %d/%d act %d/%d device_directed %d/%d; events %d/%d; max |p diff| %.2e\n",
         ok_dir, n_dir, ok_int, n_utt, ok_act, n_utt, ok_dd, n_utt, ok_ev, n_ev, maxd);
  printf("%d clips, %.1f s audio in %.0f ms (RTF %.3f)\n", n_clip, audio, wall, wall / 1000 / audio);
  arc_destroy(e);
  return 0;
}
