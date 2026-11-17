/* A minimal app loop on libarc's public API (arc.h only): a live NFT2 stream with a names list, fed 100 ms of microphone
 * audio at a time, optionally with the device's own playback as the echo reference, printing the stream events.
 *
 *   make -C c ARC_ORT=/path/to/onnxruntime        # builds c/build/nft2/nft2-app-example against build/nft2/libarc.a
 *   c/build/nft2/nft2-app-example arc-transcriber-space/nft2 mic.wav [playback.wav]
 *
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <stdio.h>
#include <stdlib.h>
#include "arc.h"
#include "../tests/wav.h"

static void print_events(arc_nft2_stream* s) {
  int n = 0;
  const arc_nft2_event* ev = arc_nft2_stream_events(s, &n);
  static const char* nm[] = {"", "partial", "entity", "eot", "barge_in", "final"};
  for (int i = 0; i < n; ++i) {
    const arc_nft2_event* e = &ev[i];
    if (e->type == ARC_EV_EOT && e->p < 0.5f) continue;
    printf("%7.2f %-8s utt %d", e->t, nm[e->type], e->utt);
    if (e->text) printf(" \"%s\"", e->text);
    if (e->type == ARC_EV_ENTITY) printf(" %s -> %s (p %.2f)", e->kind, e->entity ? e->entity : "", e->p);
    if (e->type == ARC_EV_EOT || e->type == ARC_EV_BARGE_IN) printf(" p %.2f", e->p);
    if (e->type == ARC_EV_FINAL && e->kind) printf(" [%s, leak %.2f]", e->kind, e->p);
    printf("\n");
  }
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s MODEL_DIR mic.wav [playback.wav]\n", argv[0]);
    return 2;
  }
  char err[512];
  arc_nft2* m = arc_nft2_create(argv[1], 4, err, sizeof err);
  if (!m) {
    fprintf(stderr, "%s\n", err);
    return 1;
  }
  const char* al[] = {"Oceanus Procellarum"};
  arc_name_entry kb[] = {{.name = "Tycho", .kind = "lunar_feature"},
                         {.name = "Ocean of Storms", .kind = "lunar_feature", .aliases = al, .n_aliases = 1},
                         {.name = "Bernoulli", .kind = "lunar_feature"}};
  arc_nft2_set_names(m, kb, 3);
  arc_nft2_options o = {0};
  o.live = 1;                 /* partials, early entities, end of turn */
  o.almost = 1;               /* sound-alike names as "almost" records / events */
  arc_nft2_stream* s = arc_nft2_stream_open(m, &o);
  size_t n = 0, nr = 0;
  int rate = 0;
  float* pcm = wav_read(argv[2], &n, &rate);
  float* ref = argc > 3 ? wav_read(argv[3], &nr, &rate) : NULL;
  if (!pcm) {
    fprintf(stderr, "%s: not a 16 kHz WAV\n", argv[2]);
    return 1;
  }
  if (ref) arc_nft2_stream_feed_ref(s, ref, nr, 0.0);   /* the playback starts with the stream */
  for (size_t a = 0; a < n; a += 1600) {
    arc_result* r = arc_nft2_stream_feed(s, pcm + a, a + 1600 <= n ? 1600 : n - a);
    print_events(s);
    for (int i = 0; r && i < r->n_utts; ++i)
      for (int k = 0; k < r->utts[i].n_words; ++k)
        if (r->utts[i].words[k].conf >= 0 && r->utts[i].words[k].conf < 0.5f)
          printf("        unsure word \"%s\" (conf %.2f)\n", r->utts[i].words[k].text, r->utts[i].words[k].conf);
    arc_result_free(r);
  }
  arc_result* r = arc_nft2_stream_flush(s);
  print_events(s);
  for (int i = 0; r && i < r->n_names; ++i)
    printf("name \"%s\" %s %s\n", r->names[i].heard, r->names[i].state,
           r->names[i].entity ? r->names[i].entity : (r->names[i].candidate ? r->names[i].candidate : ""));
  arc_result_free(r);
  arc_nft2_stream_close(s);
  arc_nft2_destroy(m);
  free(pcm);
  free(ref);
  return 0;
}
