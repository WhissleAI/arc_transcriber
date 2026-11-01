/* Live audio through an Arc stream: 16 kHz mono 16-bit PCM on stdin (a microphone via arecord, a file, a pipe),
 * one line per record on stdout as it happens.
 *   cc stream.c -I../../c/include ../../c/build/libarc.a -lm -pthread -o stream
 *   arecord -q -f S16_LE -r 16000 -c 1 -t raw | ./stream arc-v1.1/mini            # Arc's own transcript
 *   arecord -q -f S16_LE -r 16000 -c 1 -t raw | ./stream arc-v1.1/mini --heads    # tags only, no decoder
 *   sox clip.wav -t raw -r 16000 -c 1 -b 16 -e signed - | ./stream arc-v1.1/mini
 * Options: --heads, --threads N (default 2), --partial-ms N (default 1000; -1: none), --delay-ms N (read-out delay:
 * 0, 100, 200, 300 = default, 500, 800), --domain D (default assistant), --frames (also the 10 Hz read-out).
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arc.h"

static int g_frames = 0;

static void on_output(const arc_stream_out* o, void* user) {
  (void)user;
  const arc_utt* u = o->utt;
  switch (o->kind) {
    case ARC_OUT_PARTIAL:
      printf("%7.2f partial  #%d  %s\n", o->audio_s, o->id, u->text);
      break;
    case ARC_OUT_ENDPOINT:
      printf("%7.2f end      #%d  %.2f-%.2f\n", o->audio_s, o->id, u->start, u->end);
      break;
    case ARC_OUT_FINAL:
    case ARC_OUT_TAGS:
      printf("%7.2f %-8s #%d  %.2f-%.2f  %s  [directed %.2f, %s, %s%s]\n", o->audio_s,
             o->kind == ARC_OUT_FINAL ? "final" : "tags", o->id, u->start, u->end, u->text[0] ? u->text : "-",
             u->device_directed, u->intent.label ? u->intent.label : "-", u->act.label ? u->act.label : "-",
             o->tags_final ? "" : ", provisional");
      break;
    case ARC_OUT_EVENT:
      printf("%7.2f event    %s %.1f-%.1f (p %.2f)\n", o->audio_s, o->event->label, o->event->start, o->event->end,
             o->event->p);
      break;
    case ARC_OUT_FRAME:
      if (g_frames)
        printf("%7.2f frame    %.1f speech %.2f directed %.2f\n", o->audio_s, o->frame->t, o->frame->speech,
               o->frame->device_directed);
      break;
  }
  fflush(stdout);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: arecord -q -f S16_LE -r 16000 -c 1 -t raw | stream MINI_DIR [--heads] [--threads N] "
                    "[--partial-ms N] [--delay-ms N] [--domain D] [--frames]\n");
    return 2;
  }
  int heads = 0, threads = 2;
  arc_stream_config c = {0};
  c.mode = ARC_STREAM_FULL;
  c.domain = "assistant";
  c.partial_ms = 1000;
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--heads")) heads = 1;
    else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--partial-ms") && i + 1 < argc) c.partial_ms = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--delay-ms") && i + 1 < argc) {
      c.delay_ms = atoi(argv[++i]);
      if (!c.delay_ms) c.delay_ms = ARC_DELAY_NONE;
    } else if (!strcmp(argv[i], "--domain") && i + 1 < argc) c.domain = argv[++i];
    else if (!strcmp(argv[i], "--frames")) g_frames = 1;
  }
  char err[512];
  arc_engine* e = arc_create_ex(argv[1], threads, heads ? ARC_NO_DECODER : 0, err, sizeof err);
  if (!e) {
    fprintf(stderr, "arc: %s\n", err);
    return 1;
  }
  c.mode = heads ? ARC_STREAM_HEADS : ARC_STREAM_FULL;
  c.on_output = on_output;
  c.events = 1;
  c.frames = g_frames;
  arc_stream* s = arc_stream_open(e, &c);
  if (!s) {
    fprintf(stderr, "arc: %s\n", arc_last_error());
    return 1;
  }
  enum { CHUNK = 1600 };   /* 100 ms */
  int16_t raw[CHUNK];
  float pcm[CHUNK];
  size_t got;
  while ((got = fread(raw, 2, CHUNK, stdin)) > 0) {
    for (size_t i = 0; i < got; ++i) pcm[i] = raw[i] / 32768.0f;
    if (arc_stream_push(s, pcm, got)) {
      fprintf(stderr, "arc: %s\n", arc_last_error());
      return 1;
    }
  }
  arc_stream_finish(s);
  arc_stream_close(s);
  arc_destroy(e);
  return 0;
}
