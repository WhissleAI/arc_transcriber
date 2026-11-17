/* The device's own playback in the microphone: a partitioned-block frequency-domain NLMS echo canceller (overlap-save), the per-utterance
 * playback_leak score and the barge-in rule (arc_audio/echo.py). Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see
 * LICENSE. */
#ifndef ARC_ECHO_H
#define ARC_ECHO_H
#include <stddef.h>

typedef struct {
  int block, parts;     /* 256 samples, 32 partitions (512 ms of echo path) */
  double mu, ref_db, eps, threshold, dt_ratio, dt_after_s;
} arc_echo_cfg;
void arc_echo_defaults(arc_echo_cfg* c);

typedef struct arc_echo arc_echo;
arc_echo* arc_echo_new(const arc_echo_cfg* c);   /* NULL: defaults */
void arc_echo_free(arc_echo* e);
/* reference audio; t (s, mic clock) of its first sample, t < 0: right after the last reference sample */
int arc_echo_feed_ref(arc_echo* e, const float* pcm, size_t n, double t);
/* mic audio (arrival order): blocks as they complete; the residual grows sample-aligned with the mic */
int arc_echo_feed_mic(arc_echo* e, const float* pcm, size_t n);
/* the last partial block (zero padded); the residual is then as long as the mic */
int arc_echo_flush(arc_echo* e);
/* the residual so far (sample-aligned with the mic; *n its length); owned by e */
const float* arc_echo_residual(arc_echo* e, size_t* n);
/* playback_leak over [t0, t1): 1 - E(res) / E(mic) over frames with playback; 0 nothing played there; -1 no reference ever */
double arc_echo_leak(const arc_echo* e, double t0, double t1);
/* 1: the reference has playback in [t - span, t) */
int arc_echo_playing(const arc_echo* e, double t, double span);
/* the barge-in rule: an utterance open for >= min_s while the device plays, whose last 0.3 s (or since it opened) the
 * playback does not explain (leak < max_leak), once the filter has seen 1 s of playback */
int arc_echo_barge_in(const arc_echo* e, double t_open, double t_now, double min_s, double max_leak);
double arc_echo_threshold(const arc_echo* e);
/* seconds of playback the filter has seen */
double arc_echo_trained_s(const arc_echo* e);

#endif
