/* audio.h: request audio to 16 kHz mono float for libarc. WAV (PCM 8/16/24/32-bit, IEEE float 32/64, extensible,
 * any channel count: averaged to mono) or raw little-endian PCM16; any sample rate, resampled with a windowed-sinc
 * filter. No other container formats (no MP3 / FLAC / Opus decoder is linked: the image stays dependency-free).
 * Part of the Arc deploy service (Whissle Inc.); CC BY-NC 4.0 with the Whissle terms, as the rest of this repository. */
#ifndef ARC_DEPLOY_AUDIO_H
#define ARC_DEPLOY_AUDIO_H
#include <stddef.h>

/* 1 when the bytes start like a RIFF/WAVE file */
int audio_is_wav(const unsigned char* d, size_t n);
/* WAV bytes -> malloc'd float samples at 16 kHz mono; NULL + message in err. *rate_in gets the file's rate. */
float* audio_decode_wav(const unsigned char* d, size_t n, size_t* out_n, int* rate_in, char* err, size_t err_len);
/* raw little-endian signed 16-bit PCM, `channels` interleaved, at `rate` -> 16 kHz mono float */
float* audio_decode_pcm16(const unsigned char* d, size_t n, int rate, int channels, size_t* out_n, char* err, size_t err_len);
/* resample mono float (returns malloc'd; in == out rate: a copy) */
float* audio_resample(const float* x, size_t n, int rate_in, int rate_out, size_t* out_n);

#endif
