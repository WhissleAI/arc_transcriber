/* Speakers: arc_audio engine.py's diarization (the slot state machine's turns, or trackbank.py's TrackBank over the diar
 * embedding for the "fused" / "voice" domains) and the per-speaker pooled sums its speaker read-out is taken from.
 * Frame by frame, so the whole-clip path and streams share it.
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#ifndef ARC_SPEAKER_H
#define ARC_SPEAKER_H

#define ARC_SPK_MAX 64        /* model.py S_MAX: speaker indices pooled */
#define ARC_BANK_TRACKS 64    /* TrackBank max_tracks */

enum { ARC_DIAR_SLOTS = 0, ARC_DIAR_FUSED = 1, ARC_DIAR_VOICE = 2 };

typedef struct {
  float beta, tau_new, tau_upd;   /* config.json diarization.fused_bank (beta 0 for "voice") */
  int confirm, sw, smooth;
} arc_bank_cfg;

/* a diarization turn over frames [f0, f1): slots: speaker index (label "S<k+1>"); bank: track (label "T<k>") */
typedef struct {
  int k;
  long f0, f1;
} arc_spk_turn;

typedef struct arc_spk arc_spk;

/* mode ARC_DIAR_*; dm: pooled width (512); ns: slot logits (1 + slots); de: embedding width (bank modes) */
arc_spk* arc_spk_new(int mode, const arc_bank_cfg* bc, int dm, int ns, int de);
void arc_spk_free(arc_spk* s);
int arc_spk_mode(const arc_spk* s);
/* frame f (frames come in order from 0): spk the state machine's speaker (-1: none / no speech), speech (meta.speech >=
 * 0.5), slot the slot logits [ns], emb the diar embedding [de] (bank modes, else NULL), zs = h + pool_in.speaker(h) [dm] */
int arc_spk_frame(arc_spk* s, long f, int spk, int speech, const float* slot, const float* emb, const float* zs);
/* the end of the audio at frame f_end: the open turn closes there */
void arc_spk_finish(arc_spk* s, long f_end);
/* speakers seen (index order): their index, frame count, last frame and pooled mean (float, as model.py's .float()) */
int arc_spk_count(const arc_spk* s);
int arc_spk_info(const arc_spk* s, int i, int* index, long* last, float* mean /* [dm] */);
/* turns so far (arc_free *out): slots: the closed ones, plus with open = 1 the one still open (to the last frame);
 * bank: from the frame labels so far (final after arc_spk_finish) */
int arc_spk_turns(const arc_spk* s, int open, arc_spk_turn** out);
/* the label of turn t: "S<k+1>" or "T<k>" into buf (16 bytes) */
const char* arc_spk_label(const arc_spk* s, const arc_spk_turn* t, char* buf);

#endif
