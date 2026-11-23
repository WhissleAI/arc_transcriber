/* libarc: Arc Mini (Arc v1.1's on-device backbone; "edge" in v1.1.0) in C11, no dependencies beyond libc, libm and
 * threads.
 *
 * One model for the transcript with inline tags (names, numbers, contacts) and word times, names linked to a list,
 * numbers with values, utterance-level tags (device-directed, intent, dialogue act), sound events and per-frame speech
 * activity. Reads the release folder as published (WhissleAI/arc-v1.1, mini/ at v1.2.0, the same weights as v1.1.1;
 * edge/ at v1.1.0): config.json, tags.json, model.q4.safetensors, heads.safetensors and moonshine-streaming-tiny/.
 * Verified against arc_audio 1.2.0 (see README.md).
 *
 *   arc_engine* e = arc_create("arc-v1.1/mini", 4, err, sizeof err);
 *   arc_result* r = arc_transcribe(e, pcm, n, NULL);          // 16 kHz mono float
 *   for (int i = 0; i < r->n_utts; ++i) puts(r->utts[i].text);
 *   arc_result_free(r);
 *   arc_destroy(e);
 *
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#ifndef ARC_H
#define ARC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ARC_SAMPLE_RATE 16000

/* ABI version: the binary interface of this header (struct layouts, enum / #define values, function signatures).
 * It changes when an existing declaration changes (a field added to a struct an array of which the library returns
 * counts); new functions alone do not change it. A binding generated from this header should compare it with
 * arc_abi_version() of the library it loads. ARC_VERSION is for people (logs). */
#define ARC_ABI_VERSION 2
#define ARC_VERSION "1.5.0"
int arc_abi_version(void);
const char* arc_version(void);

typedef struct arc_engine arc_engine;

/* An inline tag over words [word_begin, word_end) of the utterance text (split on spaces). */
typedef struct {
  const char* tag;      /* "ENTITY_PERSON", "NUMBER_TIME", "CONTACT_PHONE", ... */
  int word_begin, word_end;
  const char* text;     /* those words */
} arc_tag;

/* A word with its time (cross-attention alignment) and its inline tag, if any. */
typedef struct {
  const char* text;
  float start, end;     /* seconds */
  const char* tag;      /* "ENTITY_PERSON", ... or NULL */
  int tag_begin;        /* 1: first word of its tag */
  /* Arc Transcriber Space (arc_nft2_*); else conf -1, disfl NULL, n_alts 0 */
  float conf;           /* the RNNT's probability of the word's least likely token (unbiased: a word the names bias
                         * wrote reads low), -1: none */
  const char* disfl;    /* NULL, "filler" (um, uh, "you know"), "repeat" (the earlier copy of a repeated word / pair),
                         * "fragment" (a cut-off word before the word it starts) */
  int n_alts;           /* other spellings heard for this word, most likely first (the runner-up token; with
                         * arc_nft2_options.nbest, the words other hypotheses have here) */
  const char* const* alts;
  const float* alt_p;
} arc_word;

/* A typed quantity read from the words as said ("forty five percent" -> percent 45; "two kilometers" -> distance 2
 * km; "give rover 45 more per cent" -> percent 45, relative +1). */
typedef struct {
  const char* type;     /* "number", "percent", "distance", "duration", "speed", "temperature", "degrees", "mass",
                         * "power", "energy", "voltage", "current", "multiplier" */
  double value;
  int is_int;
  const char* unit;     /* "%", "km", "m", "mi", "ft", "s", "min", "h", "km/h", "mph", "C", "F", "deg", "kg", "W",
                         * "x" ... or NULL (a plain number) */
  int relative;         /* +1 "more" / "extra", -1 "less" / "fewer", 0 */
  int word_begin, word_end;   /* words [begin, end) of the utterance */
  float start, end;
  const char* text;     /* those words */
  const char* written;  /* "75%", "2 km", "40" */
} arc_quantity;

/* A label with its probability. */
typedef struct {
  const char* label;    /* "discourse.intent.calendar_set" */
  float p;
} arc_label;

typedef struct {
  float start, end;     /* seconds */
  const char* text;     /* the transcript, punctuated and cased */
  int n_tags;
  const arc_tag* tags;
  /* utterance-level readout of the trunk (filled when the trunk ran; p < 0: not available) */
  float device_directed;     /* discourse.assistant.device_directed */
  float command_complete;    /* discourse.assistant.command_complete */
  arc_label intent;          /* best discourse.intent.* */
  arc_label act;             /* best discourse.act.* */
  float directed_head;       /* the separate device-directed head on the encoder features (beta) */
  int n_words;               /* words with times (arc_options.words / names / numbers), else 0 */
  const arc_word* words;
  /* n-best: the final decode's finished beams, best first, distinct texts (alts[0] is the decoded text before the
   * loop fix; 1 with greedy decoding). Their scores are the beam search's length-normalised log-probabilities.
   * Arc Transcriber Space with arc_nft2_options.nbest > 1: the RNNT beam's texts that read differently from text (case,
   * punctuation aside), best first; alt_scores = their share of the beam (softmax over the beam's scores, 0..1). */
  int n_alts;
  const char* const* alts;
  const float* alt_scores;
  /* Arc Transcriber Space (arc_nft2_*): the Arc decoder's own text (words come from the RNNT; their tags and casing
   * from this text), and directed_head >= the model's threshold (1 / 0; -1: not computed) */
  const char* decoder_text;
  int directed;
  /* Arc Transcriber Space read-outs (arc_nft2_*; else conf -1, clean_text NULL, 0, playback_leak -1) */
  float conf;                 /* mean word confidence, -1: none */
  const char* clean_text;     /* the text without the words tagged disfluent */
  int n_quantities;
  const arc_quantity* quantities;
  int n_dropped;              /* words the guard removed (no speech under them, a low-confidence repeat or tail) */
  float playback_leak;        /* streams fed a reference (arc_nft2_stream_feed_ref): the share of this utterance's
                               * audio explained by the device's own playback (0..1), -1: no reference */
  int playback;               /* 1: playback_leak >= the stream's threshold (the words are the device's own audio) */
  const char* tagger_json;    /* Arc Tagger read-out of the words (arc_nft2_set_tagger), JSON, or NULL */
} arc_utt;

/* A name heard in the transcript and, when it is one of the list's (arc_set_names), linked to it (Arc's "names"). */
typedef struct {
  float start, end;
  const char* tag;           /* "entity.mention.person", "entity.mention.team", ... */
  const char* heard;         /* the words, tidied ("Theo Brandt") */
  const char* entity;        /* the list's name when linked, else NULL */
  int kb_index;              /* its index in the list, -1 */
  const char* state;         /* "linked", "not_in_list" (a name, not one of the list's), "abstain" (unsure) */
  float p;                   /* confidence of the decision (rescorer), -1: none */
  const char* model_entity;  /* the pointer head's own choice (NULL: a new name), when it ran */
  float model_p;             /* its probability, -1: the head did not run */
  const char* src;           /* "decoder" (an inline tag) or "snap" (a capitalised phrase that matches the list) */
  const char* via;           /* NULL, "snap" (linked by text similarity), "text" (fuzzy text policy) */
  float snap_score;          /* -1 unless via / src snap */
  float span_p;              /* mean span probability of the tag head over the name, -1: none */
  int show;                  /* 1: worth showing (not only stop words / numbers) */
  int n_words;
  const arc_word* words;     /* the name's words (text, times) */
  /* Arc Transcriber Space with arc_nft2_options.almost: state "almost" = heard words that SOUND like a list entry
   * (candidate) but were not linked; candidate_p is the sound-alike link probability. NULL / -1 / -1 otherwise. */
  const char* candidate;
  int candidate_kb_index;
  float candidate_p;
} arc_name;

/* A number in the transcript with its value (Arc's "numbers"). */
typedef struct {
  float start, end;
  const char* kind;          /* "score", "duration", "ordinal", "time_of_day", "money", ... or NULL */
  const char* unit;          /* "minute", "points", "USD", "pm", ... or NULL */
  const char* heard;         /* the words, tidied ("102 to 98") */
  int has_value, has_value2;
  double value, value2;      /* duration in seconds, time of day in seconds after midnight, score as value-value2 */
  int value_int, value2_int; /* 1: an integer value */
  const char* state;         /* "read_text" (from the words), "read" (from the value heads), "no_words" */
  const char* kind_head;     /* the tag head's kind over the span, or NULL */
  int has_value_model;
  double value_model;        /* the value heads' reading, when it composed */
} arc_number;

typedef struct {
  const char* label;    /* "scene.silence", "meta.clipping", ... */
  float start, end, p;
} arc_event;

typedef struct {
  int n_utts;
  const arc_utt* utts;
  int n_events;
  const arc_event* events;
  int n_names;
  const arc_name* names;
  int n_numbers;
  const arc_number* numbers;
  float audio_s, wall_ms;
} arc_result;

/* dir: the mini/ folder. threads: worker threads (0 = all cores). NULL + message on failure. */
arc_engine* arc_create(const char* dir, int threads, char* err, size_t err_len);
void arc_destroy(arc_engine* e);
/* Worker threads for the next calls (0 = all cores). One thread costs the least CPU per second of audio (no spinning
 * between kernels); more threads finish sooner. Not while a call is running. */
int arc_set_threads(arc_engine* e, int threads);

/* Decoder context: names (with kinds: person team organization place work_product event_name legal_ref
 * game_entity game_map other; NULL = other), free terms, a topic and earlier text. Any may be NULL / 0. Kept until
 * changed. auto_prev: feed back what was transcribed in the last minute. */
int arc_set_context(arc_engine* e, const char* const* names, const char* const* kinds, int n_names,
                    const char* const* terms, int n_terms, const char* topic, const char* previous_text,
                    int auto_prev);

/* The list names are linked to (Arc's predict(names=...)): name, kind (person team organization place work_product
 * event_name legal_ref game_entity game_map other; NULL = other), group (a team the name belongs to, may be NULL) and
 * aliases. Also replaces the decoder bias list with the names, their last words and aliases (bias_forms). Kept until
 * changed; n = 0 clears it. */
typedef struct {
  const char* name;
  const char* kind;
  const char* group;
  const char* const* aliases;
  int n_aliases;
} arc_name_entry;
int arc_set_names(arc_engine* e, const arc_name_entry* names, int n);

/* Replaces the decoder bias list (set from the context's names by arc_set_context) with these phrases: names or any
 * vocabulary expected (command words). Used when arc_options.bias_names is 1. */
int arc_set_bias(arc_engine* e, const char* const* phrases, int n);
/* How hard the bias pulls (arc_audio bias.py DEFAULT: start_bonus 2.0 for a listed phrase's first token, cont_bonus
 * 1.5 for the next token of a phrase in progress, on log-softmaxed scores). A short list picked for this utterance
 * (live context) can take a stronger pull than a long standing one. Kept until changed; applies to later decodes. */
int arc_set_bias_strength(arc_engine* e, float start_bonus, float cont_bonus);

/* Shallow fusion with a token n-gram LM over the decoder's own ids (an "ARCLM1" file; format: docs/LIBARC.md#n-gram-lm).
 * A relative re-ranker: among the candidates the audio already finds plausible, each text piece is charged
 * weight * (its LM log-probability - the LM's favourite's), capped; tags and the end of the utterance are left to the
 * audio. It helps text worded like the LM's training text and can hurt other wording (the decoder already has its own
 * LM): measure on your audio. weight: 0.2-0.5 typical, 0 = loaded but off (output identical to no LM). topk:
 * candidates per step (0 = 16, at most 64). path NULL removes the LM. Applies to finals and stream partials decoded
 * after the call; the file is memory-mapped, so engines loading the same file share it. Returns 0, or -1 with
 * arc_last_error() (unreadable or malformed file, another tokenizer's vocabulary); the old LM then stays. */
int arc_set_lm(arc_engine* e, const char* path, float weight, int topk);
/* Change the weight of the loaded LM (0 = off); -1 if weight > 0 and no LM is loaded. Cheap: per request is fine. */
int arc_set_lm_weight(arc_engine* e, float weight);
float arc_lm_weight(const arc_engine* e);   /* the current weight; 0 when no LM is loaded */

typedef struct {
  const char* domain;   /* one of Arc's domains ("assistant", "meeting", ... "unknown"); NULL = "unknown" */
  int one_utterance;    /* 1: decode the whole clip as one utterance (a held-key command), no segmentation */
  int beams;            /* 0 = the release setting (4) */
  int trunk;            /* 1: also run the trunk for utterance tags (device-directed, intent, act) */
  int events;           /* 1: sound / channel events */
  int bias_names;       /* 1: bias the decoder towards the context's names (Arc's predict does when given names) */
  int words;            /* 1: word timestamps (arc_utt.words) */
  int names;            /* 1: names in the transcript linked to the list set by arc_set_names (implies trunk, words) */
  int numbers;          /* 1: numbers with values (implies trunk, words) */
} arc_options;

/* Memory: the trunk's recurrent state (5 SSM layers, ~1.3 MB) carried from one call to the next, as Arc's stream()
 * carries it between feed() calls, so device-directed, intent, act, events and segmentation see what came before.
 * The transcript decoder has no hidden state across utterances; its continuity is the text context (previous_text).
 * on: 1 carries the state (starting from what was saved or loaded), 0 runs each clip from zero (the default). */
int arc_memory(arc_engine* e, int on);
/* Forgets the carried state. */
void arc_memory_reset(arc_engine* e);
/* Seconds of audio the carried state has seen since it was last reset (0: empty). */
double arc_memory_seconds(arc_engine* e);
/* The carried state to / from a file, tagged with the model it belongs to (a state from another model is refused).
 * 0 on success; -1 on failure (arc_last_error), and the engine's memory is left as it was. */
int arc_memory_save(arc_engine* e, const char* path);
int arc_memory_load(arc_engine* e, const char* path);

/* Per-frame probabilities (10 Hz) for a clip, for an app's reactions: frame tags (meta.speech ...), the utterance
 * readout as it accumulates (device_directed, discourse.act.*, intent.*), the instant queries as "ev_at:<id>" /
 * "ev_imm:<id>" / "ev_rec:<id>" (discourse.assistant.command_complete, speaker.change) and the sound / channel events (meta.clipping, meta.channel.telephone, scene.silence ...). Encoder and trunk only (no
 * decoder), so a few tens of ms for a short clip. Stateless (memory is not used). NULL on failure. */
typedef struct {
  int n_frames, n_labels;
  const char* const* labels;   /* n_labels tag ids */
  const float* p;              /* [n_frames, n_labels], row-major */
} arc_frames;
arc_frames* arc_frames_compute(arc_engine* e, const float* pcm, size_t n, const char* domain);
void arc_frames_free(arc_frames* f);

/* Verification: how well each text matches a clip, as the decoder's teacher-forced log P(text | audio) (natural log,
 * EOS included) with the encoder run once over the whole clip as one utterance (the encoder states of the last
 * arc_transcribe of the same samples are reused) and the decoder sharing work between texts with a common prefix.
 * Only the domain of the context is in the prompt (no names, no earlier text): the score says what the audio
 * supports, not what the list expects. The decoder writes names inside inline tags, so a text may mark one span with
 * braces ("read {Treasure Island.}"): without ARC_SCORE_TAGS the braces are dropped; with it the span is also
 * scored inside ENTITY_PERSON and ENTITY_WORK tags and logp is the best of the three. Sums are not length-fair:
 * n_tokens (may be NULL) gives each text's scored token count. 0, or -1 (arc_last_error). */
#define ARC_SCORE_TAGS 1
/* ... or only inside ENTITY_PERSON (a text without braces: plain): one sequence per text, a cheap first pass */
#define ARC_SCORE_PERSON 2
int arc_score(arc_engine* e, const float* pcm, size_t n, const char* const* texts, int n_texts, int flags, float* logp,
              int* n_tokens);

/* A whole clip (16 kHz mono float, any length). NULL on failure (arc_last_error). */
arc_result* arc_transcribe(arc_engine* e, const float* pcm, size_t n, const arc_options* opt);
void arc_result_free(arc_result* r);

/* The date a number of kind "date" names (arc_audio's date / iso fields): its heard words then cover the month too
 * ("March third", "the third of March", "March third, twenty twenty four"). year / month / day 0: not said; iso
 * "2024-03-03", "--03-03" (month and day), "2024-03", "2024", or NULL for a day alone. Returns NULL when numbers[i]
 * carries no date. Valid until arc_result_free. (A function rather than arc_number fields, so the ABI stays 2.) */
typedef struct {
  int year, month, day;
  const char* iso;
} arc_date;
const arc_date* arc_number_date(const arc_result* r, int i);

/* Speakers (arc_audio's "diarization", "speakers" and each transcript line's "speaker"), from Arc's own heads: the
 * diarization slot head with the engine's speaker state machine (a speaker switches after 5 frames, 2 for the first),
 * or, in the domains config.json names for it ("fused": nhl cricket esports court_appellate; "voice": meeting news
 * talk), the diar embedding head with arc_audio's TrackBank (tracks confirmed after 20 frames, switched after 10). Filled
 * when the trunk ran (arc_options.trunk). Valid until arc_result_free. (Functions rather than arc_result fields, so the
 * ABI stays 2.)
 *
 * A turn: speaker "S1", "S2" ... (slot k + 1) or, with the TrackBank, "T1", "T2" ..., over [start, end) seconds (10 Hz
 * frames). */
typedef struct {
  const char* speaker;
  float start, end;
} arc_speaker_turn;
/* the most likely label of an exclusive tag group for a speaker: group "speaker.role", label "speaker.role.host" */
typedef struct {
  const char* group;
  const char* label;
  float p;
} arc_speaker_group;
/* a speaker the slot state machine found (engine.py speaker_summary): id "S1" ..., the speaker read-out (the speaker
 * query over the speaker's pooled frames) per group and its ungrouped shipped tags, and the last frame it spoke in */
typedef struct {
  const char* id;
  float last_seen;          /* seconds */
  int n_groups;
  const arc_speaker_group* groups;
  int n_tags;
  const arc_label* tags;
} arc_speaker;
/* the speakers (by index); -1 when there is no speaker read-out (trunk off, or weights without one) */
int arc_speakers(const arc_result* r, const arc_speaker** speakers);
/* the diarization turns in time order; -1 as arc_speakers */
int arc_diarization(const arc_result* r, const arc_speaker_turn** turns);
/* the speaker of utterance i: the turn it overlaps most (arc_transcriber's lines), or NULL (none overlaps) */
const char* arc_utt_speaker(const arc_result* r, int i);
/* The diarization for the next calls: "auto" (or NULL: by domain, as above), "slots", "fused" or "voice" (arc_audio's
 * predict(diarization=...)). "fused" / "voice" need the diar embedding head (Arc Mini has it). 0, or -1. */
int arc_set_diarization(arc_engine* e, const char* mode);
const char* arc_last_error(void);

/* ================================================================ load options, streaming
 *
 * arc_create_ex(dir, threads, ARC_NO_DECODER, ...) loads Arc's encoder, trunk and heads without the transcript
 * decoder and its tokenizer (less memory, faster load): arc_transcribe and arc_score then fail, and streams run in
 * ARC_STREAM_HEADS or ARC_STREAM_EXTERNAL mode. */
#define ARC_NO_DECODER 1
arc_engine* arc_create_ex(const char* dir, int threads, int flags, char* err, size_t err_len);
/* 1 when the engine has the transcript decoder (not created with ARC_NO_DECODER) */
int arc_has_decoder(const arc_engine* e);

/* A stream: push audio as it arrives (any chunk size) and get records back through one callback, run inside the push
 * call on the caller's thread. Utterances are cut by Arc's own segmenter (pauses, at most 15 s) in every mode.
 *
 *   ARC_STREAM_FULL      Arc's transcript: PARTIAL records while an utterance is open (greedy, over the streamed
 *                        encoder states), then FINAL (the same decode as arc_transcribe: identical text and tags).
 *   ARC_STREAM_HEADS     no transcript decoder: FINAL records carry the utterance's times and tags, no text.
 *   ARC_STREAM_EXTERNAL  heads plus the text of another recogniser (arc_stream_words / arc_stream_text): its words
 *                        are grouped into Arc's utterances by time, PARTIAL records follow its interim words.
 *
 * In every mode an utterance's tags (device_directed, command_complete, intent, act) come from the trunk utterance
 * it overlaps most and directed_head from the device-directed head over its frames. FINAL is sent as soon as the
 * text is final; when the trunk utterance has not closed by then (it closes 0.5 s after speech, read 0.3 s late),
 * FINAL has tags_final = 0 and its tags so far, and a TAGS record with the settled tags follows.
 *
 * Timing: FRAME records are final as they come, the read-out delay (delay_ms) plus about 0.1 s after the frame. An
 * utterance's tags are those of its last speech frame; the read-out declares the utterance over after 0.5 s without
 * speech (Arc's rule), so at its first non-speech frame a TAGS record with tags_final = 0 gives them provisionally
 * (they change only if the speech resumes), and the settled ones follow. */
typedef enum { ARC_STREAM_FULL = 0, ARC_STREAM_HEADS = 1, ARC_STREAM_EXTERNAL = 2 } arc_stream_mode;
/* ENDPOINT: Arc's segmenter closed an utterance (id, utt->start / end, no text yet). In EXTERNAL mode it is the moment
 * to ask the recogniser for its final words (SpeechAnalyzer.finalize(through: end), Android stopListening()): Apple's
 * transcriber, for one, holds a long phrase's final text until told to finalise. */
typedef enum {
  ARC_OUT_PARTIAL = 1, ARC_OUT_FINAL = 2, ARC_OUT_TAGS = 3, ARC_OUT_EVENT = 4, ARC_OUT_FRAME = 5, ARC_OUT_ENDPOINT = 6
} arc_out_kind;

/* One 100 ms frame of the trunk's read-out (it runs 0.3 s behind the audio, plus the encoder's 0.24 s look-ahead). */
typedef struct {
  float t;                   /* frame start, seconds */
  float speech;              /* meta.speech */
  int in_utterance;          /* inside an utterance of the trunk's read-out */
  float device_directed;     /* that utterance's read-out so far (-1 outside one) */
  float command_complete;    /* ev_at:discourse.assistant.command_complete in this frame (-1 outside an utterance) */
  /* every column arc_frames_compute gives for the domain, in its order: frame tags, the utterance read-out so far,
   * the instant queries ("ev_at:<id>", "ev_imm:<id>", "ev_rec:<id>") and the sound / channel events */
  int n_labels;
  const char* const* labels;   /* the stream's (valid until it is closed) */
  const float* p;              /* this frame's (valid during the callback) */
} arc_frame;

typedef struct {
  arc_out_kind kind;
  int id;                    /* PARTIAL / FINAL / TAGS / ENDPOINT: the utterance's number in the stream (0, 1, ...) */
  const arc_utt* utt;        /* PARTIAL: the text so far (and the tags so far); FINAL / TAGS: the utterance */
  int tags_final;            /* FINAL: 1 when its tags are settled (else a TAGS record follows) */
  const arc_event* event;    /* EVENT: a sound / channel event that has ended */
  const arc_frame* frame;    /* FRAME */
  double audio_s;            /* audio pushed so far when this record was made */
} arc_stream_out;
typedef void (*arc_stream_fn)(const arc_stream_out* out, void* user);

typedef struct {
  arc_stream_mode mode;
  const char* domain;        /* as arc_options.domain; NULL = "unknown" */
  arc_stream_fn on_output;
  void* user;
  int frames;                /* 1: FRAME records */
  int events;                /* 1: EVENT records (sound / channel events) */
  int partial_ms;            /* FULL: a partial every ... ms of audio while an utterance is open (0: 400; -1: none) */
  float partial_window_s;    /* FULL: partials decode at most the last ... s of the utterance (0: 7) */
  int beams;                 /* FULL: beams of the final decode (0: the release's 4; 1 greedy) */
  int bias_names;            /* FULL: as arc_options.bias_names */
  float word_wait_s;         /* EXTERNAL: an utterance's FINAL waits for the recogniser's final words to reach 0.3 s
                              * past its end (arc_stream_covered or timed final words), or the end of the audio.
                              * 0 (the default): no limit. > 0: give up ... s after it ends; words that come later
                              * then go out as an utterance of their own (out of time order). A recogniser that holds
                              * its finals until told to (Apple's) needs the ENDPOINT's finalise, or the FINALs wait. */
  int delay_ms;              /* the read-out delay: how far the trunk reads ahead of the frame it labels. One Arc was
                              * trained with: 0 (= the default, 300), 100, 200, 300, 500, 800, or ARC_DELAY_NONE.
                              * Shorter: every tag sooner, judged on less of what follows. */
} arc_stream_config;
#define ARC_DELAY_NONE (-1)

typedef struct arc_stream arc_stream;
arc_stream* arc_stream_open(arc_engine* e, const arc_stream_config* c);   /* NULL on failure (arc_last_error) */
/* 16 kHz mono float. 0, or -1 (arc_last_error). */
int arc_stream_push(arc_stream* s, const float* pcm, size_t n);
/* EXTERNAL: words of another recogniser, times in seconds from the stream's first sample. final = 0: an interim
 * hypothesis (replaces the previous one); 1: final words. A word with start < 0 has no times (Android's
 * SpeechRecognizer): such words go to the utterance open now, or else to the last one, spread over its time.
 * Final words that fall in no utterance of Arc's (a gap between two, after the last) come out in time order as
 * utterances of their own (FINAL without an ENDPOINT before it), or join the next one when they end within 0.8 s of
 * its start. */
typedef struct {
  const char* text;
  float start, end;
} arc_ext_word;
int arc_stream_words(arc_stream* s, const arc_ext_word* w, int n, int final);
/* EXTERNAL: the recogniser has finalised everything up to t seconds (the end of its final result, words or not), so
 * utterances that end before t - 0.3 s get no more words: their FINAL need not wait (arc_audio's mark_covered). */
int arc_stream_covered(arc_stream* s, float t);
/* EXTERNAL: a recogniser's text without times (split on spaces), as arc_stream_words with start = -1. */
int arc_stream_text(arc_stream* s, const char* text, int final);
/* Close the open utterance now (push-to-talk released, an external end of speech): its FINAL comes without waiting
 * for the pause. */
int arc_stream_end_turn(arc_stream* s);
/* End of the audio: everything still open is closed and sent. */
int arc_stream_finish(arc_stream* s);
void arc_stream_close(arc_stream* s);
/* Speakers so far (as arc_speakers / arc_diarization on the stream's audio; the turn still open is included, up to the
 * last frame read out; after arc_stream_finish they are what arc_transcribe gives). Valid until the next call on s. -1:
 * no speaker read-out. Callable inside the callback. */
int arc_stream_speakers(arc_stream* s, const arc_speaker** speakers);
int arc_stream_diarization(arc_stream* s, const arc_speaker_turn** turns);
/* the speaker of [start, end) seconds (a FINAL utterance's times): the turn so far that overlaps it most, or NULL.
 * Valid until the next call on s. */
const char* arc_stream_speaker(arc_stream* s, float start, float end);

/* ================================================================ Arc Transcriber Space (NFT2)
 *
 * The NFT2 backbone (the nft2/ folder of WhissleAI/arc-transcriber-space): a cache-aware streaming FastConformer with
 * its own RNNT decoder gives the words (biased towards the names list); Arc's decoder reads the same encoder (layer 9
 * tap, through Arc's adapters) with a context prompt and its inline tags are moved onto the RNNT words; Arc's net
 * links names to the list. What arc_audio's Arc.predict(wav, domain, transcript="nft2", names, context) returns:
 * utterances with words (RNNT times, decoder casing / punctuation, inline tags), names (tag, heard, entity, state,
 * p), numbers, sound events and the device-directed head (arc_utt.directed_head, .directed).
 *
 * The encoder and RNNT nets run on ONNX Runtime (build with make ARC_ORT=/path/to/onnxruntime); everything else is
 * libarc's C. Without ONNX Runtime arc_nft2_create fails with a message. One model runs one call at a time.
 *
 *   arc_nft2* m = arc_nft2_create("arc-transcriber-space/nft2", 4, err, sizeof err);
 *   arc_name_entry kb[] = {{"Tycho", "lunar_feature", NULL, NULL, 0}, {"Apollo 12", "mission", NULL, NULL, 0}};
 *   arc_nft2_set_names(m, kb, 2);
 *   arc_result* r = arc_nft2_transcribe(m, pcm, n, NULL);   // domain "assistant", context dom "space"
 *   ... r->utts[i].words[j].tag, r->names[k].entity / state
 *   arc_result_free(r);
 *   arc_nft2_destroy(m); */
typedef struct arc_nft2 arc_nft2;

/* dir: the nft2/ folder (config.json, adapters.safetensors, model.q4.safetensors, tags.json, decoder/,
 * encoder_onnx/). threads: encoder threads and libarc's pool (0 = 4). NULL + message on failure. */
arc_nft2* arc_nft2_create(const char* dir, int threads, char* err, size_t err_len);
void arc_nft2_destroy(arc_nft2* m);
/* The encoder / RNNT backend compiled in: "onnxruntime 1.23.2", or "none". */
const char* arc_nft2_backend(void);

/* The names list (kinds: lunar_feature site celestial spacecraft vehicle mission crew person place organization
 * product team law; aliases optional; see the names-list guide). Used three ways: RNNT biasing over every name and
 * alias, the decoder's kb prompt (names in order, about 56 tokens; unless the context has its own kb) and linking.
 * Kept until changed; n = 0 clears it. */
int arc_nft2_set_names(arc_nft2* m, const arc_name_entry* names, int n);

/* The decoder's context prompt (ctxprompt.build_prompt_v2): [dom][session][kb][prev]. */
typedef struct {
  const char* slot;      /* "S1" */
  const char* role;      /* "device_user" or NULL */
} arc_nft2_ses_speaker;
typedef struct {
  const char* value;     /* as written ("3", "1969") */
  const char* kind;      /* "count", "date" ... or NULL ("number") */
  int minutes;           /* age in minutes, -1 = not given */
} arc_nft2_ses_number;
typedef struct {
  const char* surface;   /* "Tycho" */
  const char* type;      /* person team org place law product term id (typed marker), other text, or NULL */
  int minutes;           /* -1 = not given */
} arc_nft2_ses_entity;
typedef struct {
  const char* dom;       /* topic words; NULL: the model's default ("space"); "": none */
  const char* prev;      /* earlier text, as said */
  int auto_prev;         /* 1: when prev is NULL, the last minute of this model's own transcripts */
  const char* const* kb;        /* an explicit kb (names / terms) and their kinds; n_kb = 0: the names list */
  const char* const* kb_kind;   /* may be NULL */
  int n_kb;
  const arc_nft2_ses_speaker* speakers;   /* session memory, most salient first; any may be 0 */
  int n_speakers;
  const arc_nft2_ses_number* numbers;
  int n_numbers;
  const arc_nft2_ses_entity* entities;
  int n_entities;
} arc_nft2_context;
/* Copies the context; NULL clears it (the model's default dom stays). Kept until changed. */
int arc_nft2_set_context(arc_nft2* m, const arc_nft2_context* c);

typedef struct {
  const char* domain;   /* Arc's domain; NULL = "assistant" (the app's) */
  int no_bias;          /* 1: RNNT words not biased towards the names list */
  int no_events;        /* 1: no sound events */
  int no_guard;         /* 1: keep every RNNT word (no no-speech gate, no tail / repetition guard) */
  int almost;           /* 1: name records with state "almost" for heard words that sound like a list entry */
  int nbest;            /* > 1: an RNNT beam of this width gives n-best texts (arc_utt.alts) and word alternatives;
                         * the words stay the greedy decode's. 0: off */
  /* streams only */
  int live;             /* 1: the open utterance is decoded as it arrives (one encoder chunk = 560 ms at a time, no
                         * re-decoding): PARTIAL, ENTITY and EOT events (arc_nft2_stream_events) */
  float playback_threshold;   /* with a reference (arc_nft2_stream_feed_ref): playback_leak at or above this marks an
                               * utterance as the device's own audio (arc_utt.playback); 0 = 0.7 */
  int playback_drop;    /* 1: such utterances are left out of the results (their FINAL event says so) */
  int echo_decode;      /* 1: utterances are decoded from the echo-cancelled audio (default: the microphone's) */
} arc_nft2_options;

/* Arc Tagger (arc-tagger-c, arctag.h) on each final: its JSON read-out of the utterance's words (entities, numbers, PII,
 * safety, intent, dialogue act, emotion, ...) in arc_utt.tagger_json. model_dir: the tagger's model folder; domain: its
 * profile ("kids" for children's turns in an app, "assistant", ...; NULL = "kids"); int8: 1 for arctag_int8.onnx. The
 * library must be built with ARC_WITH_ARCTAG (make ARC_WITH_ARCTAG=/path/to/arc-tagger-c); otherwise -1 and err says so.
 * model_dir NULL switches it off. */
int arc_nft2_set_tagger(arc_nft2* m, const char* model_dir, const char* domain, int int8, char* err, size_t errlen);

/* A whole clip (16 kHz mono float). Up to 20 s it is one utterance; longer clips are
 * cut by Arc's segmenter and each utterance decoded with 0.15 s of audio either side. NULL on failure
 * (arc_last_error). */
arc_result* arc_nft2_transcribe(arc_nft2* m, const float* pcm, size_t n, const arc_nft2_options* o);

/* Audio as it arrives: Arc's segmenter cuts utterances (0.4 s pauses, at most 15 s) and each closed utterance is
 * decoded at once. feed returns those utterances (words, tags; names / events come with flush), or a result with
 * n_utts 0. flush closes the open utterance and returns the whole stream: every utterance, names linked over all of
 * them, numbers, events and the device-directed head (as arc_nft2_transcribe on the same audio and utterances). The
 * stream keeps its audio until closed. Names and context are the model's at the time each utterance closes. */
typedef struct arc_nft2_stream arc_nft2_stream;
arc_nft2_stream* arc_nft2_stream_open(arc_nft2* m, const arc_nft2_options* o);
arc_result* arc_nft2_stream_feed(arc_nft2_stream* s, const float* pcm, size_t n);
arc_result* arc_nft2_stream_flush(arc_nft2_stream* s);
void arc_nft2_stream_close(arc_nft2_stream* s);

/* The device's own playback (TTS, sounds) as it is sent to the speaker, for echo cancellation, the per-utterance
 * playback_leak score and barge-in. t: the stream time (s, the microphone's clock: samples fed so far / 16000) at which
 * pcm[0] reaches the speaker; t < 0: right after the previous reference audio. Feed it before or while it plays; an echo
 * path up to 512 ms after t is learnt (a per-frequency NLMS filter). */
int arc_nft2_stream_feed_ref(arc_nft2_stream* s, const float* pcm, size_t n, double t);

/* Replace the names list mid-stream: the RNNT bias of the open utterance switches at once (live decoding restarts its
 * bias state), finals and linking use the new list from the next utterance that closes (= arc_nft2_set_names). */
int arc_nft2_stream_set_names(arc_nft2_stream* s, const arc_name_entry* names, int n);

/* Close the open utterance now (push-to-talk release, the app's own end of speech): its final is decoded at once and
 * returned (as feed). The stream stays open. */
arc_result* arc_nft2_stream_end_turn(arc_nft2_stream* s);

/* Stream events, produced by the last feed / feed_ref / end_turn / flush call (valid until the next call on s). */
enum {
  ARC_EV_PARTIAL = 1,   /* the open utterance's words so far (options.live); text, n_stable */
  ARC_EV_ENTITY = 2,    /* a list name, a sound-alike ("almost") or a quantity in the stable words (options.live) */
  ARC_EV_EOT = 3,       /* end-of-turn probability now (options.live: every feed while an utterance is open; and at
                         * each final) */
  ARC_EV_BARGE_IN = 4,  /* speech that is not the playback, while the device plays (needs a reference) */
  ARC_EV_FINAL = 5      /* an utterance closed (its words are in the returned arc_result) */
};
typedef struct {
  int type;             /* ARC_EV_* */
  float t;              /* stream time (s) of the audio the event is based on */
  int utt;              /* the utterance's index in the stream (0-based; the open one gets the next index) */
  const char* text;     /* PARTIAL / FINAL: the words; ENTITY: the heard words */
  int n_stable;         /* PARTIAL: leading words that no later PARTIAL changes (the last word may still grow) */
  const char* kind;     /* ENTITY: "name", "almost", "quantity"; FINAL: NULL, or "playback" (playback_leak >= threshold)
                         * / "dropped" (and options.playback_drop) */
  const char* entity;   /* ENTITY name / almost: the list entry; quantity: its written form ("75%") */
  int kb_index;         /* ENTITY name / almost: the entry's index in the list, else -1 */
  float p;              /* ENTITY: 1 for a list spelling, the sound-alike p for almost, 1 for a quantity; EOT: the
                         * probability; BARGE_IN: 1 - playback_leak of the last 0.3 s; FINAL: playback_leak or -1 */
  float start, end;     /* the words' times (s) */
  const char* qtype;    /* ENTITY quantity: "percent", "distance", ...; else NULL */
  double value;         /* ENTITY quantity: its value */
  const char* unit;     /* ENTITY quantity: its unit or NULL */
  int relative;         /* ENTITY quantity: +1 "more", -1 "less", 0 */
} arc_nft2_event;
const arc_nft2_event* arc_nft2_stream_events(arc_nft2_stream* s, int* n);

#ifdef __cplusplus
}
#endif
#endif
