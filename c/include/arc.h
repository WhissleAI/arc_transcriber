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
} arc_word;

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
  float command_complete;    /* discourse.assistant.command_complete: an instant query (ev_at), here its highest frame
                              * over the utterance through its closing pause (arc_audio reports it per frame) */
  arc_label intent;          /* best discourse.intent.* */
  arc_label act;             /* best discourse.act.* */
  float directed_head;       /* the separate device-directed head on the encoder features (beta) */
  int n_words;               /* words with times (arc_options.words / names / numbers), else 0 */
  const arc_word* words;
  /* n-best: the final decode's finished beams, best first, distinct texts (alts[0] is the decoded text before the
   * loop fix; 1 with greedy decoding). Their scores are the beam search's length-normalised log-probabilities. */
  int n_alts;
  const char* const* alts;
  const float* alt_scores;
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
  float word_wait_s;         /* EXTERNAL: wait at most ... s after an utterance for the recogniser's final words
                              * (0: 6, arc_audio's setting for external words) */
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
 * SpeechRecognizer): such words go to the utterance open now, or else to the last one, spread over its time. */
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

#ifdef __cplusplus
}
#endif
#endif
