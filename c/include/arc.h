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
  float command_complete;    /* discourse.assistant.command_complete */
  arc_label intent;          /* best discourse.intent.* */
  arc_label act;             /* best discourse.act.* */
  float directed_head;       /* the separate device-directed head on the encoder features (beta) */
  int n_words;               /* words with times (arc_options.words / names / numbers), else 0 */
  const arc_word* words;
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

/* A whole clip (16 kHz mono float, any length). NULL on failure (arc_last_error). */
arc_result* arc_transcribe(arc_engine* e, const float* pcm, size_t n, const arc_options* opt);
void arc_result_free(arc_result* r);
const char* arc_last_error(void);

#ifdef __cplusplus
}
#endif
#endif
