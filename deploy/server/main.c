/* arc-serve: Arc Mini (libarc) as an HTTP + WebSocket service. No Python, no ONNX Runtime, no network at run time.
 *
 *   arc-serve [serve] [--port N] [--host ADDR] [--model DIR]
 *
 * Routes (deploy/openapi.yaml): POST /v1/audio/transcriptions (OpenAI-compatible, plus Arc's words, inline tags, linked
 * names, number values, utterance tags and events), WS /v1/realtime, GET /health, GET /v1/health/live,
 * GET /v1/health/ready, GET /v1/models, and SageMaker's GET /ping, POST /invocations, GET /execution-parameters.
 * Built with -DARC_SERVE_TAGGER (the combined image) it also links Arc Tagger (arc-tagger-c's tagsrv.c): POST /v1/tags,
 * tags on transcripts (tags=true) and on realtime finals.
 *
 * Environment: PORT (8080), HOST (0.0.0.0), ARC_MODEL_DIR (else the first of /opt/ml/model, /repository,
 * /opt/arc/model holding Arc Mini, also in a mini/ subfolder), ARC_MODEL_ID (arc-mini), ARC_WORKERS (engines, 2),
 * ARC_THREADS (threads per engine; default cores / workers), ARC_DOMAIN (unknown), ARC_MAX_AUDIO_S (900),
 * ARC_MAX_BODY_MB (100), ARC_QUEUE_TIMEOUT_S (60), ARC_MAX_CONNECTIONS (64), ARC_ACCESS_LOG (1).
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0 with the Whissle terms; see LICENSE. */
#define _GNU_SOURCE   /* strcasestr, M_PI, strdup, strtok_r, clock_gettime on glibc with -std=c11 */
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "arc.h"
#include "audio.h"
#include "cJSON.h"
#include "http.h"
#ifdef ARC_SERVE_TAGGER
#include "arctag.h"
#include "tagsrv.h"
#endif

#ifndef ARC_DEPLOY_VERSION
#define ARC_DEPLOY_VERSION "0.1.0"
#endif

/* ------------------------------------------------------------------ settings and the engine pool */
static struct {
  char model_dir[1200];
  char model_id[256];
  char domain[64];
  int workers, threads;
  double max_audio_s;
  int queue_timeout_s;
  arc_engine** e;
  int* busy;
  pthread_mutex_t mu;
  pthread_cond_t cv;
  int ready;
  int tagger;           /* the tagger is linked and loaded */
  time_t started;
} S = {.mu = PTHREAD_MUTEX_INITIALIZER, .cv = PTHREAD_COND_INITIALIZER};

static const char* env_or(const char* k, const char* d) {
  const char* v = getenv(k);
  return v && *v ? v : d;
}

static int acquire_engine(int timeout_s) {
  struct timespec dl;
  clock_gettime(CLOCK_REALTIME, &dl);
  dl.tv_sec += timeout_s;
  pthread_mutex_lock(&S.mu);
  for (;;) {
    for (int i = 0; i < S.workers; ++i)
      if (!S.busy[i]) {
        S.busy[i] = 1;
        pthread_mutex_unlock(&S.mu);
        return i;
      }
    if (pthread_cond_timedwait(&S.cv, &S.mu, &dl) == ETIMEDOUT) {
      pthread_mutex_unlock(&S.mu);
      return -1;
    }
  }
}

/* back to the server's state: no names list, no context, no memory */
static void release_engine(int i) {
  arc_engine* e = S.e[i];
  arc_set_names(e, NULL, 0);
  arc_set_context(e, NULL, NULL, 0, NULL, 0, NULL, NULL, 0);
  arc_memory(e, 0);
  arc_memory_reset(e);
  pthread_mutex_lock(&S.mu);
  S.busy[i] = 0;
  pthread_cond_signal(&S.cv);
  pthread_mutex_unlock(&S.mu);
}

static int is_model_dir(const char* d) {
  char p[1300];
  struct stat st;
  snprintf(p, sizeof p, "%s/config.json", d);
  if (stat(p, &st)) return 0;
  snprintf(p, sizeof p, "%s/model.q4.safetensors", d);
  if (stat(p, &st)) return 0;
  snprintf(p, sizeof p, "%s/moonshine-streaming-tiny", d);
  return stat(p, &st) == 0;
}

static int find_model_dir(const char* given, char* out, size_t n) {
  const char* roots[] = {given, getenv("ARC_MODEL_DIR"), "/opt/ml/model", "/repository", "/opt/arc/model", NULL};
  const char* subs[] = {"", "/mini", "/arc-v1.1/mini", "/edge", NULL};
  for (int i = 0; i < 5; ++i) {
    if (!roots[i] || !*roots[i]) continue;
    for (int k = 0; subs[k]; ++k) {
      snprintf(out, n, "%s%s", roots[i], subs[k]);
      if (is_model_dir(out)) return 0;
    }
    if (i < 2) { snprintf(out, n, "%s", roots[i]); return -1; }
  }
  return -1;
}

/* ------------------------------------------------------------------ request options */
/* All request options end up in one cJSON object: multipart fields and query parameters as strings, JSON body keys as
 * they are, SageMaker custom attributes (k=v;k=v) as strings. */
static const char* opt_str(const cJSON* o, const char* k) {
  const cJSON* v = cJSON_GetObjectItemCaseSensitive(o, k);
  return cJSON_IsString(v) ? v->valuestring : NULL;
}

static int opt_bool(const cJSON* o, const char* k, int def) {
  const cJSON* v = cJSON_GetObjectItemCaseSensitive(o, k);
  if (cJSON_IsBool(v)) return cJSON_IsTrue(v);
  if (cJSON_IsNumber(v)) return v->valuedouble != 0;
  if (cJSON_IsString(v)) {
    const char* s = v->valuestring;
    if (!strcasecmp(s, "true") || !strcmp(s, "1") || !strcasecmp(s, "yes") || !strcasecmp(s, "on")) return 1;
    if (!strcasecmp(s, "false") || !strcmp(s, "0") || !strcasecmp(s, "no") || !strcasecmp(s, "off")) return 0;
  }
  return def;
}

static int opt_int(const cJSON* o, const char* k, int def) {
  const cJSON* v = cJSON_GetObjectItemCaseSensitive(o, k);
  if (cJSON_IsNumber(v)) return v->valueint;
  if (cJSON_IsString(v) && *v->valuestring) return atoi(v->valuestring);
  return def;
}

static void opt_set_str(cJSON* o, const char* k, const char* v) {
  if (cJSON_GetObjectItemCaseSensitive(o, k)) return;   /* the first source wins */
  cJSON_AddStringToObject(o, k, v);
}

static void add_query_opts(const http_req* r, cJSON* o) {
  static const char* K[] = {"model", "prompt", "names", "response_format", "language", "domain", "events", "words", "trunk",
                            "numbers", "link_names", "one_utterance", "beams", "terms", "topic", "tags", "tagger_domain",
                            "sample_rate", "channels", "partials", "frames", "delay_ms", NULL};
  char v[4096];
  for (int i = 0; K[i]; ++i)
    if (http_query(r, K[i], v, sizeof v)) opt_set_str(o, K[i], v);
}

static void add_custom_attributes(const http_req* r, cJSON* o) {
  const char* ca = http_header(r, "X-Amzn-SageMaker-Custom-Attributes");
  if (!ca) return;
  char* s = strdup(ca);
  char* save = NULL;
  for (char* kv = strtok_r(s, ";,", &save); kv; kv = strtok_r(NULL, ";,", &save)) {
    while (*kv == ' ') ++kv;
    char* eq = strchr(kv, '=');
    if (!eq) continue;
    *eq = 0;
    opt_set_str(o, kv, eq + 1);
  }
  free(s);
}

/* the names list: a JSON array (strings or {name, kind, group, aliases}), or names separated by newlines / commas */
typedef struct {
  arc_name_entry* e;
  int n;
  cJSON* keep;   /* owns the strings */
} names_t;

static void names_free(names_t* nl) {
  for (int i = 0; i < nl->n; ++i) free((void*)nl->e[i].aliases);
  free(nl->e);
  cJSON_Delete(nl->keep);
  memset(nl, 0, sizeof *nl);
}

static int names_parse(const cJSON* v, names_t* nl, char* err, size_t el) {
  memset(nl, 0, sizeof *nl);
  if (!v) return 0;
  cJSON* arr = NULL;
  if (cJSON_IsArray(v)) arr = cJSON_Duplicate(v, 1);
  else if (cJSON_IsString(v)) {
    const char* s = v->valuestring;
    while (isspace((unsigned char)*s)) ++s;
    if (*s == '[') {
      arr = cJSON_Parse(s);
      if (!cJSON_IsArray(arr)) { cJSON_Delete(arr); snprintf(err, el, "names: invalid JSON array"); return -1; }
    } else if (*s) {
      arr = cJSON_CreateArray();
      char* c = strdup(s);
      char* save = NULL;
      for (char* t = strtok_r(c, "\n,", &save); t; t = strtok_r(NULL, "\n,", &save)) {
        while (isspace((unsigned char)*t)) ++t;
        char* e = t + strlen(t);
        while (e > t && isspace((unsigned char)e[-1])) *--e = 0;
        if (*t) cJSON_AddItemToArray(arr, cJSON_CreateString(t));
      }
      free(c);
    }
  } else { snprintf(err, el, "names: a JSON array or a string"); return -1; }
  if (!arr) return 0;
  int n = cJSON_GetArraySize(arr);
  if (n > 5000) { cJSON_Delete(arr); snprintf(err, el, "names: at most 5000 entries"); return -1; }
  nl->keep = arr;
  nl->e = calloc((size_t)n + 1, sizeof *nl->e);
  const cJSON* it;
  cJSON_ArrayForEach(it, arr) {
    arc_name_entry* x = &nl->e[nl->n];
    if (cJSON_IsString(it)) x->name = it->valuestring;
    else if (cJSON_IsObject(it)) {
      x->name = opt_str(it, "name");
      if (!x->name) continue;
      x->kind = opt_str(it, "kind");
      x->group = opt_str(it, "group");
      const cJSON* al = cJSON_GetObjectItemCaseSensitive(it, "aliases");
      if (cJSON_IsArray(al) && cJSON_GetArraySize(al)) {
        int na = cJSON_GetArraySize(al), k = 0;
        const char** a = calloc((size_t)na, sizeof(char*));
        const cJSON* s;
        cJSON_ArrayForEach(s, al) if (cJSON_IsString(s)) a[k++] = s->valuestring;
        x->aliases = a;
        x->n_aliases = k;
      }
    } else continue;
    nl->n++;
  }
  return 0;
}

/* a list of strings: JSON array or comma-separated */
static int str_list(const cJSON* v, const char*** out, cJSON** keep) {
  *out = NULL;
  *keep = NULL;
  if (!v) return 0;
  cJSON* arr = NULL;
  if (cJSON_IsArray(v)) arr = cJSON_Duplicate(v, 1);
  else if (cJSON_IsString(v)) {
    if (v->valuestring[0] == '[') arr = cJSON_Parse(v->valuestring);
    else {
      arr = cJSON_CreateArray();
      char* c = strdup(v->valuestring);
      char* save = NULL;
      for (char* t = strtok_r(c, ",", &save); t; t = strtok_r(NULL, ",", &save)) {
        while (*t == ' ') ++t;
        if (*t) cJSON_AddItemToArray(arr, cJSON_CreateString(t));
      }
      free(c);
    }
  }
  if (!cJSON_IsArray(arr)) { cJSON_Delete(arr); return 0; }
  int n = 0;
  const char** a = calloc((size_t)cJSON_GetArraySize(arr) + 1, sizeof(char*));
  const cJSON* s;
  cJSON_ArrayForEach(s, arr) if (cJSON_IsString(s)) a[n++] = s->valuestring;
  *out = a;
  *keep = arr;
  return n;
}

/* ------------------------------------------------------------------ JSON output */
static double rnd(double x, int d) {
  double p = pow(10, d);
  return round(x * p) / p;
}

static void add_num(cJSON* o, const char* k, double v, int d) { cJSON_AddNumberToObject(o, k, rnd(v, d)); }
static void add_str_or_null(cJSON* o, const char* k, const char* v) {
  if (v) cJSON_AddStringToObject(o, k, v);
  else cJSON_AddNullToObject(o, k);
}
static void add_p_or_null(cJSON* o, const char* k, float p) {
  if (p >= 0) add_num(o, k, p, 4);
  else cJSON_AddNullToObject(o, k);
}

static cJSON* words_json(const arc_word* w, int n, const char* key, int with_tags) {
  cJSON* a = cJSON_CreateArray();
  for (int i = 0; i < n; ++i) {
    cJSON* o = cJSON_CreateObject();
    cJSON_AddStringToObject(o, key, w[i].text);
    add_num(o, "start", w[i].start, 3);
    add_num(o, "end", w[i].end, 3);
    if (with_tags && w[i].tag) {
      cJSON_AddStringToObject(o, "tag", w[i].tag);
      cJSON_AddBoolToObject(o, "tag_begin", w[i].tag_begin);
    }
    cJSON_AddItemToArray(a, o);
  }
  return a;
}

static char* full_text(const arc_result* r) {
  size_t L = 1;
  for (int i = 0; i < r->n_utts; ++i) L += strlen(r->utts[i].text) + 1;
  char* s = malloc(L);
  s[0] = 0;
  for (int i = 0; i < r->n_utts; ++i) {
    if (!*r->utts[i].text) continue;
    if (*s) strcat(s, " ");
    strcat(s, r->utts[i].text);
  }
  return s;
}

static cJSON* utt_json(const arc_utt* u, int id) {
  cJSON* o = cJSON_CreateObject();
  if (id >= 0) cJSON_AddNumberToObject(o, "id", id);
  add_num(o, "start", u->start, 3);
  add_num(o, "end", u->end, 3);
  cJSON_AddStringToObject(o, "text", u->text ? u->text : "");
  cJSON* tg = cJSON_AddArrayToObject(o, "tags");
  for (int k = 0; k < u->n_tags; ++k) {
    cJSON* t = cJSON_CreateObject();
    cJSON_AddStringToObject(t, "tag", u->tags[k].tag);
    cJSON_AddStringToObject(t, "text", u->tags[k].text ? u->tags[k].text : "");
    cJSON_AddNumberToObject(t, "word_begin", u->tags[k].word_begin);
    cJSON_AddNumberToObject(t, "word_end", u->tags[k].word_end);
    cJSON_AddItemToArray(tg, t);
  }
  if (u->intent.label) {
    cJSON* t = cJSON_AddObjectToObject(o, "intent");
    cJSON_AddStringToObject(t, "label", u->intent.label);
    add_num(t, "p", u->intent.p, 4);
  } else cJSON_AddNullToObject(o, "intent");
  if (u->act.label) {
    cJSON* t = cJSON_AddObjectToObject(o, "act");
    cJSON_AddStringToObject(t, "label", u->act.label);
    add_num(t, "p", u->act.p, 4);
  } else cJSON_AddNullToObject(o, "act");
  add_p_or_null(o, "device_directed", u->device_directed);
  add_p_or_null(o, "command_complete", u->command_complete);
  add_p_or_null(o, "directed_head", u->directed_head);
  cJSON_AddItemToObject(o, "words", words_json(u->words, u->n_words, "word", 1));
  if (u->n_alts > 1) {
    cJSON* a = cJSON_AddArrayToObject(o, "alternatives");
    for (int k = 0; k < u->n_alts; ++k) {
      cJSON* x = cJSON_CreateObject();
      cJSON_AddStringToObject(x, "text", u->alts[k]);
      add_num(x, "score", u->alt_scores[k], 4);
      cJSON_AddItemToArray(a, x);
    }
  }
  return o;
}

static cJSON* names_json(const arc_result* r) {
  cJSON* a = cJSON_CreateArray();
  for (int i = 0; i < r->n_names; ++i) {
    const arc_name* m = &r->names[i];
    cJSON* o = cJSON_CreateObject();
    cJSON_AddStringToObject(o, "heard", m->heard ? m->heard : "");
    add_str_or_null(o, "entity", m->entity);
    cJSON_AddStringToObject(o, "state", m->state ? m->state : "");
    cJSON_AddStringToObject(o, "tag", m->tag ? m->tag : "");
    if (m->kb_index >= 0) cJSON_AddNumberToObject(o, "kb_index", m->kb_index);
    else cJSON_AddNullToObject(o, "kb_index");
    add_p_or_null(o, "p", m->p);
    add_num(o, "start", m->start, 3);
    add_num(o, "end", m->end, 3);
    add_str_or_null(o, "src", m->src);
    add_str_or_null(o, "via", m->via);
    cJSON_AddBoolToObject(o, "show", m->show);
    cJSON_AddItemToArray(a, o);
  }
  return a;
}

static void add_value(cJSON* o, const char* k, int has, double v, int is_int) {
  if (!has) cJSON_AddNullToObject(o, k);
  else if (is_int) cJSON_AddNumberToObject(o, k, (double)llround(v));
  else cJSON_AddNumberToObject(o, k, v);
}

static cJSON* numbers_json(const arc_result* r) {
  cJSON* a = cJSON_CreateArray();
  for (int i = 0; i < r->n_numbers; ++i) {
    const arc_number* m = &r->numbers[i];
    cJSON* o = cJSON_CreateObject();
    cJSON_AddStringToObject(o, "heard", m->heard ? m->heard : "");
    add_str_or_null(o, "kind", m->kind);
    add_str_or_null(o, "unit", m->unit);
    add_value(o, "value", m->has_value, m->value, m->value_int);
    add_value(o, "value2", m->has_value2, m->value2, m->value2_int);
    cJSON_AddStringToObject(o, "state", m->state ? m->state : "");
    add_num(o, "start", m->start, 3);
    add_num(o, "end", m->end, 3);
    cJSON_AddItemToArray(a, o);
  }
  return a;
}

static cJSON* events_json(const arc_event* ev, int n) {
  cJSON* a = cJSON_CreateArray();
  for (int i = 0; i < n; ++i) {
    cJSON* o = cJSON_CreateObject();
    cJSON_AddStringToObject(o, "label", ev[i].label);
    add_num(o, "start", ev[i].start, 2);
    add_num(o, "end", ev[i].end, 2);
    add_num(o, "p", ev[i].p, 4);
    cJSON_AddItemToArray(a, o);
  }
  return a;
}

/* arc-transcribe --json's record (arc_audio's keys: tools/dump_names.py writes the same), for parity checks */
static cJSON* arc_json_record(const arc_result* r, const char* domain) {
  cJSON* o = cJSON_CreateObject();
  cJSON_AddStringToObject(o, "domain", domain);
  cJSON* tr = cJSON_AddArrayToObject(o, "transcript");
  for (int i = 0; i < r->n_utts; ++i) {
    const arc_utt* u = &r->utts[i];
    cJSON* x = cJSON_CreateObject();
    add_num(x, "start", u->start, 1);
    add_num(x, "end", u->end, 1);
    cJSON_AddStringToObject(x, "text", u->text);
    cJSON_AddItemToObject(x, "words", words_json(u->words, u->n_words, "text", 1));
    cJSON* tg = cJSON_AddArrayToObject(x, "tags");
    for (int k = 0; k < u->n_tags; ++k) {
      cJSON* t = cJSON_CreateArray();
      cJSON_AddItemToArray(t, cJSON_CreateString(u->tags[k].tag));
      cJSON_AddItemToArray(t, cJSON_CreateString(u->tags[k].text));
      cJSON_AddItemToArray(tg, t);
    }
    cJSON_AddItemToArray(tr, x);
  }
  cJSON* nm = cJSON_AddArrayToObject(o, "names");
  for (int i = 0; i < r->n_names; ++i) {
    const arc_name* m = &r->names[i];
    cJSON* x = cJSON_CreateObject();
    add_num(x, "start", m->start, 3);
    add_num(x, "end", m->end, 3);
    cJSON_AddStringToObject(x, "tag", m->tag);
    cJSON_AddStringToObject(x, "heard", m->heard);
    add_str_or_null(x, "entity", m->entity);
    cJSON_AddStringToObject(x, "state", m->state);
    cJSON_AddStringToObject(x, "src", m->src);
    if (m->p >= 0) add_num(x, "p", m->p, 4);
    if (m->model_p >= 0) {
      cJSON* md = cJSON_AddObjectToObject(x, "model");
      add_str_or_null(md, "entity", m->model_entity);
      add_num(md, "p", m->model_p, 4);
    }
    if (m->via) cJSON_AddStringToObject(x, "via", m->via);
    if (m->snap_score >= 0) add_num(x, "snap_score", m->snap_score, 3);
    if (m->kb_index >= 0) cJSON_AddNumberToObject(x, "kb_index", m->kb_index);
    if (m->span_p >= 0) add_num(x, "span_p", m->span_p, 4);
    cJSON_AddBoolToObject(x, "show", m->show);
    cJSON* ws = cJSON_AddArrayToObject(x, "words");
    for (int k = 0; k < m->n_words; ++k) {
      cJSON* w = cJSON_CreateObject();
      cJSON_AddStringToObject(w, "w", m->words[k].text);
      add_num(w, "start", m->words[k].start, 3);
      add_num(w, "end", m->words[k].end, 3);
      cJSON_AddItemToArray(ws, w);
    }
    cJSON_AddItemToArray(nm, x);
  }
  cJSON* nu = cJSON_AddArrayToObject(o, "numbers");
  for (int i = 0; i < r->n_numbers; ++i) {
    const arc_number* m = &r->numbers[i];
    cJSON* x = cJSON_CreateObject();
    add_num(x, "start", m->start, 3);
    add_num(x, "end", m->end, 3);
    add_str_or_null(x, "kind", m->kind);
    add_str_or_null(x, "unit", m->unit);
    cJSON_AddStringToObject(x, "heard", m->heard);
    add_value(x, "value", m->has_value, m->value, m->value_int);
    add_value(x, "value2", m->has_value2, m->value2, m->value2_int);
    cJSON_AddStringToObject(x, "state", m->state);
    add_str_or_null(x, "kind_head", m->kind_head);
    cJSON_AddItemToArray(nu, x);
  }
  cJSON_AddItemToObject(o, "audio_events", events_json(r->events, r->n_events));
  return o;
}

static void fmt_ts(char* b, size_t n, double t, char sep) {
  if (t < 0) t = 0;
  long ms = lround(t * 1000);
  snprintf(b, n, "%02ld:%02ld:%02ld%c%03ld", ms / 3600000, ms / 60000 % 60, ms / 1000 % 60, sep, ms % 1000);
}

static char* subtitles(const arc_result* r, int vtt) {
  size_t cap = 256, len = 0;
  char* s = malloc(cap);
  s[0] = 0;
  if (vtt) len += (size_t)snprintf(s, cap, "WEBVTT\n\n");
  int k = 0;
  for (int i = 0; i < r->n_utts; ++i) {
    const arc_utt* u = &r->utts[i];
    if (!*u->text) continue;
    char a[32], b[32];
    fmt_ts(a, sizeof a, u->start, vtt ? '.' : ',');
    fmt_ts(b, sizeof b, u->end, vtt ? '.' : ',');
    size_t need = strlen(u->text) + 96;
    while (len + need >= cap) cap *= 2;
    s = realloc(s, cap);
    if (vtt) len += (size_t)snprintf(s + len, cap - len, "%s --> %s\n%s\n\n", a, b, u->text);
    else len += (size_t)snprintf(s + len, cap - len, "%d\n%s --> %s\n%s\n\n", ++k, a, b, u->text);
  }
  return s;
}

/* ------------------------------------------------------------------ transcription */
typedef struct {
  int status;
  char err[1024];
  const char* err_type;
} outcome;

static void fail(outcome* oc, int status, const char* type, const char* msg) {
  oc->status = status;
  oc->err_type = type;
  snprintf(oc->err, sizeof oc->err, "%s", msg);
}

/* decode one audio payload (bytes + an optional declared content type) to 16 kHz mono */
static float* decode_audio(const unsigned char* d, size_t n, const char* ctype, const cJSON* opts, size_t* ns, outcome* oc) {
  char err[512] = "";
  float* pcm = NULL;
  if (n == 0) { fail(oc, 400, "invalid_request_error", "empty audio"); return NULL; }
  if (n >= 4 && (!memcmp(d, "fLaC", 4) || !memcmp(d, "OggS", 4) || !memcmp(d, "ID3", 3) || (d[0] == 0xFF && (d[1] & 0xE0) == 0xE0) ||
                 (n >= 8 && !memcmp(d + 4, "ftyp", 4)) || !memcmp(d, "\x1a\x45\xdf\xa3", 4))) {
    fail(oc, 415, "invalid_request_error",
         "this server decodes WAV and raw PCM16 only (no MP3 / FLAC / Ogg / MP4 / WebM decoder is linked); convert first, "
         "for example: ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav");
    return NULL;
  }
  if (audio_is_wav(d, n)) {
    int rate = 0;
    pcm = audio_decode_wav(d, n, ns, &rate, err, sizeof err);
  } else if (!ctype || !strncasecmp(ctype, "audio/pcm", 9) || !strncasecmp(ctype, "audio/l16", 9) ||
             !strncasecmp(ctype, "audio/raw", 9) || !strncasecmp(ctype, "application/octet-stream", 24) ||
             !strncasecmp(ctype, "audio/x-raw", 11)) {
    int rate = opt_int(opts, "sample_rate", 16000), ch = opt_int(opts, "channels", 1);
    if (ctype) {   /* audio/L16;rate=8000;channels=2 */
      const char* p = strcasestr(ctype, "rate=");
      if (p) rate = atoi(p + 5);
      p = strcasestr(ctype, "channels=");
      if (p) ch = atoi(p + 9);
    }
    pcm = audio_decode_pcm16(d, n, rate, ch, ns, err, sizeof err);
  } else {
    char m[640];
    snprintf(m, sizeof m, "unsupported audio type %s: send audio/wav (any PCM / float WAV) or audio/pcm (16-bit LE, ?sample_rate=)", ctype);
    fail(oc, 415, "invalid_request_error", m);
    return NULL;
  }
  if (!pcm) { fail(oc, 400, "invalid_request_error", err); return NULL; }
  if ((double)*ns / ARC_SAMPLE_RATE > S.max_audio_s) {
    char m[256];
    snprintf(m, sizeof m, "audio is %.0f s; this server takes at most %.0f s per request (ARC_MAX_AUDIO_S)", (double)*ns / ARC_SAMPLE_RATE, S.max_audio_s);
    free(pcm);
    fail(oc, 413, "invalid_request_error", m);
    return NULL;
  }
  return pcm;
}

#ifdef ARC_SERVE_TAGGER
/* the transcript's words -> arctag words (times, eou at each utterance end) */
static char* result_words_for_tagger(const arc_result* r) {
  cJSON* a = cJSON_CreateArray();
  for (int i = 0; i < r->n_utts; ++i) {
    const arc_utt* u = &r->utts[i];
    int before = cJSON_GetArraySize(a);
    if (u->n_words) {
      for (int k = 0; k < u->n_words; ++k) {
        cJSON* w = cJSON_CreateObject();
        cJSON_AddStringToObject(w, "w", u->words[k].text);
        add_num(w, "t0", u->words[k].start, 3);
        add_num(w, "t1", u->words[k].end, 3);
        cJSON_AddItemToArray(a, w);
      }
    } else {
      char* c = strdup(u->text ? u->text : "");
      char* save = NULL;
      for (char* t = strtok_r(c, " ", &save); t; t = strtok_r(NULL, " ", &save)) {
        cJSON* w = cJSON_CreateObject();
        cJSON_AddStringToObject(w, "w", t);
        cJSON_AddItemToArray(a, w);
      }
      free(c);
    }
    int after = cJSON_GetArraySize(a);
    if (after > before) cJSON_AddBoolToObject(cJSON_GetArrayItem(a, after - 1), "eou", 1);
  }
  char* s = cJSON_GetArraySize(a) ? cJSON_PrintUnformatted(a) : NULL;
  cJSON_Delete(a);
  return s;
}

static cJSON* tagger_opts(const cJSON* opts) {
  const cJSON* t = cJSON_GetObjectItemCaseSensitive(opts, "tagger");
  cJSON* o = NULL;
  if (cJSON_IsObject(t)) o = cJSON_Duplicate(t, 1);
  else if (cJSON_IsString(t) && t->valuestring[0] == '{') o = cJSON_Parse(t->valuestring);
  const char* d = opt_str(opts, "tagger_domain");
  if (d) {
    if (!o) o = cJSON_CreateObject();
    cJSON_DeleteItemFromObjectCaseSensitive(o, "domain");
    cJSON_AddStringToObject(o, "domain", d);
  }
  return o;
}
#endif

/* Transcribe one clip with the request's options; returns the response body (malloc'd) and its content type. */
static char* transcribe(const float* pcm, size_t n, const cJSON* opts, const char* default_format, const char** ctype, outcome* oc) {
  const char* fmt = opt_str(opts, "response_format");
  if (!fmt || !*fmt) fmt = default_format;
  if (strcmp(fmt, "json") && strcmp(fmt, "text") && strcmp(fmt, "verbose_json") && strcmp(fmt, "srt") && strcmp(fmt, "vtt") &&
      strcmp(fmt, "arc_json")) {
    fail(oc, 400, "invalid_request_error", "response_format: json | text | verbose_json | srt | vtt | arc_json");
    return NULL;
  }
  const char* lang = opt_str(opts, "language");
  if (lang && *lang && strncasecmp(lang, "en", 2)) {
    fail(oc, 400, "invalid_request_error", "Arc Mini transcribes English only (language=en)");
    return NULL;
  }
  const char* domain = opt_str(opts, "domain");
  if (!domain || !*domain) domain = S.domain;
  names_t nl;
  char err[512];
  if (names_parse(cJSON_GetObjectItemCaseSensitive(opts, "names"), &nl, err, sizeof err)) {
    fail(oc, 400, "invalid_request_error", err);
    return NULL;
  }
  const char** terms;
  cJSON* terms_keep;
  int n_terms = str_list(cJSON_GetObjectItemCaseSensitive(opts, "terms"), &terms, &terms_keep);
  const char* prompt = opt_str(opts, "prompt");
  const char* topic = opt_str(opts, "topic");
  int trunk = opt_bool(opts, "trunk", 1);
  arc_options o = {.domain = domain,
                   .one_utterance = opt_bool(opts, "one_utterance", 0),
                   .beams = opt_int(opts, "beams", 0),
                   .trunk = trunk,
                   .events = opt_bool(opts, "events", trunk),
                   .bias_names = nl.n > 0 || n_terms > 0,
                   .words = opt_bool(opts, "words", 1),
                   .names = opt_bool(opts, "link_names", trunk),
                   .numbers = opt_bool(opts, "numbers", trunk)};
  if (o.beams < 0 || o.beams > 16) o.beams = 0;
  int wi = acquire_engine(S.queue_timeout_s);
  if (wi < 0) {
    names_free(&nl);
    free(terms);
    cJSON_Delete(terms_keep);
    fail(oc, 503, "server_busy", "all transcription workers are busy; retry");
    return NULL;
  }
  arc_engine* e = S.e[wi];
  int rc = 0;
  if (nl.n) rc = arc_set_names(e, nl.e, nl.n);
  if (!rc && (prompt || topic || n_terms)) rc = arc_set_context(e, NULL, NULL, 0, terms, n_terms, topic, prompt, 0);
  if (!rc && n_terms && !nl.n) rc = arc_set_bias(e, terms, n_terms);
  arc_result* r = rc ? NULL : arc_transcribe(e, pcm, n, &o);
  if (!r) snprintf(err, sizeof err, "transcription failed: %s", arc_last_error());
  release_engine(wi);
  names_free(&nl);
  free(terms);
  cJSON_Delete(terms_keep);
  if (!r) { fail(oc, rc ? 400 : 500, rc ? "invalid_request_error" : "server_error", err); return NULL; }

  char* body = NULL;
  *ctype = "application/json";
  double audio_s = (double)n / ARC_SAMPLE_RATE;
  if (!strcmp(fmt, "text")) {
    char* t = full_text(r);
    body = malloc(strlen(t) + 2);
    sprintf(body, "%s\n", t);
    free(t);
    *ctype = "text/plain; charset=utf-8";
  } else if (!strcmp(fmt, "srt") || !strcmp(fmt, "vtt")) {
    body = subtitles(r, !strcmp(fmt, "vtt"));
    *ctype = !strcmp(fmt, "vtt") ? "text/vtt; charset=utf-8" : "application/x-subrip; charset=utf-8";
  } else {
    cJSON* out;
    if (!strcmp(fmt, "arc_json")) out = arc_json_record(r, domain);
    else {
      out = cJSON_CreateObject();
      char* t = full_text(r);
      if (!strcmp(fmt, "verbose_json")) {
        cJSON_AddStringToObject(out, "task", "transcribe");
        cJSON_AddStringToObject(out, "language", "english");
        add_num(out, "duration", audio_s, 3);
      }
      cJSON_AddStringToObject(out, "text", t);
      free(t);
      if (!strcmp(fmt, "verbose_json")) {
        cJSON* segs = cJSON_AddArrayToObject(out, "segments");
        cJSON* allw = cJSON_AddArrayToObject(out, "words");
        for (int i = 0; i < r->n_utts; ++i) {
          cJSON_AddItemToArray(segs, utt_json(&r->utts[i], i));
          for (int k = 0; k < r->utts[i].n_words; ++k) {
            const arc_word* w = &r->utts[i].words[k];
            cJSON* x = cJSON_CreateObject();
            cJSON_AddStringToObject(x, "word", w->text);
            add_num(x, "start", w->start, 3);
            add_num(x, "end", w->end, 3);
            if (w->tag) cJSON_AddStringToObject(x, "tag", w->tag);
            cJSON_AddNumberToObject(x, "segment", i);
            cJSON_AddItemToArray(allw, x);
          }
        }
        cJSON_AddItemToObject(out, "names", names_json(r));
        cJSON_AddItemToObject(out, "numbers", numbers_json(r));
        cJSON_AddItemToObject(out, "events", events_json(r->events, r->n_events));
      }
    }
#ifdef ARC_SERVE_TAGGER
    if (S.tagger && opt_bool(opts, "tags", 1) && strcmp(fmt, "json")) {
      char* wj = result_words_for_tagger(r);
      if (wj) {
        cJSON* to = tagger_opts(opts);
        int st = 0;
        cJSON* tg = tagsrv_tag_words(wj, to, &st);
        cJSON_Delete(to);
        free(wj);
        cJSON_AddItemToObject(out, "arc_tags", tg);
      }
    }
#endif
    cJSON_AddStringToObject(out, "model", S.model_id);
    cJSON* us = cJSON_AddObjectToObject(out, "usage");
    cJSON_AddStringToObject(us, "type", "duration");
    cJSON_AddNumberToObject(us, "seconds", ceil(audio_s));
    if (strcmp(fmt, "json")) add_num(out, "wall_ms", r->wall_ms, 1);
    body = cJSON_PrintUnformatted(out);
    cJSON_Delete(out);
  }
  arc_result_free(r);
  oc->status = 200;
  return body;
}

/* ------------------------------------------------------------------ HTTP handlers */
static void respond_outcome(http_req* r, outcome* oc, char* body, const char* ctype) {
  if (body) http_respond(r, 200, ctype, body, strlen(body));
  else http_respond_error(r, oc->status ? oc->status : 500, oc->err_type, "%s", oc->err);
  free(body);
}

/* a JSON request carrying audio: {"audio": "<base64 or data: URI>", ...options} */
static char* transcribe_json_item(const cJSON* item, const char* default_format, const char** ctype, outcome* oc) {
  const char* a = opt_str(item, "audio");
  if (!a) a = opt_str(item, "file");
  if (!a) { fail(oc, 400, "invalid_request_error", "missing audio (base64)"); return NULL; }
  const char* actype = opt_str(item, "content_type");
  if (!strncmp(a, "data:", 5)) {
    const char* comma = strchr(a, ',');
    if (!comma) { fail(oc, 400, "invalid_request_error", "bad data: URI"); return NULL; }
    a = comma + 1;
  } else if (!strncmp(a, "http://", 7) || !strncmp(a, "https://", 8) || !strncmp(a, "s3://", 5)) {
    fail(oc, 400, "invalid_request_error", "audio URLs are not fetched (the server makes no network calls): send the bytes");
    return NULL;
  }
  size_t bl = 0;
  unsigned char* bytes = b64_decode(a, strlen(a), &bl);
  if (!bytes) { fail(oc, 400, "invalid_request_error", "audio is not valid base64"); return NULL; }
  size_t ns = 0;
  float* pcm = decode_audio(bytes, bl, actype, item, &ns, oc);
  free(bytes);
  if (!pcm) return NULL;
  char* body = transcribe(pcm, ns, item, default_format, ctype, oc);
  free(pcm);
  return body;
}

static void handle_transcriptions(http_req* r, const char* default_format) {
  cJSON* opts = cJSON_CreateObject();
  outcome oc = {0};
  const char* ctype = NULL;
  char* body = NULL;
  if (http_ctype_is(r, "multipart/form-data")) {
    http_part parts[64];
    int np = http_multipart(r, parts, 64);
    if (np < 0) {
      cJSON_Delete(opts);
      http_respond_error(r, 400, "invalid_request_error", "malformed multipart body");
      return;
    }
    const http_part* file = http_part_get(parts, np, "file");
    if (!file) file = http_part_get(parts, np, "audio");
    cJSON* namearr = NULL;
    for (int i = 0; i < np; ++i) {
      if (!parts[i].name || &parts[i] == file) continue;
      char* v = http_part_str(parts, np, parts[i].name);
      if (!strcmp(parts[i].name, "names[]") || !strcmp(parts[i].name, "terms[]")) {   /* repeated fields */
        char key[16];
        snprintf(key, sizeof key, "%.*s", (int)strlen(parts[i].name) - 2, parts[i].name);
        if (!strcmp(key, "names")) {
          if (!namearr) namearr = cJSON_AddArrayToObject(opts, "names");
          char* vi = malloc(parts[i].len + 1);
          memcpy(vi, parts[i].data, parts[i].len);
          vi[parts[i].len] = 0;
          cJSON_AddItemToArray(namearr, cJSON_CreateString(vi));
          free(vi);
        } else opt_set_str(opts, key, v);
      } else {
        char* vi = malloc(parts[i].len + 1);
        memcpy(vi, parts[i].data, parts[i].len);
        vi[parts[i].len] = 0;
        opt_set_str(opts, parts[i].name, vi);
        free(vi);
      }
      free(v);
    }
    add_query_opts(r, opts);
    add_custom_attributes(r, opts);
    if (!file) {
      cJSON_Delete(opts);
      http_respond_error(r, 400, "invalid_request_error", "missing the 'file' part");
      return;
    }
    size_t ns = 0;
    float* pcm = decode_audio((const unsigned char*)file->data, file->len, file->content_type, opts, &ns, &oc);
    if (pcm) body = transcribe(pcm, ns, opts, default_format, &ctype, &oc);
    free(pcm);
  } else if (http_ctype_is(r, "application/json")) {
    cJSON* j = cJSON_ParseWithLength(r->body, r->body_len);
    if (!cJSON_IsObject(j)) {
      cJSON_Delete(j);
      cJSON_Delete(opts);
      http_respond_error(r, 400, "invalid_request_error", "invalid JSON body");
      return;
    }
    add_query_opts(r, j);
    add_custom_attributes(r, j);
    body = transcribe_json_item(j, default_format, &ctype, &oc);
    cJSON_Delete(j);
  } else {   /* raw audio bytes */
    add_query_opts(r, opts);
    add_custom_attributes(r, opts);
    size_t ns = 0;
    float* pcm = decode_audio((const unsigned char*)r->body, r->body_len, http_header(r, "Content-Type"), opts, &ns, &oc);
    if (pcm) body = transcribe(pcm, ns, opts, default_format, &ctype, &oc);
    free(pcm);
  }
  cJSON_Delete(opts);
  respond_outcome(r, &oc, body, ctype);
}

#ifdef ARC_SERVE_TAGGER
static int is_tagger_request(const cJSON* j) {
  return cJSON_GetObjectItemCaseSensitive(j, "text") || cJSON_GetObjectItemCaseSensitive(j, "words") ||
         cJSON_GetObjectItemCaseSensitive(j, "segments");
}
#endif

/* one JSON request object for /invocations: audio -> transcription, text / words / segments -> tags */
static cJSON* invoke_item(const cJSON* item, int* status) {
  outcome oc = {0};
  if (cJSON_IsObject(item) && (opt_str(item, "audio") || opt_str(item, "file"))) {
    const char* ct = NULL;
    char* body = transcribe_json_item(item, "verbose_json", &ct, &oc);
    *status = oc.status;
    if (body) {
      cJSON* o = cJSON_Parse(body);
      if (!o) { o = cJSON_CreateObject(); cJSON_AddStringToObject(o, "text", body); }
      free(body);
      return o;
    }
  }
#ifdef ARC_SERVE_TAGGER
  else if (cJSON_IsObject(item) && is_tagger_request(item)) return tagsrv_process(item, status);
  else if (cJSON_IsString(item)) {
    cJSON* t = cJSON_CreateObject();
    cJSON_AddStringToObject(t, "text", item->valuestring);
    cJSON* o = tagsrv_process(t, status);
    cJSON_Delete(t);
    return o;
  }
#endif
  else fail(&oc, 400, "invalid_request_error",
#ifdef ARC_SERVE_TAGGER
            "each request is {\"audio\": base64, ...} (transcribe) or {\"text\" | \"words\" | \"segments\": ...} (tags)"
#else
            "each request is {\"audio\": base64, ...options}; this image transcribes (no tagger linked)"
#endif
    );
  *status = oc.status;
  cJSON* o = cJSON_CreateObject();
  cJSON* e = cJSON_AddObjectToObject(o, "error");
  cJSON_AddStringToObject(e, "message", oc.err);
  cJSON_AddNumberToObject(e, "code", oc.status);
  return o;
}

static void respond_obj(http_req* r, int status, cJSON* o) {
  char* s = cJSON_PrintUnformatted(o);
  http_respond_json(r, status, s);
  free(s);
  cJSON_Delete(o);
}

static void handle_invocations(http_req* r) {
  if (http_ctype_is(r, "application/jsonlines") || http_ctype_is(r, "application/jsonl") || http_ctype_is(r, "application/x-jsonlines")) {
    size_t cap = 4096, len = 0;
    char* out = malloc(cap);
    char* save = NULL;
    for (char* line = strtok_r(r->body, "\n", &save); line; line = strtok_r(NULL, "\n", &save)) {
      while (*line == ' ' || *line == '\r') ++line;
      if (!*line) continue;
      cJSON* j = cJSON_Parse(line);
      int st = 0;
      cJSON* res = j ? invoke_item(j, &st) : NULL;
      if (!res) {
        res = cJSON_CreateObject();
        cJSON_AddStringToObject(cJSON_AddObjectToObject(res, "error"), "message", "invalid JSON line");
      }
      cJSON_Delete(j);
      char* s = cJSON_PrintUnformatted(res);
      cJSON_Delete(res);
      size_t sl = strlen(s);
      while (len + sl + 2 > cap) cap *= 2;
      out = realloc(out, cap);
      memcpy(out + len, s, sl);
      len += sl;
      out[len++] = '\n';
      free(s);
    }
    http_respond(r, 200, "application/jsonlines", out, len);
    free(out);
    return;
  }
  if (http_ctype_is(r, "application/json")) {
    cJSON* j = cJSON_ParseWithLength(r->body, r->body_len);
    if (!j) { http_respond_error(r, 400, "invalid_request_error", "invalid JSON body"); return; }
    add_custom_attributes(r, j);
    const cJSON* batch = cJSON_IsArray(j) ? j : cJSON_GetObjectItemCaseSensitive(j, "instances");
    if (!batch) batch = cJSON_GetObjectItemCaseSensitive(j, "inputs");
    if (cJSON_IsArray(batch)) {
      cJSON* out = cJSON_CreateObject();
      cJSON_AddStringToObject(out, "model", S.model_id);
      cJSON* arr = cJSON_AddArrayToObject(out, "results");
      const cJSON* it;
      cJSON_ArrayForEach(it, batch) {
        int st = 0;
        cJSON_AddItemToArray(arr, invoke_item(it, &st));
      }
      cJSON_Delete(j);
      respond_obj(r, 200, out);
      return;
    }
    int st = 0;
    cJSON* res = invoke_item(j, &st);
    cJSON_Delete(j);
    respond_obj(r, st ? st : 500, res);
    return;
  }
#ifdef ARC_SERVE_TAGGER
  if (http_ctype_is(r, "text/plain")) {
    cJSON* t = cJSON_CreateObject();
    cJSON_AddStringToObject(t, "text", r->body);
    int st = 0;
    cJSON* res = tagsrv_process(t, &st);
    cJSON_Delete(t);
    respond_obj(r, st, res);
    return;
  }
#endif
  const char* accept = http_header(r, "Accept");
  handle_transcriptions(r, accept && !strncasecmp(accept, "text/plain", 10) ? "text" : "verbose_json");
}

#ifdef ARC_SERVE_TAGGER
static void handle_tags(http_req* r) {
  cJSON* req = NULL;
  if (http_ctype_is(r, "application/json")) req = cJSON_ParseWithLength(r->body, r->body_len);
  else if (http_ctype_is(r, "text/plain") || !http_header(r, "Content-Type")) {
    req = cJSON_CreateObject();
    cJSON_AddStringToObject(req, "text", r->body);
  } else { http_respond_error(r, 415, "invalid_request_error", "send application/json or text/plain"); return; }
  if (!req) { http_respond_error(r, 400, "invalid_request_error", "invalid JSON body"); return; }
  if (cJSON_IsArray(req)) {
    cJSON* w = cJSON_CreateObject();
    cJSON_AddItemToObject(w, "inputs", req);
    req = w;
  }
  int st = 0;
  cJSON* out = tagsrv_process(req, &st);
  cJSON_Delete(req);
  respond_obj(r, st, out);
}
#endif

static void handle_models(http_req* r, const char* only) {
  cJSON* list = cJSON_CreateArray();
  cJSON* m = cJSON_CreateObject();
  cJSON_AddStringToObject(m, "id", S.model_id);
  cJSON_AddStringToObject(m, "object", "model");
  cJSON_AddNumberToObject(m, "created", (double)S.started);
  cJSON_AddStringToObject(m, "owned_by", "whissle");
  cJSON* caps = cJSON_AddArrayToObject(m, "capabilities");
  cJSON_AddItemToArray(caps, cJSON_CreateString("transcriptions"));
  cJSON_AddItemToArray(caps, cJSON_CreateString("realtime"));
  cJSON_AddStringToObject(m, "libarc", arc_version());
  cJSON_AddStringToObject(m, "backbone", "mini");
  cJSON_AddStringToObject(m, "server_version", ARC_DEPLOY_VERSION);
  cJSON_AddNumberToObject(m, "workers", S.workers);
  cJSON_AddNumberToObject(m, "threads_per_worker", S.threads);
  cJSON_AddItemToArray(list, m);
#ifdef ARC_SERVE_TAGGER
  if (S.tagger) {
    cJSON* t = cJSON_CreateObject();
    cJSON_AddStringToObject(t, "id", tagsrv_model_id());
    cJSON_AddStringToObject(t, "object", "model");
    cJSON_AddNumberToObject(t, "created", (double)S.started);
    cJSON_AddStringToObject(t, "owned_by", "whissle");
    cJSON* tc = cJSON_AddArrayToObject(t, "capabilities");
    cJSON_AddItemToArray(tc, cJSON_CreateString("tags"));
    cJSON_AddStringToObject(t, "arctag", arctag_version());
    cJSON_AddItemToArray(list, t);
  }
#endif
  if (only) {
    cJSON* it;
    cJSON_ArrayForEach(it, list) if (!strcmp(opt_str(it, "id"), only)) {
      cJSON* c = cJSON_Duplicate(it, 1);
      cJSON_Delete(list);
      respond_obj(r, 200, c);
      return;
    }
    cJSON_Delete(list);
    http_respond_error(r, 404, "invalid_request_error", "model '%s' not found", only);
    return;
  }
  cJSON* o = cJSON_CreateObject();
  cJSON_AddStringToObject(o, "object", "list");
  cJSON_AddItemToObject(o, "data", list);
  respond_obj(r, 200, o);
}

/* ------------------------------------------------------------------ WS /v1/realtime */
typedef struct {
  int fd;
  int frames;
#ifdef ARC_SERVE_TAGGER
  tagsrv_stream* tag;
#endif
} rt_ctx;

static void ws_obj(int fd, cJSON* o) {
  char* s = cJSON_PrintUnformatted(o);
  ws_send_text(fd, s);
  free(s);
  cJSON_Delete(o);
}

static void ws_err(int fd, const char* msg) {
  cJSON* o = cJSON_CreateObject();
  cJSON_AddStringToObject(o, "type", "error");
  cJSON_AddStringToObject(o, "message", msg);
  ws_obj(fd, o);
}

static void on_stream(const arc_stream_out* out, void* user) {
  rt_ctx* c = user;
  cJSON* o = cJSON_CreateObject();
  switch (out->kind) {
    case ARC_OUT_PARTIAL:
      cJSON_AddStringToObject(o, "type", "transcript.partial");
      cJSON_AddNumberToObject(o, "utterance", out->id);
      cJSON_AddStringToObject(o, "text", out->utt && out->utt->text ? out->utt->text : "");
      if (out->utt) { add_num(o, "start", out->utt->start, 3); add_num(o, "end", out->utt->end, 3); }
      break;
    case ARC_OUT_ENDPOINT:
      cJSON_AddStringToObject(o, "type", "transcript.endpoint");
      cJSON_AddNumberToObject(o, "utterance", out->id);
      if (out->utt) { add_num(o, "start", out->utt->start, 3); add_num(o, "end", out->utt->end, 3); }
      break;
    case ARC_OUT_FINAL:
    case ARC_OUT_TAGS: {
      cJSON_AddStringToObject(o, "type", out->kind == ARC_OUT_FINAL ? "transcript.final" : "transcript.tags");
      cJSON_AddNumberToObject(o, "utterance", out->id);
      cJSON_AddBoolToObject(o, "tags_final", out->kind == ARC_OUT_TAGS || out->tags_final);
      if (out->utt) {
        cJSON* u = utt_json(out->utt, -1);
        cJSON* it = u->child;
        while (it) {   /* flatten the utterance into the message */
          cJSON* next = it->next;
          cJSON_AddItemToObject(o, it->string, cJSON_DetachItemViaPointer(u, it));
          it = next;
        }
        cJSON_Delete(u);
      }
#ifdef ARC_SERVE_TAGGER
      if (out->kind == ARC_OUT_FINAL && c->tag && out->utt && out->utt->text && *out->utt->text) {
        cJSON* p = cJSON_CreateObject();
        cJSON* ws = cJSON_AddArrayToObject(p, "words");
        if (out->utt->n_words) {
          for (int k = 0; k < out->utt->n_words; ++k) {
            cJSON* w = cJSON_CreateObject();
            cJSON_AddStringToObject(w, "w", out->utt->words[k].text);
            add_num(w, "t0", out->utt->words[k].start, 3);
            add_num(w, "t1", out->utt->words[k].end, 3);
            cJSON_AddItemToArray(ws, w);
          }
        } else {
          char* cpy = strdup(out->utt->text);
          char* save = NULL;
          for (char* t = strtok_r(cpy, " ", &save); t; t = strtok_r(NULL, " ", &save)) cJSON_AddItemToArray(ws, cJSON_CreateString(t));
          free(cpy);
        }
        cJSON_AddBoolToObject(p, "endpoint", 1);
        char* js = cJSON_PrintUnformatted(p);
        cJSON_Delete(p);
        cJSON_AddItemToObject(o, "arc_tags", tagsrv_stream_push(c->tag, js));
        free(js);
      }
#endif
      break;
    }
    case ARC_OUT_EVENT:
      cJSON_AddStringToObject(o, "type", "event");
      if (out->event) {
        cJSON_AddStringToObject(o, "label", out->event->label);
        add_num(o, "start", out->event->start, 2);
        add_num(o, "end", out->event->end, 2);
        add_num(o, "p", out->event->p, 4);
      }
      break;
    case ARC_OUT_FRAME:
      if (!c->frames || !out->frame) { cJSON_Delete(o); return; }
      cJSON_AddStringToObject(o, "type", "frame");
      add_num(o, "t", out->frame->t, 2);
      add_num(o, "speech", out->frame->speech, 4);
      cJSON_AddBoolToObject(o, "in_utterance", out->frame->in_utterance);
      add_p_or_null(o, "device_directed", out->frame->device_directed);
      add_p_or_null(o, "command_complete", out->frame->command_complete);
      break;
  }
  add_num(o, "audio_s", out->audio_s, 3);
  ws_obj(c->fd, o);
}

static int push_pcm16(arc_stream* st, const unsigned char* d, size_t n) {
  size_t ns = n / 2;
  if (!ns) return 0;
  float* x = malloc(sizeof(float) * ns);
  for (size_t i = 0; i < ns; ++i) x[i] = (int16_t)(d[2 * i] | d[2 * i + 1] << 8) / 32768.0f;
  int rc = arc_stream_push(st, x, ns);
  free(x);
  return rc;
}

static int apply_session(arc_engine* e, const cJSON* ses, char* err, size_t el) {
  names_t nl;
  if (names_parse(cJSON_GetObjectItemCaseSensitive(ses, "names"), &nl, err, el)) return -1;
  int rc = 0;
  if (cJSON_GetObjectItemCaseSensitive(ses, "names")) rc = arc_set_names(e, nl.e, nl.n);
  names_free(&nl);
  const char** terms;
  cJSON* keep;
  int nt = str_list(cJSON_GetObjectItemCaseSensitive(ses, "terms"), &terms, &keep);
  const char *prompt = opt_str(ses, "prompt"), *topic = opt_str(ses, "topic");
  if (!rc && (prompt || topic || nt)) rc = arc_set_context(e, NULL, NULL, 0, terms, nt, topic, prompt, opt_bool(ses, "auto_prev", 0));
  free(terms);
  cJSON_Delete(keep);
  if (rc) snprintf(err, el, "%s", arc_last_error());
  return rc;
}

static void handle_realtime(http_req* r) {
  cJSON* q = cJSON_CreateObject();
  add_query_opts(r, q);
  int rate = opt_int(q, "sample_rate", 16000);
  if (rate != ARC_SAMPLE_RATE) {
    cJSON_Delete(q);
    http_respond_error(r, 400, "invalid_request_error", "realtime audio is 16 kHz mono PCM16 (sample_rate=16000)");
    return;
  }
  int wi = acquire_engine(atoi(env_or("ARC_STREAM_WAIT_S", "5")));
  if (wi < 0) {
    cJSON_Delete(q);
    http_respond_error(r, 503, "server_busy", "all workers are busy; retry");
    return;
  }
  arc_engine* e = S.e[wi];
  const char* domain = opt_str(q, "domain");
  char domain_buf[64];
  snprintf(domain_buf, sizeof domain_buf, "%s", domain && *domain ? domain : S.domain);
  rt_ctx ctx = {.fd = r->fd, .frames = opt_bool(q, "frames", 0)};
  arc_stream_config sc = {.mode = ARC_STREAM_FULL,
                          .domain = domain_buf,
                          .on_output = on_stream,
                          .user = &ctx,
                          .frames = ctx.frames,
                          .events = opt_bool(q, "events", 1),
                          .partial_ms = opt_int(q, "partials", 0),
                          .beams = opt_int(q, "beams", 0),
                          .bias_names = 1,
                          .delay_ms = opt_int(q, "delay_ms", 0)};
  char err[512] = "";
  if (cJSON_GetObjectItemCaseSensitive(q, "names") && apply_session(e, q, err, sizeof err)) {
    release_engine(wi);
    cJSON_Delete(q);
    http_respond_error(r, 400, "invalid_request_error", "%s", err);
    return;
  }
#ifdef ARC_SERVE_TAGGER
  if (S.tagger && opt_bool(q, "tags", 1)) {
    cJSON* to = tagger_opts(q);
    ctx.tag = tagsrv_stream_open(to, err, sizeof err);
    cJSON_Delete(to);
  }
#endif
  cJSON_Delete(q);
  arc_stream* st = arc_stream_open(e, &sc);
  if (!st) {
    snprintf(err, sizeof err, "%s", arc_last_error());
#ifdef ARC_SERVE_TAGGER
    tagsrv_stream_close(ctx.tag);
#endif
    release_engine(wi);
    http_respond_error(r, 500, "server_error", "stream: %s", err);
    return;
  }
  if (ws_accept(r)) goto done;
  {
    cJSON* o = cJSON_CreateObject();
    cJSON_AddStringToObject(o, "type", "session.created");
    cJSON_AddStringToObject(o, "model", S.model_id);
    cJSON_AddStringToObject(o, "domain", domain_buf);
    cJSON_AddNumberToObject(o, "sample_rate", ARC_SAMPLE_RATE);
    cJSON_AddStringToObject(o, "audio_format", "pcm16");
#ifdef ARC_SERVE_TAGGER
    cJSON_AddBoolToObject(o, "tags", ctx.tag != NULL);
#endif
    ws_obj(r->fd, o);
  }
  for (;;) {
    unsigned char* data;
    size_t len;
    int op = ws_recv(r->fd, &data, &len, (size_t)16 << 20);
    if (op < 0 || op == WS_CLOSE) break;
    if (op == WS_BINARY) {
      if (push_pcm16(st, data, len)) ws_err(r->fd, arc_last_error());
      free(data);
      continue;
    }
    cJSON* m = cJSON_Parse((const char*)data);
    free(data);
    if (!m) { ws_err(r->fd, "invalid JSON"); continue; }
    const char* t = opt_str(m, "type");
    if (!t) t = "";
    if (!strcmp(t, "input_audio_buffer.append")) {
      const char* a = opt_str(m, "audio");
      size_t bl = 0;
      unsigned char* b = a ? b64_decode(a, strlen(a), &bl) : NULL;
      if (!b) ws_err(r->fd, "audio: base64 PCM16 expected");
      else if (push_pcm16(st, b, bl)) ws_err(r->fd, arc_last_error());
      free(b);
    } else if (!strcmp(t, "input_audio_buffer.commit")) {
      if (arc_stream_end_turn(st)) ws_err(r->fd, arc_last_error());
    } else if (!strcmp(t, "session.update")) {
      const cJSON* ses = cJSON_GetObjectItemCaseSensitive(m, "session");
      if (apply_session(e, ses ? ses : m, err, sizeof err)) ws_err(r->fd, err);
      else {
        cJSON* o = cJSON_CreateObject();
        cJSON_AddStringToObject(o, "type", "session.updated");
        ws_obj(r->fd, o);
      }
    } else if (!strcmp(t, "input_audio_buffer.finish") || !strcmp(t, "session.close")) {
      arc_stream_finish(st);
      cJSON* o = cJSON_CreateObject();
      cJSON_AddStringToObject(o, "type", "session.closed");
      ws_obj(r->fd, o);
      ws_close(r->fd, 1000, "done");
      cJSON_Delete(m);
      break;
    } else {
      ws_err(r->fd, "unknown type (input_audio_buffer.append | input_audio_buffer.commit | input_audio_buffer.finish | "
                    "session.update | session.close), or send binary PCM16 frames");
    }
    cJSON_Delete(m);
  }
done:
  arc_stream_close(st);
#ifdef ARC_SERVE_TAGGER
  tagsrv_stream_close(ctx.tag);
#endif
  release_engine(wi);
}

/* Snowflake service functions (SPCS): {"data": [[row, audio, options_json?], ...]} -> {"data": [[row, verbose_json], ...]}.
 * audio: a path under ARC_FILE_ROOT (a stage mounted as a volume, e.g. /stage/calls/a.wav) or base64 bytes.
 *   CREATE FUNCTION arc_transcribe(path VARCHAR) RETURNS VARIANT SERVICE=arc ENDPOINT=api AS '/v1/snowflake/transcribe'; */
static unsigned char* read_stage_file(const char* path, size_t* n, outcome* oc) {
  const char* root = getenv("ARC_FILE_ROOT");
  if (!root || !*root) { fail(oc, 400, "invalid_request_error", "file paths need ARC_FILE_ROOT (the mounted stage); send base64 otherwise"); return NULL; }
  size_t rl = strlen(root);
  if (strncmp(path, root, rl) || (path[rl] != '/' && path[rl] != 0)) {
    char m[256];
    snprintf(m, sizeof m, "path outside ARC_FILE_ROOT (%s)", root);
    fail(oc, 400, "invalid_request_error", m);
    return NULL;
  }
  char real[4096], rroot[4096];
  size_t rrl;
  /* the resolved path (symlinks, "..") must still be inside the resolved root */
  if (!realpath(path, real) || !realpath(root, rroot) || strncmp(real, rroot, (rrl = strlen(rroot))) || (real[rrl] != '/' && real[rrl] != 0)) {
    fail(oc, 404, "invalid_request_error", "no such file under ARC_FILE_ROOT");
    return NULL;
  }
  FILE* f = fopen(real, "rb");
  if (!f) { fail(oc, 404, "invalid_request_error", "cannot open the file"); return NULL; }
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (len < 0 || (size_t)len > ((size_t)atoi(env_or("ARC_MAX_BODY_MB", "100")) << 20)) { fclose(f); fail(oc, 413, "invalid_request_error", "file too large"); return NULL; }
  unsigned char* b = malloc((size_t)len + 1);
  *n = fread(b, 1, (size_t)len, f);
  fclose(f);
  return b;
}

static void handle_snowflake(http_req* r, int tags) {
  cJSON* j = cJSON_ParseWithLength(r->body, r->body_len);
  const cJSON* rows = j ? cJSON_GetObjectItemCaseSensitive(j, "data") : NULL;
  if (!cJSON_IsArray(rows)) {
    cJSON_Delete(j);
    http_respond_error(r, 400, "invalid_request_error", "expected {\"data\": [[row, value, ...], ...]}");
    return;
  }
  cJSON* out = cJSON_CreateObject();
  cJSON* data = cJSON_AddArrayToObject(out, "data");
  const cJSON* row;
  cJSON_ArrayForEach(row, rows) {
    const cJSON* idx = cJSON_GetArrayItem(row, 0);
    const cJSON* val = cJSON_GetArrayItem(row, 1);
    const cJSON* opt = cJSON_GetArrayItem(row, 2);
    cJSON* item = NULL;
    if (cJSON_IsString(opt) && opt->valuestring[0] == '{') item = cJSON_Parse(opt->valuestring);
    else if (cJSON_IsObject(opt)) item = cJSON_Duplicate(opt, 1);
    if (!item) item = cJSON_CreateObject();
    cJSON* res = NULL;
    int st = 0;
    const char* v = cJSON_IsString(val) ? val->valuestring : "";
    if (tags) {
#ifdef ARC_SERVE_TAGGER
      cJSON_DeleteItemFromObjectCaseSensitive(item, "text");
      cJSON_AddStringToObject(item, "text", v);
      res = tagsrv_process(item, &st);
#endif
    } else if (v[0] == '/') {
      outcome oc = {0};
      size_t n = 0;
      unsigned char* b = read_stage_file(v, &n, &oc);
      char* body = NULL;
      if (b) {
        size_t ns = 0;
        float* pcm = decode_audio(b, n, NULL, item, &ns, &oc);
        free(b);
        const char* ct;
        if (pcm) body = transcribe(pcm, ns, item, "verbose_json", &ct, &oc);
        free(pcm);
      }
      if (body) { res = cJSON_Parse(body); if (!res) { res = cJSON_CreateObject(); cJSON_AddStringToObject(res, "text", body); } free(body); }
      else {
        res = cJSON_CreateObject();
        cJSON* e = cJSON_AddObjectToObject(res, "error");
        cJSON_AddStringToObject(e, "message", oc.err);
        cJSON_AddNumberToObject(e, "code", oc.status);
      }
    } else {
      cJSON_DeleteItemFromObjectCaseSensitive(item, "audio");
      cJSON_AddStringToObject(item, "audio", v);
      res = invoke_item(item, &st);
    }
    cJSON_Delete(item);
    cJSON* pair = cJSON_CreateArray();
    cJSON_AddItemToArray(pair, idx ? cJSON_Duplicate(idx, 1) : cJSON_CreateNull());
    cJSON_AddItemToArray(pair, res ? res : cJSON_CreateNull());
    cJSON_AddItemToArray(data, pair);
  }
  cJSON_Delete(j);
  respond_obj(r, 200, out);
}

/* ------------------------------------------------------------------ routing */
static void send_file(http_req* r, const char* path, const char* ctype) {
  FILE* fp = fopen(path, "rb");
  if (!fp) { http_respond_error(r, 404, "invalid_request_error", "not bundled"); return; }
  fseek(fp, 0, SEEK_END);
  long n = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  char* s = malloc((size_t)n + 1);
  size_t got = fread(s, 1, (size_t)n, fp);
  fclose(fp);
  http_respond(r, 200, ctype, s, got);
  free(s);
}

static void handler(http_req* r, void* user) {
  (void)user;
  const char* p = r->path;
  int get = !strcmp(r->method, "GET") || !strcmp(r->method, "HEAD"), post = !strcmp(r->method, "POST");
  if (get && (!strcmp(p, "/health") || !strcmp(p, "/v1/health/live") || !strcmp(p, "/healthz"))) {
    http_respond_json(r, 200, "{\"status\":\"ok\"}");
  } else if (get && (!strcmp(p, "/v1/health/ready") || !strcmp(p, "/ping") || !strcmp(p, "/readyz"))) {
    if (S.ready) http_respond_json(r, 200, "{\"status\":\"ready\"}");
    else http_respond_json(r, 503, "{\"status\":\"loading\"}");
  } else if (get && !strcmp(p, "/v1/models")) {
    handle_models(r, NULL);
  } else if (get && !strncmp(p, "/v1/models/", 11)) {
    handle_models(r, p + 11);
  } else if (!strcmp(p, "/v1/realtime")) {
    if (!http_is_ws_upgrade(r)) http_respond_error(r, 400, "invalid_request_error", "WebSocket upgrade required");
    else handle_realtime(r);
  } else if (!strcmp(p, "/v1/audio/transcriptions")) {
    if (!post) http_respond_error(r, 405, "invalid_request_error", "use POST");
    else handle_transcriptions(r, "json");
  } else if (!strcmp(p, "/invocations")) {
    if (!post) http_respond_error(r, 405, "invalid_request_error", "use POST");
    else handle_invocations(r);
#ifdef ARC_SERVE_TAGGER
  } else if (!strcmp(p, "/v1/tags")) {
    if (!post) http_respond_error(r, 405, "invalid_request_error", "use POST");
    else handle_tags(r);
#endif
  } else if (!strcmp(p, "/v1/snowflake/transcribe")) {
    if (!post) http_respond_error(r, 405, "invalid_request_error", "use POST");
    else handle_snowflake(r, 0);
#ifdef ARC_SERVE_TAGGER
  } else if (!strcmp(p, "/v1/snowflake/tags")) {
    if (!post) http_respond_error(r, 405, "invalid_request_error", "use POST");
    else handle_snowflake(r, 1);
#endif
  } else if (get && !strcmp(p, "/execution-parameters")) {
    char b[256];
    snprintf(b, sizeof b, "{\"MaxConcurrentTransforms\":%d,\"BatchStrategy\":\"SINGLE_RECORD\",\"MaxPayloadInMB\":%d}", S.workers,
             atoi(env_or("ARC_MAX_BODY_MB", "100")) > 100 ? 100 : atoi(env_or("ARC_MAX_BODY_MB", "100")));
    http_respond_json(r, 200, b);
  } else if (get && !strcmp(p, "/openapi.yaml")) {
    send_file(r, env_or("ARC_OPENAPI", "/opt/arc/openapi.yaml"), "application/yaml");
  } else if (get && !strcmp(p, "/")) {
    char b[512];
    snprintf(b, sizeof b, "{\"service\":\"%s\",\"model\":\"%s\",\"version\":\"%s\",\"libarc\":\"%s\",\"openapi\":\"/openapi.yaml\"}",
             S.tagger ? "arc-transcriber+tagger" : "arc-transcriber", S.model_id, ARC_DEPLOY_VERSION, arc_version());
    http_respond_json(r, 200, b);
  } else {
    http_respond_error(r, 404, "invalid_request_error", "no route %s %s", r->method, p);
  }
}

static void on_term(int sig) {
  (void)sig;
  http_server_stop();
  _exit(0);
}

int main(int argc, char** argv) {
  const char* port_s = env_or("PORT", env_or("SAGEMAKER_BIND_TO_PORT", "8080"));
  const char* host = env_or("HOST", "0.0.0.0");
  const char* model = NULL;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "serve")) continue;   /* SageMaker runs the image as `docker run IMAGE serve` */
    else if (!strcmp(argv[i], "--port") && i + 1 < argc) port_s = argv[++i];
    else if (!strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
    else if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
    else if (!strcmp(argv[i], "--healthcheck")) return http_healthcheck(atoi(port_s), "/v1/health/ready");
    else if (!strcmp(argv[i], "--version")) { printf("arc-serve %s (libarc %s)\n", ARC_DEPLOY_VERSION, arc_version()); return 0; }
    else {
      fprintf(stderr, "usage: %s [serve] [--port N] [--host ADDR] [--model DIR]\n", argv[0]);
      return 2;
    }
  }
  if (arc_abi_version() != ARC_ABI_VERSION) {
    fprintf(stderr, "arc-serve: libarc ABI %d, header %d\n", arc_abi_version(), ARC_ABI_VERSION);
    return 1;
  }
  if (find_model_dir(model, S.model_dir, sizeof S.model_dir)) {
    fprintf(stderr, "arc-serve: no Arc Mini folder (config.json, model.q4.safetensors, moonshine-streaming-tiny/) at %s; set "
                    "ARC_MODEL_DIR or mount the weights at /opt/ml/model or /repository\n", model || getenv("ARC_MODEL_DIR") ? S.model_dir : "the default paths");
    return 1;
  }
  snprintf(S.model_id, sizeof S.model_id, "%s", env_or("ARC_MODEL_ID", "arc-mini"));
  snprintf(S.domain, sizeof S.domain, "%s", env_or("ARC_DOMAIN", "unknown"));
  long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  if (ncpu < 1) ncpu = 1;
  S.workers = atoi(env_or("ARC_WORKERS", "2"));
  if (S.workers < 1) S.workers = 1;
  if (S.workers > 64) S.workers = 64;
  S.threads = atoi(env_or("ARC_THREADS", "0"));
  if (S.threads <= 0) S.threads = (int)(ncpu / S.workers) > 0 ? (int)(ncpu / S.workers) : 1;
  S.max_audio_s = atof(env_or("ARC_MAX_AUDIO_S", "900"));
  S.queue_timeout_s = atoi(env_or("ARC_QUEUE_TIMEOUT_S", "60"));
  S.started = time(NULL);
  double t0 = http_now_ms();
  S.e = calloc((size_t)S.workers, sizeof *S.e);
  S.busy = calloc((size_t)S.workers, sizeof *S.busy);
  for (int i = 0; i < S.workers; ++i) {
    char err[512];
    S.e[i] = arc_create(S.model_dir, S.threads, err, sizeof err);
    if (!S.e[i]) {
      fprintf(stderr, "arc-serve: %s: %s\n", S.model_dir, err);
      return 1;
    }
  }
#ifdef ARC_SERVE_TAGGER
  {
    char td[1200] = "", err[1024];
    const char* cand[] = {getenv("ARCTAG_MODEL_DIR"), "/opt/ml/model/arc-tagger", "/repository/arc-tagger", "/opt/arc/model/arc-tagger",
                          "/opt/arc/tagger", NULL};
    for (int i = 0; i < 5 && !*td; ++i) {
      if (!cand[i]) continue;
      char p[1300];
      struct stat st;
      snprintf(p, sizeof p, "%s/arctag_meta.json", cand[i]);
      if (!stat(p, &st)) snprintf(td, sizeof td, "%s", cand[i]);
    }
    if (!*td) {
      fprintf(stderr, "arc-serve: no Arc Tagger folder (arctag_meta.json); set ARCTAG_MODEL_DIR\n");
      return 1;
    }
    if (tagsrv_init(td, err, sizeof err)) {
      fprintf(stderr, "arc-serve: tagger %s: %s\n", td, err);
      return 1;
    }
    S.tagger = 1;
    fprintf(stderr, "arc-serve: tagger %s (%s)\n", tagsrv_model_id(), td);
  }
#endif
  S.ready = 1;
  signal(SIGTERM, on_term);
  signal(SIGINT, on_term);
  http_server_config c = {0};
  c.host = host;
  c.port = atoi(port_s);
  c.max_connections = atoi(env_or("ARC_MAX_CONNECTIONS", "64"));
  c.max_body = (size_t)atoi(env_or("ARC_MAX_BODY_MB", "100")) << 20;
  c.access_log = atoi(env_or("ARC_ACCESS_LOG", "1"));
  c.handler = handler;
  fprintf(stderr, "arc-serve %s (libarc %s): %s from %s, %d workers x %d threads, loaded in %.0f ms; listening on %s:%d\n",
          ARC_DEPLOY_VERSION, arc_version(), S.model_id, S.model_dir, S.workers, S.threads, http_now_ms() - t0, host, c.port);
  return http_serve(&c);
}
