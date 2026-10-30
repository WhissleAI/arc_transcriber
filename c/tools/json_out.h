/* JSON output shared by the tools (arc-parity, arc-transcribe --json): the result's words, names and numbers in
 * arc_audio's keys. Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#ifndef ARC_JSON_OUT_H
#define ARC_JSON_OUT_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arc.h"
#include "arc_base.h"

static void jstr(const char* s) {
  putchar('"');
  for (; *s; ++s) {
    unsigned char c = (unsigned char)*s;
    if (c == '"' || c == '\\') printf("\\%c", c);
    else if (c < 0x20) printf("\\u%04x", c);
    else putchar(c);
  }
  putchar('"');
}

/* a value as Python's json writes it: an int without a fraction, else repr-like (%.17g trimmed) */
static void jnum(double v, int is_int) {
  if (is_int) printf("%lld", (long long)llround(v));
  else {
    char b[64];
    for (int p = 1; p <= 17; ++p) {   /* shortest round-trip form */
      snprintf(b, sizeof b, "%.*g", p, v);
      if (strtod(b, NULL) == v) break;
    }
    if (!strchr(b, '.') && !strchr(b, 'e') && !strchr(b, 'n') && !strchr(b, 'i')) strcat(b, ".0");
    printf("%s", b);
  }
}

static void jstr_or_null(const char* s) {
  if (s) jstr(s);
  else printf("null");
}

/* the names list of examples/example_names.json (strings or {"name", "kind", "group", "aliases"}) */
static arc_json* load_names(const char* path, arc_name_entry** out, int* n) {
  char* text = NULL;
  size_t len = 0;
  *out = NULL;
  *n = 0;
  if (arc_read_file(path, &text, &len)) return NULL;
  arc_json* j = arc_json_parse(text, len);
  arc_free(text);
  if (!j || j->type != ARC_JARR) return j;
  arc_name_entry* e = calloc((size_t)j->n + 1, sizeof *e);
  for (int i = 0; i < j->n; ++i) {
    const arc_json* c = &j->items[i];
    if (c->type == ARC_JSTR) {
      e[i].name = c->str;
      continue;
    }
    e[i].name = arc_json_str(arc_json_get(c, "name"), "");
    e[i].kind = arc_json_str(arc_json_get(c, "kind"), NULL);
    e[i].group = arc_json_str(arc_json_get(c, "group"), NULL);
    const arc_json* al = arc_json_get(c, "aliases");
    if (al && al->type == ARC_JARR && al->n) {
      const char** a = calloc((size_t)al->n, sizeof(char*));
      for (int k = 0; k < al->n; ++k) a[k] = al->items[k].str ? al->items[k].str : "";
      e[i].aliases = a;
      e[i].n_aliases = al->n;
    }
  }
  *out = e;
  *n = j->n;
  return j;
}

static void print_words(const arc_word* w, int n) {
  printf("[");
  for (int k = 0; k < n; ++k) {
    printf("%s{\"w\": ", k ? ", " : "");
    jstr(w[k].text);
    printf(", \"start\": %.3f, \"end\": %.3f}", w[k].start, w[k].end);
  }
  printf("]");
}

static void print_names_numbers(const arc_result* r) {
  printf("\"names\": [");
  for (int i = 0; i < r->n_names; ++i) {
    const arc_name* m = &r->names[i];
    printf("%s{\"start\": %.3f, \"end\": %.3f, \"tag\": \"%s\", \"heard\": ", i ? ", " : "", m->start, m->end, m->tag);
    jstr(m->heard);
    printf(", \"entity\": ");
    jstr_or_null(m->entity);
    printf(", \"state\": \"%s\", \"src\": \"%s\"", m->state, m->src);
    if (m->p >= 0) printf(", \"p\": %.4f", m->p);
    if (m->model_p >= 0) {
      printf(", \"model\": {\"entity\": ");
      jstr_or_null(m->model_entity);
      printf(", \"p\": %.4f}", m->model_p);
    }
    if (m->via) printf(", \"via\": \"%s\"", m->via);
    if (m->snap_score >= 0) printf(", \"snap_score\": %.3f", m->snap_score);
    if (m->kb_index >= 0) printf(", \"kb_index\": %d", m->kb_index);
    if (m->span_p >= 0) printf(", \"span_p\": %.4f", m->span_p);
    printf(", \"show\": %s, \"words\": ", m->show ? "true" : "false");
    print_words(m->words, m->n_words);
    printf("}");
  }
  printf("], \"numbers\": [");
  for (int i = 0; i < r->n_numbers; ++i) {
    const arc_number* m = &r->numbers[i];
    printf("%s{\"start\": %.3f, \"end\": %.3f, \"kind\": ", i ? ", " : "", m->start, m->end);
    jstr_or_null(m->kind);
    printf(", \"unit\": ");
    jstr_or_null(m->unit);
    printf(", \"heard\": ");
    jstr(m->heard);
    printf(", \"value\": ");
    if (m->has_value) jnum(m->value, m->value_int); else printf("null");
    printf(", \"value2\": ");
    if (m->has_value2) jnum(m->value2, m->value2_int); else printf("null");
    printf(", \"state\": \"%s\", \"kind_head\": ", m->state);
    jstr_or_null(m->kind_head);
    printf("}");
  }
  printf("]");
}

#endif
