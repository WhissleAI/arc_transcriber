/* Spoken dates (arc_audio written.py parse_date / date_iso / is_month and api.py extend_date): a number record of kind
 * "date" covers its month too ("March third", "the third of March", "March third, twenty twenty four") and carries the
 * date it names. Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#ifndef ARC_DATE_H
#define ARC_DATE_H

#include "arc_tidy.h"

/* year, month, day: 0 = not said (Python's None) */
typedef struct {
  int year, month, day;
} arc_ymd;

/* written.is_month: a month name as Arc writes it (capitalised, so "may" the verb is not a month) */
int arc_is_month(const char* w);
/* written.parse_date(ws): 1 and *d, or 0 for None (the words are not a clean date) */
int arc_parse_date(const char* const* ws, int n, arc_ymd* d);
/* written.date_iso(d): "2024-03-03", "--03-03", "2024-03", "2024" into out (>= 16 bytes); 0 (out "") for None */
int arc_date_iso(const arc_ymd* d, char* out);
/* api.extend_date: the transcript words W[0, n) sorted by start, the record's words [i, j]. On a date: 1, *heard
 * replaced (malloc'd; the old one arc_free'd), *start / *end widened to the date's words (rounded to 1 ms) and *d set;
 * else 0 and nothing changed. */
int arc_extend_date(const arc_tword* W, int n, int i, int j, char** heard, double* start, double* end, arc_ymd* d);

#endif
