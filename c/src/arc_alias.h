/* The list check before a name is left unlinked: arc_audio kbsnap.py AliasIndex (used by engine.py _name_extras). The
 * heard text, its edges trimmed of command / function words, against each entry's name and aliases with punctuation,
 * case and spacing ignored ("DeMarco" = "De'Marco"), then against the parts of a name that only one entry has
 * ("Daniel" = "Daniel Brooks", "Comets" = "Riverside Comets").
 * Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE. */
#ifndef ARC_ALIAS_H
#define ARC_ALIAS_H

#include "arc_ntext.h"

typedef struct arc_alias arc_alias;

arc_alias* arc_alias_new(const arc_kbent* kb, int n);
void arc_alias_free(arc_alias* a);
/* AliasIndex.__bool__: some entry has a non-empty key */
int arc_alias_any(const arc_alias* a);
/* AliasIndex.lookup: the entry's row and via ("alias" or "partial"), or -1 (via NULL). allowed: the rows the linker
 * allowed (NULL = no restriction, as allowed=None) */
int arc_alias_lookup(const arc_alias* a, const char* heard, const int* allowed, int n_allowed, const char** via);
/* kbsnap.akey: lower case, edge punctuation and a possessive 's dropped, other non-alphanumerics removed (malloc'd) */
char* arc_alias_akey(const char* t);

#endif
