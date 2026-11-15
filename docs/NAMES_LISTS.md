# Names lists: how to build them for Arc Transcriber Space

The model links what the user says to entries of a **names list** that the app sends. For example, it links "tell me about
tycho" to your `Tycho` entry. The list is used in three places:

| Where | What it does | Which part of the list it uses |
|---|---|---|
| Word biasing | Makes the transcript prefer list names ("tycho" rather than "tie co") | The whole list, names and aliases |
| Arc decoder prompt | Helps the tagger type a mention (crater, spacecraft, mission, crew…) | Entries **in list order**, until a token budget of about 10–15 names is used |
| Linking | Sets `names[i].entity` to the list entry that was said | The whole list |

All three compare **normalized** text. Write lists with the rules below, or names will not match.

## 1. Entry format

```python
{"name": "Apollo 12", "kind": "mission", "aliases": ["Apollo twelve"]}
```

- **`name`**: the display form. It is returned unchanged as `entity` when linked, so use the exact string you want to show
  or route on.
- **`kind`** (required in practice): one of
  - `lunar_feature`: craters, maria, mountains, rilles
  - `site`: landing or impact sites, stations
  - `celestial`: planets, moons, stars, comets
  - `spacecraft` (or `vehicle`)
  - `mission`
  - `crew` (or `person`)
  - general kinds: `place`, `organization`, `product`, `team`, `law`

  An entry with no kind or an unknown kind is still biased and linked, but the tagger cannot type it. Untyped lists lose
  much of the tagging accuracy.
- **`aliases`** (optional): other ways people say the name. Each alias is biased and matched like the name. A match on an
  alias still returns `name` as `entity`.

## 2. Normalization the model applies

Each name and alias is converted to spoken-style lower case before matching:

| Rule | Example input | Normalized |
|---|---|---|
| lower case | `CAPSTONE` | `capstone` |
| digits become spoken English | `Apollo 12`, `Luna 9`, `1969` | `apollo twelve`, `luna nine`, `nineteen sixty nine` |
| years are read as years (1100–1999, 2010–2099) | `2025` | `twenty twenty five` |
| other numbers | `2000`, `2005` | `two thousand`, `two thousand five` |
| `-` becomes a space | `S-IVB` | `s ivb` |
| apostrophes are kept; `’` becomes `'` | `Chang’e 3` | `chang'e three` |
| a final possessive `'s` is ignored when matching | spoken "glushko's" | matches `Glushko` |
| **every other character is dropped, including accented letters and Greek** | `Schrödinger`, `Mons Hadley Δ` | `schr dinger`, `mons hadley` |

The transcript itself is also in spoken form (`apollo twelve`, not `Apollo 12`). Use `names[i].entity` for display and
routing, not the raw transcript words.

## 3. Rules for the app team

1. **Always set `kind`.**
2. **Put the most relevant names first**: what is on screen or in the current lesson. Only the first ~10–15 names (about
   56 tokens) reach the decoder prompt. Biasing and linking still use the whole list.
3. **Keep lists short and current**, ideally ≤ 30–50 entries. Every entry pulls the transcript toward itself, so a long
   list of unrelated names adds a little error on speech that is not about them. Replace the list when the screen or
   topic changes: `stream.set_context(names=...)` takes effect mid-stream.
4. **Give ASCII aliases for any non-ASCII name**: `{"name": "Schrödinger", "aliases": ["Schrodinger"]}`.
5. **Add an alias for any name that is not said the way it is written**:

   | Written | Spoken | Alias to add |
   |---|---|---|
   | Roman numerals: `Artemis II` | "artemis two" | `"Artemis 2"` (written `Artemis II`, it would never match) |
   | Letters said one by one | "s four b" | `S-IVB` → `"S 4 B"` |
   | Acronyms said as words | "al sep" | `ALSEP` → `"al sep"` |
   | Latin and English names | — | `Sea of Tranquility` + `"Mare Tranquillitatis"` |
   | Short names | — | `Apollo 12 ALSEP science station` + `"Apollo 12 ALSEP"` |
6. **Avoid entries made only of generic words**: "Mare", "Crater", "North", "Sea", "Valley" and similar. A mention that
   contains only generic words is not linked unless it matches an entry exactly. The model returns `state: "abstain"`;
   the app should ask the user.
7. Names shorter than 3 letters are not added as new tags by list matching. They can still be linked when the tagger tags
   them.

## 4. Reading the result

```python
out = arc.predict(wav, domain="assistant", transcript="nft2", names=NAMES)
for n in out["names"]:
    n["tag"]     # e.g. "entity.mention.lunar_feature"
    n["heard"]   # the words heard (spoken form), e.g. "apollo twelve"
    n["entity"]  # the list entry's `name`, e.g. "Apollo 12"; None if not linked
    n["state"]   # "linked" | "abstain" (ambiguous or generic: ask the user) | "not_in_list" (a name, but not in your list)
    n["p"]       # confidence
```

- **`linked`**: act on `entity`.
- **`abstain`**: show a "did you mean …?" choice.
- **`not_in_list`**: the user named something you did not list. You can search a bigger catalogue with `heard`.
