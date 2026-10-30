# Spoken book search

**What it measures.** "Read the hound of the Baskervilles", "books by Agatha Christie": the person names one of 61,476 books.
- Does Arc's transcript carry the title's or author's words?
- Does the right book come first in a catalog search?
- What does each kind of context cost?
- Separately: how well does **Arc's own name linker** find the book when given catalog titles as its list?

- **Requests** (`data/book_queries.tsv`): 50 requests.
  - 30 by title, 8 by author, 8 by genre, 2 by topic, 2 "something like".
  - The clips are 50 × 4 macOS voices = 200 (`NN_Sam.wav`, `NN_Dan.wav`, `NN_Flo.wav`, `NN_Ama.wav`; Samantha, Daniel, Flo, Aman).
- **Catalog** (`data/catalog.tsv.gz`, 1.6 MB): 61,476 Project Gutenberg books (id, title, author, popularity 0–255 from download counts), as shipped in Steer's catalog on 2026-10-29.
- **Search.** Steer's own catalog search (`BookCatalog` / `LiveContext` in Swift) isn't public. `catalog.py` is a stand-in:
  - fuzzy matching over short titles and authors, on the spelling and on its sound (metaphone);
  - a containment match for part-titles ("sherlock holmes");
  - a small popularity prior.

  It has no genre bits, topic vectors or "more like this" embeddings, so genre and topic requests aren't scored here. On the true words it finds the right book for 39/40 requests (156/160 clips); the miss is "Alice in Wonderland", which matches a play of that exact name first.
- **Context conditions** (Arc Mini, `transcribe`):
  - **none:** no names.
  - **static:** the 1,500 most-read titles and 400 authors as decoder bias, for every request.
  - **live:**
    - every 0.5 s of audio, a partial (1 beam) looks the catalog up;
    - its 40 candidates go into the decoder's names prompt and bias;
    - the final pass then decodes with them. This is what Steer for Mac 1.6.4 ships.
  - **rag2:** the whole first transcript looks the catalog up; the clip is heard again with the 20 candidates, and the second transcript is kept if it matches a book better.
  - **live+rag2.**
- **Arc's own linker** (`predict(names=...)`):
  - **(a)** the 2,000 most-read books (short titles, `work_product`) and 500 authors (`person`) as the list;
  - **(b)** the same with `kb_snap`;
  - **(c)** live RAG: pass 1 with no list, the catalog's top 20 candidates for that text become the list for pass 2.
  - The first linked entity must be the right title or author.

## Run

```bash
python evals/book_search/run.py                    # search + linker, 200 clips, about 30 min on 4 CPU cores
python evals/book_search/run.py --part search      # or --part linker
python evals/book_search/run.py --clips DIR        # your own renders, named NN_<first 3 letters of the voice>.wav
```

## Results

Words heard / right book first, on the title, author and "like" requests. Steer also scored genre and topic.

**Steer (app path)**: run on 2026-10-29 with Steer for Mac's `SteerVoiceBench --books`, libarc, Arc Mini 1.1.1 (HF rev `v1.1.1`), M3 Pro, 2 threads. It used Steer's own catalog search over all 200 clips:

| Kind (n) | none | static | relisten | live-bias | live-prompt | live+relisten | rag2 | live+rag2 | live+rag2 (prompt) |
|---|---|---|---|---|---|---|---|---|---|
| title (120) | 56% / 67% | 64% / 73% | 61% / 72% | 62% / 69% | 68% / 71% | **70% / 76%** | 62% / 69% | 67% / 72% | 68% / 72% |
| author (32) | 91% / 81% | 88% / 78% | 88% / 81% | 91% / 78% | 84% / 81% | 88% / 81% | 91% / 84% | 91% / 81% | 91% / 88% |
| like (8) | 88% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| genre (32) | – / 100% | – / 88% | – / 100% | – / 100% | – / 100% | – / 100% | – / 100% | – / 100% | – / 100% |
| topic (8) | – / 62% | – / 38% | – / 62% | – / 62% | – / 62% | – / 62% | – / 62% | – / 62% | – / 62% |
| **all (200)** | 64% / **76%** | 71% / 76% | 68% / 79% | 70% / 76% | 73% / 78% | 75% / **81%** | 69% / 78% | 73% / 78% | 74% / 80% |

Steer timing (median / p90):
- A partial decode takes 20.9 / 63.8 ms.
- A catalog lookup takes **1.8** / 46.0 ms.
- The bench prints **7.0** ms median for live+relisten (p90 346 ms when a relisten runs).
  - Reading `Books.swift`, that figure times only the relisten step: the live pass's final decode is counted under live-bias, 134.9 / 289.6 ms including its partials.
  - So "final 7 ms after speech end" is not a decode time. In the app's film take, Arc took 122 ms and the title showed 0.1 s after the last word.
- The none pass takes 52.9 / 71.0 ms; the static list costs 230 / 308 ms per request. Setting the static bias once takes 49 ms (1,900 phrases).

Example from the app: Arc corrected "Mr. High" to "Mr. Hyde" because the title was in its prompt. In the Mac demo film,
"Baska those" became *The Hound of the Baskervilles*; Arc took 122 ms and the title came up 0.1 s after the last word.

**Python reference (this PR):** `arc_audio` 1.2.0, model rev `v1.2.0` (same Mini files), CPU, 4 threads, M3 Pro, 2026-10-29. It used the stand-in search over 160 clips.

| Kind (n) | none | static | live | rag2 | live+rag2 |
|---|---|---|---|---|---|
| title (120) | 49% / 68% | 66% / 72% | 69% / 74% | 61% / 72% | 73% / 76% |
| author (32) | 78% / 69% | 75% / 66% | 81% / 72% | 81% / 69% | 81% / 72% |
| like (8) | 75% / 75% | 88% / 88% | 75% / 62% | 88% / 75% | 88% / 75% |
| **all** (160) | 56% / 68% | 69% / 72% | 72% / 73% | 66% / 71% | 76% / 75% |

Rerun timing, median / p90 (Python, CPU):
- One decode with no context: 165 / 225 ms. With the static 1,900-name bias: 248 / 358 ms.
- A live partial decode (1 beam): 84 / 126 ms.
- The stand-in catalog lookup: 28 / 40 ms. Steer's Swift index takes 1.8 ms.
- The live final decode after the request ends: 185 / 269 ms.

**Arc's own name linker** on the title and author clips (152), with the first linked entity counted:

| Setup | Right | Wrong book linked | Source |
|---|---|---|---|
| Steer's search (live prompt + catalog), for reference | **78%** (title 76%, author 81%) | – | Steer app path, above |
| Arc linker, 2,000 most-read titles as its list | 34% | 9 | Python reference, 2026-10-29 (Steer session, `kbexp`) |
| same, plus `kb_snap` | 35% | 63 | same |
| live RAG: top 20 candidates as the list, then Arc's linker | 49% (title 48/120 = 40%, author 26/32 = 81%) | 19 | same |
| **this PR's rerun**: 2,000 titles as the list | 32% (49/152; title 23/120, author 26/32) | 8 | median 1900 ms per clip |
| **this PR's rerun**: same + `kb_snap` | 34% (52/152; title 37/120, author 15/32) | 70 | median 1924 ms per clip |
| **this PR's rerun**: top 20 live-RAG candidates (stand-in search) | 53% (80/152; title 55/120, author 25/32) | 13 | median 464 ms per clip |

Why the linker misses: it was trained to link people and teams on a roster, not book titles.
- It linked "Sherlock Holmes" to the author "T. Rice Holmes".
- `kb_snap` matched "Read Frankenstein" to a title that contains the word "Read": *The Pilgrim's Progress by John Bunyan Every Child Can Read*. It made 63 wrong links.
- Its median time with a 2,000-entry list was about 1.8 s per clip, against 0.8 s with a 20-entry list.

## Known limits

- macOS voices only.
- The rerun's search is a stand-in, so compare the rerun's conditions with each other, not with Steer's absolute numbers.
- The live condition here decodes partials offline on growing prefixes. It doesn't use a stream.
