"""A simple stand-in for Steer's book catalog search (BookCatalog / LiveContext in Swift, not public).

61,476 Project Gutenberg books (data/catalog.tsv.gz: id, title, author, popularity). Fuzzy search over short titles and
authors, on the spelling and on its sound (metaphone), plus a small popularity prior. Steer's own search also has
genre bits, topic vectors and "more like this" embeddings; those are not reproduced here, so genre/topic requests are
not scored.
"""
import gzip
import os
import re
import unicodedata

import jellyfish
from rapidfuzz import fuzz, process

HERE = os.path.dirname(os.path.abspath(__file__))


STOP = {"the", "a", "an", "of", "and", "in", "on", "to", "by", "me", "book"}


def fold(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower().replace("'", "")).split())


def short(t):
    return re.split(r";|:| — | -- |, or,|, or | or, ", t)[0].strip(" ,.")


def sound(s):
    return " ".join(jellyfish.metaphone(w) for w in fold(s).split())


VERB = r"^(?:please |hey |okay |ok )?(?:can you |could you )?"
RX_AUTHOR = re.compile(VERB + r"(?:play |read |show me |find |give me )?(?:something|anything|some(?: books)?|books?|stories|stuff|works) (?:by|from|written by) (.+)$")
RX_AUTHOR2 = re.compile(VERB + r"what (?:did|has) (.+?) (?:write|written)$")
RX_LIKE = re.compile(VERB + r"(?:play |read |find |show me )?(?:something|anything|books?|a book) (?:like|similar to) (.+)$")
RX_LIST = re.compile(r"\b(books|stories|novels|tales|poetry|a book about)\b")
RX_TITLE = re.compile(VERB + r"(?:read|red|reed|rid|play|open|start|put on) (?:me )?(?:the book )?(.+)$")


def ask(text):
    """(kind, query) for a spoken book request, or None. kind: title | author | like | list"""
    t = fold(text)
    for kind, rx in (("author", RX_AUTHOR), ("author", RX_AUTHOR2), ("like", RX_LIKE)):
        m = rx.match(t)
        if m:
            return kind, m.group(1)
    if RX_LIST.search(t) and not RX_TITLE.match(t):
        return "list", t
    m = RX_TITLE.match(t)
    return ("title", m.group(1)) if m else None


class Catalog:
    def __init__(self, path=os.path.join(HERE, "data", "catalog.tsv.gz")):
        rows = [l.rstrip("\n").split("\t") for l in gzip.open(path, "rt", encoding="utf-8") if not l.startswith("#")]
        self.books = [dict(id=int(r[0]), title=r[1], author=r[2], pop=int(r[3])) for r in rows]
        self.titles, seen = [], {}
        for b in self.books:   # one entry per short title (most popular book first, as the file is ordered)
            k = fold(short(b["title"]))
            if k and k not in seen:
                seen[k] = len(self.titles)
                self.titles.append(dict(name=short(b["title"]), key=k, snd=sound(k), book=b))
        self.authors, seen = [], {}
        for b in self.books:
            if b["author"] and b["author"] not in seen:
                seen[b["author"]] = len(self.authors)
                self.authors.append(dict(name=b["author"], key=fold(b["author"]), snd=sound(b["author"]), book=b))
        self._tk = [x["key"] for x in self.titles]
        self._ts = [x["snd"] for x in self.titles]
        self._ak = [x["key"] for x in self.authors]
        self._as = [x["snd"] for x in self.authors]
        self._index = {}   # word -> short titles that have it (for queries that name part of a title)
        for i, x in enumerate(self.titles):
            for w in set(x["key"].split()):
                self._index.setdefault(w, []).append(i)

    def _search(self, q, keys, snds, items, limit):
        q = fold(q)
        if not q:
            return []
        sc = {}
        for ks, qq in ((keys, q), (snds, sound(q))):
            for _, s, i in process.extract(qq, ks, scorer=fuzz.QRatio, limit=limit * 3):
                sc[i] = max(sc.get(i, 0), s + 10 * items[i]["book"]["pop"] / 255)
        if items is self.titles:   # every content word of the query in the title ("sherlock holmes")
            ws = [w for w in q.split() if w not in STOP]
            if ws and all(w in self._index for w in ws):
                common = set(self._index[ws[0]]).intersection(*(self._index[w] for w in ws[1:]))
                for i in common:
                    extra = len(self.titles[i]["key"].split()) - len(ws)
                    sc[i] = max(sc.get(i, 0), 92 - 2 * min(extra, 6) + 10 * items[i]["book"]["pop"] / 255)
        return [(items[i], s) for i, s in sorted(sc.items(), key=lambda x: -x[1])[:limit]]

    def title(self, q, limit=1):
        return self._search(q, self._tk, self._ts, self.titles, limit)

    def author(self, q, limit=1):
        return self._search(q, self._ak, self._as, self.authors, limit)

    def candidates(self, text, k=40):
        """Live context: titles and authors that a (partial) transcript may be asking for."""
        a = ask(text)
        q = a[1] if a else fold(text)
        hits = self.title(q, k) + self.author(q, k // 2)
        hits.sort(key=lambda x: -x[1])
        out, seen = [], set()
        for item, _ in hits:
            kind = "person" if item in self.authors else "work_product"
            if item["name"] not in seen:
                seen.add(item["name"])
                out.append(dict(name=item["name"], kind=kind))
        return out[:k]

    def top(self, n_titles=1500, n_authors=400):
        return [x["name"] for x in self.titles[:n_titles]] + [x["name"] for x in self.authors[:n_authors]]

    def score(self, text):
        """How well a transcript matches a book (0-100), or None if it is not a title/author request."""
        a = ask(text)
        if not a or a[0] == "list":
            return None
        hits = self.author(a[1]) if a[0] == "author" else self.title(a[1])
        return hits[0][1] if hits else 0
