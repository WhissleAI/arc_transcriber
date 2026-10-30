"""Spoken book search: Arc's transcript (with no context, a static catalog list, or live context from the partials)
-> book request -> catalog search -> right book first? And Arc's own name linker on the same clips. See README.md.
  python evals/book_search/run.py [--part search|linker|both] [--clips DIR] [--voices Samantha,Daniel,Flo,Aman]
"""
import argparse
import collections
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
from catalog import Catalog, ask, fold, short  # noqa: E402
from arc_audio.api import context_dict  # noqa: E402


def clips(a, queries):
    out = []
    for i, (said, exp) in enumerate(queries, 1):
        for v in a.voices.split(","):
            p = os.path.join(a.clips, f"{i:02d}_{v[:3]}.wav")
            if not os.path.exists(p):
                C.render(said, v, p)
            out.append((os.path.basename(p), said, exp, C.load(p)))
    return out


def expect(exp):
    kind, rest = exp.split(":", 1)
    if kind == "title":
        t, au = rest.split("|")
        return kind, [fold(w) for w in t.split()], fold(au)
    if kind == "author":
        return kind, [fold(rest)], fold(rest)
    if kind == "like":
        sid, t = rest.split("|")
        return kind, [fold(w) for w in t.split()], int(sid)
    return kind, [], None


def right(cat, text, e):
    kind, words, key = e
    a = ask(text)
    if not a:
        return False
    if kind == "title" and a[0] == "title":
        h = cat.title(a[1])
        return bool(h) and all(w in fold(h[0][0]["book"]["title"]) for w in words) and key in fold(h[0][0]["book"]["author"])
    if kind == "author" and a[0] in ("author", "title"):
        h = cat.author(a[1])
        return bool(h) and key in fold(h[0][0]["name"])
    if kind == "like" and a[0] == "like":
        h = cat.title(a[1])
        return bool(h) and all(w in fold(h[0][0]["book"]["title"]) for w in words)
    return False


def ctx(cands):
    return dict(ctx=context_dict(names=cands, topic="assistant"))


def search_part(arc, cat, items):
    static = cat.top(1500, 400)
    conds = ["none", "static", "live", "rag2", "live+rag2"]
    heard = collections.defaultdict(lambda: collections.Counter())
    ok = collections.defaultdict(lambda: collections.Counter())
    n = collections.Counter()
    ms = collections.defaultdict(list)
    after_end, lookups, partial_ms = [], [], []
    rows = []
    for f, said, exp, wav in items:
        e = expect(exp)
        if e[0] in ("genre", "topic"):
            continue
        n[e[0]] += 1
        texts = {}
        t0 = time.perf_counter()
        texts["none"] = C.text_of(arc.transcribe(wav))
        ms["none"].append(1000 * (time.perf_counter() - t0))
        t0 = time.perf_counter()
        texts["static"] = C.text_of(arc.transcribe(wav, names=static, bias_names=True))
        ms["static"].append(1000 * (time.perf_counter() - t0))
        # live: a partial every 0.5 s looks the catalog up; its candidates become the decoder's names prompt + bias
        cands = []
        upto = 8000
        while upto < len(wav):
            t0 = time.perf_counter()
            p = C.text_of(arc.transcribe(wav[:upto], cfg=dict(final_beams=1), context=ctx(cands) if cands else None,
                                         names=[c["name"] for c in cands] or None, bias_names=bool(cands)))
            partial_ms.append(1000 * (time.perf_counter() - t0))
            t1 = time.perf_counter()
            got = cat.candidates(p, 40) if p else []
            lookups.append(1000 * (time.perf_counter() - t1))
            if got:
                cands = got
            upto += 8000
        t0 = time.perf_counter()
        texts["live"] = C.text_of(arc.transcribe(wav, context=ctx(cands) if cands else None,
                                                 names=[c["name"] for c in cands] or None, bias_names=bool(cands)))
        after_end.append(1000 * (time.perf_counter() - t0))
        ms["live"].append(after_end[-1])

        def rag2(first):   # the first transcript looks the catalog up; heard again with what it found if unclear
            before = cat.score(first)
            if before is None or before >= 95:
                return first
            got = cat.candidates(first, 20)
            if not got:
                return first
            again = C.text_of(arc.transcribe(wav, context=ctx(got), names=[c["name"] for c in got], bias_names=True))
            return again if (cat.score(again) or 0) > before + 3 else first
        t0 = time.perf_counter()
        texts["rag2"] = rag2(texts["none"])
        ms["rag2"].append(ms["none"][-1] + 1000 * (time.perf_counter() - t0))
        t0 = time.perf_counter()
        texts["live+rag2"] = rag2(texts["live"])
        ms["live+rag2"].append(ms["live"][-1] + 1000 * (time.perf_counter() - t0))
        for c in conds:
            if all(w in fold(texts[c]) for w in e[1]):
                heard[e[0]][c] += 1
            if right(cat, texts[c], e):
                ok[e[0]][c] += 1
        rows.append(dict(clip=f, said=said, exp=exp, **{c: texts[c] for c in conds},
                         right={c: right(cat, texts[c], e) for c in conds}))
        print(f, "|", texts["none"], "|", texts["live"], file=sys.stderr)
    oracle = sum(right(cat, r["said"], expect(r["exp"])) for r in rows)
    table = {}
    for k in ("title", "author", "like", "all"):
        kinds = ["title", "author", "like"] if k == "all" else [k]
        N = sum(n[x] for x in kinds)
        table[k] = {c: dict(words_pct=100 * sum(heard[x][c] for x in kinds) / max(1, N),
                            right_pct=100 * sum(ok[x][c] for x in kinds) / max(1, N), n=N) for c in conds}
    print(f"\n{len(rows)} clips (title/author/like); search right on the true words: {oracle}/{len(rows)}")
    print("kind     " + "".join(f"{c:>18}" for c in conds) + "   (words heard / right book first)")
    for k, t in table.items():
        print(f"{k:8} " + "".join(f"{t[c]['words_pct']:7.0f}% / {t[c]['right_pct']:4.0f}%  " for c in conds))
    timing = dict(ms_median={c: C.quantile(ms[c], 0.5) for c in conds}, ms_p90={c: C.quantile(ms[c], 0.9) for c in conds},
                  live_final_after_end_ms=[C.quantile(after_end, 0.5), C.quantile(after_end, 0.9)],
                  partial_decode_ms=[C.quantile(partial_ms, 0.5), C.quantile(partial_ms, 0.9)],
                  catalog_lookup_ms=[C.quantile(lookups, 0.5), C.quantile(lookups, 0.9)])
    print("timing", timing)
    return dict(table=table, oracle=[oracle, len(rows)], timing=timing, rows=rows)


def linker_part(arc, cat, items, n_list=2000):
    """Arc's own name linking (entity heads + rescorer, optionally kb_snap) with catalog titles as the list."""
    books = cat.books[:n_list]
    kb, seen = [], set()
    for b in books:
        s = short(b["title"])
        if s.lower() not in seen:
            seen.add(s.lower())
            kb.append(dict(name=s, kind="work_product", aliases=[s[4:]] if s.lower().startswith("the ") else []))
    authors = []
    for b in books:
        if b["author"] and b["author"] not in authors:
            authors.append(b["author"])
    kb += [dict(name=x, kind="person") for x in authors[: max(50, n_list // 4)]]

    def score(rs):
        t = collections.defaultdict(lambda: [0, 0, 0])
        for r in rs:
            kind, words, key = expect(r["exp"])
            if kind == "title":
                good = any(all(w in fold(x) for w in words) for x in r["linked"][:1])
            elif kind == "author":
                good = any(key in fold(x) for x in r["linked"][:1])
            else:
                continue
            t[kind][0] += 1
            t[kind][1] += good
            t[kind][2] += bool(r["linked"]) and not good
        n = sum(v[0] for v in t.values())
        return dict({k: dict(right=v[1], n=v[0], wrong_link=v[2]) for k, v in t.items()},
                    all=dict(right=sum(v[1] for v in t.values()), n=n, wrong_link=sum(v[2] for v in t.values())))

    out = {}
    for name, snap in ((f"top {n_list} titles as the list", False), (f"top {n_list} titles + kb_snap", True)):
        rs = []
        for f, said, exp, wav in items:
            if not exp.startswith(("title", "author")):
                continue
            t0 = time.perf_counter()
            o = arc.predict(wav, domain="assistant", names=kb, return_frames=False, events=None, kb_snap=snap)
            linked = [x["entity"] for x in o["names"] if x.get("state") == "linked" and x.get("entity")]
            rs.append(dict(clip=f, exp=exp, text=C.text_of(o["transcript"]), linked=linked, ms=1000 * (time.perf_counter() - t0)))
        out[name] = dict(score=score(rs), ms_median=C.quantile([r["ms"] for r in rs], 0.5), rows=rs)
        print(name, out[name]["score"], file=sys.stderr)
    # live RAG + linker: pass 1 with no list, the catalog's top 20 for that text become the list for pass 2
    rs = []
    for f, said, exp, wav in items:
        if not exp.startswith(("title", "author")):
            continue
        t0 = time.perf_counter()
        first = C.text_of(arc.transcribe(wav))
        got = cat.candidates(first, 20)
        o = arc.predict(wav, domain="assistant", names=got, return_frames=False, events=None)
        linked = [x["entity"] for x in o["names"] if x.get("state") == "linked" and x.get("entity")]
        rs.append(dict(clip=f, exp=exp, text=C.text_of(o["transcript"]), linked=linked, ms=1000 * (time.perf_counter() - t0)))
    out["top 20 candidates (live RAG) as the list"] = dict(score=score(rs), ms_median=C.quantile([r["ms"] for r in rs], 0.5), rows=rs)
    print("\nArc name linker (title/author clips): right / n, wrong links")
    for name, r in out.items():
        s = r["score"]
        print(f"  {name:42} all {s['all']['right']}/{s['all']['n']} ({C.pct(s['all']['right'], s['all']['n'])}), wrong {s['all']['wrong_link']}; "
              f"title {s.get('title', {}).get('right')}/{s.get('title', {}).get('n')}, author {s.get('author', {}).get('right')}/{s.get('author', {}).get('n')}; "
              f"median {r['ms_median']:.0f} ms")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="both", choices=["search", "linker", "both"])
    ap.add_argument("--voices", default=",".join(C.VOICES))
    ap.add_argument("--clips", default=os.path.join(C.WORK, "book_search"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "run.json"))
    a = ap.parse_args()
    queries = C.read_tsv(os.path.join(HERE, "data", "book_queries.tsv"))
    cat = Catalog()
    items = clips(a, queries)
    arc = C.load_arc(a.threads)
    out = dict(env=C.env_info(arc, a.threads), clips=len(items), catalog_books=len(cat.books))
    if a.part in ("search", "both"):
        out["search"] = search_part(arc, cat, items)
    if a.part in ("linker", "both"):
        out["linker"] = linker_part(arc, cat, items)
    C.save(a.out, out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
