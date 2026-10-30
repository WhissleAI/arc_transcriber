"""Output formats: JSON (Arc's records), plain text with inline tags, SRT and WebVTT.

written=True renders the written transcript instead (transcript_written: numbers as digits, "2:30 p.m.", "$45", "March 3"; linked
names in the list's spelling), without inline tags: the text to show, or to score against written references.

Inline tags follow each name / number span in the text:
    [Theo Brandt](name: Theo Brandt)      a name linked to your list (entity in the list's spelling)
    [Brant](name)                         a name that is not in your list
    [102 to 98](score: 102-98)            a number with its kind and value
"""
import json



def _fmt_value(n):
    v, v2, k, u = n.get("value"), n.get("value2"), n.get("kind"), n.get("unit")
    if v is None:
        return None
    num = lambda x: str(int(x)) if float(x).is_integer() else f"{x:g}"
    if k == "date" and n.get("iso"):
        return n["iso"]
    if k == "time_of_day":
        h, m = divmod(int(v) // 60, 60)
        return f"{h:02d}:{m:02d}"
    if v2 is not None:
        return f"{num(v)}-{num(v2)}"
    if k in ("duration", "clock"):
        return f"{num(v)} s"
    if k == "money" and u and u != "XXX":
        return f"{u} {num(v)}"
    if k == "percent":
        return f"{num(v)}%"
    return num(v)


def lines_of(out, gap_s=0.8, max_s=12.0, written=False):
    """transcript lines [{start, end, text, words: [{text, start, end}], speaker}] from an Arc result: the built-in transcript's
    utterances, or external words grouped at pauses / sentence ends. written=True: the lines of transcript_written."""
    tr = (out.get("transcript_written") if written else None) or out.get("transcript") or []
    if tr and "words" in tr[0]:
        lines = [dict(start=u["start"], end=u["end"], words=[dict(text=w["text"], start=w["start"], end=w["end"]) for w in u["words"]] or
                      [dict(text=u["text"], start=u["start"], end=u["end"])]) for u in tr]
    else:
        lines, cur = [], []
        for w in sorted(tr, key=lambda w: w["start"]):
            if cur and (w["start"] - cur[-1]["end"] > gap_s or w["end"] - cur[0]["start"] > max_s or cur[-1]["text"].rstrip().endswith((".", "?", "!"))):
                lines.append(cur); cur = []
            cur.append(dict(text=w["text"], start=w["start"], end=w["end"]))
        if cur:
            lines.append(cur)
        lines = [dict(start=c[0]["start"], end=c[-1]["end"], words=c) for c in lines]
    diar = out.get("diarization") or []
    for ln in lines:
        best = max(diar, key=lambda d: min(d["end"], ln["end"]) - max(d["start"], ln["start"]), default=None)
        ln["speaker"] = best["speaker"] if best and min(best["end"], ln["end"]) > max(best["start"], ln["start"]) else None
        ln["text"] = " ".join(w["text"] for w in ln["words"])
    return lines


def _bare(t):
    import re
    return re.sub(r"^[^\w$%]+|[^\w%']+$", "", str(t)).lower()


def _inside(w, r):
    m = (w["start"] + w["end"]) / 2
    return r["start"] - 0.05 <= m <= r["end"] + 0.05


def tagged_text(line, names, numbers):
    """line text with inline tags after each name / number span (records overlapping the line's words)."""
    ws = line["words"]; marks = {}
    for n in names:
        if not n.get("show", True):
            continue
        if n.get("words"):                       # the record's own words (exact placement)
            own = {(round(float(x["start"]), 2), x["w"]) for x in n["words"]}
            idx = [k for k, w in enumerate(ws) if (round(float(w["start"]), 2), w["text"]) in own]
            if not idx:                              # words from another transcript path: place by time
                idx = [k for k, w in enumerate(ws) if _inside(w, n)]
        else:
            idx = [k for k, w in enumerate(ws) if _inside(w, n)]
        if idx:
            lab = f"name: {n['entity']}" if n.get("state") == "linked" and n.get("entity") else "name"
            marks.setdefault(idx[0], []).append((idx[-1], lab))
    for n in numbers:
        val = _fmt_value(n)
        if val is None:
            continue
        idx = [k for k, w in enumerate(ws) if _inside(w, n)]
        heard = {_bare(x) for x in str(n.get("heard") or "").split()}
        while len(idx) > 1 and heard and _bare(ws[idx[-1]]["text"]) not in heard:
            idx = idx[:-1]
        while len(idx) > 1 and heard and _bare(ws[idx[0]]["text"]) not in heard:
            idx = idx[1:]
        if idx:
            marks.setdefault(idx[0], []).append((idx[-1], f"{n.get('kind') or 'number'}: {val}"))
    out, k = [], 0
    while k < len(ws):
        if k in marks:
            end, lab = max(marks[k])
            out.append(f"[{' '.join(w['text'] for w in ws[k:end + 1])}]({lab})"); k = end + 1
        else:
            out.append(ws[k]["text"]); k += 1
    return " ".join(out)


def _ts(t, sep):
    t = max(0.0, float(t)); h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d}{sep}{int(round((s - int(s)) * 1000)) % 1000:03d}"


def render(out, fmt, written=False):
    """fmt: json | txt | srt | vtt -> str. written=True: txt / srt / vtt show the written transcript (no inline tags); json always
    carries transcript_written, and each line its written text."""
    names, numbers = out.get("names") or [], out.get("numbers") or []
    written = written and bool(out.get("transcript_written"))
    lines = lines_of(out, written=written)
    tagged = (lambda l: l["text"]) if written else (lambda l: tagged_text(l, names, numbers))
    if fmt == "json":
        keep = {k: v for k, v in out.items() if k not in ("frames",)}
        lines = lines_of(out)
        wl = lines_of(out, written=True) if out.get("transcript_written") else []
        keep["lines"] = [dict(start=l["start"], end=l["end"], speaker=l["speaker"], text=l["text"], tagged=tagged_text(l, names, numbers),
                              **({"written": wl[k]["text"]} if len(wl) == len(lines) else {})) for k, l in enumerate(lines)]
        return json.dumps(keep, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    if fmt == "txt":
        return "\n".join(f"[{l['start']:7.2f}-{l['end']:7.2f}]{' ' + l['speaker'] + ':' if l['speaker'] else ''} {tagged(l)}"
                         for l in lines) + "\n"
    if fmt in ("srt", "vtt"):
        sep = "," if fmt == "srt" else "."
        cues = []
        for k, l in enumerate(lines, 1):
            txt = (f"{l['speaker']}: " if l["speaker"] else "") + tagged(l)
            head = f"{k}\n" if fmt == "srt" else ""
            cues.append(f"{head}{_ts(l['start'], sep)} --> {_ts(l['end'], sep)}\n{txt}\n")
        return ("WEBVTT\n\n" if fmt == "vtt" else "") + "\n".join(cues)
    raise ValueError("format must be json | txt | srt | vtt")
