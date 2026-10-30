"""Small reproductions of the API issues found while integrating Arc (Steer for Mac, the Pipecat service).
  python evals/api_issues/run.py
1. A short command's final transcript only arrives when the 15 s max segment closes (min_s = 1.0 s gate on the pause).
2. transcript_rewritten can duplicate a token when the linked span covers only part of the name.
3. Arc.stream() changes the process-wide sys.setswitchinterval while it runs.
4. The two device-directed numbers in one predict() output disagree.
"""
import os
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402

CLIPS = os.path.join(C.WORK, "api_issues")


def clip(text, voice="Samantha", after=0.8):
    w = C.load(C.render(text, voice, os.path.join(CLIPS, f"{voice}_{C.slug(text)}.wav")))
    return C.room(w, -60, level=0.35, before=0.5, after=after, seed=1), 0.5 + len(w) / C.SR


def main():
    arc = C.load_arc(4)
    out = {}

    # 1. final transcript of short vs longer commands, stream fed 100 ms at a time (audio clock; no worker thread)
    print("1. final transcript record after the speech ends (stream, 20 s of silence after the speech)")
    r1 = []
    for text in ["Pause.", "Open WhatsApp.", "Sleep in twenty minutes.", "Stop reading Slack for an hour."]:
        x, end = clip(text, after=20)
        s = arc.stream(domain="assistant", background=False)
        first = partial = None
        for i in range(0, len(x), 1600):
            o = s.feed(x[i:i + 1600])
            pos = (i + 1600) / C.SR - end
            if partial is None and (o.get("transcript_partial") or {}).get("text"):
                partial = pos
            if first is None and o.get("transcript"):
                first = pos
        r1.append(dict(said=text, speech_s=round(end - 0.5, 2), first_partial_s=partial, final_s=first))
        print(f"   {text:34} speech {end - 0.5:.2f} s  first partial {partial:+.1f} s  final {first:+.1f} s" if first is not None
              else f"   {text:34} speech {end - 0.5:.2f} s  final: none within 20 s")
    out["short_final"] = r1

    # 2. transcript_rewritten with a name list (the Pipecat test sentence; fictional names)
    print("2. transcript vs transcript_rewritten with names")
    x, _ = clip("Hi, it's Priya Raman, I need to move my cleaning with Doctor Okafor.")
    names = [dict(name="Priya Raman", kind="person"), dict(name="Dr. Okafor", kind="person"), dict(name="Acme Dental", kind="organization")]
    r2 = []
    for v in C.VOICES:
        x, _ = clip("Hi, it's Priya Raman, I need to move my cleaning with Doctor Okafor.", voice=v)
        o = arc.predict(x, domain="assistant", names=names, return_frames=False, events=None)
        a, b = C.text_of(o["transcript"]), C.text_of(o.get("transcript_rewritten") or [])
        ws = C.words(b)
        dup = any(ws[i] == ws[i + 1] for i in range(len(ws) - 1)) or len(ws) > len(C.words(a))
        linked = [(n.get("heard"), n.get("entity")) for n in o["names"] if n.get("state") == "linked"]
        r2.append(dict(voice=v, transcript=a, rewritten=b, linked=linked, extra_tokens=dup))
        print(f"   {v:9} transcript: {a}\n             rewritten:  {b}   linked {linked}{'   <- extra tokens' if dup else ''}")
    out["rewrite"] = r2

    # 3. the stream and the interpreter's switch interval
    before = sys.getswitchinterval()
    s = arc.stream(domain="assistant")
    x, _ = clip("Pause.")
    for i in range(0, len(x), 1600):
        s.feed(x[i:i + 1600])
    during = sys.getswitchinterval()
    s.flush()
    after = sys.getswitchinterval()
    out["switchinterval"] = dict(before=before, during_stream=during, after_flush=after)
    print(f"3. sys.getswitchinterval(): before {before}, while a stream runs {during}, after flush {after}")

    # 4. device-directed: utterance tag vs the transcript's head, on one clip
    x, _ = clip("Sleep in twenty minutes.")
    o = arc.predict(x, domain="assistant", return_frames=False, events=None)
    tag, head = C.directed_of(o)
    out["two_directed"] = dict(utterance_tag=tag, transcript_head=head)
    print(f"4. device-directed on one command: utterances[].tags {tag} vs transcript[].device_directed.p {head}")
    out["env"] = C.env_info(arc, 4)
    C.save(os.path.join(HERE, "results", "run.json"), out)


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"done in {time.time() - t:.0f} s")
