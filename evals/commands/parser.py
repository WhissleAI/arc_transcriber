"""A small stand-in for Steer's command parser (VoiceCommand.swift, ~1,500 lines of Swift, not public).

It maps a transcript to one of the command kinds in data/*.tsv. It is only here so the bench can score "did Arc's words
lead to the right command"; run.py counts a clip right when parse(heard) == expected, and only on phrases where
parse(said) == expected, so the parser's own misses are reported separately and never charged to Arc.
Ordered rules, first match wins.
"""
import re

APPS = r"(whatsapp|slack|messages|telegram|mail|safari|discord|signal|notion|zoom|this app)"

RULES = [
    ("UNKNOWN", r"\b(time in|time difference|battery life of|horoscope|flight|boil|alarm|poem|weather|who wrote|who is|who won|"
                r"why is|how tall|what does \w+ mean|how many minutes|navigate|jazz|president|text \w+ that)\b"),
    ("MUTE", rf"\b(mute|silence)\b|\b(stop|don'?t) read(ing)? {APPS}|\b(stop|don'?t) read(ing)? \w+ messages"),
    ("TIME_LEFT", r"\bhow (long|much)( longer| of the book| time)?( is)? (left|until)|\bhow much longer\b"),
    ("SLEEP", r"\bsleep\b|\b(stop|turn off)( reading)? (in|at the end|when the chapter)|\bstop when\b|\bend of the chapter\b(?!.*how long)"),
    ("REPLY", r"^(please )?(reply|answer|tell (her|him|them))\b"),
    ("OPEN", r"\b(open|switch to|show me the message)\b"),
    ("CATCH_UP", r"\b(catch me up|fill me in|did i miss|what did i miss|have i missed|anything (i should|i need to) know|"
                 r"anything new since|what'?s been happening|missed)\b"),
    ("LAST_MESSAGE", r"\bwhat did (?!you|i)\w+ say\b|\b(text|texts|texted|message|messages|messaged|write|wrote)\b"),
    ("BATTERY", r"\b(battery|charge|juice|plug in|percent|phone dying|dying)\b"),
    ("CALENDAR", r"\b(calendar|schedule|meeting|meetings|appointments?|free (tonight|at)|busy|coming up|later\b|my day|"
                 r"anything on today|what'?s next)"),
    ("DATE", r"\b(date|what day|which day|what'?s today|is it (monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b"),
    ("TIME", r"(?<!one more )\b(time|is it late)\b"),
    ("REPEAT_LAST", r"\b(say that again|repeat|one more time|again|didn'?t hear|what was that|sorry,? what|come again|"
                    r"what did you (just )?say)\b|^huh\b"),
    ("SUMMARIZE", r"\b(summar|sum it up|gist|what'?s this about)"),
    ("SLOWER", r"\b(slow|slower|too fast)\b"),
    ("FASTER", r"\b(fast|faster|speed)\b"),
    ("LOUDER", r"\b(louder|speak up|can'?t hear)\b"),
    ("QUIETER", r"\b(quieter|too loud)\b"),
    ("NEXT", r"\b(next|skip)\b"),
    ("BACK", r"\b(go back|previous|rewind)\b"),
    ("READ_SELECTION", r"\bread (this|what i selected)\b"),
    ("PAUSE", r"\b(pause|hold on|wait)\b"),
    ("RESUME", r"\b(resume|carry on|continue|keep (going|reading)|go on|play)\b"),
    ("STOP", r"\b(stop|quiet|never mind|enough|shut up)\b"),
]
RULES = [(k, re.compile(p)) for k, p in RULES]


def clean(text):
    t = text.lower().replace("’", "'")
    t = re.sub(r"^(hey |ok |okay )?steer[,.]? ", "", t)
    t = re.sub(r"[^a-z0-9' ,?]+", " ", t)
    return re.sub(r"\s+", " ", t).strip(" ,")


def parse(text):
    t = clean(text)
    if not t:
        return "NONE"
    for kind, rx in RULES:
        if rx.search(t):
            return kind
    return "UNKNOWN"
