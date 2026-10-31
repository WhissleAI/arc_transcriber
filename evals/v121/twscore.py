"""Python port of the TypeWhisper benchmark's English scoring (github.com/TypeWhisper/typewhisper-benchmark):

- normalize_text  = bench/src/metrics/normalize.ts  normalizeText(text, "en")
- wer             = bench/src/metrics/wer.ts        calculateWER (clamped to 1, both-empty 0, one-empty 1)
- digits_hit      = visualizer/scripts/build-data.mjs  digitsHit (every reference number verbatim, as a whole number)

`check_port.py` recomputes their published per-clip `werNormalized` and Digits from their published transcriptions with this
port; it must agree exactly before any number here is compared with theirs.
"""
import re

CONTRACTIONS = {
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would", "you're": "you are", "you've": "you have",
    "you'll": "you will", "you'd": "you would", "he's": "he is", "he'll": "he will", "he'd": "he would", "she's": "she is",
    "she'll": "she will", "she'd": "she would", "it's": "it is", "it'll": "it will", "it'd": "it would", "we're": "we are",
    "we've": "we have", "we'll": "we will", "we'd": "we would", "they're": "they are", "they've": "they have",
    "they'll": "they will", "they'd": "they would", "that's": "that is", "that'll": "that will", "that'd": "that would",
    "who's": "who is", "who'll": "who will", "who'd": "who would", "what's": "what is", "what'll": "what will",
    "what'd": "what would", "where's": "where is", "where'll": "where will", "where'd": "where would", "when's": "when is",
    "when'll": "when will", "when'd": "when would", "why's": "why is", "why'll": "why will", "why'd": "why would",
    "how's": "how is", "how'll": "how will", "how'd": "how would", "there's": "there is", "there'll": "there will",
    "there'd": "there would", "here's": "here is", "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not", "hadn't": "had not", "doesn't": "does not",
    "don't": "do not", "didn't": "did not", "won't": "will not", "wouldn't": "would not", "shan't": "shall not",
    "shouldn't": "should not", "can't": "cannot", "couldn't": "could not", "mustn't": "must not", "mightn't": "might not",
    "needn't": "need not", "let's": "let us", "ain't": "is not", "o'clock": "of the clock", "ma'am": "madam",
}
EN_FILLERS = {"uh", "uhm", "um", "umm", "er", "err", "ah", "ahh", "hm", "hmm", "huh", "mhm", "mm", "mmm"}
ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
        "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _hundreds(n):
    out = []
    if n >= 100:
        out.append(ONES[n // 100] + " hundred")
        n %= 100
    if n >= 20:
        out.append(TENS[n // 10] + (" " + ONES[n % 10] if n % 10 else ""))
    elif n > 0:
        out.append(ONES[n])
    return " ".join(out)


def number_to_english(n):
    if n == 0:
        return "zero"
    out = []
    if n >= 1000:
        out.append(_hundreds(n // 1000) + " thousand")
        n %= 1000
    if n > 0:
        out.append(_hundreds(n))
    return " ".join(out)


# JS \b and [a-z] are ASCII; \p{L}\p{N} are Unicode letters / numbers (Python's str.isalnum is close: it also counts
# a few numeric symbols, which never occur in these suites).
_CONTR = re.compile(r"(?<![A-Za-z0-9_])[a-z]+['’][a-z]+(?![A-Za-z0-9_])")
_INT = re.compile(r"(?<![A-Za-z0-9_])[0-9]+(?![A-Za-z0-9_])")


def normalize_text(text):
    if text == "":
        return ""
    s = text.lower()
    s = _CONTR.sub(lambda m: CONTRACTIONS.get(m.group(0).replace("’", "'"), m.group(0)), s)
    s = "".join(c if (c.isalpha() or c.isdigit() or c.isnumeric() or c.isspace()) else " " for c in s)
    s = _INT.sub(lambda m: number_to_english(int(m.group(0))) if int(m.group(0)) <= 999999 else m.group(0), s)
    s = " ".join(w for w in s.split() if w not in EN_FILLERS)
    return " ".join(s.split())


def edit_distance(a, b):
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for i in range(1, len(b) + 1):
        cur = [i] + [0] * len(a)
        bi = b[i - 1]
        for j in range(1, len(a) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[j - 1] != bi))
        prev = cur
    return prev[len(a)]


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    if not r and not h:
        return 0.0
    if not r or not h:
        return 1.0
    return min(edit_distance(r, h) / len(r), 1.0)


def wer_normalized(ref, hyp):
    return wer(normalize_text(ref), normalize_text(hyp))


# ---------------------------------------------------------------- Digits (build-data.mjs)
def number_runs(reference):
    runs, run = [], []
    for raw in reference.split():
        token = re.sub(r"[.,;:!?]+$", "", raw) if re.search(r"\d", raw) else ""
        if token:
            run.append(token)
        if not token or token != raw:
            if run:
                runs.append(run)
            run = []
    if run:
        runs.append(run)
    return runs


def _whole(pattern):
    return re.compile(r"(?<!\d)(?<!\d[.,:])" + pattern + r"(?!\d)(?![.,:]\d)")


def digits_hit(reference, transcript):
    runs = number_runs(reference)
    if not runs:
        return None
    text = transcript or ""
    return all(all(_whole(re.escape(t)).search(text) for t in run) or _whole(r"\s*".join(map(re.escape, run))).search(text)
               for run in runs)


# ---------------------------------------------------------------- suites and pooling (visualizer/src/data/metrics.ts)
EN_SUITES = ["clean-speech-en", "clean-speech-librispeech-en", "clean-speech-commonvoice-en", "noisy-environment-en",
             "whispered-lowquality-en", "hard-en", "code-dictation-en", "number-formatting-en", "punctuation-formatting-en",
             "accented-speech-en"]
DIAGNOSTIC = {"hard-en", "code-dictation-en"}       # tier "diagnostic": not in the default WER EN pool
ACCENTED = "accented-speech-en"


def pools(per_suite_rows):
    """per_suite_rows: {suite: [wer, ...]}. WER EN pools clips (not suites). Returns the PR table's two columns plus the
    core-tier pool the website shows by default."""
    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")
    allc = [w for s, ws in per_suite_rows.items() for w in ws]
    no_acc = [w for s, ws in per_suite_rows.items() if s != ACCENTED for w in ws]
    core = [w for s, ws in per_suite_rows.items() if s != ACCENTED and s not in DIAGNOSTIC for w in ws]
    return dict(wer_en_all=mean(allc), wer_en_no_acc=mean(no_acc), wer_en_core_no_acc=mean(core))
