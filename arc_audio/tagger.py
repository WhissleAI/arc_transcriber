"""Arc Tagger (arc-tagger-c) on each NFT2 final, through its C library (libarctag.so, the public arctag.h API) with ctypes:

    arc.set_tagger('/path/to/arc-tagger-c/model', lib='/path/to/libarctag.so', domain='kids')
    out = arc.predict(wav, transcript='nft2', ...)       # each utterance: 'tagger' = the tagger's read-out of its words

The read-out is the tagger's JSON (entities, numbers, PII, safety, intent, dialogue act, emotion, ...) for the utterance's
words with their times and confidences. libarc does the same with ARC_WITH_ARCTAG (arc_nft2_set_tagger, arc_utt.tagger_json).
Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
"""
import ctypes
import json
import os


class _Opt(ctypes.Structure):
    _fields_ = [('size', ctypes.c_size_t), ('precision', ctypes.c_int), ('threads', ctypes.c_int),
                ('model_file', ctypes.c_char_p), ('backend', ctypes.c_int)]


class ArcTagger:
    def __init__(self, model_dir, lib=None, domain='kids', int8=True):
        lib = lib or os.environ.get('ARCTAG_LIB') or 'libarctag.so'
        self.L = L = ctypes.CDLL(lib)
        L.arctag_options_init.argtypes = [ctypes.POINTER(_Opt)]
        L.arctag_create_ex.restype = ctypes.c_void_p
        L.arctag_create_ex.argtypes = [ctypes.c_char_p, ctypes.POINTER(_Opt), ctypes.c_char_p, ctypes.c_size_t]
        L.arctag_set_option.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        L.arctag_tag_words.restype = ctypes.c_void_p
        L.arctag_tag_words.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.arctag_free_str.argtypes = [ctypes.c_void_p]
        L.arctag_destroy.argtypes = [ctypes.c_void_p]
        o = _Opt()
        L.arctag_options_init(ctypes.byref(o))
        o.precision = 1 if int8 else 0
        o.backend = 1                       # native: plain C, no ONNX Runtime needed
        err = ctypes.create_string_buffer(512)
        self.h = L.arctag_create_ex(str(model_dir).encode(), ctypes.byref(o), err, 512)
        if not self.h:
            raise RuntimeError(f'arc tagger: {err.value.decode(errors="replace")}')
        if L.arctag_set_option(self.h, b'domain', str(domain or 'kids').encode()):
            raise ValueError(f'arc tagger: unknown domain {domain}')

    def tag_words(self, words):
        """utterance words (dicts with text, start, end, conf) -> the tagger's read-out (dict)"""
        if not words:
            return None
        ws = []
        for i, w in enumerate(words):
            d = {'w': str(w['text']), 't0': round(float(w['start']), 3), 't1': round(float(w['end']), 3)}
            if w.get('conf') is not None:
                d['conf'] = round(float(w['conf']), 4)
            d['final'] = True
            if i == len(words) - 1:
                d['eou'] = True
            ws.append(d)
        p = self.L.arctag_tag_words(self.h, json.dumps(ws).encode())
        if not p:
            return None
        try:
            return json.loads(ctypes.string_at(p).decode('utf-8', errors='replace'))
        finally:
            self.L.arctag_free_str(p)

    def close(self):
        if getattr(self, 'h', None):
            self.L.arctag_destroy(self.h)
            self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
