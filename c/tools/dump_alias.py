"""Fixtures for c/tests/test_alias.c: arc_audio kbsnap.py AliasIndex (the list check engine.py _name_extras makes before a
name is left unlinked) and akey, on a table of lists and heard texts: whole names and aliases with punctuation / case /
spacing / possessive variants, unique and shared partial names, command and stop words at the edges, stop / common /
short words that never form a partial, and the linker's allowed rows narrowing an ambiguous match.

    python tools/dump_alias.py tests/data/alias.jsonl

Lines: {"k": "akey", "text", "key"} and {"k": "lookup", "kb", "heard", "allowed" (null = no restriction), "row", "via"}.
Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from arc_audio.kbsnap import AliasIndex, akey  # noqa: E402

KBS = {
    'people': [dict(name='Daniel Brooks', aliases=['Dan Brooks']), dict(name='Priya Raman'), dict(name="De'Marco Hale"),
               dict(name='Nora Ellis'), dict(name='Riverside Comets', kind='team'), dict(name='Mary-Kate O\'Neil')],
    'two_daniels': [dict(name='Daniel Brooks'), dict(name='Daniel Ortiz'), dict(name='Priya Raman')],
    'same_name': [dict(name='Alex Kim', kind='person'), dict(name='Alex Kim', kind='team'), dict(name='Jordan Lee')],
    'alias_is_part': [dict(name='Brooks', kind='team'), dict(name='Daniel Brooks'), dict(name='Ortiz Group', kind='organization')],
    'common_words': [dict(name='The Will Smith Show', kind='work_product'), dict(name='Al Green'), dict(name='Big Time Rush'),
                     dict(name='Jo Ng'), dict(name='Apollo 12', kind='mission'), dict(name='Apollo 14', kind='mission')],
    'three_words': [dict(name='Mary Ann Evans'), dict(name='Ann Arbor', kind='place'), dict(name='Juan Carlos Ortiz Jr.')],
    'spacing': [dict(name='McDonald Hall', kind='place'), dict(name='Le Bron', aliases=['LeBron James']), dict(name='Jean-Luc Picard')],
}

HEARD = {
    'people': ['Daniel', 'daniel', 'Daniel.', "Daniel's", 'Daniel’s', 'call Daniel', 'Call Daniel please', 'Brooks', 'Dan',
               'Dan Brooks', 'dan brooks', 'Daniel Brooks', 'Priya', 'Raman', 'Priya Raman.', 'DeMarco', "De'Marco", 'Hale',
               'DeMarco Hale', 'demarcohale', 'De Marco Hale', 'Comets', 'the Comets', 'Riverside', 'Ellis', 'Nora', 'Mary-Kate',
               'MaryKate', 'O\'Neil', 'oneil', 'Mary Kate', 'the', 'call', 'please call', 'Danny', '', 'Daniel Ellis',
               '"Brooks,"', '(Nora)'],
    'two_daniels': ['Daniel', 'Brooks', 'Ortiz', 'Daniel Ortiz', 'Priya', 'remind Daniel'],
    'same_name': ['Alex Kim', 'Alex', 'Kim', 'alexkim', 'Jordan'],
    'alias_is_part': ['Brooks', 'Daniel', 'Ortiz', 'the Brooks', 'Group'],
    'common_words': ['Will', 'Smith', 'Will Smith', 'Smith Show', 'Show', 'Al', 'Green', 'Big', 'Time', 'Rush', 'Big Time', 'Jo',
                     'Ng', 'Apollo', '12', 'Apollo 12', 'apollo twelve'],
    'three_words': ['Mary', 'Ann', 'Evans', 'Mary Ann', 'Ann Evans', 'Mary Evans', 'Arbor', 'Juan Carlos', 'Carlos Ortiz',
                    'Ortiz', 'Jr', 'Juan Carlos Ortiz', 'juancarlos'],
    'spacing': ['McDonald', 'Mc Donald', 'mcdonald hall', 'Mc Donald Hall', 'LeBron', 'Le Bron', 'lebron james', 'Bron',
                'Picard', 'Jean Luc', 'JeanLuc', 'Jean-Luc', 'Luc'],
}

ALLOWED = [None, 'first', 'second', 'none', 'all']


def allowed_rows(mode, kb):
    return {None: None, 'first': [0], 'second': [1], 'none': [], 'all': list(range(len(kb)))}[mode]


def main():
    out = sys.argv[1]
    n = 0
    with open(out, 'w') as f:
        texts = sorted({h for hs in HEARD.values() for h in hs} | {c['name'] for kb in KBS.values() for c in kb} |
                       {"it's", "O'Brien's", "rock'n'roll", "  spaced\tout  ", "[Brackets]", "Ünïcode Name", "3:30", "a.b.c."})
        for t in texts:
            f.write(json.dumps(dict(k='akey', text=t, key=akey(t))) + '\n')
            n += 1
        for name, kb in KBS.items():
            idx = AliasIndex(kb)
            for h in HEARD[name]:
                for mode in ALLOWED:
                    al = allowed_rows(mode, kb)
                    hit = idx.lookup(h, None if al is None else set(al))
                    f.write(json.dumps(dict(k='lookup', list=name, kb=kb, heard=h, allowed=al, row=None if hit is None else hit[0],
                                            via=None if hit is None else hit[1])) + '\n')
                    n += 1
    print(out, n, 'cases', file=sys.stderr)


if __name__ == '__main__':
    main()
