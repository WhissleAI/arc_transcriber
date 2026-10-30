"""Tokenizer reference cases: TSV lines 'E<TAB>text<TAB>ids' (encode, add_special_tokens=False) and
'D<TAB>ids<TAB>text' (decode, skip_special_tokens=True, stripped).

Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
"""
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
meta = json.load(open(sys.argv[2]))
texts = [l.split('\t')[2].strip() for l in open(sys.argv[3])]
texts += ['', 'a', 'Priya Raman', 'Dr. Okafor', 'Whissle', 'naïve café — “quoted” text', '日本語 and emoji 🎉', '  double  spaces ',
          "it's 10:30 on Tuesday", 'pick and roll', 'Theo Brandt', 'nba', 'hello\nworld', 'x' * 50, 'Ωmega ñ ü ß']
with open(sys.argv[4], 'w') as f:
    for t in texts:
        f.write('E\t' + json.dumps(t) + '\t' + ' '.join(map(str, tok(t, add_special_tokens=False).input_ids)) + '\n')
    for m in meta:
        f.write('D\t' + ' '.join(map(str, m['ids'])) + '\t' + json.dumps(tok.decode(m['ids'], skip_special_tokens=True).strip()) + '\n')
