import json, os
import torch
GROUP = 32

def _unpack_nibbles(p, n):
    return torch.stack([p & 15, p >> 4], 1).reshape(-1)[:n]

def _dequant(codes, s, zp, m):
    R, C, pad = m['rc']
    D = (codes.float() - zp.float()[:, None]) * s[:, None]
    D = D.reshape(R, C + pad)[:, :C]
    shape, axis = (m['shape'], m['axis'])
    if axis == -1:
        D = D.reshape(shape)
    else:
        a = axis % len(shape)
        moved = list(shape)
        last = moved.pop(a)
        moved.append(last)
        D = D.reshape(moved).movedim(-1, a)
    return D.to(getattr(torch, m['dtype'])).contiguous()

def load_q4(path):
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(path, 'pt') as f:
        meta = f.metadata() or {}
    layout = json.loads(meta.get('int4_layout', '{}'))
    g = int(meta.get('group_size', GROUP))
    raw = load_file(path)
    out = {}
    for name, m in layout.items():
        R, C, pad = m['rc']
        n = R * (C + pad) // g
        codes = _unpack_nibbles(raw.pop(name + '.q'), n * g).reshape(n, g)
        zp = _unpack_nibbles(raw.pop(name + '.z'), n)
        out[name] = _dequant(codes, raw.pop(name + '.s'), zp, m)
    out.update(raw)
    return out

def load_state(folder, stem='model'):
    plain = os.path.join(folder, stem + '.safetensors')
    if os.path.exists(plain):
        from safetensors.torch import load_file
        return load_file(plain)
    q = os.path.join(folder, stem + '.q4.safetensors')
    if os.path.exists(q):
        return load_q4(q)
    raise FileNotFoundError(f'no {stem}.safetensors or {stem}.q4.safetensors in {folder}')

def load_moonshine(folder, decoder_only=False):
    from transformers import MoonshineStreamingForConditionalGeneration, MoonshineStreamingConfig, GenerationConfig
    if os.path.exists(os.path.join(folder, 'model.safetensors')) and (not decoder_only):
        return MoonshineStreamingForConditionalGeneration.from_pretrained(folder)
    cfg = MoonshineStreamingConfig.from_pretrained(folder)
    try:
        from transformers.initialization import no_init_weights
        with no_init_weights():
            m = MoonshineStreamingForConditionalGeneration(cfg)
    except ImportError:
        m = MoonshineStreamingForConditionalGeneration(cfg)
    sd = load_state(folder)
    miss, unexp = m.load_state_dict(sd, strict=False)
    if cfg.tie_word_embeddings:
        m.tie_weights()
        miss = [k for k in miss if k != 'proj_out.weight']
    if decoder_only:
        miss = [k for k in miss if not k.startswith('model.encoder.')]
    if miss or unexp:
        raise RuntimeError(f'weights mismatch: missing {miss[:5]} unexpected {unexp[:5]}')
    if os.path.exists(os.path.join(folder, 'generation_config.json')):
        m.generation_config = GenerationConfig.from_pretrained(folder)
    return m.eval()
