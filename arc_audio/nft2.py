"""Arc Transcriber on the NFT2 backbone (backbone 'nft2', transcript 'nft2').

NFT2 = a cache-aware streaming FastConformer hybrid (114M, 17 x 512, chunked_limited attention, causal convolutions) fine-tuned with its
own RNNT decoder. One encoder pass over an utterance feeds two readers:
  * the RNNT greedy decoder (optionally biased toward a names list with a boosting trie) -> the FINAL WORDS (transcript='nft2');
  * the Arc layer: adapters (decoder memory, Arc frames, event tap) on the same encoder output -> the Arc decoder (inline tags /
    entity spans, read with the context prompt), the Arc net (span checks, linking), the event and device-directed heads.
Decoder tags are mapped onto the RNNT words by word alignment (tagdec.map_spans), like decoder spans on external words.

Encoder: the full-sequence forward with the chunked_limited attention mask equals NeMo's cache-aware streaming output (same
receptive field per frame; the streaming buffer and caches only change when the computation happens). Frame i of the encoder output
covers [80 (i - 1), 80 i) ms; the first frame (which sees only mel frame 0) is dropped for the Arc layer (drop0) so Arc frame i sits on
[80 i, 80 i + 80) ms. The RNNT reads every frame.
Encoder backends (config.json backbone.encoder, Arc(..., encoder=...), or $ARC_NFT2_ENCODER):
  * 'onnx' (default): the streaming ONNX export in backbone.onnx_dir (encoder + RNNT graphs, tokens) on CPU onnxruntime
    (pip install "arc-transcriber[space]");
  * 'torch': transformers' NemotronAsrStreamingEncoder with the weights of a local NFT2 .nemo ($ARC_NFT2_NEMO or backbone.nemo).
    Nothing here needs NeMo: weights, config and tokenizer are read from the .nemo tar.
"""
import json, os, queue, re, tarfile, threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR, HOP, NFFT, WIN = 16000, 160, 512, 400
FRAME_STRIDE = 1280
_REN = [('.pre_encode.conv.0.', '.subsampling.conv_in.'), ('.pre_encode.conv.2.', '.subsampling.layers.0.depthwise_conv.'),
        ('.pre_encode.conv.3.', '.subsampling.layers.0.pointwise_conv.'), ('.pre_encode.conv.5.', '.subsampling.layers.1.depthwise_conv.'),
        ('.pre_encode.conv.6.', '.subsampling.layers.1.pointwise_conv.'), ('.pre_encode.out.', '.subsampling.linear.'),
        ('.pre_encode.conv.', '.subsampling.layers.'), ('.self_attn.linear_q.', '.self_attn.q_proj.'),
        ('.self_attn.linear_k.', '.self_attn.k_proj.'), ('.self_attn.linear_v.', '.self_attn.v_proj.'),
        ('.self_attn.linear_out.', '.self_attn.o_proj.'), ('.self_attn.linear_pos.', '.self_attn.relative_k_proj.'),
        ('.self_attn.pos_bias_u', '.self_attn.bias_u'), ('.self_attn.pos_bias_v', '.self_attn.bias_v'), ('.conv.batch_norm.', '.conv.norm.')]
LORA_TARGETS = dict(q=('self_attn.q_proj',), k=('self_attn.k_proj',), v=('self_attn.v_proj',), o=('self_attn.o_proj',),
                    ff=('feed_forward1.linear1', 'feed_forward1.linear2', 'feed_forward2.linear1', 'feed_forward2.linear2'),
                    pos=('self_attn.relative_k_proj',))


# ------------------------------------------------------------------------------------------------------------- .nemo reading
def fetch_nemo(spec):
    """backbone spec -> local .nemo path: $ARC_NFT2_NEMO, spec['nemo'] (absolute, or relative to the model folder), or a HF download"""
    p = os.environ.get('ARC_NFT2_NEMO')
    if p and os.path.exists(p):
        return p
    p = spec.get('nemo')
    if p:
        p = p if os.path.isabs(p) else os.path.join(spec.get('_dir', ''), p)
        if os.path.exists(p):
            return p
    if spec.get('repo'):
        from huggingface_hub import hf_hub_download
        return hf_hub_download(spec['repo'], spec['file'], revision=spec.get('revision'))
    raise FileNotFoundError("nft2 backbone: set ARC_NFT2_NEMO or backbone.nemo in config.json to the NFT2 .nemo file")


def read_nemo(path, keep=('encoder.', 'preprocessor.', 'decoder.', 'joint.')):
    """.nemo tar -> (state dict, model_config dict, tokenizer.vocab lines, tokenizer.model bytes); nothing is extracted to disk"""
    import yaml
    with tarfile.open(path) as tf:
        mem = {m.name: m for m in tf.getmembers()}
        get = lambda suf: next((tf.extractfile(m).read() for n, m in mem.items() if n.endswith(suf)), None)
        sd = torch.load(io_bytes(get('model_weights.ckpt')), map_location='cpu', weights_only=True)
        mc = yaml.safe_load(get('model_config.yaml').decode())
        voc = (get('tokenizer.vocab') or b'').decode('utf-8').splitlines()
        spm = get('tokenizer.model')
    return {k: v for k, v in sd.items() if k.startswith(tuple(keep))}, mc, voc, spm


def io_bytes(b):
    import io
    return io.BytesIO(b)


def map_keys(sd):
    out = {}
    for k, v in sd.items():
        if not k.startswith('encoder.'):
            continue
        k2 = '.' + k[len('encoder.'):]
        for a, b in _REN:
            k2 = k2.replace(a, b)
        out[k2[1:]] = v
    return out


def encoder_config(mc, att_ctx):
    """NemotronAsrStreamingEncoderConfig from the .nemo encoder config (NeMo defaults for unset keys: use_bias / xscaling True)"""
    from transformers import NemotronAsrStreamingEncoderConfig
    ec, pc = mc['encoder'], mc['preprocessor']
    assert ec.get('att_context_style') == 'chunked_limited' and ec.get('conv_context_size') == 'causal' and str(pc.get('normalize')) == 'NA', 'nft2: not a cache-aware streaming FastConformer'
    d = int(ec['d_model'])
    bias = bool(ec.get('use_bias', True))
    return NemotronAsrStreamingEncoderConfig(hidden_size=d, num_hidden_layers=int(ec['n_layers']), num_attention_heads=int(ec['n_heads']), num_key_value_heads=int(ec['n_heads']), intermediate_size=d * int(ec.get('ff_expansion_factor', 4)), hidden_act='silu', attention_bias=bias, convolution_bias=bias, conv_kernel_size=int(ec['conv_kernel_size']), subsampling_factor=int(ec['subsampling_factor']), subsampling_conv_channels=int(ec['subsampling_conv_channels']), num_mel_bins=int(ec['feat_in']), subsampling_conv_kernel_size=3, subsampling_conv_stride=2, dropout=0.0, dropout_positions=0.0, layerdrop=0.0, activation_dropout=0.0, attention_dropout=0.0, max_position_embeddings=int(ec.get('pos_emb_max_len', 5000)), scale_input=bool(ec.get('xscaling', True)), sliding_window=int(att_ctx[0]) + 1, default_num_lookahead_tokens=int(att_ctx[1]))


def upper_weights(enc_sd_hf, upper, ucfg):
    """trained encoder layers of an Arc export (pk_lora.safetensors) -> {hf name: tensor} to load over the base encoder.
    LoRA (r > 0): W + alpha / r * B A on the target linears (merged); full-top (r = 0): the layer tensors as saved."""
    if not upper:
        return {}
    r = int(ucfg.get('r', 0))
    if r == 0:
        return {k: v for k, v in upper.items()}
    sc = float(ucfg['alpha']) / r
    out = {}
    for k in upper:
        if k.endswith('.lora_A'):
            n = k[:-len('.lora_A')]
            out[n + '.weight'] = enc_sd_hf[n + '.weight'].float() + sc * (upper[n + '.lora_B'].float() @ upper[k].float())
    return out


def chunk_mask(T, att_ctx, device):
    L, R = att_ctx
    C = R + 1
    lc = L // C if L >= 0 else 10000
    c = torch.arange(T, device=device) // C
    d = c[:, None] - c[None, :]
    return (d >= 0) & (d <= lc)


class LogMel(nn.Module):
    """NeMo AudioToMelSpectrogramPreprocessor, eval path, normalize NA (log-mel only; pads 0)"""

    def __init__(self, window, fb):
        super().__init__()
        self.register_buffer('window', window.float(), persistent=False)
        self.register_buffer('fb', fb.float().reshape(fb.shape[-2], fb.shape[-1]), persistent=False)

    @torch.no_grad()
    def forward(self, x, lengths):
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()
            B, L = x.shape
            flen = lengths // HOP
            tm = torch.arange(L, device=x.device)[None] < lengths[:, None]
            x = torch.cat([x[:, :1], x[:, 1:] - 0.97 * x[:, :-1]], 1).masked_fill(~tm, 0.0)
            s = torch.stft(x, n_fft=NFFT, hop_length=HOP, win_length=WIN, center=True, window=self.window, return_complex=True, pad_mode='constant')
            s = torch.view_as_real(s)
            p = torch.sqrt(s.pow(2).sum(-1)).pow(2.0)
            m = torch.log(torch.matmul(self.fb, p) + 2 ** (-24))
            v = torch.arange(m.shape[-1], device=x.device)[None] < flen[:, None]
            m = m.masked_fill(~v[:, None], 0.0)
        return (m.transpose(1, 2).contiguous(), flen)


class FCEncoder(nn.Module):
    """cache-aware streaming FastConformer encoder (full-sequence forward with the chunked_limited mask).
    forward(x (B, L), lengths) -> H (B, T, d) every encoder frame (incl. frame 0), valid mask (B, T), Htap (B, T, d) after layer `tap`"""

    def __init__(self, sd, mc, att_ctx=(70, 6), tap=9, device='cpu', half=None, upper=None):
        super().__init__()
        from transformers import NemotronAsrStreamingEncoder
        cfg = encoder_config(mc, att_ctx)
        cfg._attn_implementation = 'sdpa'
        try:
            from transformers.initialization import no_init_weights
            with no_init_weights():
                self.enc = NemotronAsrStreamingEncoder(cfg)
        except ImportError:
            self.enc = NemotronAsrStreamingEncoder(cfg)
        hsd = map_keys(sd)
        hsd.update(upper or {})
        miss, unexp = self.enc.load_state_dict(hsd, strict=False)
        miss = [k for k in miss if not k.endswith('inv_freq')]
        if miss or unexp:
            raise RuntimeError(f'nft2 encoder weights mismatch: {miss[:5]} {unexp[:5]}')
        self.mel = LogMel(sd['preprocessor.featurizer.window'], sd['preprocessor.featurizer.fb'])
        self.att_ctx = tuple(int(v) for v in att_ctx)
        self.tap = int(tap)
        self.d_model = int(cfg.hidden_size)
        self.in_scale = float(getattr(self.enc, 'input_scale', 1.0))
        half = str(device).startswith('cuda') if half is None else half
        self.dtype = torch.float16 if half else torch.float32
        self.enc.to(device=device, dtype=self.dtype).eval()
        self.mel.to(device)
        for p in self.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, x, lengths, want_tap=True):
        feats, flen = self.mel(x, lengths)
        e = self.enc
        dev = x.device
        am = torch.arange(feats.shape[1], device=dev)[None] < flen[:, None]
        with torch.autocast(dev.type, enabled=False):
            h = e.subsampling(feats.to(self.dtype), am.long())
            if self.in_scale != 1.0:
                h = h * self.in_scale
            pos = e.encode_positions(h)
            om = e._get_output_attention_mask(am.long(), target_length=h.shape[1])
            cm = chunk_mask(h.shape[1], self.att_ctx, dev)
            att = ((cm[None] & om[:, None, :]) | ~om[:, :, None])[:, None]
            amr = (~om)[:, None, :]
            keep = om[..., None]
            h = h.masked_fill(~keep, 0.0)
            htap = h if want_tap and self.tap == 0 else None
            for i, layer in enumerate(e.layers):
                h = layer(h, attention_mask=att, all_masked_rows=amr, position_embeddings=pos).masked_fill(~keep, 0.0)
                if want_tap and i + 1 == self.tap:
                    htap = h
        return (h, om, htap)


# ------------------------------------------------------------------------------------------------------------- ONNX encoder
def _calc_len(n, k=3):
    """mel frames -> encoder frames of the full-sequence forward (three CAUSAL stride-2 / kernel-3 convolutions, left pad 2, right pad 1)"""
    for _ in range(k):
        n = n // 2 + 1
    return max(0, n)


class OnnxEncoder:
    """the NFT2 encoder from the cache-aware STREAMING ONNX export (layout: encoder.onnx,
    encoder_first.onnx, encoder.weights, frontend.npz, config.json), run on CPU with onnxruntime; same interface as FCEncoder.

    The utterance is fed exactly like the on-device stream (nft_onnx_stream.py): chunk 0 = encoder_first.onnx on the first 49 mel
    frames, then encoder.onnx on [9 cached mel frames | 56 new] (560 ms, 7 encoder frames, 480 ms lookahead) with the layer caches
    carried; the final partial chunk is always run. Frame i of the output = frame i of the full-sequence chunked-mask forward (same
    7-frame chunk grid; it differs only by the subsampling's missing right context at chunk edges). The Arc tap (output of layer
    `tap`) is added as an extra graph output in memory (no files written). Both graphs share one copy of the weights.
    The last result is memoised on the audio bytes: Nft2Finals.decode and Arc._feats then cost one encoder pass for a <= 30 s clip.
    Inputs are cut to a multiple of 80 samples (as Nft2Backbone.utterance; the mel frame count n // 160 is unchanged)."""

    def __init__(self, onnx_dir, tap=9, threads=4, device='cpu'):
        import onnxruntime as ort
        self.dir = onnx_dir
        self.cfg = c = json.load(open(os.path.join(onnx_dir, 'config.json')))
        fe = np.load(os.path.join(onnx_dir, 'frontend.npz'))
        self.mel = LogMel(torch.as_tensor(fe['window']), torch.as_tensor(fe['fb']))
        self.tap = int(tap)
        self.d_model = int(c['d_model'])
        self.dev = torch.device(device)
        self.chunk, self.pre_n = int(c['shift_size_frames'][1]), int(c['pre_encode_cache_size'][1])
        fg = c.get('first_chunk_graph')
        self.first_n = int(c.get('first_chunk_frames') or 0) if fg and os.path.exists(os.path.join(onnx_dir, fg)) else 0
        self.tap_name = f'/layers.{self.tap - 1}/norm_out/LayerNormalization_output_0' if self.tap > 0 else None
        self.threads = int(threads)
        self._mm = {}
        self._blobs = {}
        names = ['encoder.onnx'] + ([fg] if self.first_n else [])
        self.sess = {n: self._session(ort, n) for n in names}
        self.enc = self.sess['encoder.onnx']
        self.enc0 = self.sess.get(fg) if self.first_n else None
        self.outs = ['outputs', 'encoded_lengths', 'cache_last_channel_next', 'cache_last_time_next', 'cache_last_channel_next_len'] + ([self.tap_name] if self.tap_name else [])
        self._last = (None, None)

    def _session(self, ort, fn):
        """graph (no external data) + the tap output -> session; external initializers handed over as shared read-only OrtValues"""
        import onnx
        from onnx import helper
        m = onnx.load(os.path.join(self.dir, fn), load_external_data=False)
        g = m.graph
        def blob(t):
            kv = {e.key: e.value for e in t.external_data}
            key = (kv['location'], int(kv.get('offset', 0)), int(kv['length']), t.data_type, tuple(t.dims))
            if key[0] not in self._mm:
                self._mm[key[0]] = np.memmap(os.path.join(self.dir, key[0]), np.uint8, 'r')
            return key, self._mm[key[0]][key[1]:key[1] + key[2]].view(helper.tensor_dtype_to_np_dtype(t.data_type)).reshape(t.dims)
        for n in g.node:                                   # external tensors in node attributes (Constant): inline them
            for at in n.attribute:
                if at.type == onnx.AttributeProto.TENSOR and at.t.data_location == onnx.TensorProto.EXTERNAL:
                    _, arr = blob(at.t)
                    at.t.raw_data = arr.tobytes()
                    at.t.data_location = onnx.TensorProto.DEFAULT
                    del at.t.external_data[:]
        so = ort.SessionOptions()
        so.intra_op_num_threads = self.threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        folder = os.path.abspath(self.dir)
        for t in g.initializer:
            if t.data_location == onnx.TensorProto.EXTERNAL:
                key, arr = blob(t)
                if key not in self._blobs:
                    self._blobs[key] = ort.OrtValue.ortvalue_from_numpy(arr)
                so.add_initializer(t.name, self._blobs[key])
                # the Hugging Face cache stores files as symlinks into blobs/, and onnxruntime refuses external data outside the
                # model folder: point the (overridden) location at the real file and its folder
                real = os.path.realpath(os.path.join(self.dir, key[0]))
                for e in t.external_data:
                    if e.key == 'location':
                        e.value = os.path.basename(real)
                folder = os.path.dirname(real)
        so.add_session_config_entry('session.model_external_initializers_file_folder_path', folder)
        if self.tap_name:
            if not any(self.tap_name in n.output for n in g.node):
                raise RuntimeError(f'nft2 onnx: no tensor {self.tap_name} in {fn} (tap {self.tap})')
            g.output.append(helper.make_tensor_value_info(self.tap_name, onnx.TensorProto.FLOAT, None))
        return ort.InferenceSession(m.SerializeToString(), so, providers=['CPUExecutionProvider'])

    def train(self, mode=True):
        return self

    def eval(self):
        return self

    def encode_mel(self, M):
        """mel frames (n, 80) float32 -> (H (T, d), Htap (T, d) or None) numpy, T = full-sequence encoder length"""
        c = self.cfg
        L, d = int(c['n_layers']), self.d_model
        cch = np.zeros((1, L, int(c['last_channel_cache_size']), d), np.float32)
        ctm = np.zeros((1, L, d, int(c['conv_cache'])), np.float32)
        clen = np.zeros(1, np.int64)
        pre = np.zeros((self.pre_n, M.shape[1]), np.float32)
        hs, ts = [], []
        i = 0
        n = M.shape[0]
        T = _calc_len(n)
        while i < n:
            first = i == 0 and self.first_n > 0
            new = M[i:i + (self.first_n if first else self.chunk)]
            x = new if first else np.concatenate([pre, new], 0)
            pre = np.concatenate([pre, x], 0)[-self.pre_n:]
            r = (self.enc0 if first else self.enc).run(self.outs, {'audio_signal': np.ascontiguousarray(x.T[None]), 'length': np.array([x.shape[0]], np.int64),
                                                                  'cache_last_channel': cch, 'cache_last_time': ctm, 'cache_last_channel_len': clen})
            out, olen, cch, ctm, clen = r[:5]
            k = int(olen[0])
            hs.append(out[0, :, :k].T)
            if self.tap_name:
                ts.append(r[5][0, :k])
            i += new.shape[0]
        def fit(z):
            z = np.concatenate(z, 0) if z else np.zeros((0, d), np.float32)
            if z.shape[0] < T:                              # (never seen) pad with the last frame
                z = np.concatenate([z, np.repeat(z[-1:] if len(z) else np.zeros((1, d), np.float32), T - z.shape[0], 0)], 0)
            return z[:T]
        return fit(hs), (fit(ts) if self.tap_name else None)

    @torch.no_grad()
    def __call__(self, x, lengths, want_tap=True):
        """x (B, L) audio, lengths (B,) -> H (B, T, d), valid mask (B, T), Htap (B, T, d) | None (torch, on the backbone device)"""
        xs = x.detach().float().cpu().numpy()
        Hs, Ts = [], []
        for b in range(xs.shape[0]):
            n = max(80, int(lengths[b]) // 80 * 80)
            y = np.zeros(n, np.float32)
            m = min(n, int(lengths[b]))
            y[:m] = xs[b, :m]
            key = (n, hash(y.tobytes()))
            if self._last[0] == key:
                H, Ht = self._last[1]
            else:
                M, fl = self.mel(torch.as_tensor(y)[None], torch.tensor([n]))
                H, Ht = self.encode_mel(M[0, :int(fl[0])].numpy())
                self._last = (key, (H, Ht))
            Hs.append(H)
            Ts.append(Ht)
        T = max(h.shape[0] for h in Hs)
        Hb = np.zeros((len(Hs), T, self.d_model), np.float32)
        om = np.zeros((len(Hs), T), bool)
        Tb = np.zeros_like(Hb) if want_tap and self.tap_name else None
        for b, (h, ht) in enumerate(zip(Hs, Ts)):
            Hb[b, :len(h)] = h
            om[b, :len(h)] = True
            if Tb is not None:
                Tb[b, :len(h)] = ht
        tt = lambda z: torch.from_numpy(z).to(self.dev)
        return (tt(Hb), tt(om), tt(Tb) if Tb is not None else None)


class OnnxRNNT:
    """the RNNT prediction net + joint from the same ONNX export (decoder.onnx, joint_enc.onnx, joint_net.onnx); greedy() is
    RNNT.greedy (same loop, same biasing rule) on onnxruntime"""

    def __init__(self, onnx_dir, threads=1):
        import onnxruntime as ort
        c = json.load(open(os.path.join(onnx_dir, 'config.json')))
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        mk = lambda n: ort.InferenceSession(os.path.join(onnx_dir, n), so, providers=['CPUExecutionProvider'])
        self.jenc, self.dec, self.jnet = mk('joint_enc.onnx'), mk('decoder.onnx'), mk('joint_net.onnx')
        self.blank = int(c['blank_id'])
        self.max_sym = int(c.get('max_symbols_per_frame', 10))
        self.L, self.H = int(c['pred_rnn_layers']), int(c['pred_hidden'])

    def init_state(self):
        return dict(g=None, hc=None, b=None)

    def _pred(self, y, hc):
        if hc is None:
            hc = (np.zeros((self.L, 1, self.H), np.float32), np.zeros((self.L, 1, self.H), np.float32))
        g, h, c = self.dec.run(None, {'y': np.array([[y]], np.int64), 'h': hc[0], 'c': hc[1]})
        return (g[0], (h, c))

    def _joint(self, f, g):
        return self.jnet.run(None, {'f': np.ascontiguousarray(f), 'g': np.ascontiguousarray(np.broadcast_to(g, (f.shape[0], g.shape[-1])))})[0]

    def greedy(self, H, st=None, trie=None, alpha=2.0):
        st = st or self.init_state()
        if st['g'] is None:
            st['g'], st['hc'] = self._pred(self.blank, None)
        if trie is not None and st['b'] is None:
            st['b'] = trie.root
        H = H.detach().float().cpu().numpy() if torch.is_tensor(H) else np.asarray(H, np.float32)
        ids, frames = ([], [])
        T = H.shape[0]
        if T == 0:
            return (ids, frames, st)
        Fe = self.jenc.run(None, {'enc': np.ascontiguousarray(H[None])})[0][0]
        t = 0
        while t < T:
            lg = self._joint(Fe[t:], st['g'])
            k = lg.argmax(-1)
            nb = np.nonzero(k != self.blank)[0]
            if len(nb) == 0:
                break
            t += int(nb[0])
            row = lg[int(nb[0])]
            sym = 0
            while True:
                kk = int(row.argmax())
                if kk == self.blank:
                    break
                if trie is not None:
                    kk = int(np.argmax(row[:self.blank] + alpha * trie.scores(st['b'])))
                    st['b'] = trie.advance(st['b'], kk)
                ids.append(kk)
                frames.append(t)
                st['g'], st['hc'] = self._pred(kk, st['hc'])
                sym += 1
                if sym >= self.max_sym:
                    break
                row = self._joint(Fe[t:t + 1], st['g'])[0]
            t += 1
        return (ids, frames, st)


def onnx_pieces(onnx_dir):
    """Pieces from the ONNX export's tokens.txt + tokenizer.model (piece scores from sentencepiece when installed)"""
    toks = [l.rstrip('\n').rsplit(' ', 1)[0] for l in open(os.path.join(onnx_dir, 'tokens.txt'), encoding='utf-8')]
    cfg = json.load(open(os.path.join(onnx_dir, 'config.json')))
    toks = toks[:int(cfg['vocab_size'])]
    sp_path = os.path.join(onnx_dir, 'tokenizer.model')
    spm_bytes = open(sp_path, 'rb').read() if os.path.exists(sp_path) else None
    sc = _spm_scores(spm_bytes) if spm_bytes else []
    if len(sc) < len(toks):
        sc = sc + [0.0] * (len(toks) - len(sc))
    return Pieces([f'{t}\t{v}' for t, v in zip(toks, sc)], spm_bytes)


def _spm_scores(b):
    """piece scores from a sentencepiece ModelProto (field 1 = repeated SentencePiece{1: piece, 2: float score}) without
    sentencepiece / protobuf; the Viterbi fallback of Pieces.encode needs them to segment like the real tokenizer"""
    import struct
    def varint(b, i):
        v = s_ = 0
        while True:
            c = b[i]
            i += 1
            v |= (c & 127) << s_
            s_ += 7
            if c < 128:
                return v, i
    out, i = [], 0
    while i < len(b):
        k, i = varint(b, i)
        f, w = k >> 3, k & 7
        if w == 0:
            _, i = varint(b, i)
        elif w == 1:
            i += 8
        elif w == 5:
            i += 4
        elif w == 2:
            n, i = varint(b, i)
            if f == 1:
                sub, j, sc = b[i:i + n], 0, 0.0
                while j < len(sub):
                    k2, j = varint(sub, j)
                    f2, w2 = k2 >> 3, k2 & 7
                    if w2 == 5:
                        if f2 == 2:
                            sc = struct.unpack('<f', sub[j:j + 4])[0]
                        j += 4
                    elif w2 == 0:
                        _, j = varint(sub, j)
                    elif w2 == 2:
                        n2, j = varint(sub, j)
                        j += n2
                    elif w2 == 1:
                        j += 8
                out.append(float(sc))
            i += n
        else:
            break
    return out


# ------------------------------------------------------------------------------------------------------------- RNNT
class RNNT(nn.Module):
    """NeMo RNNTDecoder (LSTM prediction net, blank_as_pad) + RNNTJoint (relu), greedy decoding (max_symbols per frame)"""

    def __init__(self, sd, mc):
        super().__init__()
        dc, jc = mc['decoder'], mc['joint']
        V = int(dc['vocab_size'])
        H = int(dc['prednet']['pred_hidden'])
        L = int(dc['prednet'].get('pred_rnn_layers', 1))
        jh = int(jc['jointnet']['joint_hidden'])
        assert jc['jointnet'].get('activation', 'relu') == 'relu'
        self.blank = V
        self.max_sym = int(((mc.get('decoding') or {}).get('greedy') or {}).get('max_symbols', 10) or 10)
        self.embed = nn.Embedding(V + 1, H)
        self.lstm = nn.LSTM(H, H, L)
        self.jenc = nn.Linear(int(jc['jointnet']['encoder_hidden']), jh)
        self.jpred = nn.Linear(H, jh)
        self.jout = nn.Linear(jh, V + 1)
        p = 'decoder.prediction.'
        with torch.no_grad():
            self.embed.weight.copy_(sd[p + 'embed.weight'])
            for n, t in self.lstm.named_parameters():
                t.copy_(sd[p + 'dec_rnn.lstm.' + n])
            for m, k in ((self.jenc, 'joint.enc'), (self.jpred, 'joint.pred'), (self.jout, 'joint.joint_net.2')):
                m.weight.copy_(sd[k + '.weight'])
                m.bias.copy_(sd[k + '.bias'])
        self.eval()
        for q in self.parameters():
            q.requires_grad_(False)

    def init_state(self):
        return dict(g=None, hc=None, b=None)

    def _pred(self, y, hc):
        dev = self.embed.weight.device
        e = self.embed(torch.tensor([[y]], device=dev)) if y != self.blank else torch.zeros(1, 1, self.embed.embedding_dim, device=dev)
        o, hc = self.lstm(e, hc)
        return (self.jpred(o[0, 0]), hc)

    @torch.no_grad()
    def greedy(self, H, st=None, trie=None, alpha=2.0):
        """H (T, d_enc) encoder frames -> (token ids, frame index per token, state); st carries across calls (streaming).
        Biasing (trie): the blank / non-blank decision is unbiased, the non-blank choice gets alpha * trie score (NeMo greedy fusion)."""
        st = st or self.init_state()
        if st['g'] is None:
            st['g'], st['hc'] = self._pred(self.blank, None)
        if trie is not None and st['b'] is None:
            st['b'] = trie.root
        Fe = self.jenc(H.float())
        ids, frames = ([], [])
        T = Fe.shape[0]
        t = 0
        while t < T:
            lg = self.jout(F.relu(Fe[t:] + st['g'][None]))
            k = lg.argmax(-1)
            nb = (k != self.blank).nonzero()
            if len(nb) == 0:
                break
            t += int(nb[0, 0])
            row = lg[int(nb[0, 0])]
            sym = 0
            while True:
                kk = int(row.argmax())
                if kk == self.blank:
                    break
                if trie is not None:
                    bv = torch.as_tensor(trie.scores(st['b']), device=row.device)
                    kk = int((row[:self.blank] + alpha * bv).argmax())
                    st['b'] = trie.advance(st['b'], kk)
                ids.append(kk)
                frames.append(t)
                st['g'], st['hc'] = self._pred(kk, st['hc'])
                sym += 1
                if sym >= self.max_sym:
                    break
                row = self.jout(F.relu(Fe[t] + st['g']))
            t += 1
        return (ids, frames, st)


class BoostTrie:
    """per-utterance word boosting (CPU port of NeMo's boosting tree: context_score for the first token, context_score * depth_scaling +
    log(depth) for later ones, Aho-Corasick back-off on a mismatch; 0 from a completed phrase)"""

    class Node:
        __slots__ = ('next', 'token_score', 'node_score', 'is_end', 'fail', 'id')

        def __init__(self, i, ts=0.0, ns=0.0, end=False):
            self.next = {}
            self.token_score = ts
            self.node_score = ns
            self.is_end = end
            self.fail = None
            self.id = i

    def __init__(self, token_lists, vocab, context_score=1.0, depth_scaling=2.0):
        from collections import deque
        self.V = vocab
        self.root = self.Node(0)
        n = 0
        for toks in token_lists:
            node = self.root
            for i, t in enumerate(toks):
                end = i == len(toks) - 1
                if t not in node.next:
                    ts = context_score * depth_scaling + np.log(i + 1) if i > 0 else context_score
                    n += 1
                    node.next[t] = self.Node(n, ts, node.node_score + ts, end)
                else:
                    c = node.next[t]
                    c.token_score = max(context_score, c.token_score)
                    c.node_score = node.node_score + c.token_score
                    c.is_end = c.is_end or end
                node = node.next[t]
        q = deque()
        for c in self.root.next.values():
            c.fail = self.root
            q.append(c)
        while q:
            cur = q.popleft()
            for t, c in cur.next.items():
                f = cur.fail
                while f is not self.root and t not in f.next:
                    f = f.fail
                c.fail = f.next[t] if t in f.next and f.next[t] is not c else self.root
                q.append(c)
        self._vec = {}

    @staticmethod
    def _backoff(s):
        return 0.0 if s.is_end else s.fail.node_score - s.node_score

    def scores(self, s):
        v = self._vec.get(s.id)
        if v is None:
            v = np.full(self.V, np.nan, np.float32)
            acc = 0.0
            node = s
            while True:
                for t, c in node.next.items():
                    if np.isnan(v[t]):
                        v[t] = acc + c.token_score
                if node is self.root:
                    break
                acc += self._backoff(node)
                node = node.fail
            v[np.isnan(v)] = acc
            self._vec[s.id] = v
        return v

    def advance(self, s, t):
        while True:
            if t in s.next:
                return s.next[t]
            if s is self.root:
                return s
            s = s.fail


_ONES = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
_TENS = 'twenty thirty forty fifty sixty seventy eighty ninety'.split()


def num_words(n):
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10 - 2] + ('' if n % 10 == 0 else ' ' + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + ' hundred' + ('' if n % 100 == 0 else ' ' + num_words(n % 100))
    if n < 10000 and n % 1000 and 1100 <= n < 2000 or 2010 <= n < 2100:
        return num_words(n // 100) + ' ' + (num_words(n % 100) if n % 100 >= 10 else 'oh ' + _ONES[n % 100] if n % 100 else 'hundred')
    if n < 1000000:
        return num_words(n // 1000) + ' thousand' + ('' if n % 1000 == 0 else ' ' + num_words(n % 1000))
    return ' '.join(_ONES[int(d)] for d in str(n))


class Pieces:
    """the NFT2 SentencePiece unigram vocabulary: ids -> words, text -> ids (sentencepiece when installed, else a Viterbi unigram
    segmentation over the same piece scores)"""

    def __init__(self, vocab_lines, spm_bytes=None):
        self.p, self.s = ([], [])
        for l in vocab_lines:
            a, _, b = l.rpartition('\t')
            self.p.append(a)
            self.s.append(float(b))
        self.idx = {p: i for i, p in enumerate(self.p)}
        self.maxlen = max((len(p) for p in self.p))
        self.sp = None
        if spm_bytes:
            try:
                import sentencepiece as spm
                self.sp = spm.SentencePieceProcessor(model_proto=spm_bytes)
            except Exception:
                self.sp = None

    @staticmethod
    def norm(text):
        """to the RNNT output style: lower case, spoken numbers ('Apollo 11' -> 'apollo eleven'), letters / apostrophes only"""
        t = (text or '').lower().replace('’', "'").replace('-', ' ')
        t = re.sub('\\d+', lambda m: ' ' + num_words(int(m.group(0))) + ' ', t)
        return ' '.join(re.sub("[^a-z' ]+", ' ', t).split())

    def encode(self, text):
        t = self.norm(text)
        if not t:
            return []
        if self.sp is not None:
            return list(self.sp.encode(t))
        s = '▁' + t.replace(' ', '▁')
        n = len(s)
        best = [(-1e30, -1, -1)] * (n + 1)
        best[0] = (0.0, -1, -1)
        for i in range(n):
            if best[i][0] <= -1e29:
                continue
            for j in range(i + 1, min(n, i + self.maxlen) + 1):
                k = self.idx.get(s[i:j])
                if k is not None and k != 0:
                    v = best[i][0] + self.s[k]
                    if v > best[j][0]:
                        best[j] = (v, i, k)
            if best[i + 1][0] <= -1e29:
                best[i + 1] = (best[i][0] - 20.0, i, 0)
        out, j = ([], n)
        while j > 0:
            _, i, k = best[j]
            if not (k == 0 and out and out[-1] == 0):
                out.append(k)
            j = i
        return out[::-1]

    def words(self, ids, frames, t0=0.0, frame_s=0.08):
        """token ids + encoder frame of each -> [dict(text, start, end)] (encoder frame f covers [(f - 1) * 80, f * 80) ms)"""
        out = []
        for k, f in zip(ids, frames):
            p = self.p[k] if 0 <= k < len(self.p) else ''
            st = t0 + max(0.0, (f - 1) * frame_s)
            en = t0 + max(frame_s, f * frame_s)
            if p.startswith('▁') or not out:
                out.append(dict(text=p.lstrip('▁'), start=st, end=en))
            else:
                out[-1]['text'] += p
                out[-1]['end'] = en
        return [dict(w, start=round(w['start'], 3), end=round(max(w['end'], w['start'] + 0.04), 3)) for w in out if w['text']]


# ------------------------------------------------------------------------------------------------------------- backbone
class Nft2Backbone(nn.Module):
    """the NFT2 encoder (+ RNNT, vocabulary) and, when the Arc export was trained on a different / adapted encoder, that encoder.
    __call__ = the Arc view of server.Backbone (drop0 frames, 80 ms grid): (P, mask, Ptap)"""

    def __init__(self, nemo, tap=9, att_ctx=(70, 6), device='cpu', half=None, arc_nemo=None, upper=None, upper_cfg=None, encoder='torch',
                 onnx_dir=None, arc_onnx_dir=None, onnx_threads=4):
        super().__init__()
        self.encoder = encoder
        if encoder == 'onnx':
            # ONNX (CPU) backbone: words from onnx_dir's streaming encoder + RNNT graphs; the Arc layer reads arc_onnx_dir's encoder when
            # given (e.g. fp32 for the Arc layer, int8 for the words), else the same frames. No .nemo / torch encoder is loaded.
            if upper or (arc_nemo is not None and os.path.realpath(arc_nemo) != os.path.realpath(nemo or '')):
                raise ValueError('nft2 encoder=onnx: the export has its own Arc-layer encoder (arc_nemo / encoder_upper); use encoder=torch')
            self.enc = OnnxEncoder(onnx_dir, tap, onnx_threads, device)
            self.rnnt = OnnxRNNT(onnx_dir, 1)
            self.pieces = onnx_pieces(onnx_dir)
            same = not arc_onnx_dir or os.path.realpath(arc_onnx_dir) == os.path.realpath(onnx_dir)
            self.arc_enc = None if same else OnnxEncoder(arc_onnx_dir, tap, onnx_threads, device)
            self.tap = int(tap)
            self.d_model = self.enc.d_model
            self.dev = torch.device(device)
            return
        sd, mc, voc, spm = read_nemo(nemo)
        self.enc = FCEncoder(sd, mc, att_ctx, tap, device, half)
        self.rnnt = RNNT(sd, mc).to(device)
        self.pieces = Pieces(voc, spm)
        self.arc_enc = None
        same = arc_nemo is None or os.path.realpath(arc_nemo) == os.path.realpath(nemo)
        if not same or upper:
            asd, amc = (sd, mc) if same else read_nemo(arc_nemo, keep=('encoder.', 'preprocessor.'))[:2]
            up = upper_weights(map_keys(asd), upper, upper_cfg or {})
            self.arc_enc = FCEncoder(asd, amc, att_ctx, tap, device, half, upper=up)
        del sd
        self.tap = int(tap)
        self.d_model = self.enc.d_model
        self.dev = torch.device(device)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, x, lengths, want_tap=True):
        h, om, ht = (self.arc_enc or self.enc)(x, lengths, want_tap)
        return (h[:, 1:], om[:, 1:], ht[:, 1:] if ht is not None else None)

    @torch.no_grad()
    def utterance(self, wav, trie=None, alpha=2.0, t0=0.0, want_tap=False):
        """one encoder pass over an utterance -> dict(words (RNNT), ids, frames, P (Arc frames, drop0), Ptap, n)"""
        n = max(80, len(wav) // 80 * 80)
        y = np.zeros(n, np.float32)
        y[:min(n, len(wav))] = wav[:n]
        x = torch.as_tensor(y, device=self.dev)[None]
        L = torch.tensor([n], device=self.dev)
        h, om, ht = self.enc(x, L, want_tap and self.arc_enc is None)
        T = int(om[0].sum())
        ids, frames, _ = self.rnnt.greedy(h[0, :T], trie=trie, alpha=alpha)
        out = dict(ids=ids, frames=frames, words=self.pieces.words(ids, frames, t0), n=n)
        if self.arc_enc is not None:
            h, om, ht = self.arc_enc(x, L, want_tap)
        out.update(P=h[:, 1:], Ptap=ht[:, 1:] if ht is not None else None, mask=om[:, 1:])
        return out

    def trie(self, phrases, context_score=1.0):
        toks = [self.pieces.encode(p) for p in phrases or []]
        toks = [t for t in toks if t]
        return BoostTrie(toks, self.rnnt.blank, context_score) if toks else None


def load_backbone(path, bc, device):
    """config.json backbone section -> Nft2Backbone (+ the Arc-layer encoder overrides of the export, if any)"""
    bc = dict(bc, _dir=path)
    enc = os.environ.get('ARC_NFT2_ENCODER') or bc.get('encoder') or 'onnx'
    if enc == 'onnx':
        d = lambda p: p if not p or os.path.isabs(p) else os.path.join(path, p)
        od = os.environ.get('ARC_NFT2_ONNX') or d(bc.get('onnx_dir'))
        ad = os.environ.get('ARC_NFT2_ARC_ONNX') or d(bc.get('arc_onnx_dir'))
        if not od:
            raise FileNotFoundError('nft2 encoder=onnx: set onnx_dir (or $ARC_NFT2_ONNX) to the streaming ONNX export folder')
        if bc.get('arc_nemo') or bc.get('encoder_upper'):
            raise ValueError('nft2 encoder=onnx: this export has its own Arc-layer encoder (arc_nemo / encoder_upper); use encoder=torch')
        th = int(os.environ.get('ARC_NFT2_ONNX_THREADS') or bc.get('onnx_threads') or torch.get_num_threads())
        return Nft2Backbone(None, tap=int(bc.get('tap', 9)), device=device, encoder='onnx', onnx_dir=od, arc_onnx_dir=ad, onnx_threads=th)
    nemo = fetch_nemo(bc)
    up, ucfg, arc_nemo = (None, None, None)
    if bc.get('arc_nemo'):
        arc_nemo = fetch_nemo(dict(bc, nemo=bc['arc_nemo'], repo=None)) if not os.environ.get('ARC_NFT2_ARC_NEMO') else os.environ['ARC_NFT2_ARC_NEMO']
    if bc.get('encoder_upper'):
        from safetensors.torch import load_file
        ucfg = dict(bc['encoder_upper'])
        up = load_file(os.path.join(path, ucfg['file']))
    return Nft2Backbone(nemo, tap=int(bc.get('tap', 9)), att_ctx=tuple(bc.get('att_ctx', (70, 6))), device=device, half=bc.get('half'), arc_nemo=arc_nemo, upper=up, upper_cfg=ucfg)


# ------------------------------------------------------------------------------------------------------------- finals
def _wkey(w):
    k = re.sub("[^\\w']", '', str(w).lower())
    k = re.sub("'s$", '', k).strip("'")
    return re.sub('\\d+', lambda m: num_words(int(m.group(0))).replace(' ', ''), k)


def map_spans_rnnt(src_words, spans, dst_words, max_block=3, min_ratio=0.5):
    """decoder spans (tag, a, b over src_words) -> spans over the RNNT words. Exact word matches as tagdec.map_spans (keys without
    possessive 's, digits as words); a span inside a small REPLACED block (<= max_block words on both sides: 'Burgess' / 'byrgius',
    'South East' / 'southeast', '1130' / 'eleven thirty') maps onto that block's RNNT words. Spans in larger mismatches are dropped."""
    import difflib
    if not spans:
        return []
    a_, b_ = ([_wkey(w) for w in src_words], [_wkey(w) for w in dst_words])
    m, rep = ({}, set())
    for tag_, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a_, b_, autojunk=False).get_opcodes():
        if tag_ == 'equal':
            for k in range(i2 - i1):
                m[i1 + k] = (j1 + k, j1 + k + 1)
        elif tag_ == 'replace' and i2 - i1 <= max_block and j2 - j1 <= max_block:
            for k in range(i1, i2):
                rep.add(k)
                if i2 - i1 == j2 - j1:
                    m[k] = (j1 + k - i1, j1 + k - i1 + 1)
                else:
                    m[k] = (j1, j2)
    out = []
    for t, a, b in spans:
        js = [m[j] for j in range(a, b) if j in m]
        if js:
            sp = (t, min(x for x, _ in js), max(y for _, y in js))
            if any(j in rep for j in range(a, b)):          # fuzzy part: the heard RNNT words must sound / spell alike
                r = difflib.SequenceMatcher(None, ''.join(a_[a:b]), ''.join(b_[sp[1]:sp[2]]), autojunk=False).ratio()
                if r < min_ratio:
                    continue
            if not out or sp[1] >= out[-1][2]:
                out.append(sp)
    return out


def _char_align(a_, b_):
    """word keys a_ / b_ -> (A, B, src word of each A char, dst word of each B char, B char aligned to each A char or None): difflib
    character alignment of the space-joined keys, replaced runs mapped proportionally"""
    import difflib
    A, B = (' '.join(a_), ' '.join(b_))
    aw = [i for i, k in enumerate(a_) for _ in range(len(k) + 1)][:len(A)]
    bw = [j for j, k in enumerate(b_) for _ in range(len(k) + 1)][:len(B)]
    cm = [None] * len(A)
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
        if op == 'equal':
            for k in range(i2 - i1):
                cm[i1 + k] = j1 + k
        elif op == 'replace':
            for k in range(i2 - i1):
                cm[i1 + k] = j1 + min(j2 - j1 - 1, int(k * (j2 - j1) / (i2 - i1)))
    return (A, B, aw, bw, cm)


def map_spans_char(src_words, spans, dst_words, min_ratio=0.5, cover=0.5):
    """decoder spans (tag, a, b over src_words) -> spans over the RNNT words by a CHARACTER alignment of the two word-key strings
    (keys as map_spans_rnnt). A span takes the RNNT words that receive at least `cover` of their characters (or half of the span's
    characters) from it; spans whose words differ are kept only when the two spellings are alike (difflib ratio >= min_ratio).
    Handles mismatch blocks of any size ('apollo 19 launch start' / 'apollo nine') and split / merged words ('Chandrayaan-3')."""
    import difflib
    if not spans:
        return []
    a_, b_ = ([_wkey(w) for w in src_words], [_wkey(w) for w in dst_words])
    A, B, aw, bw, cm = _char_align(a_, b_)
    out = []
    for t, a, b in spans:
        cs = [c for c in range(len(A)) if a <= aw[c] < b and A[c] != ' ']
        hit = {}
        for c in cs:
            d = cm[c]
            if d is not None and B[d] != ' ':
                hit[bw[d]] = hit.get(bw[d], 0) + 1
        js = [j for j, n in hit.items() if n >= cover * max(1, len(b_[j])) or n >= 0.5 * len(cs)]
        if not js:
            continue
        j1, j2 = (min(js), max(js) + 1)
        s1, s2 = (''.join(a_[a:b]), ''.join(b_[j1:j2]))
        if s1 != s2 and difflib.SequenceMatcher(None, s1, s2, autojunk=False).ratio() < min_ratio:
            continue
        if not out or j1 >= out[-1][2]:
            out.append((t, j1, j2))
    return out


_SPACE_TAGS = ('ENTITY_LUNAR', 'ENTITY_SPACECRAFT', 'ENTITY_MISSION', 'ENTITY_CELESTIAL')
KIND_TAG = dict(lunar_feature='ENTITY_LUNAR', site='ENTITY_LUNAR', celestial='ENTITY_CELESTIAL', spacecraft='ENTITY_SPACECRAFT',
                vehicle='ENTITY_SPACECRAFT', mission='ENTITY_MISSION', person='ENTITY_PERSON', crew='ENTITY_PERSON', place='ENTITY_PLACE',
                team='ENTITY_TEAM', organization='ENTITY_ORG', org='ENTITY_ORG', product='ENTITY_PRODUCT', law='ENTITY_LAW')


def unify_types(spans, dst_words):
    """one entity type per surface within an utterance: the decoder sometimes tags the same name with two types ('Kepler' LUNAR, then
    'Kepler' PERSON); the space type wins (else the most frequent, first on a tie)"""
    import collections
    key = lambda a, b: tuple(_wkey(w) for w in dst_words[a:b])
    g = collections.defaultdict(list)
    for t, a, b in spans:
        g[key(a, b)].append(t)
    pick = {}
    for k, ts in g.items():
        if len(set(ts)) > 1:
            sp = [t for t in ts if t in _SPACE_TAGS]
            pick[k] = collections.Counter(sp or ts).most_common(1)[0][0]
    return [(pick.get(key(a, b), t), a, b) for t, a, b in spans]


def list_keys(names):
    """names list (Arc._kb dicts) -> {word-key tuple of the name / each alias: entity tag of its kind}"""
    out = {}
    for c in names or []:
        t = KIND_TAG.get(str(c.get('kind', '')))
        if not t:
            continue
        for x in [c.get('name', '')] + [str(v) for v in c.get('aliases') or []]:
            k = tuple(w for w in (_wkey(v) for v in Pieces.norm(str(x)).split()) if w)
            if k:
                out.setdefault(k, t)
    return out


def heard_by_decoder(src_words, dst_words):
    """-> f(i, j): the decoder word keys (joined) aligned to RNNT words [i, j) by the character alignment ('' when none)"""
    a_, b_ = ([_wkey(w) for w in src_words], [_wkey(w) for w in dst_words])
    A, B, aw, bw, cm = _char_align(a_, b_)
    src_of = {}
    for c, d in enumerate(cm):
        if d is not None and A[c] != ' ' and B[d] != ' ':
            src_of.setdefault(bw[d], set()).add(aw[c])
    def f(i, j):
        s = set().union(*[src_of.get(k, set()) for k in range(i, j)])
        return ''.join(a_[min(s):max(s) + 1]) if s else ''
    return f


def list_spans(spans, dst_words, lk, min_chars=4, add=True, heard=None, add_ratio=0.5, long_chars=0):
    """names-list entries spelled out in the RNNT words (longest first, non-overlapping): entity tags overlapping an occurrence are
    replaced by ONE tag over the whole entry with the entry's kind ('Maury' + 'Homer' PERSON on 'mare humorum' -> 'mare humorum' LUNAR,
    'apollo twelve' MISSION in 'apollo twelve lander' -> the lander); with add, untagged occurrences (>= min_chars letters) get the tag
    when the decoder heard something alike there (heard(i, j) vs the entry: difflib ratio >= add_ratio; the RNNT is biased toward the
    list, the decoder is not, so this drops list words the RNNT made up: 'you're just getting' -> 'stubby'). Entries of >= long_chars
    letters (0: off, the default) spelled out exactly are added without that check. Opt-in: it links more names said in a
    space lesson, but on general speech the bias writes long list names ('space launch system') that it would then tag."""
    import difflib
    keys = [_wkey(w) for w in dst_words]
    sp = list(spans)
    used = set()
    for k, t in sorted(lk.items(), key=lambda x: -len(x[0])):
        n = len(k)
        for i in range(len(keys) - n + 1):
            if tuple(keys[i:i + n]) != k or any(j in used for j in range(i, i + n)):
                continue
            ov = [s for s in sp if s[1] < i + n and s[2] > i]
            ent = [s for s in ov if s[0].startswith('ENTITY_')]
            if not ent and add and not ov and len(''.join(k)) >= min_chars and heard is not None:
                ok = (long_chars > 0 and len(''.join(k)) >= long_chars) or \
                    difflib.SequenceMatcher(None, heard(i, i + n), ''.join(k), autojunk=False).ratio() >= add_ratio
            else:
                ok = False
            if ent or ok:
                sp = [s for s in sp if s not in ov] if ent else sp
                sp.append((t, i, i + n))
                used.update(range(i, i + n))
    return sorted(sp, key=lambda x: x[1])


def case_from(words, dec_words):
    """copy the decoder's casing / punctuation onto RNNT words whose normalised form matches (RNNT output is lower case, no punctuation)"""
    import difflib
    key = lambda w: re.sub("[^\\w']", '', str(w).lower())
    a = [key(w['text']) for w in words]
    b = [key(w) for w in dec_words]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            if a[i + k]:
                words[i + k]['text'] = dec_words[j + k]
    return words


class Nft2Finals:
    """final words (RNNT) + Arc decoder tags per closed utterance; the external-final-word source of transcript='nft2'
    (same interface as parakeet.ParakeetFinals: submit / poll / drain / transcribe_offline)"""

    def __init__(self, arc, tr, names=None, bias=True, alpha=None, background=False):
        self.arc, self.tr = (arc, tr)
        c = (arc.cfg.get('transcription') or {}).get('nft2') or {}
        self.c = dict(dict(alpha=2.0, context_score=1.0, case_from_decoder=True, whole_max_s=20.0, edge_pad_s=0.15, char_align=True,
                           unify_types=True, list_spans=True, list_add=True, list_min_chars=3, list_add_ratio=0.5, list_add_long_chars=0,
                           tail_pad_s=0.5), **c)
        if alpha is not None:
            self.c['alpha'] = float(alpha)
        self.bias = bool(bias)
        self.trie = None
        self.set_names(names)
        self.q, self.res = (queue.Queue(), queue.Queue())
        self.th = None
        self.err = None
        if background:
            self.th = threading.Thread(target=self._loop, daemon=True, name='arc-nft2')
            self.th.start()

    def set_names(self, names):
        """names list -> RNNT boosting trie over the full names + aliases (alpha / context_score from config transcription.nft2)"""
        from .api import Arc
        ph = []
        self.lk = list_keys(Arc._kb(names)) if names else {}
        for c in Arc._kb(names):
            ph += [str(c.get('name', ''))] + [str(x) for x in c.get('aliases') or []]
        self.trie = self.arc.bb.trie([p for p in ph if p.strip()], self.c['context_score']) if self.bias and names else None

    def _tail(self, y, t0, f1):
        """a streamed utterance + silence so the input runs tail_pad_s past its last frame f1: the RNNT emits
        a word's last pieces a few hundred ms after it is spoken, so a segment cut 0.15 s after the speech loses its last word
        ('near Sinus me' for 'near Sinus Medii')"""
        y = np.asarray(y, np.float32)
        need = int(round(((f1 + 1) / 10 + float(self.c['tail_pad_s']) - t0) * SR)) - len(y)
        return np.concatenate([y, np.zeros(need, np.float32)]) if need > 0 else y

    @torch.no_grad()
    def decode(self, y, t0=0.0, f0=None, f1=None, want=False, tail=False):
        """utterance audio -> utterance dict (text, words with inline tags, tag_spans, decoder text); one encoder pass. tail: y is a
        cut utterance; silence is added after it (_tail)"""
        from .moon import states_output
        from .server import mem_with_pos
        from .tagdec import map_spans, annotate
        from .transcribe import loopfix
        arc, tr = (self.arc, self.tr)
        f0 = int(round(t0 * 10)) if f0 is None else f0
        f1 = max(f0, int(round((t0 + len(y) / SR) * 10)) - 1) if f1 is None else f1
        r = arc.bb.utterance(self._tail(y, t0, f1) if tail else np.asarray(y, np.float32), trie=self.trie, alpha=float(self.c['alpha']), t0=t0)
        words = r['words']
        nv = max(1, int(r['mask'][0].sum()))
        M = mem_with_pos(arc.heads, arc.moon.model.decoder.pos_emb, r['P'][:, :nv])   # valid frames only (= the encoder mask)
        tr._t_now = f0 / 10
        raw = tr._gen(dict(encoder_outputs=states_output(tr.m, M.float())), len(y), None, final=True)
        dec_ids = list(tr.last_ids)
        txt = loopfix(raw) if tr.c.get('loopfix', False) else raw
        sp = tr.tagged_spans(raw) if tr.tagged else []
        if sp and txt != raw:
            sp = map_spans(raw.split(), sp, txt.split())
        if txt:
            tr.hist = (tr.hist + [(f0 / 10, (f1 + 1) / 10, txt)])[-64:]
        dst = [w['text'] for w in words]
        wsp = (map_spans_char if self.c.get('char_align') else map_spans_rnnt)(txt.split(), sp, dst) if sp else []
        if self.c.get('unify_types'):
            wsp = unify_types(wsp, dst)
        if self.lk and self.c.get('list_spans'):
            wsp = list_spans(wsp, dst, self.lk, int(self.c['list_min_chars']), bool(self.c.get('list_add')),
                             heard_by_decoder(txt.split(), dst), float(self.c['list_add_ratio']),
                             int(self.c.get('list_add_long_chars') or 0))
        if self.c.get('case_from_decoder') and txt:
            words = case_from(words, txt.split())
        ev = annotate([dict(w, is_final=True) for w in words], wsp)
        u = dict(start=round(f0 / 10, 1), end=round((f1 + 1) / 10, 1), f0=f0, f1=f1, text=' '.join((w['text'] for w in ev)), words=ev, tag_spans=wsp, final=True, source='nft2', decoder_text=txt)
        if want:
            u['_dec_ids'] = dec_ids
            u['_rnnt_ids'] = r['ids']
            u['_P'] = r['P']
        return u

    def _decode(self, y, t0, f0, f1):
        return self.decode(y, t0, f0, f1, tail=True)

    def _loop(self):
        while True:
            job = self.q.get()
            if job is None:
                return
            try:
                self.res.put(self._decode(*job))
            except BaseException:
                import traceback
                self.err = traceback.format_exc()

    def submit(self, y, t0, f0, f1):
        if self.th is None:
            self.res.put(self._decode(y, t0, f0, f1))
        else:
            self.q.put((np.array(y, np.float32), t0, f0, f1))

    def poll(self, words):
        if self.err:
            raise RuntimeError('nft2 worker failed:\n' + self.err)
        out = []
        while True:
            try:
                u = self.res.get_nowait()
            except queue.Empty:
                break
            words.add(u['words'], 'nft2')
            words.mark_covered(u['end'])
            out.append(u)
        return out

    def drain(self, words):
        if self.th is not None:
            self.q.put(None)
            self.th.join()
        return self.poll(words)

    def transcribe_offline(self, wav, arc=None):
        """audio <= whole_max_s: one utterance (clip-length input); longer: energy segmentation, +-edge_pad_s around each"""
        from .transcribe import UttSegmenter
        from .api import frame_db, FR
        arc = arc or self.arc
        if len(wav) <= float(self.c['whole_max_s']) * SR:
            return [self.decode(wav, 0.0, 0, max(0, (len(wav) + FR - 1) // FR - 1))] if len(wav) >= 800 else []
        seg = UttSegmenter(arc.cfg.get('transcription', {}))
        closed = []
        n = (len(wav) + FR - 1) // FR
        for f in range(n):
            closed += seg.feed(f, frame_db(wav[f * FR:(f + 1) * FR]))
        closed += seg.flush()
        pad = int(float(self.c['edge_pad_s']) * SR)
        out = []
        for f0, f1, _ in closed:
            a = max(0, f0 * FR - pad)
            b = min(len(wav), (f1 + 1) * FR + pad)
            out.append(self.decode(wav[a:b], a / SR, f0, f1))
        return out
