import torch

def is_streaming_backbone(moon):
    return 'Streaming' in type(moon).__name__

def encoder_out(moon, x):
    n = x.shape[1] // 80 * 80
    x = x[:, :max(n, 80)]
    return moon.model.encoder(x, attention_mask=torch.ones(x.shape, dtype=torch.long, device=x.device))

def generate_kwargs(moon, x):
    n = max(80, x.shape[1] // 80 * 80)
    x = x[:, :n]
    return dict(input_values=x, attention_mask=torch.ones(x.shape, dtype=torch.long, device=x.device))

def states_output(moon, h):
    from transformers.models.moonshine_streaming.modeling_moonshine_streaming import MoonshineStreamingEncoderModelOutput as O
    return O(last_hidden_state=h.clone(), attention_mask=torch.ones(h.shape[:2], dtype=torch.bool, device=h.device))
