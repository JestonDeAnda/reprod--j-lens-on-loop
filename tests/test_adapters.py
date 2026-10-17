from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from loop_jlens import HuginnLensModel, OuroLensModel, validate_adapter
from loop_jlens.compat import default_rope_parameters, install_runtime_compat


class Tokenizer:
    bos_token_id = None

    def __call__(self, text, return_tensors, truncation, max_length):
        ids = [ord(character) % 16 for character in text[:max_length]]
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))

    def decode(self, ids):
        return "".join(chr(int(index) + 65) for index in ids)


class Block(nn.Module):
    def __init__(self, width: int, offset: float) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(width) * (1 + offset))

    def forward(self, hidden_states, *args, **kwargs):
        return torch.tanh(self.linear(hidden_states) + 0.01)


class ToyOuroBody(nn.Module):
    def __init__(self, width=4, physical=3, loops=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, width)
        self.layers = nn.ModuleList(Block(width, index / 10) for index in range(physical))
        self.norm = nn.LayerNorm(width)
        self.total_ut_steps = loops
        self.gradient_checkpointing = False

    def forward(self, input_ids, use_cache=False):
        hidden = self.embed_tokens(input_ids)
        for _ in range(self.total_ut_steps):
            for layer in self.layers:
                hidden = layer(hidden)
        return hidden


class ToyOuro(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.model = ToyOuroBody()
        self.lm_head = nn.Linear(4, 16, bias=False)


class ToyHuginn(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.config = SimpleNamespace(n_embd=width, test_time_noise=0.1)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(16, width),
                "prelude": nn.ModuleList([Block(width, 0.1)]),
                "core_block": nn.ModuleList([Block(width, 0.2), Block(width, 0.3)]),
                "coda": nn.ModuleList([Block(width, 0.4)]),
                "ln_f": nn.LayerNorm(width),
            }
        )
        self.lm_head = nn.Linear(width, 16, bias=False)
        self.gradient_checkpointing = False

    def initialize_state(self, input_embeds, scale: float = 1.0):
        # Matches tomg-group-umd/huginn-0125: one independent draw per batch row.
        del scale
        return torch.randn_like(input_embeds)

    def forward(self, input_ids, num_steps, use_cache=False):
        hidden = self.transformer.wte(input_ids)
        for block in self.transformer.prelude:
            hidden = block(hidden)
        state = self.initialize_state(hidden)
        no_grad_steps, grad_steps = num_steps
        with torch.no_grad():
            for _ in range(no_grad_steps):
                for block in self.transformer.core_block:
                    state = block(state + hidden)
        for _ in range(grad_steps):
            for block in self.transformer.core_block:
                state = block(state + hidden)
        for block in self.transformer.coda:
            state = block(state)
        state = self.transformer.ln_f(state)
        return SimpleNamespace(logits=self.lm_head(state))


def test_ouro_virtual_depth_and_validation():
    adapter = OuroLensModel(ToyOuro(), Tokenizer())
    assert adapter.n_layers == 6
    report = validate_adapter(adapter, "a sufficiently long prompt", max_seq_len=24)
    assert report.virtual_layers == 6
    assert report.nonzero_gradients


def test_huginn_virtual_depth_determinism_and_gradients():
    adapter = HuginnLensModel(ToyHuginn(), Tokenizer(), recurrences=3, seed=123)
    assert adapter.n_layers == 1 + 2 * 3 + 1
    report = validate_adapter(adapter, "a sufficiently long prompt", max_seq_len=24)
    assert report.virtual_layers == 8
    assert report.deterministic
    assert report.max_repeat_error == 0


def test_huginn_disables_test_time_noise():
    model = ToyHuginn()
    HuginnLensModel(model, Tokenizer(), recurrences=2)
    assert model.config.test_time_noise == 0


def test_huginn_replicated_batch_shares_one_latent_init():
    from jlens.hooks import ActivationRecorder

    adapter = HuginnLensModel(ToyHuginn(), Tokenizer(), recurrences=2, seed=5)
    ids = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    recorded = list(range(adapter.n_layers))
    with torch.no_grad(), ActivationRecorder(adapter.layers, at=recorded) as recorder:
        adapter.forward(ids)
    for activation in recorder.activations.values():
        assert torch.equal(activation[0], activation[1])
        assert torch.equal(activation[0], activation[2])


def test_default_rope_parameters_match_original_formula():
    config = SimpleNamespace(rope_theta=10000.0, hidden_size=32, num_attention_heads=4, head_dim=8)
    inv_freq, scale = default_rope_parameters(config)
    exponent = torch.arange(0, 8, 2, dtype=torch.float) / 8
    expected = 1.0 / (10000.0 ** exponent)
    assert scale == 1.0
    assert torch.allclose(inv_freq, expected)


def test_ouro_remote_rotary_init_does_not_require_library_method():
    from transformers.modeling_utils import PreTrainedModel

    install_runtime_compat()

    class RemoteRotaryEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.rope_type = "default"
            self.config = SimpleNamespace(
                rope_theta=10000.0,
                hidden_size=32,
                num_attention_heads=4,
                head_dim=8,
            )
            self.register_buffer("inv_freq", torch.zeros(4))
            self.register_buffer("original_inv_freq", torch.zeros(4))

    module = RemoteRotaryEmbedding()
    host = SimpleNamespace(config=SimpleNamespace(initializer_range=0.02))
    PreTrainedModel._init_weights(host, module)
    expected, _ = default_rope_parameters(module.config)
    assert torch.allclose(module.inv_freq, expected)
    assert torch.allclose(module.original_inv_freq, expected)
