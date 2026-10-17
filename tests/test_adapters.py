from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from loop_jlens import HuginnLensModel, OuroLensModel, validate_adapter


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

    def forward(self, input_ids, num_steps, use_cache=False):
        hidden = self.transformer.wte(input_ids)
        for block in self.transformer.prelude:
            hidden = block(hidden)
        state = torch.randn_like(hidden)
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
