"""Virtual-unrolling adapters for the two recurrent models in arXiv:2609.01924.

The adapters expose each firing of a shared physical block as a separate
``nn.Module`` hook point.  This lets the unmodified ``jlens`` fitting code see
the recurrent computation as an ordinary deep residual stack.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn


class VirtualBlock(nn.Module):
    """Identity hook point representing one firing of a physical block."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


def _first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"block output has unsupported type {type(output)!r}")


def _replace_first_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return tensor
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    if isinstance(output, list):
        return [tensor, *output[1:]]
    raise TypeError(f"block output has unsupported type {type(output)!r}")


class _VirtualLensModel:
    """Shared implementation of the model-independent ``jlens`` protocol."""

    def __init__(self, hf_model: nn.Module, tokenizer: Any) -> None:
        self._hf_model = hf_model
        self.tokenizer = tokenizer
        self._physical: list[tuple[nn.Module, Callable[[int], int]]] = []
        self._expected_firings: list[int] = []
        hf_model.eval()
        for parameter in hf_model.parameters():
            parameter.requires_grad_(False)
        if hasattr(hf_model, "gradient_checkpointing_disable"):
            hf_model.gradient_checkpointing_disable()

    @property
    def input_device(self) -> torch.device:
        return self._embed_tokens.weight.device

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        return encoded.input_ids.to(self.input_device)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        weight = self._lm_head.weight
        residual = residual.to(device=weight.device, dtype=weight.dtype)
        return self._lm_head(self._final_norm(residual))

    @contextmanager
    def _route_virtual_blocks(self):
        counters = [0] * len(self._physical)
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def make_hook(slot: int, locate: Callable[[int], int]):
            def hook(module: nn.Module, inputs: Any, output: Any) -> Any:
                firing = counters[slot]
                counters[slot] += 1
                virtual_index = locate(firing)
                if virtual_index >= len(self.layers):
                    raise RuntimeError(
                        f"{type(module).__name__} fired too many times "
                        f"(unexpected firing {firing})"
                    )
                tensor = self.layers[virtual_index](_first_tensor(output))
                return _replace_first_tensor(output, tensor)

            return hook

        try:
            for slot, (block, locate) in enumerate(self._physical):
                handles.append(block.register_forward_hook(make_hook(slot, locate)))
            yield
            if counters != self._expected_firings:
                raise RuntimeError(
                    "virtual unrolling mismatch: "
                    f"observed firings {counters}, expected {self._expected_firings}"
                )
        finally:
            for handle in handles:
                handle.remove()


class OuroLensModel(_VirtualLensModel):
    """Expose an Ouro stack looped ``n_loops`` times as virtual depth."""

    def __init__(self, hf_model: nn.Module, tokenizer: Any) -> None:
        super().__init__(hf_model, tokenizer)
        text_model = hf_model.model
        physical_layers: Sequence[nn.Module] = text_model.layers
        self.n_physical_layers = len(physical_layers)
        self.n_loops = int(text_model.total_ut_steps)
        self.n_layers = self.n_physical_layers * self.n_loops
        self.d_model = int(hf_model.config.hidden_size)
        self.layers = nn.ModuleList(VirtualBlock() for _ in range(self.n_layers))
        self._embed_tokens = text_model.embed_tokens
        self._final_norm = text_model.norm
        self._lm_head = hf_model.lm_head
        text_model.gradient_checkpointing = False

        for physical_index, block in enumerate(physical_layers):
            self._physical.append(
                (
                    block,
                    lambda firing, i=physical_index: firing * self.n_physical_layers + i,
                )
            )
            self._expected_firings.append(self.n_loops)

    def forward(self, input_ids: torch.Tensor) -> Any:
        with self._route_virtual_blocks():
            return self._hf_model.model(input_ids=input_ids, use_cache=False)


class HuginnLensModel(_VirtualLensModel):
    """Expose Huginn's prelude + recurrent core + coda as virtual depth."""

    def __init__(
        self,
        hf_model: nn.Module,
        tokenizer: Any,
        *,
        recurrences: int = 16,
        seed: int = 0,
    ) -> None:
        if recurrences < 1:
            raise ValueError("recurrences must be positive")
        super().__init__(hf_model, tokenizer)
        transformer = hf_model.transformer
        prelude: Sequence[nn.Module] = transformer.prelude
        core: Sequence[nn.Module] = transformer.core_block
        coda: Sequence[nn.Module] = transformer.coda
        self.n_prelude = len(prelude)
        self.n_core = len(core)
        self.n_coda = len(coda)
        self.recurrences = recurrences
        self.seed = seed
        self.n_layers = self.n_prelude + self.n_core * recurrences + self.n_coda
        self.d_model = int(hf_model.config.n_embd)
        self.layers = nn.ModuleList(VirtualBlock() for _ in range(self.n_layers))
        self._embed_tokens = transformer.wte
        self._final_norm = transformer.ln_f
        self._lm_head = hf_model.lm_head
        hf_model.gradient_checkpointing = False
        if hasattr(hf_model.config, "test_time_noise"):
            hf_model.config.test_time_noise = 0

        for index, block in enumerate(prelude):
            self._physical.append((block, lambda firing, i=index: i))
            self._expected_firings.append(1)
        for index, block in enumerate(core):
            self._physical.append(
                (
                    block,
                    lambda firing, i=index: self.n_prelude + firing * self.n_core + i,
                )
            )
            self._expected_firings.append(recurrences)
        coda_start = self.n_prelude + self.n_core * recurrences
        for index, block in enumerate(coda):
            self._physical.append((block, lambda firing, i=index: coda_start + i))
            self._expected_firings.append(1)

    @contextmanager
    def _seeded_rng(self):
        cuda_devices = sorted(
            {
                parameter.device.index
                for parameter in self._hf_model.parameters()
                if parameter.device.type == "cuda" and parameter.device.index is not None
            }
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(self.seed)
            yield

    def forward(self, input_ids: torch.Tensor) -> Any:
        # A scalar num_steps sends every recurrence through Huginn's no-grad
        # phase.  (0, r) is required so Jacobians reach all recurrent firings.
        with self._seeded_rng(), self._route_virtual_blocks():
            return self._hf_model(
                input_ids=input_ids,
                num_steps=(0, self.recurrences),
                use_cache=False,
            )


def from_looped_hf(
    hf_model: nn.Module,
    tokenizer: Any,
    *,
    architecture: str = "auto",
    recurrences: int = 16,
    seed: int = 0,
) -> OuroLensModel | HuginnLensModel:
    """Create the appropriate virtual-unrolling adapter for a loaded model."""

    if architecture == "auto":
        if hasattr(getattr(hf_model, "model", None), "total_ut_steps"):
            architecture = "ouro"
        elif hasattr(getattr(hf_model, "transformer", None), "core_block"):
            architecture = "huginn"
        else:
            raise ValueError(f"cannot identify looped architecture {type(hf_model).__name__}")
    if architecture == "ouro":
        return OuroLensModel(hf_model, tokenizer)
    if architecture == "huginn":
        return HuginnLensModel(
            hf_model, tokenizer, recurrences=recurrences, seed=seed
        )
    raise ValueError("architecture must be one of: auto, ouro, huginn")
