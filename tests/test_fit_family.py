from __future__ import annotations

import sys
from pathlib import Path

from loop_jlens import HuginnLensModel, OuroLensModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fit_family  # noqa: E402

from tests.test_adapters import Tokenizer, ToyHuginn, ToyOuro  # noqa: E402


def test_source_window_keeps_layers_nearest_the_target():
    ouro = OuroLensModel(ToyOuro(), Tokenizer())
    family = fit_family.target_family(ouro, [1], full_sources=False, source_window=2)
    assert family == [("loop-1", 2, [0, 1])]

    huginn = HuginnLensModel(ToyHuginn(), Tokenizer(), recurrences=4)
    family = fit_family.target_family(huginn, [4], full_sources=False, source_window=2)
    target = huginn.n_layers - 1
    assert family == [("recurrence-4", target, [target - 2, target - 1])]


def test_source_window_rejects_non_positive():
    ouro = OuroLensModel(ToyOuro(), Tokenizer())
    try:
        fit_family.target_family(ouro, [1], full_sources=True, source_window=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")
