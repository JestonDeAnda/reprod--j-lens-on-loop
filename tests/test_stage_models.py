import json

from scripts.stage_models_dsw import is_complete


def test_incomplete_sharded_model_without_index_is_detected(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model-00001-of-00003.safetensors").touch()
    (tmp_path / "model-00003-of-00003.safetensors").touch()

    assert not is_complete(tmp_path)

    (tmp_path / "model-00002-of-00003.safetensors").touch()
    assert is_complete(tmp_path)


def test_index_controls_completeness(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}")
    index = {"weight_map": {"a": "first.safetensors", "b": "second.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    (tmp_path / "first.safetensors").touch()

    assert not is_complete(tmp_path)

    (tmp_path / "second.safetensors").touch()
    assert is_complete(tmp_path)
