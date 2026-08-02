import pytest

from triage_eg.common.config import load_yaml_config, validate_required_keys


def test_load_config_and_resolve_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("AIC_TEST_ROOT", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text("root: ${AIC_TEST_ROOT}\nnested:\n  value: 4\n", encoding="utf-8")
    config = load_yaml_config(config_path)
    assert config == {"root": str(tmp_path), "nested": {"value": 4}}
    validate_required_keys(config, ["root", "nested.value"])


def test_missing_environment_variable_is_clear(tmp_path, monkeypatch):
    monkeypatch.delenv("AIC_MISSING_TEST", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("root: ${AIC_MISSING_TEST}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="AIC_MISSING_TEST"):
        load_yaml_config(config_path)
