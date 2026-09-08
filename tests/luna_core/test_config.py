import textwrap

from luna_core.config import LunaConfig, load_config


def test_load_config_parses_yaml(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            server:
              port: 9999
            llm:
              active_provider: ollama
              providers:
                ollama:
                  type: ollama
                  model: test-model
            """
        )
    )
    cfg = load_config(cfg_path)
    assert isinstance(cfg, LunaConfig)
    assert cfg.server.port == 9999
    assert cfg.llm.providers["ollama"].model == "test-model"


def test_load_config_applies_defaults_for_missing_sections(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("{}")
    cfg = load_config(cfg_path)
    assert cfg.wake_word.enabled is False
    assert cfg.wake_word.model_path is None
    assert cfg.permissions.default_level == "deny"
    assert cfg.tts.engine == "piper"
