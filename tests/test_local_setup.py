from qusa.utils import config


def test_config_loads_project_env_without_overwriting_exports(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("QUSA_SETUP_FROM_FILE", raising=False)
    monkeypatch.setenv("QUSA_SETUP_EXPORTED", "shell-value")
    (tmp_path / ".env").write_text(
        '# Local settings\nQUSA_SETUP_FROM_FILE="file-value"\n'
        'QUSA_SETUP_EXPORTED=file-value\n'
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("data: {}\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert config.load_config(config_path) == {"data": {}}
    import os
    assert os.environ["QUSA_SETUP_FROM_FILE"] == "file-value"
    assert os.environ["QUSA_SETUP_EXPORTED"] == "shell-value"


def test_config_loads_without_env_file(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("data: {}\n")
    assert config.load_config(config_path) == {"data": {}}
