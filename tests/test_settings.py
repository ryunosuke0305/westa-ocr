import os

from app import settings


def test_load_env_file_overrides_environment(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# comment should be ignored",
                "export FOO=bar",
                'BAR="baz qux"',
                "EMPTY=",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("FOO", "old")
    monkeypatch.setenv("BAR", "old")
    monkeypatch.delenv("EMPTY", raising=False)

    loaded = settings.load_env_file(path=env_path)

    assert loaded == {"FOO": "bar", "BAR": "baz qux", "EMPTY": ""}
    assert os.getenv("FOO") == "bar"
    assert os.getenv("BAR") == "baz qux"
    assert os.getenv("EMPTY") == ""


def test_save_env_file_quotes_and_skips_none(tmp_path):
    env_path = tmp_path / ".env"

    settings.save_env_file(
        {
            "SIMPLE": "value",
            "SPACE": "hello world",
            "QUOTE": 'a"b',
            "MULTILINE": "line1\nline2",
            "SKIP": None,
        },
        path=env_path,
    )

    content = env_path.read_text(encoding="utf-8").splitlines()

    assert "SIMPLE=value" in content
    assert 'SPACE="hello world"' in content
    assert 'QUOTE="a\\"b"' in content
    assert 'MULTILINE="line1\\nline2"' in content
    assert all(not line.startswith("SKIP=") for line in content)

    # Ensure the generated file can be loaded back without errors.
    loaded = settings.load_env_file(path=env_path)
    assert loaded["SPACE"] == "hello world"
    assert loaded["QUOTE"] == 'a"b'
    assert loaded["MULTILINE"] == "line1\nline2"
