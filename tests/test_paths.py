"""Tests for tts_books.paths — XDG resolution and legacy migration.

Every test uses monkeypatch to set HOME (and optionally XDG_CONFIG_HOME)
to a temp dir so the user's real ~/bin/ and ~/.config/ are never touched.
Because paths.py caches nothing at module scope after the refactor,
we can freely re-invoke config_dir() and migrate_legacy() per-test.
"""

import importlib
from pathlib import Path


def _fresh_paths_module(monkeypatch, tmp_home: Path, xdg_config: Path | None = None):
    """Reload tts_books.paths so its module-level constants pick up the
    monkeypatched HOME / XDG_CONFIG_HOME. Necessary because APP_CONFIG_PATH
    etc. are evaluated at module import time."""
    monkeypatch.setenv("HOME", str(tmp_home))
    if xdg_config is not None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config))
    else:
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    import tts_books.paths as paths
    return importlib.reload(paths)


class TestConfigDir:
    def test_uses_xdg_config_home_when_set(self, monkeypatch, tmp_path):
        xdg = tmp_path / "xdg"
        paths = _fresh_paths_module(monkeypatch, tmp_home=tmp_path, xdg_config=xdg)
        assert paths.config_dir() == xdg / "tts-books"

    def test_falls_back_to_dot_config(self, monkeypatch, tmp_path):
        paths = _fresh_paths_module(monkeypatch, tmp_home=tmp_path)
        assert paths.config_dir() == tmp_path / ".config" / "tts-books"

    def test_creates_directory_if_missing(self, monkeypatch, tmp_path):
        paths = _fresh_paths_module(monkeypatch, tmp_home=tmp_path)
        d = paths.config_dir()
        assert d.is_dir()

    def test_empty_xdg_config_home_falls_back(self, monkeypatch, tmp_path):
        """Empty string is treated as unset per XDG spec convention."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("XDG_CONFIG_HOME", "")
        import tts_books.paths as paths
        paths = importlib.reload(paths)
        # Empty string is falsy in Python; our impl uses `or` which treats
        # it as unset. Validates that behavior.
        assert paths.config_dir() == tmp_path / ".config" / "tts-books"


class TestDataDir:
    def test_uses_xdg_data_home_when_set(self, monkeypatch, tmp_path):
        xdg = tmp_path / "xdg-data"
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
        import tts_books.paths as paths
        paths = importlib.reload(paths)
        assert paths.data_dir() == xdg / "tts-books"

    def test_falls_back_to_local_share(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        import tts_books.paths as paths
        paths = importlib.reload(paths)
        assert paths.data_dir() == tmp_path / ".local" / "share" / "tts-books"


class TestExposedConstants:
    def test_all_three_are_str_type(self, monkeypatch, tmp_path):
        """gui.py uses `PATH + '.tmp'` so exposed constants must be str,
        not Path (Path + str raises TypeError)."""
        paths = _fresh_paths_module(monkeypatch, tmp_home=tmp_path)
        assert isinstance(paths.APP_CONFIG_PATH, str)
        assert isinstance(paths.QUEUE_PATH, str)
        assert isinstance(paths.PRON_DICT_PATH, str)

    def test_all_three_live_under_config_dir(self, monkeypatch, tmp_path):
        paths = _fresh_paths_module(monkeypatch, tmp_home=tmp_path)
        cfg = paths.config_dir()
        assert Path(paths.APP_CONFIG_PATH).parent == cfg
        assert Path(paths.QUEUE_PATH).parent == cfg
        assert Path(paths.PRON_DICT_PATH).parent == cfg

    def test_expected_filenames(self, monkeypatch, tmp_path):
        paths = _fresh_paths_module(monkeypatch, tmp_home=tmp_path)
        assert Path(paths.APP_CONFIG_PATH).name == "app.json"
        assert Path(paths.QUEUE_PATH).name == "queue.json"
        assert Path(paths.PRON_DICT_PATH).name == "pronunciation.json"


class TestMigrateLegacy:
    def _seed_legacy(self, tmp_home: Path) -> dict[str, Path]:
        """Create fake ~/bin/tts-book.* files. Return {kind: legacy_path}."""
        bin_dir = tmp_home / "bin"
        bin_dir.mkdir(parents=True)
        files = {
            "config": bin_dir / "tts-book.config",
            "queue": bin_dir / "tts-book.queue.json",
            "pron": bin_dir / "tts-pronunciation.json",
        }
        files["config"].write_text('{"instance_count": 3}')
        files["queue"].write_text('[]')
        files["pron"].write_text('{"foo": "bar"}')
        return files

    def test_copies_all_three_when_none_exist_at_new(self, monkeypatch, tmp_path):
        legacy = self._seed_legacy(tmp_path)
        paths = _fresh_paths_module(monkeypatch, tmp_home=tmp_path)
        copied = paths.migrate_legacy()

        assert len(copied) == 3
        assert Path(paths.APP_CONFIG_PATH).read_text() == '{"instance_count": 3}'
        assert Path(paths.QUEUE_PATH).read_text() == '[]'
        assert Path(paths.PRON_DICT_PATH).read_text() == '{"foo": "bar"}'
        # Legacy files must remain in place (copy, not move — safety net)
        assert legacy["config"].exists()
        assert legacy["queue"].exists()
        assert legacy["pron"].exists()

    def test_is_idempotent(self, monkeypatch, tmp_path):
        self._seed_legacy(tmp_path)
        paths = _fresh_paths_module(monkeypatch, tmp_home=tmp_path)
        paths.migrate_legacy()  # first run
        second = paths.migrate_legacy()  # should be a no-op
        assert second == []

    def test_partial_migration_only_copies_missing(self, monkeypatch, tmp_path):
        """If the user manually copied one file, migrate should copy the
        remaining two without overwriting the manual one."""
        self._seed_legacy(tmp_path)
        paths = _fresh_paths_module(monkeypatch, tmp_home=tmp_path)
        # Pre-populate one target with different content
        Path(paths.APP_CONFIG_PATH).write_text('{"manual": true}')
        copied = paths.migrate_legacy()
        assert len(copied) == 2
        # Manual content preserved -- legacy did NOT overwrite
        assert Path(paths.APP_CONFIG_PATH).read_text() == '{"manual": true}'

    def test_noop_when_no_legacy_files(self, monkeypatch, tmp_path):
        paths = _fresh_paths_module(monkeypatch, tmp_home=tmp_path)
        assert paths.migrate_legacy() == []

    def test_returns_legacy_new_pairs(self, monkeypatch, tmp_path):
        self._seed_legacy(tmp_path)
        paths = _fresh_paths_module(monkeypatch, tmp_home=tmp_path)
        copied = paths.migrate_legacy()
        for legacy_path, new_path in copied:
            assert legacy_path.exists()  # legacy still there
            assert Path(new_path).exists()  # new created
