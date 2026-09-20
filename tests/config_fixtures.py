"""The shipped configuration, copied somewhere a test may edit it.

Four tests grew their own `copytree` of `config/`; when a new required file appears under
it, one place learns about it rather than four.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from backend.infrastructure.config.settings import REPO_ROOT


def config_copy(tmp_path: Path) -> Path:
    """A mutable copy of the shipped config directory, ready for `config_dir=`."""
    config = tmp_path / "config"
    shutil.copytree(REPO_ROOT / "config", config)
    return config
