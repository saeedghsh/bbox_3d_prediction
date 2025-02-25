# pylint: disable=missing-module-docstring, missing-function-docstring
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from unittest.mock import patch

import pytest

from config.configuration_reader import read_config


@pytest.mark.parametrize(
    "yaml_content, expected",
    [
        ("key: value", {"key": "value"}),
        ("list:\n  - item1\n  - item2", {"list": ["item1", "item2"]}),
        ("42", 42),
        ("null", None),
        ("", None),
    ],
)
def test_read_config(yaml_content: str, expected: Any) -> None:
    with NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as temp_file:
        temp_file.write(yaml_content)
        temp_file_path = Path(temp_file.name)

    try:
        assert read_config(temp_file_path) == expected
    finally:
        temp_file_path.unlink()  # Cleanup


@pytest.mark.parametrize("is_colab", [True, False])
def test_read_config_resolves_paths(is_colab: bool) -> None:
    yaml_content = """
    dir:
        local: "local_path"
        colab: "colab_path"
    other_var: 1
    """
    expected = {
        "dir": "colab_path" if is_colab else "local_path",
        "other_var": 1,
    }

    with NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as temp_file:
        temp_file.write(yaml_content)
        temp_file_path = Path(temp_file.name)

    try:
        with patch.dict(sys.modules, {"google.colab": object()} if is_colab else {}):
            assert read_config(temp_file_path) == expected
    finally:
        temp_file_path.unlink()  # Cleanup
