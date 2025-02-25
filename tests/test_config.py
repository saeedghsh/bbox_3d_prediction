# pylint: disable=missing-module-docstring, missing-function-docstring
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

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
