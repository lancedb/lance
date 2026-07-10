# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import pyarrow as pa
import pytest


class ManifestOpenProbe:
    def __init__(self, path: Path):
        self.path = str(path)

    def read(self):
        import lance

        try:
            lance.dataset(self.path)
        except Exception as error:
            return type(error).__name__, str(error)
        return None


@pytest.mark.compat
def test_previous_release_reader_fails_closed(tmp_path, venv_factory):
    path = tmp_path / "columnar-manifest.lance"
    lance.write_dataset(
        pa.table({"value": [1, 2, 3]}), path, data_storage_version="2.3"
    )

    previous = venv_factory.get_venv("9.0.0b20")
    result = previous.execute_method(ManifestOpenProbe(path), "read")

    assert result is not None, "the previous release unexpectedly read the new manifest"
    exception_type, message = result
    assert exception_type in {"OSError", "ValueError"}
    assert message
    assert "panic" not in message.lower()
