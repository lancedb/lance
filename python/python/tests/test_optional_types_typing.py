# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Type-checking regression tests for optional dependency input types.

This module is part of the pyright target configured in ``pyproject.toml``, so
a regression in the annotations below fails the repository type check and not
only the runtime suite.

The rejected cases are pinned with ``pyright: ignore`` comments and
``reportUnnecessaryTypeIgnoreComment``, so the file also fails if an annotation
becomes *too* permissive and one of them stops being an error. They sit in a
``TYPE_CHECKING`` block because they are invalid at runtime; the matching
runtime assertions live in ``test_dataset.py``.

Like ``test_fragment_typing.py``, this module does not import ``pytest``: the
lint workflow installs pyright without the test dependencies.
"""

# pyright: reportUnnecessaryTypeIgnoreComment=true

from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd
from lance.util import sanitize_ts

if TYPE_CHECKING:
    import lance

    def _check_accepted_asof_types() -> None:
        # ``sanitize_ts`` is exercised at runtime below; this pins the public
        # entry point that forwards to it.
        lance.dataset("memory://unused", asof=pd.Timestamp("2026-01-01"))
        lance.dataset("memory://unused", asof=pd.NaT)
        lance.dataset("memory://unused", asof=datetime(2026, 1, 1))
        lance.dataset("memory://unused", asof="2026-01-01")

    def _check_rejected_asof_types() -> None:
        # `DatetimeIndex` has a `to_pydatetime` method, so it satisfies a
        # structural timestamp annotation without being a valid instant.
        index = pd.DatetimeIndex(["2026-01-01"])
        lance.dataset(
            "memory://unused",
            asof=index,  # pyright: ignore[reportArgumentType]
        )
        sanitize_ts(object())  # pyright: ignore[reportArgumentType]


def test_sanitize_ts_accepts_pandas_timestamp() -> None:
    # The stubs bundled with pandas type this constructor as
    # ``Timestamp | NaTType``, so both halves have to satisfy ``ts_types``.
    result: datetime = sanitize_ts(pd.Timestamp("2026-01-01"))

    assert result == datetime(2026, 1, 1)


def test_sanitize_ts_accepts_datetime_and_str() -> None:
    assert sanitize_ts(datetime(2026, 1, 1)) == datetime(2026, 1, 1)
    assert sanitize_ts("2026-01-01 00:00:00") == datetime(2026, 1, 1)
