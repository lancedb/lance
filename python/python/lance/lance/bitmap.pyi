# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from typing import Iterable, Iterator, Optional, Union

import pyarrow as pa

class Bitmap:
    """
    A set of non-negative integers backed by a Rust ``RoaringBitmap``.

    Cheap to clone and to pass into Lance APIs that accept a bitmap, since no
    per-value Python object is created. Mutating methods (``add``,
    ``discard``, ``update``) copy-on-write: cloning a ``Bitmap`` and mutating
    one copy never affects the other.
    """

    def __init__(
        self,
        values: Optional[
            Union[Iterable[int], pa.Array, pa.ChunkedArray, "Bitmap"]
        ] = None,
    ) -> None: ...
    def __len__(self) -> int: ...
    def __contains__(self, value: int) -> bool: ...
    def __iter__(self) -> Iterator[int]: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def add(self, value: int) -> None: ...
    def discard(self, value: int) -> None: ...
    def update(self, values: Iterable[int]) -> None: ...
