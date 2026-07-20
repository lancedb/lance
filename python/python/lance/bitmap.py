# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

# Both names refer to the same class: `Bitmap` for use in type hints and
# isinstance checks, and the lowercase `bitmap` constructor-style alias
# (like `list`/`set`) for `from lance.bitmap import bitmap; bitmap([1, 2])`.
from .lance import Bitmap as Bitmap
from .lance import Bitmap as bitmap

__all__ = ["Bitmap", "bitmap"]
