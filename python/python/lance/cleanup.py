# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from typing import List, Tuple

from .lance import _cleanup_partial_writes


def cleanup_partial_writes(objects: List[Tuple[str, str]]):
    """Cleans up partial writes from a list of objects.

    These writes can be discovered using the
    :class:`lance.progress.FragmentWriteProgress` class.

    Parameters
    ----------
    objects : List[Tuple[str, str]]
        A list of tuples of (fragment_id, multipart_id) to clean up.
    """
    _cleanup_partial_writes(objects)
