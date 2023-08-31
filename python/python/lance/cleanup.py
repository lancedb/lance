#  Copyright (c) 2023. Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

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
