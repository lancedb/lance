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

from pathlib import Path
from typing import Optional, Union

from .dataset import LanceDataset, __version__, write_dataset


__all__ = [
    "LanceDataset",
    "__version__",
    "write_dataset",
    "dataset",
]


def dataset(uri: Union[str, Path], version: Optional[int] = None) -> LanceDataset:
    """
    Opens the Lance dataset from the address specified.

    Parameters
    ----------
    uri : str
        Address to the Lance dataset.
    version : optional, int or str
        If specified, loads the version of the Lance dataset specified by the integer id or the tag.
        Else, loads the latest version.
    """
    return LanceDataset(uri, version)
