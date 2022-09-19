# Copyright 2022 Lance Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Callable, List, Optional, Union
from urllib.parse import urlparse

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset
import pyarrow.fs

try:
    import torch
    from torch.utils.data import IterableDataset
except ImportError as e:
    raise ImportError("Please install pytorch", e)

import lance
from lance import dataset
from lance.types.image import is_image

__all__ = ["LanceDataset"]


def to_tensor(arr: pa.Array):
    """Convert pyarrow array to Pytorch Tensors"""
    # TODO: arrow.to_numpy(writable=True) makes a new copy of data.
    # Investigate how to directly perform zero-copy into Torch Tensor.
    if pa.types.is_dictionary(arr.type):
        return arr.indices.to_numpy()
    if is_image(arr.type):
        return [img.to_pil() for img in arr.tolist()]
    np_arr = arr.to_numpy(zero_copy_only=False, writable=True)
    if pa.types.is_binary(arr.type) or pa.types.is_large_binary(arr.type):
        return np_arr.astype(np.bytes_)
    elif pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
        return np_arr.astype(np.str_)
    else:
        return np_arr


class LanceDataset(IterableDataset):
    """An PyTorch IterableDataset.

    See:
    https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    """

    def __init__(
        self,
        root: Union[str, Path],
        columns: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        filter: Optional[pc.Expression] = None,
        transform: Optional[Callable] = None,
    ):
        """LanceDataset

        Parameters
        ----------
        root : str or Path
            The root URI
        columns : list of str, optional
            List of the column names.
        batch_size : int, optional
            The batch size.
        filter : filter
            The filter to apply to the scanner.
        transform : Callable, optional
            Apply transform to each of the example.
        """
        self.root = root
        # Handle local relative path
        if isinstance(root, str) and not urlparse(root).scheme:
            self.root = Path(root)
        self.columns = columns if columns else []
        self.filter = filter
        self.batch_size = batch_size
        self.transform = transform

        self._dataset: pa.dataset.FileSystemDataset = None
        self._fs: Optional[pyarrow.fs.FileSystem] = None
        self._files: Optional[List[str]] = None

    def __repr__(self):
        return f"LanceDataset(root={self.root})"

    def _setup_dataset(self):
        """Lazy loading dataset in different process."""
        if self._files:
            return self._files

        self._fs, _ = pyarrow.fs.FileSystem.from_uri(self.root)
        self._files = dataset(self.root).files
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            # Split the work using at the files level for now.
            rank = worker_info.id
            num_workers = worker_info.num_workers
            self._files = [
                self._files[i] for i in range(rank, len(self._files), num_workers)
            ]

    def __iter__(self):
        """Yield dataset"""
        self._setup_dataset()
        for file_uri in self._files:
            ds = lance.dataset(
                file_uri,
                filesystem=self._fs,
            )
            scan = ds.scanner(
                columns=self.columns, batch_size=self.batch_size, filter=self.filter
            )
            for batch in scan.to_reader():
                tensors = [to_tensor(arr) for arr in batch.columns]
                if len(tensors) == 1:
                    # Only one column to return
                    yield tensors[0]
                yield tensors
