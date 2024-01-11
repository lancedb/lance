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

import atexit
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Optional, Union

import pyarrow as pa


class CachedDataset:
    def __init__(
        self,
        stream: Iterable[pa.RecordBatch],
        cache: Optional[Union[str, Path, bool]] = None,
    ):
        self.cache_dir: Optional[TemporaryDirectory] = None
        if cache is None or cache is True:
            self.cache_dir = TemporaryDirectory(
                prefix="lance-torch-dataset",
            )
        elif isinstance(cache, (str, Path)):
            self.cache_dir = TemporaryDirectory(
                prefix="lance-torch-dataset",
                dir=cache,
            )
        else:
            raise ValueError(f"Unsupported cache type: {type(cache)}")
        self.cache_file = None
        self.stream = stream
        self.finished_origin_stream = False

        atexit.register(lambda x: x.close(), self)

    def close(self):
        """Close the dataset and delete tmp files"""
        if self.cache_dir is not None:
            self.cache_dir.cleanup()
            self.cache_dir = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_type is not None:
            raise

    def __del__(self):
        self.close()

    def __iter__(self):
        if self.cache_file is None:
            # First iteration.
            writer: Optional[pa.ipc.RecordBatchFileWriter] = None
            for batch in self.stream:
                if writer is None:
                    self.cache_file = Path(self.cache_dir.name) / "cache.arrow"
                    writer = pa.ipc.new_stream(str(self.cache_file), batch.schema)
                writer.write(batch)
                yield batch
                del batch
            writer.close()
            self.finished_origin_stream = True
        else:
            # Follow up iteration
            if not self.finished_origin_stream:
                raise RuntimeError(
                    "CachedDataset: the iteration over original data has not finished"
                )
            reader = pa.ipc.open_stream(self.cache_file)
            for batch in reader:
                yield batch
                del batch
