#  Copyright (c) 2024. Lance Developers
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

"""Preparing Benchmark Data"""

from abc import ABC, abstractmethod
from subprocess import check_call
from pathlib import Path
from typing import Tuple
import numpy as np

import lance


class BenchmarkData(ABC):
    @abstractmethod
    def base(self) -> lance.LanceDataset:
        """Return the base lance dataset.

        Base dataset is used to store the vectors.
        """

    @abstractmethod
    def ground_truth(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the queries and ground truth.

        Returns query and ground truth vectors.
        """


class Text2Image_10M(BenchmarkData):
    """Text2image(10M) dataset from BigANN benchmark."""

    URL = "s3://lancedb-public/text2image-10m.lance"
    GT_URL = "s3://lancedb-public/text2image-10m-queries.lance"

    cached_dir = Path("cached_data")

    @property
    def base_dir(self) -> Path:
        return self.cached_dir / "text2image-10m.lance"

    @property
    def ground_truth_dir(self) -> Path:
        return self.cached_dir / "text2image-10m-queries.lance"

    def prepare(self) -> lance.LanceDataset:
        if not self.cached_dir.exists():
            self.cached_dir.mkdir()

        if not self.base_dir.exists():
            check_call(
                f"aws s3 cp --recursive --no-sign-request {self.URL} {self.base_dir}",
                shell=True,
            )

        if not self.ground_truth_dir.exists():
            check_call(
                f"aws s3 cp --recursive --no-sign-request {self.GT_URL} {self.ground_truth_dir}",
                shell=True,
            )

    def base(self) -> lance.LanceDataset:
        self.prepare()
        return lance.LanceDataset(self.base_dir)

    def ground_truth(self) -> Tuple[np.ndarray]:
        self.prepare()
        ds = lance.LanceDataset(self.ground_truth_dir)
        tbl = ds.to_table()
        queries = np.stack(tbl["vector"].to_numpy())
        ground_truth = np.stack(tbl["ground_truth"].to_numpy())

        return (queries, ground_truth)


DATASETS = {
    "text2image-10M": Text2Image_10M(),
}
