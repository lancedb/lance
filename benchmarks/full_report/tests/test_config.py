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

import numpy as np
from os import path
from vbench.experiment import Benchmark


def test_config(tempdir, random_dataset):
    query = np.random.standard_normal((10, 128)).astype(np.float32)
    gt = np.arange(10)
    query_path = path.join(tempdir, "query.npy")
    np.save(query_path, query)
    gt_path = path.join(tempdir, "gt.npy")
    np.save(gt_path, gt)

    config = {
        "name": "test",
        "data": random_dataset.uri,
        "desc": "test",
        "test_cases": {
            "test": {
                "query": query_path,
                "ground_truth": gt_path,
            }
        },
    }

    benchmark = Benchmark(**config)

    assert len(benchmark.test_cases) == 1
    assert benchmark.test_cases["test"].name == "test"
    assert (benchmark.test_cases["test"].ground_truth == np.arange(10)).all()
    assert (benchmark.test_cases["test"].query == query).all()

    assert benchmark.data.to_table() == random_dataset.to_table()
