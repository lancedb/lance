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

"""This file contains the experiment configuration for the benchmarking
framework. The configuration is loaded from a yaml file and converted
into a Benchmark object."""

from typing import Any, Dict
import attrs
import lance
import numpy as np


@attrs.define
class AnnTest:
    name: str = attrs.field()
    query: np.ndarray = attrs.field(converter=np.load)
    ground_truth: np.ndarray = attrs.field(converter=np.load)


def _load_dataset(params: str | Dict[str, Any]) -> lance.LanceDataset:
    if isinstance(params, dict):
        return lance.LanceDataset(**params)
    elif isinstance(params, str):
        return lance.dataset(params)
    else:
        raise ValueError(f"Invalid dataset parameter: {params}")


@attrs.define
class Benchmark:
    name: str = attrs.field()
    data: lance.LanceDataset = attrs.field(converter=_load_dataset)
    desc: str = attrs.field()
    test_cases: Dict[str, AnnTest] = attrs.field(
        converter=lambda configs: {
            name: AnnTest(name=name, **config) for name, config in configs.items()
        }
    )
