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

"""
This file contains the runner for the benchmarking framework.
"""

from typing import Any, Dict

import attrs
import logging
import functools
import numpy as np
import time
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

from .experiment import Benchmark


@attrs.define
class RecallStats:
    dataset_name: str
    test_name: str
    index_params: Dict[str, Any]
    query_params: Dict[str, Any]

    recall: float

    latency_mean: float
    latency_std: float

    throughput: float


@attrs.define
class BenchmarkRunner:
    benchmark: Benchmark = attrs.field(kw_only=True)
    test_name: str = attrs.field(kw_only=True)
    query_params: Dict[str, Any] = attrs.field(kw_only=True)

    num_threads: int = attrs.field(default=1, kw_only=True)

    # not actually used, just for returning this in the stats
    index_params: Dict[str, Any] = attrs.field(kw_only=True)

    def run(self) -> RecallStats:
        logging.info("running test %s on %s", self.test_name, self.benchmark.name)

        test_case = self.benchmark.test_cases[self.test_name]

        k_gt = test_case.ground_truth.shape[1]

        count = 0
        total_hits = 0

        latency_sum = 0
        latency_squared_sum = 0

        def _test(q, gt):
            nonlocal latency_sum, latency_squared_sum
            start = time.perf_counter()
            res = self.benchmark.data.to_table(
                nearest={"q": q, **self.query_params},
                with_row_id=True,
                columns=[],
            )["_rowid"].to_numpy()
            end = time.perf_counter()

            elapsed = end - start
            latency_sum += elapsed
            latency_squared_sum += elapsed**2
            return len(np.intersect1d(res, gt))

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futs = []

            # warmup
            for _ in range(100 * self.num_threads):
                futs.append(
                    executor.submit(
                        functools.partial(
                            _test,
                            np.random.standard_normal(test_case.query.shape[1]),
                            list(range(k_gt)),
                        )
                    )
                )
            wait(futs)

            futs = []
            for q, gt in zip(test_case.query, test_case.ground_truth):
                futs.append(executor.submit(functools.partial(_test, q, gt)))

            pbar = tqdm(total=test_case.query.shape[0], desc="recall: 0.0000")
            for fut in as_completed(futs):
                total_hits += fut.result()
                count += 1
                pbar.update()
                pbar.set_description(f"recall: {total_hits / k_gt / count:.4f}")

        recall = total_hits / k_gt / count
        latency_mean = latency_sum / count
        # E[(X - E[X]) ^ 2] = E[X^2] - E[X]^2
        latency_std = np.sqrt(latency_squared_sum / count - latency_mean**2)
        return RecallStats(
            dataset_name=self.benchmark.name,
            test_name=self.test_name,
            index_params=self.index_params,
            query_params={
                "num_threads_in_runner": self.num_threads,
                **self.query_params,
            },
            recall=recall,
            latency_mean=latency_mean,
            latency_std=latency_std,
            throughput=count / latency_sum,
        )
