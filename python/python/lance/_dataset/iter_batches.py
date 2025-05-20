# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Optional

import pyarrow as pa


class PrefetchIterator:
    def __init__(
        self,
        iterator: Iterator[pa.RecordBatch],
        prefetch_window: int,
        batch_size: Optional[int],
    ):
        self.iterator = iterator
        self.prefetch_window = prefetch_window
        self.batch_size = batch_size
        self.sliding_window = deque()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.prefetch_future = None
        self._prefetch_initial()

    def _prefetch_initial(self):
        """Prefill initial window with optimized exception handling"""
        count = 0
        try:
            while count < self.prefetch_window:
                batch = next(self.iterator)
                self.sliding_window.append(batch)
                count += 1
        except StopIteration:
            pass

        if self.sliding_window:
            self._trigger_async_prefetch()

    def _trigger_async_prefetch(self):
        """Trigger async prefetch task"""
        if not self.prefetch_future or self.prefetch_future.done():
            self.prefetch_future = self.executor.submit(self._prefetch_task)

    def _prefetch_task(self):
        """Background prefetch task with batch fetching optimization"""
        try:
            # Prefetch multiple batches at once to reduce loop iterations
            remaining = self.prefetch_window - len(self.sliding_window)
            for _ in range(remaining):
                batch = next(self.iterator)
                self.sliding_window.append(batch)
        except StopIteration:
            pass

    def __iter__(self):
        return self

    def __next__(self) -> pa.RecordBatch:
        if not self.sliding_window:
            raise StopIteration

        # Batch-trigger prefetch to reduce async calls
        if len(self.sliding_window) <= self.prefetch_window // 2:
            self._trigger_async_prefetch()

        return self.sliding_window.popleft()

    def __del__(self):
        self.executor.shutdown(wait=False)
