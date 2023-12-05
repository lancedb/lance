from multiprocessing import Process, Queue
from typing import Callable

from torch.utils.data import IterableDataset


def _worker_ep(
    dataset_creator: Callable[[], IterableDataset],
    queue: Queue,
):
    dataset = dataset_creator()
    while True:
        for item in dataset:
            queue.put(item)
        queue.put(None)


class AsyncDataset(IterableDataset):
    def __init__(
        self,
        dataset_creator: Callable[[], IterableDataset],
        *,
        queue_size: int = 4,
    ):
        self.dataset_creator = dataset_creator
        self.queue = Queue(maxsize=queue_size)
        self.started = False

    def _start(self):
        if self.started:
            return
        self.started = True
        self.worker = Process(
            target=_worker_ep,
            args=(self.dataset_creator, self.queue),
        ).start()

    def __iter__(self):
        self._start()
        while (val := self.queue.get()) is not None:
            yield val
