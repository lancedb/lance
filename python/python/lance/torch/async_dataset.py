import contextlib
from multiprocessing import Process, Queue, Value
from typing import Callable, Iterable

from torch.utils.data import IterableDataset


def _worker_ep(
    dataset_creator: Callable[[], IterableDataset],
    queue: Queue,
    shutdown: Value,
):
    dataset = dataset_creator()
    while not shutdown.value:
        for item in dataset:
            queue.put(item)
        queue.put(None)

    # put one last None so that the iterator knows to stop
    queue.put(None)


# This class is similar to torch.utils.data.Dataloader
# except for a few things:
# 1. We only use 1 worker
# 2. The worker process is the same one after each complete iteration.
#    This helps when dataset contains internal state like caching.
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

        self.shutdown = Value("b", False)

    def _start(self):
        if self.started:
            return
        self.started = True
        self.worker = Process(
            target=_worker_ep,
            args=(self.dataset_creator, self.queue, self.shutdown),
        )
        self.worker.start()

    def __iter__(self):
        self._start()
        while (val := self.queue.get()) is not None:
            yield val

    def close(self):
        self.shutdown.value = True
        for _ in self:
            pass
        self.queue.close()
        self.worker.join()
        self.worker.close()


@contextlib.contextmanager
def async_dataset(
    dataset_creator: Callable[[], IterableDataset],
    *,
    queue_size: int = 4,
) -> Iterable[AsyncDataset]:
    try:
        ds = AsyncDataset(dataset_creator, queue_size=queue_size)
        yield ds
    finally:
        ds.close()
