import logging
import os

import torch.distributed as dist
import torch.utils.data


def get_dist_world_size() -> int:
    """
    Get the number of processes in the distributed training setup.

    Returns:
        int: The number of distributed processes if distributed training is initialized,
             otherwise 1.
    """
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_dist_rank() -> int:
    """
    Get the rank of the current process in the distributed training setup.

    Returns:
        int: The rank of the current process if distributed training is initialized,
             otherwise 0.
    """
    if dist.is_initialized():
        return int(dist.get_rank())
    return 0


def get_dist_local_rank() -> int:
    """
    Get the local rank of the current process in the distributed training setup.

    Returns:
        int: The local rank of the current process within its node.
    """
    return get_dist_rank() % get_dist_local_world_size()


def get_dist_local_world_size() -> int:
    """
    Get the number of processes in the local node for distributed training.

    Returns:
        int: The number of processes in the local node, or 1 if not specified.
    """
    return int(os.environ.get("LOCAL_WORLD_SIZE", 1))


def get_mp_world_size() -> int:
    """
    Get the number of worker processes for the current DataLoader.

    Returns:
        int: The number of worker processes if running in a DataLoader worker,
             otherwise 1.
    """
    if (worker_info := torch.utils.data.get_worker_info()) is not None:
        return worker_info.num_workers
    return 1


def get_mp_rank() -> int:
    """
    Get the rank of the current DataLoader worker process.

    Returns:
        int: The rank of the current DataLoader worker if running in a worker process,
             otherwise 0.
    """
    if (worker_info := torch.utils.data.get_worker_info()) is not None:
        return worker_info.id
    return 0


def is_rank_zero() -> bool:
    """
    Check if the current process is the global rank zero process.

    Returns:
        bool: True if the current process is the global rank zero, False otherwise.
    """
    return get_dist_rank() == 0


def rank_zero_log(message: str, logger: logging.Logger, log_level: int = logging.INFO) -> None:
    """
    Log a message only from the rank zero process.

    Args:
        message (str): The message to log.
        logger (logging.Logger): The logger object to use.
        log_level (int): The logging level to use. Defaults to logging.INFO.
    """
    if is_rank_zero():
        logger.log(log_level, message)


def is_local_leader() -> bool:
    """
    Check if the current process is the local leader (local rank zero).

    Returns:
        bool: True if the current process is the local leader, False otherwise.
    """
    return get_dist_local_rank() == 0


def is_mp_local_leader() -> bool:
    """
    Check if the current process is the multiprocessing local leader.

    Returns:
        bool: True if the current process is the multiprocessing local leader, False otherwise.
    """
    return get_mp_rank() == 0


def safe_barrier() -> None:
    """
    Perform a barrier operation if distributed training is initialized.

    This function is safe to call even if distributed training is not initialized.
    """
    if dist.is_initialized():
        dist.barrier()
