# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

""" Pytorch Distributed Utilities """

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
    return get_dist_rank() % get_dist_local_world_size()


def get_dist_local_world_size() -> int:
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
