# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import os
from pathlib import Path


def wipe_os_cache(dataset_uri: str):
    if dataset_uri.startswith("/"):
        path = dataset_uri
    elif dataset_uri.startswith("file://"):
        path = Path(dataset_uri.removeprefix("file://"))
    else:
        return

    if not hasattr(os, "posix_fadvise"):
        raise NotImplementedError("posix_fadvise not available on this platform")

    POSIX_FADV_DONTNEED = 4  # Tell kernel we don't need this data in cache

    directory = Path(path)

    file_iterator = directory.rglob("*")

    for filepath in file_iterator:
        # Skip directories, symlinks, and non-regular files
        if not filepath.is_file():
            continue

        with open(filepath, "rb") as f:
            fd = f.fileno()
            # offset=0, length=0 means drop entire file from cache
            os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
