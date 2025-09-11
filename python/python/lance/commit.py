# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from contextlib import AbstractContextManager
from typing import Callable

CommitLock = Callable[[int], AbstractContextManager]


class CommitConflictError(Exception):
    pass
