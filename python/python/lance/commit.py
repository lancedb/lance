# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from contextlib import AbstractContextManager
from typing import Callable

CommitLock = Callable[[int], AbstractContextManager[None]]


class CommitConflictError(OSError):
    """A commit conflicted with a concurrent transaction.

    Subclasses :class:`OSError` so existing ``except OSError`` handlers keep
    working. ``retryable`` is ``True`` when the transaction was preempted and
    can be retried against the newer version, ``False`` when the conflict is
    incompatible and retrying will not help.
    """

    def __init__(self, message: str = "", retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class DuplicateTransactionError(OSError):
    """A transaction committed after the read version already applied some of
    this commit's idempotency keys (see ``idempotency_keys`` on
    :meth:`lance.LanceDataset.commit`).

    Not retryable: committing would apply the same work twice. ``version`` is
    the version that carries the keys and ``keys`` the keys it shares with this
    commit. The version may be this writer's own earlier commit, when a commit
    that already landed is retried (for example after a commit whose status was
    reported unknown); callers tell the two cases apart by their own commit
    identity.
    """

    def __init__(self, message: str, version: int, keys: list[str]):
        super().__init__(message)
        self.version = version
        self.keys = keys
