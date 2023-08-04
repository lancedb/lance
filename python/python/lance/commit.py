#  Copyright (c) 2023. Lance Developers
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

from abc import ABC, abstractmethod


class CommitLock(ABC):
    """
    A lock that can be used to prevent concurrent commits to the same version.

    This is only necessary for storage engines that do not support atomic commits,
    which right now is just AWS S3.
    """

    @abstractmethod
    def lock(self, version: int) -> "CommitLease":
        """
        Attempt to lock the table for the given version.

        If it is already locked by another transaction, wait until it is unlocked.
        Once it is unlocked, raise :class:`CommitConflictError` if the version
        has already been committed. Otherwise, return the lock.

        To prevent poisoned locks, it's recommended to set a timeout on the lock
        of at least 30 seconds.

        It is not required that the lock tracks the version. It is provided in
        case the locking is handled by a catalog service that needs to know the
        current version of the table.
        """
        pass


class CommitLease(ABC):
    @abstractmethod
    def release(self, success: bool) -> None:
        """
        Release the lease.

        Params
        ------
        success: bool
            Whether the commit was successful.

        It is not required that the lock uses success. This is to tell catalog
        services whether the new version should be considered committed or not.
        """
        pass


class CommitConflictError(Exception):
    pass
