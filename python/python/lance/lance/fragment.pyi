# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from typing import Literal, Optional

class DeletionFile:
    """
    Metadata for a deletion file.

    The deletion file contains the row ids that are tombstoned.

    Attributes
    ----------
    read_version : int
        The read version of the deletion file.
    id : int
        A unique identifier for the deletion file, used to prevent collisions.
    num_deleted_rows : int
        The number of rows that are deleted.
    file_type : str
        The type of deletion file. "array" is used for small sets, while
        "bitmap" is used for large sets.
    """

    read_version: int
    id: int
    num_deleted_rows: int
    file_type: Literal["array", "bitmap"]

    def __init__(
        self,
        read_version: int,
        id: int,
        file_type: Literal["array", "bitmap"],
        num_deleted_rows: int,
    ): ...
    def asdict(self) -> dict:
        """Get a dictionary representation of the deletion file."""
        ...
    def path(self, fragment_id: int, base_uri: Optional[str] = None) -> str:
        """
        Get the path to the deletion file.

        Parameters
        ----------
        fragment_id : int
            The fragment id.
        base_uri : str, optional
            The base URI to use for the path. If not provided, a relative path
            is returned.

        Returns
        -------
        str
            The path to the deletion file.
        """
        ...

    def json(self) -> str:
        """Get a JSON representation of the deletion file.

        Returns
        -------
        str

        Warning
        -------
        The JSON representation is not guaranteed to be stable across versions.
        """
        ...

    @classmethod
    def from_json(json: str) -> DeletionFile:
        """
        Load a deletion file from a JSON representation.

        Parameters
        ----------
        json : str
            The JSON representation of the deletion file.

        Returns
        -------
        DeletionFile
        """
        ...

    def __reduce__(self) -> tuple: ...

class RowIdMeta:
    def json(self) -> str:
        """Get a JSON representation of the row id metadata.

        Returns
        -------
        str

        Warning
        -------
        The JSON representation is not guaranteed to be stable across versions.
        """
        ...

    @classmethod
    def from_json(json: str) -> RowIdMeta:
        """
        Load row id metadata from a JSON representation.

        Parameters
        ----------
        json : str
            The JSON representation of the row id metadata.

        Returns
        -------
        RowIdMeta
        """
        ...

    def __reduce__(self) -> tuple: ...
