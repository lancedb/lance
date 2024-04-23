# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

# ruff: noqa: F821

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Optional

from .lance import _cleanup_partial_writes

if TYPE_CHECKING:
    # We don't import directly because of circular import
    from .fragment import FragmentMetadata


class FragmentWriteProgress(ABC):
    """Progress tracking for Writing a Dataset or Fragment.

    Warns
    -----
    This tracking class is experimental and may change in the future.
    """

    def _do_begin(
        self, fragment_json: str, multipart_id: Optional[str] = None, **kwargs
    ):
        """Called when a new fragment is created"""
        from .fragment import FragmentMetadata

        fragment = FragmentMetadata.from_json(fragment_json)
        return self.begin(fragment, multipart_id, **kwargs)

    @abstractmethod
    def begin(
        self, fragment: "FragmentMetadata", multipart_id: Optional[str] = None, **kwargs
    ) -> None:
        """Called when a new fragment is about to be written.

        Parameters
        ----------
        fragment : FragmentMetadata
            The fragment that is open to write to. The fragment id might not
            yet be assigned at this point.
        multipart_id : str, optional
            The multipart id that will be uploaded to cloud storage. This may be
            used later to abort incomplete uploads if this fragment write fails.
        kwargs: dict, optional
            Extra keyword arguments to pass to the implementation.

        Returns
        -------
        None
        """
        pass

    def _do_complete(self, fragment_json: str, **kwargs):
        """Called when a fragment is completed"""
        from .fragment import FragmentMetadata

        fragment = FragmentMetadata.from_json(fragment_json)
        return self.complete(fragment, **kwargs)

    @abstractmethod
    def complete(self, fragment: "FragmentMetadata", **kwargs) -> None:
        """Callback when a fragment is completely written.

        Parameters
        ----------
        fragment : FragmentMetadata
            The fragment that is open to write to.
        kwargs: dict, optional
            Extra keyword arguments to pass to the implementation.
        """
        pass


class NoopFragmentWriteProgress(FragmentWriteProgress):
    """No-op implementation of WriteProgressTracker.

    This is the default implementation.
    """

    def begin(
        self, fragment: "FragmentMetadata", multipart_id: Optional[str] = None, **kargs
    ):
        pass

    def complete(self, fragment: "FragmentMetadata", **kwargs):
        pass


class FileSystemFragmentWriteProgress(FragmentWriteProgress):
    """Progress tracking for Writing a Dataset or Fragment.

    Warns
    -----
    This tracking class is experimental and will change in the future.

    This implementation writes a JSON file to track in-progress state
    to the filesystem for each fragment.


    """

    PROGRESS_EXT: str = ".in_progress"

    def __init__(self, base_uri: str, metadata: Optional[Dict[str, str]] = None):
        """Create a FileSystemFragmentWriteProgress tracker.

        Parameters
        ----------
        base_uri : str
            The base directory to write the progress files to. Two files will be created
            under this directory: a Fragment file, and a JSON file to track progress.
        metadata : dict, optional
            Extra metadata for this Progress tracker instance. Can be used to track
            distributed worker where this tracker is running.
        """
        from pyarrow.fs import FileSystem

        fs, path = FileSystem.from_uri(base_uri)
        self._fs = fs
        self._base_path: str = path
        self._metadata = metadata if metadata else {}

    def _in_progress_path(self, fragment: "FragmentMetadata") -> str:
        return os.path.join(
            self._base_path, f"fragment_{fragment.id}{self.PROGRESS_EXT}"
        )

    def _fragment_file(self, fragment: "FragmentMetadata") -> str:
        return os.path.join(self._base_path, f"fragment_{fragment.id}.json")

    def begin(
        self, fragment: "FragmentMetadata", multipart_id: Optional[str] = None, **kwargs
    ):
        """Called when a new fragment is created.

        Parameters
        ----------
        fragment : FragmentMetadata
            The fragment that is open to write to.
        multipart_id : str, optional
            The multipart id to upload this fragment to cloud storage.
        """

        self._fs.create_dir(self._base_path, recursive=True)

        with self._fs.open_output_stream(self._in_progress_path(fragment)) as out:
            progress_data = {
                "fragment_id": fragment.id,
                "multipart_id": multipart_id if multipart_id else "",
                "metadata": self._metadata,
            }
            out.write(json.dumps(progress_data).encode("utf-8"))

        with self._fs.open_output_stream(self._fragment_file(fragment)) as out:
            out.write(json.dumps(fragment.to_json()).encode("utf-8"))

    def complete(self, fragment: "FragmentMetadata", **kwargs):
        """Called when a fragment is completed"""
        self._fs.delete_file(self._in_progress_path(fragment))

    def cleanup_partial_writes(self, dataset_uri: str) -> int:
        """
        Finds all in-progress files and cleans up any partially written data
        files. This is useful for cleaning up after a failed write.

        Parameters
        ----------
        dataset_uri : str
            The URI of the table to clean up.

        Returns
        -------
        int
            The number of partial writes cleaned up.
        """
        from pyarrow.fs import FileSelector

        from .fragment import FragmentMetadata

        marker_paths = []
        objects = []
        selector = FileSelector(self._base_path)
        for info in self._fs.get_file_info(selector):
            path = info.path
            if path.endswith(self.PROGRESS_EXT):
                marker_paths.append(path)
                with self._fs.open_input_stream(path) as f:
                    progress_data = json.loads(f.read().decode("utf-8"))

                json_path = path.rstrip(self.PROGRESS_EXT) + ".json"
                with self._fs.open_input_stream(json_path) as f:
                    fragment_metadata = FragmentMetadata.from_json(
                        f.read().decode("utf-8")
                    )
                objects.append(
                    (
                        fragment_metadata.data_files()[0].path(),
                        progress_data["multipart_id"],
                    )
                )

        _cleanup_partial_writes(dataset_uri, objects)

        for path in marker_paths:
            self._fs.delete_file(path)
            json_path = path.rstrip(self.PROGRESS_EXT) + ".json"
            self._fs.delete_file(json_path)

        return len(objects)
