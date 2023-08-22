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

# ruff: noqa: F821

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Dict, Optional


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
        self._base_path = path
        self._metadata = metadata if metadata else {}

    def _in_progress_path(self, fragment: "FragmentMetadata"):
        return self._base_path / f"fragment_{fragment.id}.in_progress"

    def _fragment_file(self, fragment: "FragmentMetadata"):
        return self._base_path / f"fragment_{fragment.id}.json"

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

        self._fs.create_dir(self._base_path)
        with self._fs.open_output_stream(self._in_progress_path(fragment)) as out:
            progress_data = {
                "fragment_id": fragment.id,
                "multipart_id": multipart_id if multipart_id else "",
                "metadata": self._metadata,
            }
            json.dump(progress_data, out)
        with self._fs.open_input_stream(self._fragment_file(fragment)) as out:
            out.write(fragment.to_json()).encode("utf-8")

    def complete(self, fragment: "FragmentMetadata", **kwargs):
        """Called when a fragment is completed"""
        self._fs.delete(self._in_progress_path(fragment))
