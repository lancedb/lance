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


class FragmentWriteProgress(ABC):
    """Progress tracking for Writing a Dataset or Fragment"""

    def _do_begin(self, fragment_json: str):
        """Called when a new fragment is created"""
        from .fragment import FragmentMetadata

        fragment = FragmentMetadata.from_json(fragment_json)
        return self.begin(fragment)

    @abstractmethod
    def begin(self, fragment: "FragmentMetadata"):
        """Called when a new fragment is created"""
        pass

    def _do_complete(self, fragment_json: str):
        """Called when a fragment is completed"""
        from .fragment import FragmentMetadata

        fragment = FragmentMetadata.from_json(fragment_json)
        return self.complete(fragment)

    @abstractmethod
    def complete(self, fragment: "FragmentMetadata"):
        """Called when a fragment is completed"""
        pass


class NoopFragmentWriteProgress(FragmentWriteProgress):
    """No-op implementation of WriteProgressTracker"""

    def begin(self, fragment: "FragmentMetadata"):
        pass

    def complete(self, fragment: "FragmentMetadata"):
        pass


class FileSystemFragmentWriteProgress(FragmentWriteProgress):
    """Progress tracking for Writing a Dataset or Fragment.

    This implementation writes a JSON file to the filesystem for each fragment.
    """

    def __init__(self, base_uri: str):
        from pyarrow.fs import FileSystem

        fs, path = FileSystem.from_uri(base_uri)
        self._fs = fs
        self._base_path = path

    def _in_progress_path(self, fragment: "FragmentMetadata"):
        return self._base_path / f"fragment_{fragment.id}.in_progress"

    def _fragment_file(self, fragment: "FragmentMetadata"):
        return self._base_path / f"fragment_{fragment.id}.json"

    def begin(self, fragment: "FragmentMetadata"):
        """Called when a new fragment is created"""

        self._fs.create_dir(self._base_path)
        with self._fs.open_output_stream(self._in_progress_path(fragment)) as out:
            out.write(str(fragment.id).encode("utf-8"))
        with self._fs.open_input_stream(self._fragment_file(fragment)) as out:
            out.write(fragment.to_json()).encode("utf-8")

    def complete(self, fragment: "FragmentMetadata"):
        """Called when a fragment is completed"""
        self._fs.delete(self._in_progress_path(fragment))
