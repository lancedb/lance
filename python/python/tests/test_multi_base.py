# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Tests for multi-base dataset functionality.
"""

import shutil
import tempfile
import uuid
from pathlib import Path

import lance
import pandas as pd
import pytest
from lance import DatasetBasePath


class TestMultiBase:
    """Test multi-base dataset functionality with local file system."""

    def setup_method(self):
        """Set up test directories for each test."""
        self.test_dir = tempfile.mkdtemp()
        self.test_id = str(uuid.uuid4())[:8]

        # Create primary and additional path directories
        self.primary_uri = str(Path(self.test_dir) / "primary")
        self.path1_uri = str(Path(self.test_dir) / f"path1_{self.test_id}")
        self.path2_uri = str(Path(self.test_dir) / f"path2_{self.test_id}")
        self.path3_uri = str(Path(self.test_dir) / f"path3_{self.test_id}")

        # Create directories
        for uri in [self.primary_uri, self.path1_uri, self.path2_uri, self.path3_uri]:
            Path(uri).mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        """Clean up test directories after each test."""
        if hasattr(self, "test_dir"):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_data(self, num_rows=500, id_offset=0):
        """Create test data for multi-base tests."""
        return pd.DataFrame(
            {
                "id": range(id_offset, id_offset + num_rows),
                "value": [f"value_{i}" for i in range(id_offset, id_offset + num_rows)],
                "score": [i * 0.1 for i in range(id_offset, id_offset + num_rows)],
            }
        )

    def test_multi_base_create_and_read(self):
        """Test creating a multi-base dataset and reading it back."""
        data = self.create_test_data(500)

        # Create dataset with multi-base layout
        dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
                DatasetBasePath(self.path3_uri, name="path3"),
            ],
            target_bases=["path2"],  # Write data to path2
            max_rows_per_file=100,  # Force multiple fragments
        )

        assert dataset is not None
        assert dataset.uri == self.primary_uri

        # Verify we can read the data back
        result = dataset.to_table().to_pandas()
        assert len(result) == 500

        # Verify data integrity
        pd.testing.assert_frame_equal(
            result.sort_values("id").reset_index(drop=True),
            data.sort_values("id").reset_index(drop=True),
        )

    def test_multi_base_append_mode(self):
        """Test appending data to a multi-base dataset."""
        # Create initial dataset
        initial_data = self.create_test_data(300)

        dataset = lance.write_dataset(
            initial_data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
            ],
            target_bases=["path1"],  # Write to path1
            max_rows_per_file=100,
        )

        # Create additional data to append
        append_data = self.create_test_data(100, id_offset=300)

        # Append to different path
        updated_dataset = lance.write_dataset(
            append_data,
            dataset,
            mode="append",
            target_bases=["path2"],  # Write to path2
            max_rows_per_file=50,
        )

        # Verify total data
        result = updated_dataset.to_table().to_pandas()
        assert len(result) == 400

        # Verify all data is present
        expected_ids = set(range(400))
        actual_ids = set(result["id"].tolist())
        assert actual_ids == expected_ids

    def test_multi_base_overwrite_mode_inherits_bases(self):
        """Test OVERWRITE mode inherits existing base configuration."""
        # Create initial dataset with multi-base configuration
        initial_data = self.create_test_data(200)

        lance.write_dataset(
            initial_data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
            ],
            target_bases=["path1"],  # Write to path1
            max_rows_per_file=100,
        )

        # Create new data for overwrite
        overwrite_data = self.create_test_data(150, id_offset=100)

        # Overwrite - should inherit existing bases (path1, path2)
        # Write to path2 (existing base, referenced by name)
        updated_dataset = lance.write_dataset(
            overwrite_data,
            self.primary_uri,
            mode="overwrite",
            target_bases=["path2"],  # Reference existing base by name
            max_rows_per_file=75,
        )

        # Verify overwritten data
        result = updated_dataset.to_table().to_pandas()
        assert len(result) == 150

        # Verify data content
        expected_ids = set(range(100, 250))
        actual_ids = set(result["id"].tolist())
        assert actual_ids == expected_ids

        # Verify base_paths are preserved from initial dataset
        base_paths = updated_dataset._ds.base_paths()
        assert len(base_paths) == 2
        assert any(bp.name == "path1" for bp in base_paths.values())
        assert any(bp.name == "path2" for bp in base_paths.values())

    def test_multi_base_overwrite_mode_primary_path_default(self):
        """Test OVERWRITE mode defaults to primary path when no target."""
        # Create initial dataset with explicit data file bases
        initial_data = self.create_test_data(100)

        lance.write_dataset(
            initial_data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
            ],
            target_bases=["path1"],  # Write to path1
            max_rows_per_file=50,
        )

        # Overwrite without specifying target - should use primary path
        # This clears the old base_paths and writes to primary path only
        overwrite_data = self.create_test_data(75, id_offset=200)

        updated_dataset = lance.write_dataset(
            overwrite_data,
            self.primary_uri,
            mode="overwrite",
            # No target_bases specified - data goes to primary path
            # Old bases (path1, path2) are NOT preserved
            max_rows_per_file=25,
        )

        # Verify overwritten data
        result = updated_dataset.to_table().to_pandas()
        assert len(result) == 75

        # Verify data content
        expected_ids = set(range(200, 275))
        actual_ids = set(result["id"].tolist())
        assert actual_ids == expected_ids

        # Verify base_paths are preserved from previous manifest
        base_paths = updated_dataset._ds.base_paths()
        # Old path1 and path2 ARE preserved in manifest
        assert len(base_paths) == 2
        assert any(bp.name == "path1" for bp in base_paths.values())
        assert any(bp.name == "path2" for bp in base_paths.values())

        # Verify that data files were written to primary path (not to path1 or path2)
        fragments = list(updated_dataset.get_fragments())
        for fragment in fragments:
            # Fragment files should be in primary path (base_id should be None)
            data_file = fragment.data_files()[0]
            assert data_file.base_id is None, (
                f"Expected data file to be in primary path (base_id=None), "
                f"but got base_id={data_file.base_id}"
            )

        # Since base_paths exist, we can append to them by name
        append_data = self.create_test_data(25, id_offset=300)

        final_dataset = lance.write_dataset(
            append_data,
            updated_dataset,
            mode="append",
            target_bases=["path2"],  # Specify which base to use
        )

        final_result = final_dataset.to_table().to_pandas()
        assert len(final_result) == 100

    def test_multi_base_append_mode_primary_path_default(self):
        """Test that APPEND mode defaults to primary path when no target specified."""
        # Create initial dataset with explicit data file bases
        initial_data = self.create_test_data(100)

        dataset = lance.write_dataset(
            initial_data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
            ],
            target_bases=["path1"],  # Write to path1
            max_rows_per_file=50,
        )

        # Append without specifying target - should use primary path
        append_data = self.create_test_data(75, id_offset=100)

        updated_dataset = lance.write_dataset(
            append_data,
            dataset,
            mode="append",
            # No target_bases specified - data goes to primary path
            max_rows_per_file=25,
        )

        # Verify appended data
        result = updated_dataset.to_table().to_pandas()
        assert len(result) == 175

        # Verify data content
        expected_ids = set(range(175))
        actual_ids = set(result["id"].tolist())
        assert actual_ids == expected_ids

        # Verify base_paths are preserved from previous manifest
        base_paths = updated_dataset._ds.base_paths()
        assert len(base_paths) == 2
        assert any(bp.name == "path1" for bp in base_paths.values())
        assert any(bp.name == "path2" for bp in base_paths.values())

        # Verify new data files were written to primary path
        # (not to path1 or path2)
        fragments = list(updated_dataset.get_fragments())
        # The first 2 fragments should be in path1 (from initial write)
        # The remaining fragments should be in primary path (from append)
        primary_path_fragments = 0
        path1_fragments = 0

        for fragment in fragments:
            data_file = fragment.data_files()[0]
            if data_file.base_id is None:
                primary_path_fragments += 1
            else:
                base_path = base_paths.get(data_file.base_id)
                if base_path and base_path.name == "path1":
                    path1_fragments += 1

        assert path1_fragments == 2, (
            f"Expected 2 fragments in path1, got {path1_fragments}"
        )
        assert primary_path_fragments >= 3, (
            f"Expected at least 3 fragments in primary path, "
            f"got {primary_path_fragments}"
        )

    def test_multi_base_is_dataset_root_flag(self):
        """Test is_dataset_root flag on DatasetBasePath."""
        # Create initial dataset with one base marked as dataset_root
        initial_data = self.create_test_data(100)

        dataset = lance.write_dataset(
            initial_data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.path1_uri, name="path1", is_dataset_root=True),
                DatasetBasePath(self.path2_uri, name="path2", is_dataset_root=False),
            ],
            target_bases=["path1"],
            max_rows_per_file=50,
        )

        # Verify base_paths configuration
        base_paths = dataset._ds.base_paths()
        assert len(base_paths) == 2

        # Find path1 and path2 in base_paths
        path1_base = None
        path2_base = None
        for base_path in base_paths.values():
            if base_path.name == "path1":
                path1_base = base_path
            elif base_path.name == "path2":
                path2_base = base_path

        assert path1_base is not None, "path1 base not found"
        assert path2_base is not None, "path2 base not found"

        # Verify is_dataset_root flags
        assert path1_base.is_dataset_root is True, (
            f"Expected path1.is_dataset_root=True, got {path1_base.is_dataset_root}"
        )
        assert path2_base.is_dataset_root is False, (
            f"Expected path2.is_dataset_root=False, got {path2_base.is_dataset_root}"
        )

        # Verify data is readable
        result = dataset.to_table().to_pandas()
        assert len(result) == 100

    def test_multi_base_target_by_path_uri(self):
        """Test using path URIs instead of names in target_bases."""
        # Create initial dataset with named bases
        initial_data = self.create_test_data(100)

        dataset = lance.write_dataset(
            initial_data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
            ],
            target_bases=["path1"],  # Write to path1 using name
            max_rows_per_file=50,
        )

        # Get the base_paths to find the actual path URI for path2
        base_paths = dataset._ds.base_paths()
        path2_base = None
        for base_path in base_paths.values():
            if base_path.name == "path2":
                path2_base = base_path
                break

        assert path2_base is not None, "path2 base not found"

        # Append using the path URI instead of name
        append_data = self.create_test_data(50, id_offset=100)

        updated_dataset = lance.write_dataset(
            append_data,
            dataset,
            mode="append",
            target_bases=[path2_base.path],  # Use path URI instead of name
            max_rows_per_file=25,
        )

        # Verify appended data
        result = updated_dataset.to_table().to_pandas()
        assert len(result) == 150

        # Verify data content
        expected_ids = set(range(150))
        actual_ids = set(result["id"].tolist())
        assert actual_ids == expected_ids

        # Verify that new fragments are in path2 (not primary or path1)
        fragments = list(updated_dataset.get_fragments())
        base_paths_updated = updated_dataset._ds.base_paths()

        path1_fragments = 0
        path2_fragments = 0

        for fragment in fragments:
            data_file = fragment.data_files()[0]
            if data_file.base_id is not None:
                base_path = base_paths_updated.get(data_file.base_id)
                if base_path and base_path.name == "path1":
                    path1_fragments += 1
                elif base_path and base_path.name == "path2":
                    path2_fragments += 1

        assert path1_fragments == 2, (
            f"Expected 2 fragments in path1, got {path1_fragments}"
        )
        assert path2_fragments == 2, (
            f"Expected 2 fragments in path2, got {path2_fragments}"
        )

    def test_validation_errors(self):
        """Test validation errors for invalid multi-base configurations."""
        data = self.create_test_data(100)

        # Test 1: Target reference not found in new bases for CREATE mode
        with pytest.raises(Exception, match="not found"):
            lance.write_dataset(
                data,
                self.primary_uri,
                mode="create",
                initial_bases=[
                    DatasetBasePath(self.path1_uri, name="path1"),
                    DatasetBasePath(self.path2_uri, name="path2"),
                ],
                target_bases=["path3"],  # Not in new bases
            )

        # Test 2: DatasetBasePath in APPEND mode
        # First create a base dataset
        base_dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
            ],
            target_bases=["path1"],
        )

        with pytest.raises(Exception, match="Cannot register new bases in Append mode"):
            lance.write_dataset(
                data,
                base_dataset,
                mode="append",
                initial_bases=[
                    DatasetBasePath(
                        name="path3", path=self.path3_uri
                    ),  # New base not allowed
                ],
                target_bases=["path3"],
            )

    def test_fragment_distribution(self):
        """Test that fragments are correctly distributed and readable."""
        data = self.create_test_data(1000)

        dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            initial_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
            ],
            target_bases=["path1"],
            max_rows_per_file=100,  # Should create 10 fragments
        )

        # Verify fragment count
        fragments = list(dataset.get_fragments())
        assert len(fragments) >= 10  # At least 10 fragments

        # Verify all data is readable
        result = dataset.to_table().to_pandas()
        assert len(result) == 1000

        # Test scanning with filters
        filtered_result = dataset.scanner(filter="id < 500").to_table().to_pandas()
        assert len(filtered_result) == 500
        assert all(filtered_result["id"] < 500)
