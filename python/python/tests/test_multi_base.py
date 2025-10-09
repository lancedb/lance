# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Tests for multi-base dataset functionality.
"""

import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import lance
import pandas as pd
import pyarrow as pa
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
            target_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
                DatasetBasePath(self.path3_uri, name="path3"),
                "path2",  # Write data to path2
            ],
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
            target_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
                "path1",  # Write to path1
            ],
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
            target_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
                "path1",  # Write to path1
            ],
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
        """Test that OVERWRITE mode defaults to primary path when no target specified."""
        # Create initial dataset with explicit data file bases
        initial_data = self.create_test_data(100)

        lance.write_dataset(
            initial_data,
            self.primary_uri,
            mode="create",
            target_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
                "path1",  # Write to path1
            ],
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

    def test_concurrent_appends(self):
        """Test concurrent appends to multi-base dataset."""
        # Create initial dataset
        initial_data = self.create_test_data(100)

        lance.write_dataset(
            initial_data,
            self.primary_uri,
            mode="create",
            target_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
                DatasetBasePath(self.path3_uri, name="path3"),
                "path1",  # Write to path1
            ],
            max_rows_per_file=50,
        )

        def append_worker(worker_id, target_path_name, start_id, num_rows):
            """Worker function to append data concurrently."""
            worker_data = self.create_test_data(num_rows, id_offset=start_id)

            # Load dataset fresh for this worker
            worker_dataset = lance.dataset(self.primary_uri)

            # Perform append
            return lance.write_dataset(
                worker_data,
                worker_dataset,
                mode="append",
                target_bases=[target_path_name],  # Reference by name
                max_rows_per_file=25,
            )

        # Configure concurrent append tasks
        append_tasks = [
            {"worker_id": 1, "target_path": "path1", "start_id": 100, "num_rows": 50},
            {"worker_id": 2, "target_path": "path2", "start_id": 150, "num_rows": 50},
            {"worker_id": 3, "target_path": "path3", "start_id": 200, "num_rows": 50},
        ]

        # Execute concurrent appends
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_task = {
                executor.submit(
                    append_worker,
                    task["worker_id"],
                    task["target_path"],
                    task["start_id"],
                    task["num_rows"],
                ): task
                for task in append_tasks
            }

            for future in as_completed(future_to_task):
                results.append(future.result())

        # Verify final dataset integrity
        final_dataset = lance.dataset(self.primary_uri)
        final_data = final_dataset.to_table().to_pandas()

        # Should have initial 100 + 3 * 50 = 250 rows
        assert len(final_data) == 250

        # Verify all expected IDs are present
        expected_ids = set(range(250))
        actual_ids = set(final_data["id"].tolist())
        assert actual_ids == expected_ids

    def test_validation_errors(self):
        """Test validation errors for invalid multi-base configurations."""
        data = self.create_test_data(100)

        # Test 1: Target reference not found in new bases for CREATE mode
        with pytest.raises(Exception, match="not found"):
            lance.write_dataset(
                data,
                self.primary_uri,
                mode="create",
                target_bases=[
                    DatasetBasePath(self.path1_uri, name="path1"),
                    DatasetBasePath(self.path2_uri, name="path2"),
                    "path3",  # Not in new bases
                ],
            )

        # Test 2: DatasetBasePath in APPEND mode
        # First create a base dataset
        base_dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            target_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
                "path1",
            ],
        )

        with pytest.raises(Exception, match="Cannot register new bases in Append mode"):
            lance.write_dataset(
                data,
                base_dataset,
                mode="append",
                target_bases=[
                    DatasetBasePath(
                        name="path3", path=self.path3_uri
                    ),  # New base not allowed
                    "path3",
                ],
            )

    def test_fragment_distribution(self):
        """Test that fragments are correctly distributed and readable."""
        data = self.create_test_data(1000)

        dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            target_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
                "path1",
            ],
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

    def test_schema_consistency(self):
        """Test that schema is consistent across multi-base operations."""
        # Create dataset with specific schema
        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("name", pa.string()),
                pa.field("score", pa.float64()),
            ]
        )

        data = pa.table(
            {
                "id": [1, 2, 3, 4, 5],
                "name": ["a", "b", "c", "d", "e"],
                "score": [1.0, 2.0, 3.0, 4.0, 5.0],
            },
            schema=schema,
        )

        dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            target_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
                "path1",
            ],
        )

        # Verify schema
        assert dataset.schema == schema

        # Append with same schema to different path
        append_data = pa.table(
            {"id": [6, 7, 8], "name": ["f", "g", "h"], "score": [6.0, 7.0, 8.0]},
            schema=schema,
        )

        updated_dataset = lance.write_dataset(
            append_data, dataset, mode="append", target_bases=["path2"]
        )

        # Schema should remain consistent
        assert updated_dataset.schema == schema

        # Verify all data
        result = updated_dataset.to_table()
        assert len(result) == 8
        assert result.schema == schema

    def test_dataset_reopening(self):
        """Test reopening a multi-base dataset."""
        data = self.create_test_data(300)

        # Create multi-base dataset
        dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            target_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
                "path1",
            ],
            max_rows_per_file=100,
        )

        # Close and reopen dataset
        del dataset
        reopened_dataset = lance.dataset(self.primary_uri)

        # Verify data is still accessible
        result = reopened_dataset.to_table().to_pandas()
        assert len(result) == 300

        # Verify we can still append to different paths
        append_data = self.create_test_data(50, id_offset=300)

        updated_dataset = lance.write_dataset(
            append_data,
            reopened_dataset,
            mode="append",
            target_bases=["path2"],  # Reference by name
        )

        final_result = updated_dataset.to_table().to_pandas()
        assert len(final_result) == 350

    def test_single_path_mode(self):
        """Test behavior with single path configuration."""
        data = self.create_test_data(100)

        # Test with single path (should work like normal dataset)
        dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            target_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                "path1",
            ],
        )

        result = dataset.to_table().to_pandas()
        assert len(result) == 100

        # Verify data integrity
        pd.testing.assert_frame_equal(
            result.sort_values("id").reset_index(drop=True),
            data.sort_values("id").reset_index(drop=True),
        )

    def test_primary_path_write_rejection(self):
        """Test that writing to primary path URI is rejected."""
        data = self.create_test_data(50)

        # Create dataset with explicit data file bases
        dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            target_bases=[
                DatasetBasePath(self.path1_uri, name="path1"),
                DatasetBasePath(self.path2_uri, name="path2"),
                "path1",
            ],
        )

        # Try to append using primary path URI (should fail - not in registered bases)
        with pytest.raises(Exception, match="not found"):
            lance.write_dataset(
                data,
                dataset,
                mode="append",
                target_bases=[self.primary_uri],  # Primary path URI not registered
            )
