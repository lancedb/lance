# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Tests for multi-paths dataset functionality.
"""

import tempfile
import shutil
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import lance
import pandas as pd
import pyarrow as pa
import pytest


class TestMultiPaths:
    """Test multi-paths dataset functionality with local file system."""

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
        if hasattr(self, 'test_dir'):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_data(self, num_rows=500, id_offset=0):
        """Create test data for multi-paths tests."""
        return pd.DataFrame({
            'id': range(id_offset, id_offset + num_rows),
            'value': [f'value_{i}' for i in range(id_offset, id_offset + num_rows)],
            'score': [i * 0.1 for i in range(id_offset, id_offset + num_rows)]
        })

    def test_multi_paths_create_and_read(self):
        """Test creating a multi-paths dataset and reading it back."""
        data = self.create_test_data(500)
        
        # Create dataset with multi-paths layout
        dataset = lance.write_dataset(
            data, 
            self.primary_uri,
            mode="create",
            initial_data_paths=[self.path1_uri, self.path2_uri, self.path3_uri],
            target_data_paths=[self.path2_uri],  # Write data to path2
            max_rows_per_file=100  # Force multiple fragments
        )
        
        assert dataset is not None
        assert dataset.uri == self.primary_uri
        
        # Verify we can read the data back
        result = dataset.to_table().to_pandas()
        assert len(result) == 500
        
        # Verify data integrity
        pd.testing.assert_frame_equal(
            result.sort_values('id').reset_index(drop=True),
            data.sort_values('id').reset_index(drop=True)
        )

    def test_multi_paths_append_mode(self):
        """Test appending data to a multi-paths dataset."""
        # Create initial dataset
        initial_data = self.create_test_data(300)
        
        dataset = lance.write_dataset(
            initial_data, 
            self.primary_uri,
            mode="create",
            initial_data_paths=[self.path1_uri, self.path2_uri],
            target_data_paths=[self.path1_uri],
            max_rows_per_file=100
        )
        
        # Create additional data to append
        append_data = self.create_test_data(100, id_offset=300)
        
        # Append to different path
        updated_dataset = lance.write_dataset(
            append_data,
            dataset,
            mode="append",
            target_data_paths=[self.path2_uri],  # Write to path2
            max_rows_per_file=50
        )
        
        # Verify total data
        result = updated_dataset.to_table().to_pandas()
        assert len(result) == 400
        
        # Verify all data is present
        expected_ids = set(range(400))
        actual_ids = set(result['id'].tolist())
        assert actual_ids == expected_ids

    def test_multi_paths_overwrite_mode_with_new_paths(self):
        """Test overwriting with new path configuration."""
        # Create initial dataset
        initial_data = self.create_test_data(200)
        
        dataset = lance.write_dataset(
            initial_data, 
            self.primary_uri,
            mode="create",
            initial_data_paths=[self.path1_uri, self.path2_uri],
            target_data_paths=[self.path1_uri],
            max_rows_per_file=100
        )
        
        # Create new data for overwrite with new paths
        overwrite_data = self.create_test_data(150, id_offset=100)
        
        # Overwrite with new path configuration
        updated_dataset = lance.write_dataset(
            overwrite_data,
            self.primary_uri,
            mode="overwrite",
            initial_data_paths=[self.path2_uri, self.path3_uri],  # New paths
            target_data_paths=[self.path3_uri],  # Write to path3
            max_rows_per_file=75
        )
        
        # Verify overwritten data
        result = updated_dataset.to_table().to_pandas()
        assert len(result) == 150
        
        # Verify data content
        expected_ids = set(range(100, 250))
        actual_ids = set(result['id'].tolist())
        assert actual_ids == expected_ids

    def test_multi_paths_overwrite_mode_primary_path_default(self):
        """Test that OVERWRITE mode defaults to primary path when no target specified."""
        # Create initial dataset with explicit data file bases
        initial_data = self.create_test_data(100)
        
        dataset = lance.write_dataset(
            initial_data, 
            self.primary_uri,
            mode="create",
            initial_data_paths=[self.path1_uri, self.path2_uri],
            target_data_paths=[self.path1_uri],
            max_rows_per_file=50
        )
        
        # Overwrite without specifying target - should use primary path
        overwrite_data = self.create_test_data(75, id_offset=200)
        
        updated_dataset = lance.write_dataset(
            overwrite_data,
            self.primary_uri,
            mode="overwrite",
            # No initial_data_paths or target_data_paths specified
            max_rows_per_file=25
        )
        
        # Verify overwritten data
        result = updated_dataset.to_table().to_pandas()
        assert len(result) == 75
        
        # Verify data content
        expected_ids = set(range(200, 275))
        actual_ids = set(result['id'].tolist())
        assert actual_ids == expected_ids
        
        # Verify we can still append to registered paths
        append_data = self.create_test_data(25, id_offset=300)
        
        final_dataset = lance.write_dataset(
            append_data,
            updated_dataset,
            mode="append",
            target_data_paths=[self.path2_uri]
        )
        
        final_result = final_dataset.to_table().to_pandas()
        assert len(final_result) == 100

    def test_concurrent_appends(self):
        """Test concurrent appends to multi-paths dataset."""
        # Create initial dataset
        initial_data = self.create_test_data(100)
        
        dataset = lance.write_dataset(
            initial_data, 
            self.primary_uri,
            mode="create",
            initial_data_paths=[self.path1_uri, self.path2_uri, self.path3_uri],
            target_data_paths=[self.path1_uri],
            max_rows_per_file=50
        )
        
        def append_worker(worker_id, target_path, start_id, num_rows):
            """Worker function to append data concurrently."""
            worker_data = self.create_test_data(num_rows, id_offset=start_id)
            
            # Load dataset fresh for this worker
            worker_dataset = lance.dataset(self.primary_uri)
            
            # Perform append
            return lance.write_dataset(
                worker_data,
                worker_dataset,
                mode="append",
                target_data_paths=[target_path],
                max_rows_per_file=25
            )
        
        # Configure concurrent append tasks
        append_tasks = [
            {'worker_id': 1, 'target_path': self.path1_uri, 'start_id': 100, 'num_rows': 50},
            {'worker_id': 2, 'target_path': self.path2_uri, 'start_id': 150, 'num_rows': 50},
            {'worker_id': 3, 'target_path': self.path3_uri, 'start_id': 200, 'num_rows': 50},
        ]
        
        # Execute concurrent appends
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_task = {
                executor.submit(
                    append_worker, 
                    task['worker_id'], 
                    task['target_path'], 
                    task['start_id'], 
                    task['num_rows']
                ): task for task in append_tasks
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
        actual_ids = set(final_data['id'].tolist())
        assert actual_ids == expected_ids

    def test_validation_errors(self):
        """Test validation errors for invalid multi-paths configurations."""
        data = self.create_test_data(100)
        
        # Test 1: initial_data_paths without target_data_paths in CREATE mode
        with pytest.raises(Exception, match="target_data_paths must also be specified"):
            lance.write_dataset(
                data,
                self.primary_uri,
                mode="create",
                initial_data_paths=[self.path1_uri, self.path2_uri]
                # target_data_paths missing
            )
        
        # Test 2: target_data_paths not in initial_data_paths
        with pytest.raises(Exception, match="not found in initial_data_paths"):
            lance.write_dataset(
                data,
                self.primary_uri,
                mode="create",
                initial_data_paths=[self.path1_uri, self.path2_uri],
                target_data_paths=[self.path3_uri]  # Not in initial list
            )
        
        # Test 3: target_data_paths without initial_data_paths in CREATE mode
        with pytest.raises(Exception, match="initial_data_paths must also be specified"):
            lance.write_dataset(
                data,
                self.primary_uri,
                mode="create",
                target_data_paths=[self.path1_uri]
                # initial_data_paths missing
            )
        
        # Test 4: initial_data_paths in APPEND mode
        # First create a base dataset
        base_dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            initial_data_paths=[self.path1_uri, self.path2_uri],
            target_data_paths=[self.path1_uri]
        )
        
        with pytest.raises(Exception, match="initial_data_paths should not be provided.*Append"):
            lance.write_dataset(
                data,
                base_dataset,
                mode="append",
                initial_data_paths=[self.path1_uri, self.path2_uri],  # Should not be provided
                target_data_paths=[self.path2_uri]
            )
        
        # Test 5: Write to dataset with explicit data file bases without target_data_paths
        with pytest.raises(Exception, match="explicit data file bases.*target_data_paths"):
            lance.write_dataset(
                data,
                base_dataset,
                mode="append"
                # No target_data_paths specified
            )

    def test_fragment_distribution(self):
        """Test that fragments are correctly distributed and readable."""
        data = self.create_test_data(1000)
        
        dataset = lance.write_dataset(
            data, 
            self.primary_uri,
            mode="create",
            initial_data_paths=[self.path1_uri, self.path2_uri],
            target_data_paths=[self.path1_uri],
            max_rows_per_file=100  # Should create 10 fragments
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
        assert all(filtered_result['id'] < 500)

    def test_schema_consistency(self):
        """Test that schema is consistent across multi-paths operations."""
        # Create dataset with specific schema
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("name", pa.string()),
            pa.field("score", pa.float64()),
        ])
        
        data = pa.table({
            "id": [1, 2, 3, 4, 5],
            "name": ["a", "b", "c", "d", "e"],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0]
        }, schema=schema)
        
        dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            initial_data_paths=[self.path1_uri, self.path2_uri],
            target_data_paths=[self.path1_uri]
        )
        
        # Verify schema
        assert dataset.schema == schema
        
        # Append with same schema to different path
        append_data = pa.table({
            "id": [6, 7, 8],
            "name": ["f", "g", "h"],
            "score": [6.0, 7.0, 8.0]
        }, schema=schema)
        
        updated_dataset = lance.write_dataset(
            append_data,
            dataset,
            mode="append",
            target_data_paths=[self.path2_uri]
        )
        
        # Schema should remain consistent
        assert updated_dataset.schema == schema
        
        # Verify all data
        result = updated_dataset.to_table()
        assert len(result) == 8
        assert result.schema == schema

    def test_dataset_reopening(self):
        """Test reopening a multi-paths dataset."""
        data = self.create_test_data(300)
        
        # Create multi-paths dataset
        dataset = lance.write_dataset(
            data, 
            self.primary_uri,
            mode="create",
            initial_data_paths=[self.path1_uri, self.path2_uri],
            target_data_paths=[self.path1_uri],
            max_rows_per_file=100
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
            target_data_paths=[self.path2_uri]
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
            initial_data_paths=[self.path1_uri],
            target_data_paths=[self.path1_uri]
        )
        
        result = dataset.to_table().to_pandas()
        assert len(result) == 100
        
        # Verify data integrity
        pd.testing.assert_frame_equal(
            result.sort_values('id').reset_index(drop=True),
            data.sort_values('id').reset_index(drop=True)
        )

    def test_primary_path_write_rejection(self):
        """Test that writing to primary path URI is rejected."""
        data = self.create_test_data(50)
        
        # Create dataset with explicit data file bases
        dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            initial_data_paths=[self.path1_uri, self.path2_uri],
            target_data_paths=[self.path1_uri]
        )
        
        # Try to append to the primary path URI (should fail)
        with pytest.raises(Exception, match="not found in existing"):
            lance.write_dataset(
                data,
                dataset,
                mode="append",
                target_data_paths=[self.primary_uri]  # Primary path URI
            )