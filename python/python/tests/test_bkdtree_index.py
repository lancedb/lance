# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Tests for BKD Tree spatial index on GeoArrow Point data.

This module tests:
1. Creating GeoArrow Point data
2. Writing to Lance dataset
3. Creating a BKD tree index on GeoArrow Point column
4. Querying with spatial filters
5. Verifying the index is used in query execution
"""

import os

import numpy as np
import pyarrow as pa
import pytest

import lance

geoarrow = pytest.importorskip("geoarrow.pyarrow")


@pytest.fixture
def geoarrow_data():
    """Create GeoArrow Point test data with known cities and random points."""
    np.random.seed(42)
    num_points = 5000

    # Generate random points across the US
    # US bounding box: lng [-125, -65], lat [25, 50]
    lng_vals = np.random.uniform(-125, -65, num_points)
    lat_vals = np.random.uniform(25, 50, num_points)

    # Add known cities at the beginning for testing
    known_cities = [
        {
            "id": 1,
            "city": "San Francisco",
            "lng": -122.4194,
            "lat": 37.7749,
            "population": 883305,
        },
        {
            "id": 2,
            "city": "Los Angeles",
            "lng": -118.2437,
            "lat": 34.0522,
            "population": 3898747,
        },
        {
            "id": 3,
            "city": "New York",
            "lng": -74.0060,
            "lat": 40.7128,
            "population": 8336817,
        },
        {
            "id": 4,
            "city": "Chicago",
            "lng": -87.6298,
            "lat": 41.8781,
            "population": 2746388,
        },
        {
            "id": 5,
            "city": "Houston",
            "lng": -95.3698,
            "lat": 29.7604,
            "population": 2304580,
        },
    ]

    # Replace first 5 points with known cities
    for i, city in enumerate(known_cities):
        lng_vals[i] = city["lng"]
        lat_vals[i] = city["lat"]

    start_location = geoarrow.point().from_geobuffers(None, lng_vals, lat_vals)

    # Create IDs and city names
    ids = list(range(1, num_points + 1))
    cities = [
        known_cities[i]["city"] if i < len(known_cities) else f"Point_{i + 1}"
        for i in range(num_points)
    ]
    populations = [
        known_cities[i]["population"]
        if i < len(known_cities)
        else np.random.randint(10000, 1000000)
        for i in range(num_points)
    ]

    table = pa.table(
        {
            "id": ids,
            "city": cities,
            "start_location": start_location,
            "population": populations,
        }
    )

    return table, known_cities


def test_write_geoarrow_to_lance(tmp_path, geoarrow_data):
    """Test writing GeoArrow Point data to Lance dataset."""
    table, _ = geoarrow_data
    dataset_path = tmp_path / "geo_dataset"

    ds = lance.write_dataset(table, dataset_path)

    # Verify data was written correctly
    loaded_table = ds.to_table()
    assert len(loaded_table) == len(table)
    assert loaded_table.schema.equals(table.schema)


def test_create_bkdtree_index(tmp_path, geoarrow_data):
    """Test creating a BKD tree index on GeoArrow Point column."""
    table, _ = geoarrow_data
    dataset_path = tmp_path / "geo_dataset"

    ds = lance.write_dataset(table, dataset_path)

    # Create BKD tree index
    ds.create_scalar_index(column="start_location", index_type="BKDTREE")

    # Verify index was created
    indexes = ds.list_indices()
    assert len(indexes) > 0

    # Check that index files exist
    index_dir = dataset_path / "_indices"
    assert index_dir.exists()
    index_files = list(index_dir.rglob("*"))
    assert len(index_files) > 0


def test_spatial_query_broad_bbox(tmp_path, geoarrow_data):
    """Test spatial query with broad bounding box covering multiple cities."""
    table, known_cities = geoarrow_data
    dataset_path = tmp_path / "geo_dataset"

    ds = lance.write_dataset(table, dataset_path)
    ds.create_scalar_index(column="start_location", index_type="BKDTREE")

    # Query with broad bbox covering western US
    # Should include San Francisco and Los Angeles
    sql = """
    SELECT id, city, population 
    FROM dataset 
    WHERE st_intersects(start_location, bbox(-125, 30, -115, 45))
    """

    query = ds.sql(sql).build()
    result = query.to_batch_records()

    assert result is not None
    result_table = pa.Table.from_batches(result)

    # Should get many results with random points
    assert len(result_table) > 100

    # Should include known cities in the bbox
    cities = result_table.column("city").to_pylist()
    assert "San Francisco" in cities
    assert "Los Angeles" in cities


def test_spatial_query_tight_bbox(tmp_path, geoarrow_data):
    """Test spatial query with tight bounding box around single city."""
    table, known_cities = geoarrow_data
    dataset_path = tmp_path / "geo_dataset"

    ds = lance.write_dataset(table, dataset_path)
    ds.create_scalar_index(column="start_location", index_type="BKDTREE")

    # Query with tight bbox around San Francisco only
    # SF is at (-122.4194, 37.7749)
    sql = """
    SELECT id, city, population 
    FROM dataset 
    WHERE st_intersects(start_location, bbox(-123, 37, -122, 38))
    """

    query = ds.sql(sql).build()
    result = query.to_batch_records()

    assert result is not None
    result_table = pa.Table.from_batches(result)

    # Should include San Francisco
    cities = result_table.column("city").to_pylist()
    assert "San Francisco" in cities

    # From known cities, should only include San Francisco
    known_cities_in_result = [
        c
        for c in cities
        if c in ["San Francisco", "Los Angeles", "New York", "Chicago", "Houston"]
    ]
    assert known_cities_in_result == ["San Francisco"]


def test_spatial_query_uses_index(tmp_path, geoarrow_data):
    """Test that spatial queries use the BKD tree index via EXPLAIN ANALYZE."""
    table, _ = geoarrow_data
    dataset_path = tmp_path / "geo_dataset"

    ds = lance.write_dataset(table, dataset_path)
    ds.create_scalar_index(column="start_location", index_type="BKDTREE")

    # Run EXPLAIN ANALYZE to verify index usage
    explain_sql = """
    EXPLAIN ANALYZE SELECT id, city, population 
    FROM dataset 
    WHERE st_intersects(start_location, bbox(-125, 30, -115, 45))
    """

    query = ds.sql(explain_sql).build()
    result = query.to_batch_records()

    assert result is not None
    explain_table = pa.Table.from_batches(result)
    assert len(explain_table) > 0

    # Check if index was used in the execution plan
    # The plan is in the second column
    plan_text = str(explain_table.column(1).to_pylist()[0])

    # Look for evidence of index usage
    # (The exact string may vary, adjust based on actual output)
    assert "ScalarIndexQuery" in plan_text or "start_location_idx" in plan_text, (
        f"Index not detected in execution plan: {plan_text}"
    )


def test_spatial_query_empty_result(tmp_path, geoarrow_data):
    """Test spatial query with bbox that doesn't intersect any points."""
    table, _ = geoarrow_data
    dataset_path = tmp_path / "geo_dataset"

    ds = lance.write_dataset(table, dataset_path)
    ds.create_scalar_index(column="start_location", index_type="BKDTREE")

    # Query with bbox outside the US (e.g., over the Pacific Ocean)
    sql = """
    SELECT id, city, population 
    FROM dataset 
    WHERE st_intersects(start_location, bbox(-180, -10, -175, -5))
    """

    query = ds.sql(sql).build()
    result = query.to_batch_records()

    # Should return empty result or very small result
    if result:
        result_table = pa.Table.from_batches(result)
        assert len(result_table) == 0


def test_index_file_structure(tmp_path, geoarrow_data):
    """Test that BKD tree index creates expected file structure."""
    table, _ = geoarrow_data
    dataset_path = tmp_path / "geo_dataset"

    ds = lance.write_dataset(table, dataset_path)
    ds.create_scalar_index(column="start_location", index_type="BKDTREE")

    index_dir = dataset_path / "_indices"
    assert index_dir.exists()

    # Check for index subdirectories
    index_subdirs = [d for d in index_dir.iterdir() if d.is_dir()]
    assert len(index_subdirs) > 0

    # Check that index files exist and have content
    for subdir in index_subdirs:
        files = list(subdir.glob("*"))
        assert len(files) > 0

        # Verify files have content
        for f in files:
            if f.is_file():
                assert f.stat().st_size > 0
