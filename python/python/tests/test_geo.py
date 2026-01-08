# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("geoarrow.rust.core")
from geoarrow.rust.core import (
    linestring,
    linestrings,
    point,
    points,
    polygon,
    polygons,
)


def test_geo_types(tmp_path: Path):
    uri = str(tmp_path / "test_geo_types.lance")
    # Points
    points_2d = points([np.random.rand(3), np.random.rand(3)])

    # LineStrings
    line_offsets = np.array([0, 2, 6, 10], dtype=np.int32)
    linestrings_2d = linestrings([np.random.rand(10), np.random.rand(10)], line_offsets)

    # Polygons
    ring_offsets = np.array([0, 3, 7, 12], dtype=np.int32)
    geom_offsets = np.array([0, 1, 2, 3], dtype=np.int32)
    polygons_2d = polygons(
        [np.random.rand(12), np.random.rand(12)],
        ring_offsets=ring_offsets,
        geom_offsets=geom_offsets,
    )

    schema = pa.schema(
        [
            pa.field(point("xy")).with_name("geometry_points"),
            pa.field(linestring("xy")).with_name("geometry_lines"),
            pa.field(polygon("xy")).with_name("geometry_polygons_2d"),
        ]
    )
    table = pa.Table.from_arrays(
        [points_2d, linestrings_2d, polygons_2d], schema=schema
    )
    lance.write_dataset(table, uri)
    ds = lance.dataset(uri)
    assert ds.schema.field(0) == table.schema.field(0)
    assert ds.schema.field(1) == table.schema.field(1)
    assert ds.schema.field(2) == table.schema.field(2)

    read_table = ds.to_table()
    assert read_table.schema.field(0) == table.schema.field(0)
    assert read_table.schema.field(1) == table.schema.field(1)
    assert read_table.schema.field(2) == table.schema.field(2)

    assert (
        read_table.schema.field(0).metadata[b"ARROW:extension:name"]
        == b"geoarrow.point"
    )
    assert (
        read_table.schema.field(1).metadata[b"ARROW:extension:name"]
        == b"geoarrow.linestring"
    )
    assert (
        read_table.schema.field(2).metadata[b"ARROW:extension:name"]
        == b"geoarrow.polygon"
    )

    assert read_table.num_rows == 3


def test_geo_sql(tmp_path: Path):
    # Points
    points_2d = points([np.array([1.0]), np.array([2.0])])

    # LineStrings
    line_offsets = np.array([0, 2], dtype=np.int32)
    linestrings_2d = linestrings(
        [np.array([3.0, 4.0]), np.array([5.0, 0.0])], line_offsets
    )

    schema = pa.schema(
        [
            pa.field(point("xy")).with_name("point"),
            pa.field(linestring("xy")).with_name("linestring"),
        ]
    )
    table = pa.Table.from_arrays([points_2d, linestrings_2d], schema=schema)
    ds = lance.write_dataset(table, str(tmp_path / "test_geo_udf_distance.lance"))

    batches = (
        ds.sql("SELECT St_Distance(point, linestring) as dist FROM dataset")
        .build()
        .to_batch_records()
    )
    assert len(batches) == 1
    result = batches[0].to_pydict()
    assert result["dist"]
    assert np.allclose(
        np.array(result["dist"]), np.array([2.5495097567963922]), atol=1e-8
    )


def test_rtree_index(tmp_path: Path):
    # LineStrings
    num_lines = 10000
    line_offsets = np.arange(num_lines + 1, dtype=np.int32) * 2
    linestrings_2d = linestrings(
        [np.random.randn(num_lines * 2) * 100, np.random.randn(num_lines * 2) * 100],
        line_offsets,
    )
    assert len(linestrings_2d) == num_lines

    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field(linestring("xy")).with_name("linestring"),
        ]
    )
    table = pa.Table.from_arrays(
        [np.arange(num_lines, dtype=np.int64), linestrings_2d], schema=schema
    )
    ds = lance.write_dataset(table, str(tmp_path / "test_rtree_index.lance"))

    def query(ds: lance.LanceDataset, has_index=False):
        sql = """
              SELECT `id`, linestring
              FROM dataset
              WHERE
              St_Intersects(linestring, ST_GeomFromText('LINESTRING ( 2 0, 0 2 )'))
              """

        batches = ds.sql("EXPLAIN ANALYZE " + sql).build().to_batch_records()
        explain = pa.Table.from_batches(batches).to_pandas().to_string()

        if has_index:
            assert "ScalarIndexQuery" in explain
        else:
            assert "ScalarIndexQuery" not in explain

        batches = ds.sql(sql).build().to_batch_records()
        return pa.Table.from_batches(batches)

    table_without_index = query(ds)

    ds.create_scalar_index("linestring", "RTREE")

    table_with_index = query(ds, has_index=True)

    assert table_with_index == table_without_index
