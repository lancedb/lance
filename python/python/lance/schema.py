# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import json
from typing import Any, Dict

import pyarrow as pa

from .lance import LanceSchema as LanceSchema
from .lance import _json_to_schema, _schema_to_json


def schema_to_json(schema: pa.Schema) -> Dict[str, Any]:
    """
    Converts a pyarrow schema to a JSON string.

    Parameters
    ----------
    """
    return json.loads(_schema_to_json(schema))


def json_to_schema(schema_json: Dict[str, Any]) -> pa.Schema:
    """
    Converts a JSON string to a PyArrow schema.

    Parameters
    ----------
    schema_json: Dict[str, Any]
        The JSON payload to convert to a PyArrow Schema.
    """
    return _json_to_schema(json.dumps(schema_json))


_LANCE_MAP_KEYS_SORTED_KEY = b"lance:map_keys_sorted"


# PyArrow's C stream conversion can drop MapType.keys_sorted. Lance stores a
# metadata marker on sorted map fields and restores the PyArrow schema at API
# boundaries where users observe schemas directly.
def _restore_map_keys_sorted_type(
    data_type: pa.DataType, keys_sorted: bool
) -> pa.DataType:
    if pa.types.is_map(data_type):
        key_field = _restore_map_keys_sorted_field(data_type.key_field)
        item_field = _restore_map_keys_sorted_field(data_type.item_field)
        return pa.map_(
            key_field,
            item_field,
            keys_sorted=keys_sorted or data_type.keys_sorted,
        )
    if pa.types.is_struct(data_type):
        return pa.struct([_restore_map_keys_sorted_field(field) for field in data_type])
    if pa.types.is_list(data_type):
        return pa.list_(_restore_map_keys_sorted_field(data_type.value_field))
    if pa.types.is_large_list(data_type):
        return pa.large_list(_restore_map_keys_sorted_field(data_type.value_field))
    if pa.types.is_fixed_size_list(data_type):
        return pa.list_(
            _restore_map_keys_sorted_field(data_type.value_field),
            data_type.list_size,
        )
    return data_type


def _restore_map_keys_sorted_field(field: pa.Field) -> pa.Field:
    metadata = field.metadata
    keys_sorted = False
    if metadata is not None and metadata.get(_LANCE_MAP_KEYS_SORTED_KEY) == b"true":
        keys_sorted = True
        metadata = {
            key: value
            for key, value in metadata.items()
            if key != _LANCE_MAP_KEYS_SORTED_KEY
        }
        if not metadata:
            metadata = None

    return pa.field(
        field.name,
        _restore_map_keys_sorted_type(field.type, keys_sorted),
        field.nullable,
        metadata=metadata,
    )


def _restore_map_keys_sorted_schema(schema: pa.Schema) -> pa.Schema:
    if not _schema_needs_map_keys_sorted_restore(schema):
        return schema

    return pa.schema(
        [_restore_map_keys_sorted_field(field) for field in schema],
        metadata=schema.metadata,
    )


def _restore_map_keys_sorted_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    if batch.num_columns == 0 or not _schema_needs_map_keys_sorted_restore(
        batch.schema
    ):
        return batch

    return pa.RecordBatch.from_arrays(
        batch.columns,
        schema=_restore_map_keys_sorted_schema(batch.schema),
    )


def _restore_map_keys_sorted_table(table: pa.Table) -> pa.Table:
    if table.num_columns == 0 or not _schema_needs_map_keys_sorted_restore(
        table.schema
    ):
        return table

    return pa.Table.from_arrays(
        table.columns,
        schema=_restore_map_keys_sorted_schema(table.schema),
    )


def _table_from_restored_batches(batches: list[pa.RecordBatch]) -> pa.Table:
    return _restore_map_keys_sorted_table(pa.Table.from_batches(batches))


def _restore_map_keys_sorted_reader(
    reader: pa.RecordBatchReader,
) -> pa.RecordBatchReader:
    if not _schema_needs_map_keys_sorted_restore(reader.schema):
        return reader

    return pa.RecordBatchReader.from_batches(
        _restore_map_keys_sorted_schema(reader.schema),
        (_restore_map_keys_sorted_batch(batch) for batch in reader),
    )


def _schema_needs_map_keys_sorted_restore(schema: pa.Schema) -> bool:
    return any(_field_needs_map_keys_sorted_restore(field) for field in schema)


def _field_needs_map_keys_sorted_restore(field: pa.Field) -> bool:
    if (
        field.metadata is not None
        and field.metadata.get(_LANCE_MAP_KEYS_SORTED_KEY) == b"true"
    ):
        return True

    data_type = field.type
    if pa.types.is_map(data_type):
        return _field_needs_map_keys_sorted_restore(
            data_type.key_field
        ) or _field_needs_map_keys_sorted_restore(data_type.item_field)
    if pa.types.is_struct(data_type):
        return any(_field_needs_map_keys_sorted_restore(child) for child in data_type)
    if (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_fixed_size_list(data_type)
    ):
        return _field_needs_map_keys_sorted_restore(data_type.value_field)
    return False
