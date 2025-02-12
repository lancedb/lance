# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Union

import pyarrow as pa
from pyarrow import RecordBatch

from . import dataset
from .dependencies import _check_for_pandas
from .dependencies import pandas as pd

if TYPE_CHECKING:
    ReaderLike = Union[
        pd.Timestamp,
        pa.Table,
        pa.dataset.Dataset,
        pa.dataset.Scanner,
        pa.RecordBatch,
        Iterable[RecordBatch],
        pa.RecordBatchReader,
    ]


def _casting_recordbatch_iter(
    input_iter: Iterable[pa.RecordBatch], schema: pa.Schema
) -> Iterable[pa.RecordBatch]:
    """
    Wrapper around an iterator of record batches. If the batches don't match the
    schema, try to cast them to the schema. If that fails, raise an error.

    This is helpful for users who might have written the iterator with default
    data types in PyArrow, but specified more specific types in the schema. For
    example, PyArrow defaults to float64 for floating point types, but Lance
    uses float32 for vectors.
    """
    for batch in input_iter:
        if not isinstance(batch, pa.RecordBatch):
            raise TypeError(f"Expected RecordBatch, got {type(batch)}")
        if batch.schema != schema:
            try:
                # RecordBatch doesn't have a cast method, but table does.
                batch = pa.Table.from_batches([batch]).cast(schema).to_batches()[0]
            except pa.lib.ArrowInvalid:
                raise ValueError(
                    f"Input RecordBatch iterator yielded a batch with schema that "
                    f"does not match the expected schema.\nExpected:\n{schema}\n"
                    f"Got:\n{batch.schema}"
                )
        yield batch


def _coerce_reader(
    data_obj: ReaderLike, schema: Optional[pa.Schema] = None
) -> pa.RecordBatchReader:
    if _check_for_pandas(data_obj) and isinstance(data_obj, pd.DataFrame):
        return pa.Table.from_pandas(data_obj, schema=schema).to_reader()
    elif isinstance(data_obj, pa.Table):
        return data_obj.to_reader()
    elif isinstance(data_obj, pa.RecordBatch):
        return pa.Table.from_batches([data_obj]).to_reader()
    elif isinstance(data_obj, dataset.LanceDataset):
        return data_obj.scanner().to_reader()
    elif isinstance(data_obj, pa.dataset.Dataset):
        return pa.dataset.Scanner.from_dataset(data_obj).to_reader()
    elif isinstance(data_obj, pa.dataset.Scanner):
        return data_obj.to_reader()
    elif isinstance(data_obj, pa.RecordBatchReader):
        return data_obj
    elif (
        type(data_obj).__module__.startswith("polars")
        and data_obj.__class__.__name__ == "DataFrame"
    ):
        return data_obj.to_arrow().to_reader()
    # for other iterables, assume they are of type Iterable[RecordBatch]
    elif isinstance(data_obj, Iterable):
        if schema is not None:
            data = _casting_recordbatch_iter(data_obj, schema)
            return pa.RecordBatchReader.from_batches(schema, data)
        else:
            raise ValueError(
                "Must provide schema to write dataset from RecordBatch iterable"
            )
    else:
        raise TypeError(
            f"Unknown data type {type(data_obj)}. "
            "Please check "
            "https://lancedb.github.io/lance/read_and_write.html "
            "to see supported types."
        )
