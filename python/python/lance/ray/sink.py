# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import logging
from typing import TYPE_CHECKING, Iterable, List, Optional, Union, Literal, Tuple
from itertools import chain

import pyarrow as pa

import lance
from lance.fragment import FragmentMetadata, LanceFragment

from ..dependencies import ray

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["LanceDatasink"]


class LanceDatasink(ray.data.Datasink):
    """Lance Ray Datasink.

    Write a Ray dataset to lance.

    Parameters
    ----------
    uri : str
        The base URI of the dataset.
    schema : pyarrow.Schema
        The schema of the dataset.
    mode : str, optional
        The write mode. Default is 'append'.
        Choices are 'append', 'create', 'overwrite'.
    max_rows_per_file : int, optional
        The maximum number of rows per file. Default is 1024 * 1024.


    Examples
    --------

    .. code-block:: python

        import ray
        import lance
        import pyarrow as pa

        ray.init()

        schema = pa.schema([pa.field("id", pa.int64()), pa.field("str", pa.string())])
        sink = lance.ray.sink.LanceDatasink("path/to/dataset", schema=schema)
        (
            ray.data
            .range(10)
            .map(lambda x: {"id": x["id"], "str": f"str-{x['id']}"})
            .write_datasink(sink)
        )

    """

    NAME = "Lance"

    def __init__(
        self,
        uri: str,
        schema: Optional[pa.Schema] = None,
        mode: Literal["create", "append", "overwrite"] = "create",
        max_rows_per_file: int = 1024 * 1024,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.uri = uri
        self.schema = schema
        self.mode = mode

        self.max_rows_per_file = max_rows_per_file
        # if mode is append, read_version is read from existing dataset.
        self.read_version: int | None = None

    @property
    def supports_distributed_writes(self) -> bool:
        return True

    @property
    def num_rows_per_write(self) -> int:
        return self.max_rows_per_file

    def get_name(self) -> str:
        return self.NAME

    def write(
        self,
        blocks: Iterable[Union[pa.Table, "pd.DataFrame"]],
        ctx,
    ):
        from ..dependencies import _PANDAS_AVAILABLE
        from ..dependencies import pandas as pd

        if self.schema is None:
            first = next(blocks)
            if _PANDAS_AVAILABLE and isinstance(first, pd.DataFrame):
                schema = pa.Schema.from_pandas(first)
            else:
                schema = first.schema
            blocks = chain([first], blocks)
        else:
            schema = self.schema

        # TODO: use v2 writer later.
        def record_batch_converter():
            for block in blocks:
                if _PANDAS_AVAILABLE and isinstance(block, pd.DataFrame):
                    tbl = pa.Table.from_pandas(block, schema=schema)
                else:
                    tbl = block
                yield from tbl.to_batches()

        reader = pa.RecordBatchReader.from_batches(schema, record_batch_converter())
        fragment = LanceFragment.create(self.uri, reader, fragment_id=0, schema=schema)
        ret_schema = None if ctx.task_idx != 0 else schema
        return (fragment, ret_schema)

    def on_write_start(self):
        if self.mode == "append":
            ds = lance.LanceDataset(self.uri)
            self.read_version = ds.version
            if self.schema is None:
                self.schema = ds.schema

    def on_write_complete(
        self, write_results: List[Tuple[FragmentMetadata, Optional[pa.Schema]]]
    ):
        logging.info(
            "%s.on_write_complete: committing %s fragments", self, len(write_results)
        )

        schema = self.schema
        if schema is None:
            for rst in write_results:
                if rst[1] is not None:
                    schema = rst[1]
                    break
        if schema is None:
            raise ValueError("schema is not provided or inferred from data.")

        fragments = [rst[0] for rst in write_results]
        if self.mode in set(["create", "overwrite"]):
            op = lance.LanceOperation.Overwrite(schema, fragments)
        elif self.mode == "append":
            op = lance.LanceOperation.Append(fragments)
        lance.LanceDataset.commit(self.uri, op, read_version=self.read_version)
