# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import pickle
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import pyarrow as pa

import lance
from lance.fragment import DEFAULT_MAX_BYTES_PER_FILE, FragmentMetadata, write_fragments

from ..dependencies import ray

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["LanceDatasink", "LanceFragmentWriter", "LanceCommitter", "write_lance"]


def _pd_to_arrow(
    df: Union[pa.Table, "pd.DataFrame", Dict], schema: Optional[pa.Schema]
) -> pa.Table:
    """Convert a pandas DataFrame to pyarrow Table."""
    from ..dependencies import _PANDAS_AVAILABLE
    from ..dependencies import pandas as pd

    if isinstance(df, dict):
        return pa.Table.from_pydict(df, schema=schema)
    if _PANDAS_AVAILABLE and isinstance(df, pd.DataFrame):
        tbl = pa.Table.from_pandas(df, schema=schema)
        tbl.schema = tbl.schema.remove_metadata()
        return tbl
    return df


def _write_fragment(
    stream: Iterable[Union[pa.Table, "pd.Pandas"]],
    uri: str,
    *,
    schema: Optional[pa.Schema] = None,
    max_rows_per_file: int = 1024 * 1024,
    max_bytes_per_file: Optional[int] = None,
    max_rows_per_group: int = 1024,  # Only useful for v1 writer.
    data_storage_version: Optional[str] = None,
    storage_options: Optional[Dict[str, Any]] = None,
) -> Tuple[FragmentMetadata, pa.Schema]:
    from ..dependencies import _PANDAS_AVAILABLE
    from ..dependencies import pandas as pd

    if schema is None:
        first = next(stream)
        if _PANDAS_AVAILABLE and isinstance(first, pd.DataFrame):
            schema = pa.Schema.from_pandas(first).remove_metadata()
        elif isinstance(first, Dict):
            tbl = pa.Table.from_pydict(first)
            schema = tbl.schema.remove_metadata()
        else:
            schema = first.schema
        stream = chain([first], stream)

    def record_batch_converter():
        for block in stream:
            tbl = _pd_to_arrow(block, schema)
            yield from tbl.to_batches()

    max_bytes_per_file = (
        DEFAULT_MAX_BYTES_PER_FILE if max_bytes_per_file is None else max_bytes_per_file
    )

    reader = pa.RecordBatchReader.from_batches(schema, record_batch_converter())
    fragments = write_fragments(
        reader,
        uri,
        schema=schema,
        max_rows_per_file=max_rows_per_file,
        max_rows_per_group=max_rows_per_group,
        max_bytes_per_file=max_bytes_per_file,
        data_storage_version=data_storage_version,
        storage_options=storage_options,
    )
    return [(fragment, schema) for fragment in fragments]


class _BaseLanceDatasink(ray.data.Datasink):
    """Base Lance Ray Datasink."""

    def __init__(
        self,
        uri: str,
        schema: Optional[pa.Schema] = None,
        mode: Literal["create", "append", "overwrite"] = "create",
        storage_options: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.uri = uri
        self.schema = schema
        self.mode = mode

        self.read_version: int | None = None
        self.storage_options = storage_options

    @property
    def supports_distributed_writes(self) -> bool:
        return True

    def on_write_start(self):
        if self.mode == "append":
            ds = lance.LanceDataset(self.uri, storage_options=self.storage_options)
            self.read_version = ds.version
            if self.schema is None:
                self.schema = ds.schema

    def on_write_complete(
        self,
        write_results: List[List[Tuple[str, str]]],
    ):
        fragments = []
        schema = None
        for batch in write_results:
            for fragment_str, schema_str in batch:
                fragment = pickle.loads(fragment_str)
                fragments.append(fragment)
                schema = pickle.loads(schema_str)
        # Check weather writer has fragments or not.
        # Skip commit when there are no fragments.
        if not schema:
            return
        if self.mode in set(["create", "overwrite"]):
            op = lance.LanceOperation.Overwrite(schema, fragments)
        elif self.mode == "append":
            op = lance.LanceOperation.Append(fragments)
        lance.LanceDataset.commit(
            self.uri,
            op,
            read_version=self.read_version,
            storage_options=self.storage_options,
        )


class LanceDatasink(_BaseLanceDatasink):
    """Lance Ray Datasink.

    Write a Ray dataset to lance.

    If we expect to write larger-than-memory files,
    we can use `LanceFragmentWriter` and `LanceCommitter`.

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
    data_storage_version: optional, str, default None
        The version of the data storage format to use. Newer versions are more
        efficient but require newer versions of lance to read.  The default is
        "legacy" which will use the legacy v1 version.  See the user guide
        for more details.
    use_legacy_format : optional, bool, default None
        Deprecated method for setting the data storage version. Use the
        `data_storage_version` parameter instead.
    storage_options : Dict[str, Any], optional
        The storage options for the writer. Default is None.
    """

    NAME = "Lance"

    def __init__(
        self,
        uri: str,
        schema: Optional[pa.Schema] = None,
        mode: Literal["create", "append", "overwrite"] = "create",
        max_rows_per_file: int = 1024 * 1024,
        data_storage_version: Optional[str] = None,
        use_legacy_format: Optional[bool] = None,
        storage_options: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            uri,
            schema=schema,
            mode=mode,
            storage_options=storage_options,
            *args,
            **kwargs,
        )

        if use_legacy_format is not None:
            import warnings

            warnings.warn(
                "The `use_legacy_format` parameter is deprecated. Use the "
                "`data_storage_version` parameter instead.",
                DeprecationWarning,
            )

            if use_legacy_format:
                data_storage_version = "legacy"
            else:
                data_storage_version = "stable"

        self.max_rows_per_file = max_rows_per_file
        self.data_storage_version = data_storage_version
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
        _ctx,
    ):
        fragments_and_schema = _write_fragment(
            blocks,
            self.uri,
            schema=self.schema,
            max_rows_per_file=self.max_rows_per_file,
            data_storage_version=self.data_storage_version,
            storage_options=self.storage_options,
        )
        return [
            (pickle.dumps(fragment), pickle.dumps(schema))
            for fragment, schema in fragments_and_schema
        ]


class LanceFragmentWriter:
    """Write a fragment to one of Lance fragment.

    This Writer can be used in case to write large-than-memory data to lance,
    in distributed fashion.

    Parameters
    ----------
    uri : str
        The base URI of the dataset.
    transform : Callable[[pa.Table], Union[pa.Table, Generator]], optional
        A callable to transform the input batch. Default is None.
    schema : pyarrow.Schema, optional
        The schema of the dataset.
    max_rows_per_file : int, optional
        The maximum number of rows per file. Default is 1024 * 1024.
    max_bytes_per_file : int, optional
        The maximum number of bytes per file. Default is 90GB.
    max_rows_per_group : int, optional
        The maximum number of rows per group. Default is 1024.
        Only useful for v1 writer.
    data_storage_version: optional, str, default None
        The version of the data storage format to use. Newer versions are more
        efficient but require newer versions of lance to read.  The default
        (None) will use the 2.0 version.  See the user guide for more details.
    use_legacy_format : optional, bool, default None
        Deprecated method for setting the data storage version. Use the
        `data_storage_version` parameter instead.
    storage_options : Dict[str, Any], optional
        The storage options for the writer. Default is None.

    """

    def __init__(
        self,
        uri: str,
        *,
        transform: Optional[Callable[[pa.Table], Union[pa.Table, Generator]]] = None,
        schema: Optional[pa.Schema] = None,
        max_rows_per_file: int = 1024 * 1024,
        max_bytes_per_file: Optional[int] = None,
        max_rows_per_group: Optional[int] = None,  # Only useful for v1 writer.
        data_storage_version: Optional[str] = None,
        use_legacy_format: Optional[bool] = False,
        storage_options: Optional[Dict[str, Any]] = None,
    ):
        if use_legacy_format is not None:
            import warnings

            warnings.warn(
                "The `use_legacy_format` parameter is deprecated. Use the "
                "`data_storage_version` parameter instead.",
                DeprecationWarning,
            )

            if use_legacy_format:
                data_storage_version = "legacy"
            else:
                data_storage_version = "stable"

        self.uri = uri
        self.schema = schema
        self.transform = transform if transform is not None else lambda x: x

        self.max_rows_per_group = max_rows_per_group
        self.max_rows_per_file = max_rows_per_file
        self.max_bytes_per_file = max_bytes_per_file
        self.data_storage_version = data_storage_version
        self.storage_options = storage_options

    def __call__(self, batch: Union[pa.Table, "pd.DataFrame"]) -> Dict[str, Any]:
        """Write a Batch to the Lance fragment."""

        transformed = self.transform(batch)
        if not isinstance(transformed, Generator):
            transformed = (t for t in [transformed])

        fragments = _write_fragment(
            transformed,
            self.uri,
            schema=self.schema,
            max_rows_per_file=self.max_rows_per_file,
            max_rows_per_group=self.max_rows_per_group,
            data_storage_version=self.data_storage_version,
            storage_options=self.storage_options,
        )
        return pa.Table.from_pydict(
            {
                "fragment": [pickle.dumps(fragment) for fragment, _ in fragments],
                "schema": [pickle.dumps(schema) for _, schema in fragments],
            }
        )


class LanceCommitter(_BaseLanceDatasink):
    """Lance Commiter as Ray Datasink.

    This is used with `LanceFragmentWriter` to write large-than-memory data to
    lance file.
    """

    @property
    def num_rows_per_write(self) -> int:
        return 1

    def get_name(self) -> str:
        return f"LanceCommiter({self.mode})"

    def write(
        self,
        blocks: Iterable[Union[pa.Table, "pd.DataFrame"]],
        _ctx,
    ):
        """Passthrough the fragments to commit phase"""
        v = []
        for block in blocks:
            # If block is empty, skip to get "fragment" and "schema" filed
            if len(block) == 0:
                continue

            for fragment, schema in zip(
                block["fragment"].to_pylist(), block["schema"].to_pylist()
            ):
                v.append((fragment, schema))
        return v


def write_lance(
    data: ray.data.Dataset,
    output_uri: str,
    *,
    schema: Optional[pa.Schema] = None,
    transform: Optional[
        Callable[[pa.Table], Union[pa.Table, Generator[None, pa.Table, None]]]
    ] = None,
    max_rows_per_file: int = 1024 * 1024,
    max_bytes_per_file: Optional[int] = None,
    storage_options: Optional[Dict[str, Any]] = None,
    data_storage_version: Optional[str] = None,
) -> None:
    """Write Ray dataset at scale.

    This method wraps the `LanceFragmentWriter` and `LanceCommitter` to write
    large-than-memory ray data to lance files.

    Parameters
    ----------
    data : ray.data.Dataset
        The dataset to write.
    output_uri : str
        The output dataset URI.
    transform : Callable[[pa.Table], Union[pa.Table, Generator]], optional
        A callable to transform the input batch. Default is identity function.
    schema : pyarrow.Schema, optional
        If provided, the schema of the dataset. Otherwise, it will be inferred.
    max_rows_per_file: int, optional
        The maximum number of rows per file. Default is 1024 * 1024.
    max_bytes_per_file: int, optional
        The maximum number of bytes per file. Default is 90GB.
    storage_options : Dict[str, Any], optional
        The storage options for the writer. Default is None.
    data_storage_version: optional, str, default None
        The version of the data storage format to use. Newer versions are more
        efficient but require newer versions of lance to read.  The default
        (None) will use the 2.0 version.  See the user guide for more details.
    """
    data.map_batches(
        LanceFragmentWriter(
            output_uri,
            schema=schema,
            transform=transform,
            max_rows_per_file=max_rows_per_file,
            max_bytes_per_file=max_bytes_per_file,
            storage_options=storage_options,
            data_storage_version=data_storage_version,
        ),
        batch_size=max_rows_per_file,
    ).write_datasink(
        LanceCommitter(output_uri, schema=schema, storage_options=storage_options)
    )


def _register_hooks():
    """Register lance hook to Ray for better integration.

    You can use `ray.data.Dataset.write_lance` to write Ray dataset to lance.
    Example:

    ```python
    import ray
    import lance
    from lance.ray.sink import _register_hooks

    _register_hooks()

    ray.data.range(10)
        .map(lambda x: {"id": x["id"], "str": f"str-{x['id']}"})
        .write_lance("~/data.lance")
    ```
    """
    ray.data.Dataset.write_lance = write_lance
