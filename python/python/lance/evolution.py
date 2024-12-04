# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from typing import TYPE_CHECKING, Any, Dict, List

import pyarrow as pa

from . import LanceDataset, LanceOperation, write_dataset
from .fragment import write_fragments
from .lance.align import (
    AlignFragmentsPlan,
)
from .types import _coerce_reader

if TYPE_CHECKING:
    from .types import ReaderLike


class DataShard:
    """
    A shard of new data to be added to a dataset.

    This is returned by the ``add_data`` method of an ``AddColumnsJob``.  This should
    be collected and passed to the ``finish_adding_data`` method of the job.

    This should be treated as an opaque pickleable object.
    """

    def __init__(self, fragments):
        self._fragments = fragments


class AddColumnsJob:
    """
    A job to add new columns to a dataset.

    This job can be used to distribute the work of adding new columns to a dataset
    across multiple workers.  The job is created by calling the ``create`` method.
    This job operates in two phases.

    In the first phase the new data is calculated.  The new data is written to a
    temporary dataset.  Once all the new data is calculated the job enters the second
    phase where the new data is aligned with the target dataset.  The alignment phase
    reads from the temporary dataset and rewrites the data in fragments aligned with
    the target dataset.  Both phases can be distributed across multiple workers.

    The job itself should be pickled and sent to any number of workers.  The workers
    can then call the ``add_data`` method to create a data shard containing some new
    values for the new columns.  These data shards should be collected and passed to
    some finalization step which will call the ``finish_adding_data`` method to create
    an ``AlignColumnsPlan`` which can be used to commit the new data to the dataset.
    Details on the alignment phase can be found in the documentation for
    ``AlignColumnsPlan``.

    It is not required that a value be given for every row in the target dataset.  If a
    value is not given for a row then the value for the new columns will be set to null.
    New columns created using the ``AddColumnsJob`` will be appended to the end of the
    schema of the target dataset and will always be nullable columns.  If desired, then
    ``LanceDataset.alter_columns`` can be used to change the nullability of the new
    columns after they are added.

    Modifications to he target dataset can be tolerated while the new data is being.
    Any new rows added after the job has been distributed to workers will have null for
    the new columns.  New values calculated for rows that have since been deleted will
    be ignored.  Note: if a row is used to calculate new values, and then that row is
    updated before the new data is committed, then the new values (based on the old row)
    will still be inserted.
    """

    def __init__(
        self,
        target: LanceDataset,
        source: LanceDataset,
        join_key: str,
    ):
        self.target = target
        self.source = source
        self.join_key = join_key

    @staticmethod
    def create(
        target,
        source_uri: str,
        new_schema: pa.Schema,
        join_key: str,
        *,
        overwrite_tmp: bool = False,
    ) -> "AddColumnsJob":
        """
        Creates a new AddColumnsJob instance to add new columns to ``target``.

        Parameters
        ----------

        target : LanceDataset
            The dataset to which the new columns will be added.

        source_uri: str
            The URI of the temporary dataset to which the new data will be written.

            There must not be a dataset at this URI unless ``overwrite_tmp`` is set
            to True in which case the existing dataset will be overwritten.

            The temporary dataset must be in a location that is accessible to all
            workers that will be adding data to the job.

        new_schema : pa.Schema
            The schema of the new columns to be added.  This schema must contain
            the join key column.

        join_key : str
            The column used to join the new data with the existing data.  There are
            special values that can be used here that will affect the behavior of the
            job.

            If the join_key value is ``_dataset_offset`` then the new data should have
            a ``_dataset_offset`` column which indicates the offset of the data in the
            target dataset.

            If the join key value is ``_rowid`` then the new data should have a
            ``_rowid`` column which indicates the row id of the data in the target
            dataset.

            If the join key value is anything else then the both the new data and the
            target dataset must have a column with the same name.  A join will be
            performed on this column.
        """
        if new_schema.field(0).name != join_key:
            if any(f.name == join_key for f in new_schema):
                raise ValueError(
                    f"If the join_key ({join_key}) is in the new_schema it "
                    "must be the first column"
                )
            if join_key == "_rowid" or join_key == "_dataset_offset":
                new_schema = pa.schema(
                    [pa.field(join_key, pa.uint64())] + list(iter(new_schema))
                )
            else:
                # We can't infer type so we require it to be in the schema
                raise ValueError(
                    f"join_key ({join_key}) must be included in new_schema "
                    "when not using _rowid or _dataset_offset"
                )
        source = write_dataset(
            [],
            source_uri,
            schema=new_schema,
            mode="create",
            data_storage_version=target.data_storage_version,
        )
        return AddColumnsJob(target, source, join_key)

    @staticmethod
    def _disallow_arg(args: Dict[Any, Any], arg: str):
        if arg in args:
            raise TypeError(f"{arg} cannot be used in add_data")

    def add_data(self, data: "ReaderLike", **kwargs) -> DataShard:
        """
        Adds new data.  The new data must contain the join key column.

        Returns a DataShard.  This can be pickled and should be collected and
        passed to the ``finish_adding_data`` method.
        """
        # This must always be append
        self._disallow_arg(kwargs, "mode")
        # This is inherited from the base dataset
        self._disallow_arg(kwargs, "data_storage_version")
        # Soon to be deprecated
        self._disallow_arg(kwargs, "use_legacy_format")

        reader = _coerce_reader(data)
        first_col = reader.schema.field(0)
        if first_col.name != self.join_key:
            raise ValueError(
                "First column in the new data must be "
                f"the join key column {self.join_key}"
            )

        frags = write_fragments(data, self.source)
        return DataShard(frags)

    def finish_adding_data(
        self, shards: List[DataShard]
    ) -> (AlignFragmentsPlan, LanceDataset):
        all_frags = []
        for shard in shards:
            all_frags.extend(shard._fragments)

        op = LanceOperation.Append(all_frags)
        # TODO: Ideally we should fail here if finish_adding_data is called twice
        # but we don't have a good way to do that yet (two appends will be allowed)
        source = LanceDataset.commit(self.source, op, read_version=1)
        return (
            AlignFragmentsPlan.create(source._ds, self.target._ds, self.join_key),
            source,
        )
