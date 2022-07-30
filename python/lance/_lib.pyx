# distutils: language = c++

from typing import Union, Optional
import collections

from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from pathlib import Path
import pyarrow as pa
from pyarrow import Table
from pyarrow._dataset cimport FileFormat, Dataset, CTaggedRecordBatchIterator, Fragment
from pyarrow._compute cimport Expression, _bind
from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow cimport CTable, COutputStream, CExpression
from pyarrow.includes.libarrow_dataset cimport CFileFormat, CDataset, CScanner, CTaggedRecordBatch
from pyarrow.lib cimport *


cdef extern from "<optional>" namespace "std" nogil:
    # Backport https://github.com/cython/cython/blob/master/Cython/Includes/libcpp/optional.pxd
    # before cython 3.0 release
    cdef cppclass nullopt_t:
        nullopt_t()

    cdef nullopt_t nullopt

    cdef cppclass optional[T]:
        ctypedef T value_type
        optional()
        optional(nullopt_t)
        optional(optional &) except +
        optional(T &) except +
        bool has_value()
        T& value()
        T& value_or[U](U& default_value)
        void swap(optional &)
        void reset()
        T& emplace(...)
        T& operator *()
        #T* operator->() # Not Supported
        optional& operator=(optional &)
        optional& operator=[U](U &)


cdef extern from "lance/arrow/file_lance.h" namespace "lance" nogil:
    cdef cppclass CLanceFileFormat "::lance::arrow::LanceFileFormat"(
        CFileFormat):
        pass

    cdef cppclass CFileWriteOptions "::lance::arrow::FileWriteOptions"


cdef extern from "lance/arrow/scanner.h" namespace "lance::arrow" nogil:
    cdef cppclass LCScannerBuilder "::lance::arrow::ScannerBuilder":
        LCScannerBuilder(shared_ptr[CDataset]) except +
        void Project(const vector[string]& columns)
        void Filter(CExpression filter)
        void Limit(int64_t limit, int64_t offset)
        CResult[shared_ptr[CScanner]] Finish()


class TaggedRecordBatch(collections.namedtuple(
        "TaggedRecordBatch", ["record_batch", "fragment"])):
    """
    Copied from pyarrow._dataset.pyx
    """

cdef class TaggedRecordBatchIterator(_Weakrefable):
    """Copied from pyarrow._dataset.pyx"""
    cdef:
        object iterator_owner
        shared_ptr[CTaggedRecordBatchIterator] iterator

    def __init__(self):
        raise TypeError("Can not run py-init")

    @staticmethod
    cdef wrap(object owner, CTaggedRecordBatchIterator iterator):
        cdef TaggedRecordBatchIterator self = \
            TaggedRecordBatchIterator.__new__(TaggedRecordBatchIterator)
        self.iterator_owner = owner
        self.iterator = make_shared[CTaggedRecordBatchIterator](
            move(iterator))
        return self

    def __iter__(self):
        return self

    def __next__(self):
        cdef CTaggedRecordBatch batch
        with nogil:
            batch = GetResultValue(move(self.iterator.get().Next()))
        if batch.record_batch == NULL:
            raise StopIteration
        return TaggedRecordBatch(
            record_batch=pyarrow_wrap_batch(batch.record_batch),
            fragment=Fragment.wrap(batch.fragment))

# We have to copy Scanner implement since it is not public API.
cdef class Scanner(_Weakrefable):
    """A materialized scan operation with context and options bound.

    A scanner is the class that glues the scan tasks, data fragments and data
    sources together.

    Parameters
    ----------
    dataset : Dataset
        Dataset to scan.
    columns : list of str or dict, default None
        The columns to project. This can be a list of column names to
        include (order and duplicates will be preserved), or a dictionary
        with {{new_column_name: expression}} values for more advanced
        projections.

        The list of columns or expressions may use the special fields
        `__batch_index` (the index of the batch within the fragment),
        `__fragment_index` (the index of the fragment within the dataset),
        `__last_in_fragment` (whether the batch is last in fragment), and
        `__filename` (the name of the source file or a description of the
        source fragment).

        The columns will be passed down to Datasets and corresponding data
        fragments to avoid loading, copying, and deserializing columns
        that will not be required further down the compute chain.
        By default all of the available columns are projected.
        Raises an exception if any of the referenced column names does
        not exist in the dataset's Schema.
    filter : Expression, default None
        Scan will return only the rows matching the filter.
        If possible the predicate will be pushed down to exploit the
        partition information or internal metadata found in the data
        source, e.g. Parquet statistics. Otherwise filters the loaded
        RecordBatches before yielding them.
    batch_size : int, default 128Ki
        The maximum row count for scanned record batches. If scanned
        record batches are overflowing memory then this method can be
        called to reduce their size.
    use_threads : bool, default True
        If enabled, then maximum parallelism will be used determined by
        the number of available CPU cores.
    use_async : bool, default True
        This flag is deprecated and is being kept for this release for
        backwards compatibility.  It will be removed in the next release.
    memory_pool : MemoryPool, default None
        For memory allocations, if required. If not specified, uses the
        default pool.
    """

    cdef:
        shared_ptr[CScanner] wrapped
        CScanner* scanner

    def __init__(self):
        raise TypeError("Can not use py-init")

    cdef void init(self, const shared_ptr[CScanner]& sp):
        self.wrapped = sp
        self.scanner = sp.get()

    @staticmethod
    cdef wrap(const shared_ptr[CScanner]& sp):
        cdef Scanner self = Scanner.__new__(Scanner)
        self.init(sp)
        return self

    cdef inline shared_ptr[CScanner] unwrap(self):
        return self.wrapped

    @property
    def dataset_schema(self):
        """The schema with which batches will be read from fragments."""
        return pyarrow_wrap_schema(
            self.scanner.options().get().dataset_schema)

    @property
    def projected_schema(self):
        """
        The materialized schema of the data, accounting for projections.

        This is the schema of any data returned from the scanner.
        """
        return pyarrow_wrap_schema(
            self.scanner.options().get().projected_schema)

    def to_batches(self):
        """
        Consume a Scanner in record batches.

        Returns
        -------
        record_batches : iterator of RecordBatch
        """
        def _iterator(batch_iter):
            for batch in batch_iter:
                yield batch.record_batch
        # Don't make ourselves a generator so errors are raised immediately
        return _iterator(self.scan_batches())

    def scan_batches(self):
        """
        Consume a Scanner in record batches with corresponding fragments.

        Returns
        -------
        record_batches : iterator of TaggedRecordBatch
        """
        cdef CTaggedRecordBatchIterator iterator
        with nogil:
            iterator = move(GetResultValue(self.scanner.ScanBatches()))
        # Don't make ourselves a generator so errors are raised immediately
        return TaggedRecordBatchIterator.wrap(self, move(iterator))

    def to_table(self):
        """
        Convert a Scanner into a Table.

        Use this convenience utility with care. This will serially materialize
        the Scan result in memory before creating the Table.

        Returns
        -------
        Table
        """
        cdef CResult[shared_ptr[CTable]] result

        with nogil:
            result = self.scanner.ToTable()

        return pyarrow_wrap_table(GetResultValue(result))

    def take(self, object indices):
        """
        Select rows of data by index.

        Will only consume as many batches of the underlying dataset as
        needed. Otherwise, this is equivalent to
        ``to_table().take(indices)``.

        Parameters
        ----------
        indices : Array or array-like
            indices of rows to select in the dataset.

        Returns
        -------
        Table
        """
        cdef CResult[shared_ptr[CTable]] result
        cdef shared_ptr[CArray] c_indices

        if not isinstance(indices, pa.Array):
            indices = pa.array(indices)
        c_indices = pyarrow_unwrap_array(indices)

        with nogil:
            result = self.scanner.TakeRows(deref(c_indices))
        return pyarrow_wrap_table(GetResultValue(result))

    def head(self, int num_rows):
        """
        Load the first N rows of the dataset.

        Parameters
        ----------
        num_rows : int
            The number of rows to load.

        Returns
        -------
        Table
        """
        cdef CResult[shared_ptr[CTable]] result
        with nogil:
            result = self.scanner.Head(num_rows)
        return pyarrow_wrap_table(GetResultValue(result))

    def count_rows(self):
        """
        Count rows matching the scanner filter.

        Returns
        -------
        count : int
        """
        cdef CResult[int64_t] result
        with nogil:
            result = self.scanner.CountRows()
        return GetResultValue(result)


def BuildScanner(
        dataset: Dataset,
        columns: Optional[list[str]] = None,
        filter: Optional[Expression] = None,
        limit: Optional[int] = None,
        offset: int = 0,
):
    cdef shared_ptr[CDataset] cdataset = dataset.unwrap()
    cdef shared_ptr[LCScannerBuilder] builder = shared_ptr[LCScannerBuilder](
        new LCScannerBuilder(cdataset))

    if columns:
        builder.get().Project(columns)
    if filter is not None:
        builder.get().Filter(_bind(filter, dataset.schema()))
    if limit is not None:
        builder.get().Limit(limit, offset)

    scanner = GetResultValue(builder.get().Finish())
    return Scanner.wrap(scanner)


cdef extern from "lance/arrow/writer.h" namespace "lance::arrow" nogil:
    CStatus CWriteTable "::lance::arrow::WriteTable"(
            const CTable& table,
            shared_ptr[COutputStream] sink,
            const c_string& primary_key,
            optional[CFileWriteOptions] options)


cdef class LanceFileFormat(FileFormat):
    def __init__(self):
        self.init(shared_ptr[CFileFormat](new CLanceFileFormat()))

    def equals(self, LanceFileFormat other):
        return True

    @property
    def default_extname(self):
        return "lance"

    def __reduce__(self):
        return LanceFileFormat, tuple()

def WriteTable(table: Table,
               sink: Union[str, Path],
               primary_key: str):
    arrow_table = pyarrow_unwrap_table(table)
    cdef shared_ptr[COutputStream] out
    get_writer(sink, &out)
    cdef string pk = primary_key.encode("utf-8")

    cdef optional[CFileWriteOptions] options = nullopt
    with nogil:
        check_status(CWriteTable(deref(arrow_table), out, pk, options))
