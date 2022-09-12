# distutils: language = c++

from typing import Optional, Union

from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.memory cimport const_pointer_cast, shared_ptr
from libcpp.string cimport string

from pathlib import Path

from pyarrow import Table

from pyarrow._dataset cimport (
    CDataset,
    CFileWriteOptions,
    CScanner,
    Dataset,
    FileFormat,
    FileFragment,
    FileWriteOptions,
    Partitioning,
)

from pyarrow._dataset import Scanner, _forbid_instantiation

from pyarrow._compute cimport Expression, _bind
from pyarrow._fs cimport FileSystem
from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow cimport COutputStream, CTable
from pyarrow.includes.libarrow_dataset cimport (
    CFileFormat,
    CFileFragment,
    CFileSystem,
    CFileSystemDataset,
    CFragment,
)
from pyarrow.lib cimport (
    CExpression,
    GetResultValue,
    RecordBatchReader,
    Schema,
    check_status,
    get_writer,
    pyarrow_unwrap_table,
)

from pyarrow.lib import frombytes, tobytes


cdef Expression _true = Expression._scalar(True)

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

    cdef cppclass CLanceFileWriteOptions "::lance::arrow::FileWriteOptions":
        CLanceFileWriteOptions() except +
        int batch_size


cdef extern from "lance/arrow/writer.h" namespace "lance::arrow" nogil:
    CStatus CWriteTable "::lance::arrow::WriteTable"(
        const CTable& table,
        shared_ptr[COutputStream] sink,
        CLanceFileWriteOptions options)


cdef extern from "lance/arrow/scanner.h" namespace "lance::arrow" nogil:
    cdef cppclass LScannerBuilder "::lance::arrow::ScannerBuilder":
        LScannerBuilder(shared_ptr[CDataset]) except +
        CStatus Project(const vector[string]& columns)
        CStatus Filter(CExpression filter)
        CStatus BatchSize(int64_t batch_size)
        CStatus Limit(int64_t limit, int64_t offset)
        CResult[shared_ptr[CScanner]] Finish()

def BuildScanner(
    dataset: Dataset,
    columns: Optional[list[str]] = None,
    filter: Optional[Expression] = None,
    batch_size: Optional[int] = None,
    limit: Optional[int] = None,
    offset: int = 0,
):
    cdef shared_ptr[CDataset] cdataset = dataset.unwrap()
    cdef shared_ptr[LScannerBuilder] builder = shared_ptr[LScannerBuilder](
        new LScannerBuilder(cdataset))

    if columns:
        builder.get().Project([tobytes(c) for c in columns])
    if filter is not None:
        builder.get().Filter(_bind(filter, dataset.schema))
    if batch_size is not None:
        builder.get().BatchSize(batch_size)
    if limit is not None:
        builder.get().Limit(limit, offset)
    scanner = GetResultValue(builder.get().Finish())
    creader = GetResultValue(scanner.get().ToRecordBatchReader())
    reader = RecordBatchReader()
    reader.reader = creader
    return Scanner.from_batches(reader)

cdef class LanceFileWriteOptions(FileWriteOptions):
    @staticmethod
    cdef wrap(const shared_ptr[CFileWriteOptions]& sp):
        cdef LanceFileWriteOptions self = LanceFileWriteOptions.__new__(LanceFileWriteOptions)
        self.init(sp)
        return self

    @property
    def format(self):
        return LanceFileFormat()

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

    def make_write_options(self):
        return LanceFileWriteOptions.wrap(self.format.DefaultWriteOptions())

def WriteTable(table: Table,
               sink: Union[str, Path],
               batch_size: int):
    arrow_table = pyarrow_unwrap_table(table)
    cdef shared_ptr[COutputStream] out
    get_writer(sink, &out)

    cdef CLanceFileWriteOptions options = CLanceFileWriteOptions()
    options.batch_size = batch_size
    with nogil:
        check_status(CWriteTable(deref(arrow_table), out, options))

cdef class FileSystemDataset(Dataset):
    """
    A Dataset of Lance fragments.

    A LanceDataset is composed of one or more FileFragment.

    Parameters
    ----------
    fragments : list[Fragments]
        List of fragments to consume.
    schema : Schema
        The top-level schema of the Dataset.
    filesystem : FileSystem
        FileSystem of the fragments.
    root_partition : Expression, optional
        The top-level partition of the DataDataset.
    """

    cdef:
        CFileSystemDataset * filesystem_dataset

    def __init__(self):
        _forbid_instantiation(self.__class__)

    @property
    def filesystem(self):
        return FileSystem.wrap(self.filesystem_dataset.filesystem())

    @property
    def partitioning(self):
        """
        The partitioning of the Dataset source, if discovered.

        If the FileSystemDataset is created using the ``dataset()`` factory
        function with a partitioning specified, this will return the
        finalized Partitioning object from the dataset discovery. In all
        other cases, this returns None.
        """
        c_partitioning = self.filesystem_dataset.partitioning()
        if c_partitioning.get() == nullptr:
            return None
        try:
            return Partitioning.wrap(c_partitioning)
        except TypeError:
            # e.g. type_name "default"
            return None

    cdef void init(self, const shared_ptr[CDataset]& sp):
        Dataset.init(self, sp)
        self.filesystem_dataset = <CFileSystemDataset *> sp.get()

    @staticmethod
    cdef wrap(const shared_ptr[CDataset]& sp):
        cdef Dataset ds = FileSystemDataset.__new__(FileSystemDataset)
        ds.init(sp)
        return ds

    def __reduce__(self):
        return FileSystemDataset, (
            list(self.get_fragments()),
            self.schema,
            self.format,
            self.filesystem,
            self.partition_expression
        )

    @classmethod
    def from_paths(cls, paths, schema=None, format=None,
                   filesystem=None, partitions=None, root_partition=None):
        """A Dataset created from a list of paths on a particular filesystem.

        Parameters
        ----------
        paths : list of str
            List of file paths to create the fragments from.
        schema : Schema
            The top-level schema of the DataDataset.
        format : FileFormat
            File format to create fragments from, currently only
            ParquetFileFormat, IpcFileFormat, and CsvFileFormat are supported.
        filesystem : FileSystem
            The filesystem which files are from.
        partitions : list[Expression], optional
            Attach additional partition information for the file paths.
        root_partition : Expression, optional
            The top-level partition of the DataDataset.
        """
        cdef:
            FileFragment fragment

        if root_partition is None:
            root_partition = _true

        for arg, class_, name in [
            (schema, Schema, 'schema'),
            (format, FileFormat, 'format'),
            (filesystem, FileSystem, 'filesystem'),
            (root_partition, Expression, 'root_partition')
        ]:
            if not isinstance(arg, class_):
                raise TypeError(
                    "Argument '{0}' has incorrect type (expected {1}, "
                    "got {2})".format(name, class_.__name__, type(arg))
                )

        partitions = partitions or [_true] * len(paths)

        if len(paths) != len(partitions):
            raise ValueError(
                'The number of files resulting from paths_or_selector '
                'must be equal to the number of partitions.'
            )

        fragments = [
            format.make_fragment(path, filesystem, partitions[i])
            for i, path in enumerate(paths)
        ]
        return FileSystemDataset(fragments, schema, format,
                                 filesystem, root_partition)

    @property
    def files(self):
        """List of the files"""
        cdef vector[c_string] files = self.filesystem_dataset.files()
        return [frombytes(f) for f in files]

    @property
    def format(self):
        """The FileFormat of this source."""
        cdef FileFormat format = LanceFileFormat.__new__(LanceFileFormat)
        format.init(self.filesystem_dataset.format())
        return format

    def head(self, n: int, offset: int = 0) -> Table:
        scanner = self.scanner(limit=n, offset=offset)
        return scanner.to_table()

    def scanner(self, *args, **kwargs):
        return BuildScanner(self, *args, **kwargs)

def _wrap_dataset(Dataset dataset not None):
    cdef shared_ptr[CDataset] copy = dataset.unwrap()
    return FileSystemDataset.wrap(move(copy))
