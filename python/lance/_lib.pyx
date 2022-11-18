# distutils: language = c++

from typing import Callable, Optional, List, Dict
from pathlib import Path

import pyarrow
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.memory cimport shared_ptr, static_pointer_cast
from libcpp.string cimport string
from libc.stdint cimport uint64_t, int64_t

from pyarrow import Table
from pyarrow._dataset cimport (
    CDataset,
    CFileWriteOptions,
    CScanner,
    Dataset,
    FileFormat,
    FileWriteOptions,
)
from pyarrow._dataset import Scanner, _forbid_instantiation
from pyarrow._compute cimport Expression, _bind
from pyarrow._fs cimport FileSystem
from pyarrow.includes.common cimport CStatus, CResult, vector, move
from pyarrow.includes.libarrow_dataset cimport (
    CFileFormat,
    CFileSystemDatasetWriteOptions,
)
from pyarrow.includes.libarrow_fs cimport CFileSystem
from pyarrow.lib cimport (
    CArray,
    CExpression,
    CField,
    CRecordBatch,
    Field,
    GetResultValue,
    RecordBatchReader,
    check_status,
    pyarrow_wrap_batch,
    pyarrow_unwrap_field,
    pyarrow_unwrap_array,
)
from pyarrow.lib import tobytes
from pyarrow.util import _stringify_path

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


cdef extern from "lance/arrow/updater.h" namespace "lance::arrow" nogil:
    cdef cppclass CUpdater "::lance::arrow::Updater":
        CResult[shared_ptr[CRecordBatch]] Next();

        CStatus UpdateBatch(const shared_ptr[CArray] arr);

        CResult[shared_ptr[CLanceDataset]] Finish();

    cdef cppclass CUpdaterBuilder "::lance::arrow::UpdaterBuilder":
        CResult[shared_ptr[CUpdater]] Finish();


cdef class Updater:
    cdef shared_ptr[CUpdater] sp_updater

    @staticmethod
    cdef wrap(const shared_ptr[CUpdater]& up):
        cdef Updater self = Updater.__new__(Updater)
        self.sp_updater = move(up)
        return self

    def update_batch(self, data: pyarrow.Array):
        cdef shared_ptr[CArray] arr = pyarrow_unwrap_array(data)
        with nogil:
            check_status(self.sp_updater.get().UpdateBatch(arr))

    def finish(self):
        cdef shared_ptr[CLanceDataset] new_dataset = GetResultValue(
            self.sp_updater.get().Finish()
        )
        return FileSystemDataset.wrap(static_pointer_cast[CDataset, CLanceDataset](new_dataset))

    def __iter__(self):
        return self

    def __next__(self) -> pyarrow.Table:
        cdef shared_ptr[CRecordBatch] c_batch
        c_batch = GetResultValue(self.sp_updater.get().Next())
        if c_batch.get() == NULL:
            raise StopIteration
        batch = pyarrow_wrap_batch(c_batch)
        return pyarrow.Table.from_batches([batch])


cdef extern from "lance/arrow/dataset.h" namespace "lance::arrow" nogil:
    cdef cppclass CDatasetVersion "::lance::arrow::DatasetVersion":
        uint64_t version() const;

    cdef cppclass CLanceDataset "::lance::arrow::LanceDataset":
        enum WriteMode "WriteMode":
            CREATE "::lance::arrow::LanceDataset::WriteMode::kCreate"
            APPEND "::lance::arrow::LanceDataset::WriteMode::kAppend"
            OVERWRITE "::lance::arrow::LanceDataset::WriteMode::kOverwrite"

        @staticmethod
        CStatus Write(
                const CFileSystemDatasetWriteOptions& write_options,
                shared_ptr[CDataset] dataset,
                WriteMode mode)

        @staticmethod
        CResult[shared_ptr[CLanceDataset]] Make(
                const shared_ptr[CFileSystem]& fs,
                const string& base_uri,
                optional[uint64_t] version,
        )

        CDatasetVersion version() const;

        CResult[CDatasetVersion] latest_version() const;

        CResult[vector[CDatasetVersion]] versions() const;

        CResult[shared_ptr[CUpdaterBuilder]] NewUpdate(const shared_ptr[CField]& field) const;

cdef _dataset_version_to_json(CDatasetVersion cdv):
    return {
        "version": cdv.version(),
    }

cdef class FileSystemDataset(Dataset):
    """Lance Dataset.
    """
    cdef:
        CLanceDataset * lance_dataset

    def __init__(self):
        _forbid_instantiation(self.__class__)

    cdef void init(self, const shared_ptr[CDataset]& sp):
        Dataset.init(self, sp)
        self.lance_dataset = <CLanceDataset *> sp.get()

    @staticmethod
    cdef wrap(const shared_ptr[CDataset]& sp):
        cdef FileSystemDataset ds = FileSystemDataset.__new__(FileSystemDataset)
        ds.init(sp)
        return ds

    def head(self, n: int, offset: int = 0, **kwargs) -> Table:
        scanner = self.scanner(limit=n, offset=offset, **kwargs)
        return scanner.to_table()

    def scanner(self, *args, **kwargs):
        return BuildScanner(self, *args, **kwargs)

    def versions(self) -> List[Dict]:
        """Fetch all versions of this dataset."""
        cdef:
            vector[CDatasetVersion] c_versions
        c_versions = GetResultValue(self.lance_dataset.versions())

        return [_dataset_version_to_json(cv) for cv in c_versions]

    @property
    def version(self) -> Dict:
        """Get the current version of the dataset."""
        cdef:
            CDatasetVersion c_version
        c_version = self.lance_dataset.version()
        return _dataset_version_to_json(c_version)

    def latest_version(self) -> Dict:
        """Get the latest version of the dataset."""
        c_version = GetResultValue(self.lance_dataset.latest_version())
        return _dataset_version_to_json(c_version)

    def append_column(self, field: Field, func: Callable[[pyarrow.Table], pyarrow.Array]) -> FileSystemDataset:
        """Append a new column.

        Parameters
        ----------
        field : pyarrow.Field
            The name and schema of the newly added column.
        func : Callback[[pyarrow.Table], pyarrow.Array]
            A function / callback that takes in a Batch and produces an Array. The generated array must
            have the same length as the input batch.
        """
        cdef:
            shared_ptr[CUpdater] c_updater
            shared_ptr[CField] c_field

        c_field = pyarrow_unwrap_field(field)
        c_updater = move(GetResultValue(GetResultValue(move(self.lance_dataset.NewUpdate(c_field))).get().Finish()))
        updater = Updater.wrap(c_updater)
        for table in updater:
            arr = func(table)
            updater.update_batch(arr)
        return updater.finish()

def _lance_dataset_write(
        Dataset data,
        object base_dir not None,
        FileSystem filesystem not None,
        str mode not None,
        int max_rows_per_file,
        int min_rows_per_group,
        int max_rows_per_group,
):
    """Wraps 'LanceDataset::Write'.

    Parameters
    ----------
    data : pyarrow.dataset.Dataset
        Dataset to write.
    base_dir : str or Path
        The directory to write dataset to.
    filesystem : pyarrow.fs.FileSystem
        Pyarrow File System instance to write dataset.
    mode : str
        Write mode, can be "create", "append" or "overwrite".
    max_rows_per_file : int
    min_rows_per_group : int
    max_rows_per_group : int
    """

    cdef:
        CFileSystemDatasetWriteOptions c_options
        shared_ptr[CScanner] c_scanner
        shared_ptr[CDataset] c_dataset
        CLanceDataset.WriteMode c_mode
        FileWriteOptions write_options

    c_dataset = data.unwrap()

    fmt = LanceFileFormat()
    write_options = fmt.make_write_options()

    c_options.base_dir = tobytes(_stringify_path(base_dir))
    c_options.filesystem = filesystem.unwrap()
    c_options.file_write_options = write_options.unwrap()
    c_options.create_dir = True
    if max_rows_per_file > 0:
        c_options.max_rows_per_file = max_rows_per_file
    if min_rows_per_group > 0:
        c_options.min_rows_per_group = min_rows_per_group
    c_options.max_rows_per_group = max_rows_per_group

    # Default value
    c_mode = CLanceDataset.WriteMode.CREATE
    if mode == "append":
        c_mode = CLanceDataset.WriteMode.APPEND
    elif mode == "overwrite":
        c_mode = CLanceDataset.WriteMode.OVERWRITE

    with nogil:
        check_status(CLanceDataset.Write(c_options, c_dataset, c_mode))

def _lance_dataset_make(
        FileSystem filesystem not None,
        object base_uri not None,
        bool has_version,
        uint64_t version
):
    cdef:
        shared_ptr[CLanceDataset] c_dataset
        shared_ptr[CDataset] c_base_dataset
        shared_ptr[CFileSystem] c_filesystem
        optional[uint64_t] c_version

    c_filesystem = filesystem.unwrap()
    base_dir = tobytes(_stringify_path(base_uri))
    if has_version:
        c_version = optional[uint64_t](version)

    c_dataset = GetResultValue(CLanceDataset.Make(c_filesystem, base_dir, c_version))
    c_base_dataset = static_pointer_cast[CDataset, CLanceDataset](c_dataset)
    return FileSystemDataset.wrap(move(c_base_dataset))
