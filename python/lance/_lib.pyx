# distutils: language = c++

from typing import Union, Optional

from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from pathlib import Path
from pyarrow import Table
from pyarrow._dataset cimport FileFormat, CScanner, CDataset, Dataset
from pyarrow._dataset import Scanner
from pyarrow.includes.common cimport *
from pyarrow._compute cimport Expression, _bind
from pyarrow.includes.libarrow cimport CTable, COutputStream
from pyarrow.includes.libarrow_dataset cimport CFileFormat
from pyarrow.lib cimport GetResultValue, check_status, pyarrow_unwrap_table, get_writer, RecordBatchReader, CExpression
from pyarrow.lib import tobytes


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


cdef extern from "lance/arrow/writer.h" namespace "lance::arrow" nogil:
    CStatus CWriteTable "::lance::arrow::WriteTable"(
            const CTable& table,
            shared_ptr[COutputStream] sink,
            const c_string& primary_key,
            optional[CFileWriteOptions] options)


cdef extern from "lance/arrow/scanner.h" namespace "lance::arrow" nogil:
    cdef cppclass LScannerBuilder "::lance::arrow::ScannerBuilder":
        LScannerBuilder(shared_ptr[CDataset]) except +
        void Project(const vector[string]& columns)
        void Filter(CExpression filter)
        void Limit(int64_t limit, int64_t offset)
        CResult[shared_ptr[CScanner]] Finish()

def BuildScanner(
        dataset: Dataset,
        columns: Optional[list[str]] = None,
        filter: Optional[Expression] = None,
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
    if limit is not None:
        builder.get().Limit(limit, offset)

    scanner = GetResultValue(builder.get().Finish())
    creader = GetResultValue(scanner.get().ToRecordBatchReader())
    reader = RecordBatchReader()
    reader.reader = creader
    return Scanner.from_batches(reader)


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
