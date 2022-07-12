# distutils: language = c++

from typing import Union

from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from pathlib import Path
from pyarrow import Table
from pyarrow._dataset cimport FileFormat
from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow cimport CTable, COutputStream
from pyarrow.includes.libarrow_dataset cimport CFileFormat
from pyarrow.lib cimport pyarrow_unwrap_table, check_status, get_writer


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
