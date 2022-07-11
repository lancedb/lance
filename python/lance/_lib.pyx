# distutils: language = c++

from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow_dataset cimport CFileFormat
from pyarrow.includes.libarrow cimport CTable, COutputStream
from pyarrow._dataset cimport FileFormat


cdef extern from "<optional>" namespace "std" nogil:
    # Backport https://github.com/cython/cython/blob/master/Cython/Includes/libcpp/optional.pxd
    # before cython 3.0 release
    cdef cppclass nullopt_t:
        nullopt_t()

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
    CStatus WriteTable(const CTable& table,
                       shared_ptr[COutputStream] sink,
                       const string& primary_key,
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
