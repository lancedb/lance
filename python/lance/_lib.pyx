from libcpp.memory cimport shared_ptr
from pyarrow.includes.libarrow_dataset cimport CFileFormat
from pyarrow._dataset cimport FileFormat


cdef extern from "lance/arrow/file_lance.h" namespace "lance" nogil:

    cdef cppclass CLanceFileFormat "::lance::arrow::LanceFileFormat"(
            CFileFormat):
        pass


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

