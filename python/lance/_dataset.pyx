# distutils: language = c++

from typing import List, Dict

from pyarrow._dataset cimport Dataset, CDataset
from pyarrow._dataset import _forbid_instantiation, CScanner, Scanner
from pyarrow._fs cimport FileSystem
from pyarrow.lib import tobytes
from pyarrow.lib cimport check_status
from pyarrow.includes.common cimport CStatus
from pyarrow.includes.libarrow_dataset cimport CScanner, CFileSystemDatasetWriteOptions
from pyarrow.util import _stringify_path
from libcpp.memory cimport shared_ptr
from pyarrow import Table

from lance._lib import BuildScanner

cdef extern from "lance/arrow/dataset.h" namespace "lance::arrow" nogil:
    cdef cppclass CLanceDataset "::lance::arrow::LanceDataset":
        enum WriteMode "::lance::arrow::LanceDataset::WriteMode":
            CREATE "::lance::arrow::LanceDataset::WriteMode::kCreate"
            pass

        @ staticmethod
        CStatus Write(
                const CFileSystemDatasetWriteOptions& write_options,
                shared_ptr[CScanner] scanner,
                WriteMode mode)



cdef class LanceDataset(Dataset):
    """Lance Dataset.
    """
    cdef:
        CLanceDataset * lance_dataset

    def __init__(self):
        _forbid_instantiation(self.__class__)

    @staticmethod
    cdef wrap(const shared_ptr[CDataset]& sp):
        # cdef Dataset ds = FileSystemDataset.__new__(FileSystemDataset)
        # ds.init(sp)
        # return ds
        return None

    def head(self, n: int, offset: int = 0) -> Table:
        scanner = self.scanner(limit=n, offset=offset)
        return scanner.to_table()

    def scanner(self, *args, **kwargs):
        return BuildScanner(self, *args, **kwargs)

    def versions(self) -> List[Dict]:
        """Fetch all versions of this dataset."""
        pass

def _lance_dataset_write(
        Scanner data not None,
        object base_dir not None,
        FileSystem filesystem not None,
        str mode not None,
):
    """Wraps 'LanceDataset::Write'. """

    cdef:
        CFileSystemDatasetWriteOptions c_options
        shared_ptr[CScanner] c_scanner
        CLanceDataset.WriteMode c_mode

    c_options.base_dir = tobytes(_stringify_path(base_dir))
    c_options.filesystem = filesystem.unwrap()
    c_options.create_dir = True

    if mode == "create":
        c_mode = CLanceDataset.WriteMode.CREATE
    c_scanner = data.unwrap()
    with nogil:
        check_status(CLanceDataset.Write(c_options, c_scanner, c_mode))
