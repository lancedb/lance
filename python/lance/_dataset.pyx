# distutils: language = c++

cdef extern from "lance/arrow/dataset.h" namespace "lance::arrow" nogil:
    cdef cppclass CLanceDataset "::lance::arrow::LanceDataset":
        pass


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


def write_dataset():
    """"""
    pass