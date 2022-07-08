from typing import Union

import pyarrow as pa
import pyarrow.dataset as ds
from lance.lib import LanceFileFormat

__all__ = ["dataset", "write_table"]


def dataset(uri: str):
    """
    Create an Arrow Dataset from the given lance uri.

    Parameters
    ----------
    uri: str
        The uri to the lance data
    """
    fmt = LanceFileFormat()
    return ds.dataset(uri, format=fmt)


def write_table(table: pa.Table, destination: str):
    """Write an Arrow Table into the destination.

    Parameters
    ----------
    table : pa.Table
        Apache Arrow Table
    sink : 

    """
    pass
