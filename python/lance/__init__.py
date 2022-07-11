import pyarrow.dataset as ds
from lance.lib import LanceFileFormat


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
