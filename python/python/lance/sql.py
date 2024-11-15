# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
import pyarrow as pa

from .dataset import LanceDataset
from .lance import SqlQueryBuilder as LanceSqlQueryBuilder


class SqlQueryBuilder:
    def __init__(self, query_str: str):
        self.inner = LanceSqlQueryBuilder(query_str)

    def with_lance_dataset(self, ds: LanceDataset) -> "SqlQueryBuilder":
        self.inner.with_lance_dataset(ds._ds)
        return self

    def to_table(self) -> pa.Table:
        return self.inner.execute().read_all()


def query(query_str: str) -> SqlQueryBuilder:
    """
    Create an SQL query builder from a query string.

    Parameters
    ----------
    query_str: str
        The SQL query string.

    Returns
    -------
    SqlQueryBuilder
        The SQL query builder object.

    Examples
    --------
    >>> import pyarrow as pa
    >>> import lance
    >>>
    >>> ds = lance.write_dataset(pa.table("a", [1, 2, 3], "b": [4, 5, 6]))
    >>> query = lance.query("SELECT SUM(a) FROM ds WHERE b > 4")
    >>> table = query.to_table()
    >>> print(table)
    # pyarrow.Table
    a
    3
    """
    return SqlQueryBuilder(query_str)
