# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
import pyarrow as pa

from .dataset import LanceDataset
from .lance import SqlQueryBuilder as LanceSqlQueryBuilder


class SqlQueryBuilder:
    """A tool for building SQL queries"""

    def __init__(self, query_str: str):
        """Create a new SQL query builder from a query string."""
        self.inner = LanceSqlQueryBuilder(query_str)

    def with_dataset(self, alias: str, ds: LanceDataset) -> "SqlQueryBuilder":
        """Adds a dataset to the query's context with a given alias."""
        self.inner.with_dataset(alias, ds._ds)
        return self

    def to_table(self) -> pa.Table:
        """Execute the query and return the result as a table."""
        return self.inner.execute().read_all()


def query(query_str: str) -> SqlQueryBuilder:
    """
    Create an SQL query builder from a query string.

    Note: This is an experimental feature.  The API may change in future
    versions or be removed entirely.  The most stable way to execute SQL
    against Lance datasets is to use another tool such as DuckDB.  This
    tool is primarily intended for simple exploration and prototyping.

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
    >>> query = lance.query("SELECT SUM(a) FROM d1 WHERE b > 4")
    >>> table = query.with_dataset("d1", ds).to_table()
    >>> print(table)
    # pyarrow.Table
    a
    3
    """
    return SqlQueryBuilder(query_str)
