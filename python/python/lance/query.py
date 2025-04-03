# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors


import abc
from enum import Enum
from typing import Optional

from .lance import PyFullTextQuery


class FullTextQueryType(Enum):
    MATCH = "match"
    MATCH_PHRASE = "match_phrase"
    BOOST = "boost"
    MULTI_MATCH = "multi_match"


class FullTextOperator(Enum):
    AND = "AND"
    OR = "OR"


class FullTextQuery(abc.ABC):
    _inner: PyFullTextQuery

    @property
    def inner(self) -> PyFullTextQuery:
        """
        Get the inner query object.

        Returns
        -------
        PyFullTextQuery
            The inner query object.
        """
        return self._inner

    @abc.abstractmethod
    def query_type(self) -> FullTextQueryType:
        """
        Get the query type of the query.

        Returns
        -------
        str
            The type of the query.
        """


class MatchQuery(FullTextQuery):
    def __init__(
        self,
        query: str,
        column: str,
        *,
        boost: float = 1.0,
        fuzziness: int = 0,
        max_expansions: int = 50,
        operator: FullTextOperator = FullTextOperator.OR,
    ):
        """
        Match query for full-text search.

        Parameters
        ----------
        query : str
            The query string to match against.
        column : str
            The name of the column to match against.
        boost : float, default 1.0
            The boost factor for the query.
            The score of each matching document is multiplied by this value.
        fuzziness : int, optional
            The maximum edit distance for each term in the match query.
            Defaults to 0 (exact match).
            If None, fuzziness is applied automatically by the rules:
                - 0 for terms with length <= 2
                - 1 for terms with length <= 5
                - 2 for terms with length > 5
        max_expansions : int, optional
            The maximum number of terms to consider for fuzzy matching.
            Defaults to 50.
        """
        self._inner = PyFullTextQuery.match_query(
            query,
            column,
            boost=boost,
            fuzziness=fuzziness,
            max_expansions=max_expansions,
            operator=operator.value,
        )

    def query_type(self) -> FullTextQueryType:
        return FullTextQueryType.MATCH


class PhraseQuery(FullTextQuery):
    def __init__(self, query: str, column: str):
        """
        Phrase query for full-text search.

        Parameters
        ----------
        query : str
            The query string to match against.
        column : str
            The name of the column to match against.
        """
        self._inner = PyFullTextQuery.phrase_query(query, column)

    def query_type(self) -> FullTextQueryType:
        return FullTextQueryType.MATCH_PHRASE


class BoostQuery(FullTextQuery):
    def __init__(
        self,
        positive: FullTextQuery,
        negative: FullTextQuery,
        *,
        negative_boost: float = 0.5,
    ):
        """
        Boost query for full-text search.

        Parameters
        ----------
        positive : dict
            The positive query object.
        negative : dict
            The negative query object.
        negative_boost : float, default 0.5
            The boost factor for the negative query.
        """
        self._inner = PyFullTextQuery.boost_query(
            positive.inner, negative.inner, negative_boost
        )

    def query_type(self) -> FullTextQueryType:
        return FullTextQueryType.BOOST


class MultiMatchQuery(FullTextQuery):
    def __init__(
        self,
        query: str,
        columns: list[str],
        *,
        boosts: Optional[list[float]] = None,
        operator: FullTextOperator = FullTextOperator.OR,
    ):
        """
        Multi-match query for full-text search.

        Parameters
        ----------
        query : str | list[Query]
            If a string, the query string to match against.

        columns : list[str]
            The list of columns to match against.

        boosts : list[float], optional
            The list of boost factors for each column. If not provided,
            all columns will have the same boost factor.
        operator : FullTextOperator, default OR
            The operator to use for combining the query results.
            Can be either `AND` or `OR`.
            It would be applied to all columns individually.
            For example, if the operator is `AND`,
            then the query "hello world" is equal to
            `match("hello AND world", column1) OR match("hello AND world", column2)`.
        """
        self._inner = PyFullTextQuery.multi_match_query(
            query, columns, boosts=boosts, operator=operator.value
        )

    def query_type(self) -> FullTextQueryType:
        return FullTextQueryType.MULTI_MATCH
