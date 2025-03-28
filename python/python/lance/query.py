# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors


import abc
from enum import Enum
from typing import Optional


class FullTextQueryType(Enum):
    MATCH = "match"
    MATCH_PHRASE = "match_phrase"
    BOOST = "boost"
    MULTI_MATCH = "multi_match"


class FullTextQuery(abc.ABC):
    @abc.abstractmethod
    def query_type(self) -> FullTextQueryType:
        """
        Get the query type of the query.

        Returns
        -------
        str
            The type of the query.
        """

    @abc.abstractmethod
    def to_dict(self) -> dict:
        """
        Convert the query to a dictionary.

        Returns
        -------
        dict
            The query as a dictionary.
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
        self.column = column
        self.query = query
        self.boost = boost
        self.fuzziness = fuzziness
        self.max_expansions = max_expansions

    def query_type(self) -> FullTextQueryType:
        return FullTextQueryType.MATCH

    def to_dict(self) -> dict:
        return {
            "match": {
                self.column: {
                    "query": self.query,
                    "boost": self.boost,
                    "fuzziness": self.fuzziness,
                    "max_expansions": self.max_expansions,
                }
            }
        }


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
        self.column = column
        self.query = query

    def query_type(self) -> FullTextQueryType:
        return FullTextQueryType.MATCH_PHRASE

    def to_dict(self) -> dict:
        return {
            "match_phrase": {
                self.column: self.query,
            }
        }


class BoostQuery(FullTextQuery):
    def __init__(
        self,
        positive: FullTextQuery,
        negative: FullTextQuery,
        negative_boost: float,
    ):
        """
        Boost query for full-text search.

        Parameters
        ----------
        positive : dict
            The positive query object.
        negative : dict
            The negative query object.
        negative_boost : float
            The boost factor for the negative query.
        """
        self.positive = positive
        self.negative = negative
        self.negative_boost = negative_boost

    def query_type(self) -> FullTextQueryType:
        return FullTextQueryType.BOOST

    def to_dict(self) -> dict:
        return {
            "boost": {
                "positive": self.positive.to_dict(),
                "negative": self.negative.to_dict(),
                "negative_boost": self.negative_boost,
            }
        }


class MultiMatchQuery(FullTextQuery):
    def __init__(
        self,
        query: str,
        columns: list[str],
        *,
        boosts: Optional[list[float]] = None,
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
        """
        self.query = query
        self.columns = columns
        if boosts is None:
            boosts = [1.0] * len(columns)
        self.boosts = boosts

    def query_type(self) -> FullTextQueryType:
        return FullTextQueryType.MULTI_MATCH

    def to_dict(self) -> dict:
        return {
            "multi_match": {
                "query": self.query,
                "columns": self.columns,
                "boost": self.boosts,
            }
        }
