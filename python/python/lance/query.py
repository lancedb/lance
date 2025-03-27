# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors


from typing import Optional


class Query:
    def __init__(self, query: dict):
        self._inner = query

    def query_type(self) -> str:
        """
        Get the query type of the query.

        Returns
        -------
        str
            The type of the query.
        """
        return list(self._inner.keys())[0]

    @classmethod
    def match_query(cls, query: str, field: str, *, boost=1.0, fuzziness=0) -> "Query":
        """
        Match query for full-text search.

        Parameters
        ----------
        query : str
            The query string to match against.
        field : str
            The name of the field to match against.
        boost : float, default 1.0
            The boost factor for the query.
        fuzziness : int, optional
            The maximum edit distance for each term in the match query.
            Defaults to 0 (exact match).
            If None, fuzziness is applied automatically by the rules:
                - 0 for terms with length <= 2
                - 1 for terms with length <= 5
                - 2 for terms with length > 5

        Returns
        -------
        Query
            The match query object.
        """

        return Query(
            {
                "match": {
                    field: {
                        "query": query,
                        "boost": boost,
                        "fuzziness": fuzziness,
                    }
                }
            }
        )

    @classmethod
    def phrase_query(cls, query: str, field: str) -> "Query":
        """
        Phrase query for full-text search.

        Parameters
        ----------
        query : str
            The query string to match against.
        field : str
            The name of the field to match against.

        Returns
        -------
        Query
            The phrase query object.
        """
        return Query(
            {
                "match_phrase": {
                    field: query,
                }
            }
        )

    @classmethod
    def boost_query(
        cls, positive: "Query", negative: "Query", negative_boost: float
    ) -> "Query":
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

        Returns
        -------
        Query
            The boost query object.
        """
        return Query(
            {
                "boost": {
                    "positive": positive._inner,
                    "negative": negative._inner,
                    "negative_boost": negative_boost,
                }
            }
        )

    @classmethod
    def multi_match_query(
        cls,
        query: str,
        fields: list[str],
        *,
        boosts: Optional[list[float]] = None,
    ) -> "Query":
        """
        Multi-match query for full-text search.

        Parameters
        ----------
        query : str | list[Query]
            If a string, the query string to match against.

        fields : list[str]
            The list of fields to match against.

        boosts : list[float], optional
            The list of boost factors for each field. If not provided,
            all fields will have the same boost factor.

        Returns
        -------
        Query
            The multi-match query object.
        """
        if boosts is not None:
            if len(fields) != len(boosts):
                raise ValueError("The number of fields and boosts must be the same.")

        return Query(
            {
                "multi_match": {
                    "query": query,
                    "fields": fields,
                    "boost": boosts,
                }
            }
        )
