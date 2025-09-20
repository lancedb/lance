# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Graph query functionality for Lance datasets.

This module provides Cypher query support for interpreting Lance datasets
as property graphs. It allows you to configure how tabular data maps to
nodes and relationships, then query using Cypher syntax.

Example:
    >>> import lance
    >>> from lance.graph import GraphConfig, CypherQuery
    >>>
    >>> # Configure graph mapping
    >>> config = GraphConfig.builder() \\
    ...     .with_node_label("Person", "person_id") \\
    ...     .with_relationship("KNOWS", "person_id", "friend_id") \\
    ...     .build()
    >>>
    >>> # Create and execute query
    >>> query = CypherQuery("MATCH (p:Person) WHERE p.age > 30 RETURN p.name") \\
    ...     .with_config(config)
    >>>
    >>> # Get generated SQL
    >>> sql = query.to_sql()
"""

import lance

# Import the base classes
_GraphConfig = lance.lance.graph.GraphConfig
_GraphConfigBuilder = lance.lance.graph.GraphConfigBuilder
_CypherQuery = lance.lance.graph.CypherQuery


class GraphConfigBuilder:
    """Builder for GraphConfig with fluent interface support."""

    def __init__(self):
        self._builder = _GraphConfigBuilder()

    def with_node_label(self, label: str, id_field: str):
        """Add a node label mapping.

        Args:
            label: The node label (e.g., "Person")
            id_field: The field in the dataset that serves as the node ID

        Returns:
            Self for method chaining
        """
        self._builder.with_node_label(label, id_field)
        return self

    def with_relationship(self, rel_type: str, source_field: str, target_field: str):
        """Add a relationship mapping.

        Args:
            rel_type: The relationship type (e.g., "KNOWS")
            source_field: The field containing source node IDs
            target_field: The field containing target node IDs

        Returns:
            Self for method chaining
        """
        self._builder.with_relationship(rel_type, source_field, target_field)
        return self

    def build(self):
        """Build the GraphConfig.

        Returns:
            GraphConfig: The configured graph config
        """
        inner_config = self._builder.build()
        return GraphConfig(inner_config)


class GraphConfig:
    """Graph configuration for interpreting Lance datasets as property graphs."""

    def __init__(self, inner):
        self._inner = inner

    @staticmethod
    def builder():
        """Create a new GraphConfig builder.

        Returns:
            GraphConfigBuilder: A new builder instance
        """
        return GraphConfigBuilder()

    def node_labels(self):
        """Get node labels.

        Returns:
            List[str]: List of node labels
        """
        return self._inner.node_labels()

    def relationship_types(self):
        """Get relationship types.

        Returns:
            List[str]: List of relationship types
        """
        return self._inner.relationship_types()

    def __repr__(self):
        return self._inner.__repr__()


class CypherQuery:
    """Cypher query interface for Lance datasets."""

    def __init__(self, query_text: str):
        """Create a new Cypher query.

        Args:
            query_text: The Cypher query string
        """
        self._inner = _CypherQuery(query_text)

    def with_config(self, config: GraphConfig):
        """Set the graph configuration.

        Args:
            config: The graph configuration

        Returns:
            CypherQuery: A new query instance with the config set
        """
        # Extract the inner config from our wrapper
        inner_config = config._inner
        new_inner = self._inner.with_config(inner_config)

        # Create a new CypherQuery wrapper
        new_query = CypherQuery.__new__(CypherQuery)
        new_query._inner = new_inner
        return new_query

    def with_parameter(self, key: str, value):
        """Add a query parameter.

        Args:
            key: Parameter name
            value: Parameter value (will be converted to JSON)

        Returns:
            CypherQuery: A new query instance with the parameter added
        """
        new_inner = self._inner.with_parameter(key, value)
        new_query = CypherQuery.__new__(CypherQuery)
        new_query._inner = new_inner
        return new_query

    def query_text(self):
        """Get the query text.

        Returns:
            str: The query text
        """
        return self._inner.query_text()

    def parameters(self):
        """Get query parameters.

        Returns:
            dict: Query parameters
        """
        return self._inner.parameters()

    def to_sql(self):
        """Convert query to SQL.

        Returns:
            str: The generated SQL query
        """
        return self._inner.to_sql()

    def execute(self, datasets):
        """Execute query against Lance datasets.

        Args:
            datasets (dict): Dictionary mapping table names to Lance datasets

        Returns:
            pyarrow.Table: Query results as Arrow table
        """
        # Always use Rust execution - no Python fallback
        return self._inner.execute(datasets)

    def variables(self):
        """Get variables used in the query.

        Returns:
            List[str]: List of variables
        """
        return self._inner.variables()

    def node_labels(self):
        """Get node labels referenced in the query.

        Returns:
            List[str]: List of node labels
        """
        return self._inner.node_labels()

    def relationship_types(self):
        """Get relationship types referenced in the query.

        Returns:
            List[str]: List of relationship types
        """
        return self._inner.relationship_types()

    def __repr__(self):
        return self._inner.__repr__()


__all__ = ["GraphConfig", "GraphConfigBuilder", "CypherQuery"]
