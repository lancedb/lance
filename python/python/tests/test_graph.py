# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import os

import lance
import pandas as pd
import pytest
from lance.graph import CypherQuery, GraphConfig


@pytest.fixture
def graph_env(tmp_path):
    people_data = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Carol", "David"],
            "age": [28, 34, 29, 42],
            "city": ["New York", "San Francisco", "New York", "Chicago"],
        }
    )

    companies_data = pd.DataFrame(
        {
            "company_id": [101, 102, 103],
            "company_name": ["TechCorp", "DataInc", "CloudSoft"],
            "industry": ["Technology", "Analytics", "Cloud"],
        }
    )

    employment_data = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "company_id": [101, 101, 102, 103],
            "position": ["Engineer", "Designer", "Manager", "Director"],
            "salary": [120000, 95000, 130000, 180000],
        }
    )

    friendship_data = pd.DataFrame(
        {
            "person1_id": [1, 1, 2, 3],
            "person2_id": [2, 3, 4, 4],
            "friendship_type": ["close", "casual", "close", "casual"],
            "years_known": [5, 2, 3, 1],
        }
    )

    config = (
        GraphConfig.builder()
        .with_node_label("Person", "person_id")
        .with_node_label("Company", "company_id")
        .with_relationship("WORKS_FOR", "person_id", "company_id")
        .with_relationship("FRIEND_OF", "person1_id", "person2_id")
        .build()
    )

    people_ds = lance.write_dataset(
        people_data, os.path.join(str(tmp_path), "people.lance")
    )
    companies_ds = lance.write_dataset(
        companies_data, os.path.join(str(tmp_path), "companies.lance")
    )
    employment_ds = lance.write_dataset(
        employment_data, os.path.join(str(tmp_path), "employment.lance")
    )
    friendship_ds = lance.write_dataset(
        friendship_data, os.path.join(str(tmp_path), "friendship.lance")
    )

    datasets = {
        "Person": people_ds,
        "Company": companies_ds,
        "WORKS_FOR": employment_ds,
        "FRIEND_OF": friendship_ds,
    }

    return config, datasets, people_data


def test_basic_node_selection(graph_env):
    config, datasets, _ = graph_env
    query = CypherQuery("MATCH (p:Person) RETURN p.name, p.age").with_config(config)
    result = query.execute({"Person": datasets["Person"]})
    df = result.to_pandas()
    assert len(df) == 4
    assert set(df.columns) == {"name", "age"}
    assert "Alice" in set(df["name"].values)


def test_filtered_query(graph_env):
    config, datasets, _ = graph_env
    query = CypherQuery(
        "MATCH (p:Person) WHERE p.age > 30 RETURN p.name, p.age"
    ).with_config(config)
    result = query.execute({"Person": datasets["Person"]})
    df = result.to_pandas()
    assert len(df) == 2
    assert set(df["name"].values) == {"Bob", "David"}
    assert all(age > 30 for age in df["age"])


def test_relationship_query(graph_env):
    config, datasets, _ = graph_env
    # Alias outputs to stable column names regardless of internal qualification
    query = CypherQuery(
        "MATCH (p:Person)-[:WORKS_FOR]->(c:Company) "
        "RETURN p.person_id AS person_id, p.name AS name, c.company_id AS company_id"
    ).with_config(config)

    result = query.execute(
        {
            "Person": datasets["Person"],
            "Company": datasets["Company"],
            "WORKS_FOR": datasets["WORKS_FOR"],
        }
    )
    df = result.to_pandas()
    assert len(df) == 4
    assert list(df["person_id"].values) == [1, 2, 3, 4]
    assert list(df["company_id"].values) == [101, 101, 102, 103]


def test_friendship_direct_and_network(graph_env):
    config, datasets, _ = graph_env
    # Direct friends of Alice (person_id = 1)
    query_direct = CypherQuery(
        "MATCH (a:Person)-[:FRIEND_OF]->(b:Person) "
        "WHERE a.person_id = 1 "
        "RETURN b.person_id AS friend_id"
    ).with_config(config)

    result_direct = query_direct.execute(
        {
            "Person": datasets["Person"],
            "FRIEND_OF": datasets["FRIEND_OF"],
        }
    )
    df_direct = result_direct.to_pandas()
    assert set(df_direct["friend_id"].values) == {2, 3}

    # Full friendship edges
    query_edges = CypherQuery(
        "MATCH (f:Person)-[r:FRIEND_OF]->(t:Person) "
        "RETURN f.person_id AS person1_id, t.person_id AS person2_id"
    ).with_config(config)

    result_edges = query_edges.execute(
        {
            "Person": datasets["Person"],
            "FRIEND_OF": datasets["FRIEND_OF"],
        }
    )
    df_edges = result_edges.to_pandas()
    got = set(
        (int(a), int(b)) for a, b in zip(df_edges["person1_id"], df_edges["person2_id"])
    )
    assert got == {(1, 2), (1, 3), (2, 4), (3, 4)}


def test_two_hop_friends_of_friends(graph_env):
    config, datasets, _ = graph_env
    query = CypherQuery(
        "MATCH (a:Person)-[:FRIEND_OF]->(b:Person)-[:FRIEND_OF]->(c:Person) "
        "WHERE a.person_id = 1 "
        "RETURN a.person_id AS a_id, b.person_id AS b_id, c.person_id AS c_id"
    ).with_config(config)

    result = query.execute(
        {
            "Person": datasets["Person"],
            "FRIEND_OF": datasets["FRIEND_OF"],
        }
    )
    df = result.to_pandas()
    # Expect two 2-hop paths ending at 4
    assert set(df["c_id"].values) == {4}


@pytest.mark.xfail(reason="Variable-length path (*1..2) support pending in executor")
def test_variable_length_path(graph_env):
    config, datasets, _ = graph_env
    query = CypherQuery(
        "MATCH (p1:Person)-[:FRIEND_OF*1..2]-(p2:Person) "
        "RETURN p1.person_id AS p1, p2.person_id AS p2"
    ).with_config(config)
    _ = query.execute(
        {
            "Person": datasets["Person"],
            "FRIEND_OF": datasets["FRIEND_OF"],
        }
    )
