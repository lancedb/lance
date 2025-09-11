# Benchmark performance Lance vs Parquet w/ Tpch Q1 and Q6
import lance
import duckdb

import sys
import time

Q1 = """
SELECT
    l_returnflag,
    l_linestatus,
    sum(l_quantity) as sum_qty,
    sum(l_extendedprice) as sum_base_price,
    sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
    avg(l_quantity) as avg_qty,
    avg(l_extendedprice) as avg_price,
    avg(l_discount) as avg_disc,
    count(*) as count_order
FROM
    lineitem
WHERE
    l_shipdate <= date '1998-12-01' - interval '90' day
GROUP BY
    l_returnflag,
    l_linestatus
ORDER BY
    l_returnflag,
    l_linestatus;
"""

Q6 = """
SELECT
    sum(l_extendedprice * l_discount) as revenue
FROM
    lineitem
WHERE
    l_shipdate >= date '1994-01-01'
    AND l_shipdate < date '1994-01-01' + interval '1' year
    AND l_discount between 0.06 - 0.01 AND 0.06 + 0.01
    AND l_quantity < 24;
"""

num_args = len(sys.argv)
assert num_args == 2

query = ""
if sys.argv[1] == "q1":
    query = Q1
elif sys.argv[1] == "q6":
    query = Q6
else:
    sys.exit("We only support Q1 and Q6 for now")

print("------------------BENCHMARK TPCH " + sys.argv[1] + "-------------------\n")
##### Lance #####
start1 = time.time()
# read from lance and create a relation from it
lineitem = lance.dataset("./dataset/lineitem.lance")
res1 = duckdb.sql(query).df()
end1 = time.time()

print("Lance Latency: ", str(round(end1 - start1, 3)) + "s")
print(res1)

##### Parquet #####
lineitem = None
start2 = time.time()
# read from parquet and create a view instead of table from it
duckdb.sql(
    "CREATE VIEW lineitem AS SELECT * FROM read_parquet('./dataset/lineitem_sf1.parquet');"
)
res2 = duckdb.sql(query).df()
end2 = time.time()

print("Parquet Latency: ", str(round(end2 - start2, 3)) + "s")
print(res2)
