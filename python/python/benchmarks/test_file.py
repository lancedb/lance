#  Copyright (c) 2023. Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from datetime import datetime

import duckdb
import lance
import pyarrow.compute as pc
import pyarrow.parquet as pq
from lance.file import LanceFileReader
from lance.tracing import trace_to_chrome

# rand_doubles = np.random.rand(100_000_000)
# table = pa.table({"a": rand_doubles})
# pq.write_table(table, "/tmp/tinyrand.parquet")

trace_to_chrome(file="/tmp/foo.trace", level="trace")

table = pq.read_table("/tmp/tinyrand.parquet")

# pq.write_table(table, "/tmp/tinyrand1g.parquet", row_group_size=200 * 1024 * 1024)

# print(f"There are {table.num_columns} columns")
# non_string_fields = [field.name for field in table.schema if field.type != pa.string()]
# table = table.select(non_string_fields)
# print(f"There are {table.num_columns} non-string columns")

conn = duckdb.connect()
# start = datetime.now()
# for _ in range(5):
#     print(
#         conn.execute(
#             "SELECT SUM(passenger_count) FROM '/tmp/nyctaxi-tiny.parquet'"
#         ).fetchall()[0][0]
#     )
# end = datetime.now()
# avg_duration = (end - start).total_seconds() / 5.0
# print(f"DuckDB average duration: {avg_duration}s")

# start = datetime.now()
# for _ in range(5):
#     df = pl.read_parquet("/tmp/nyctaxi-tiny.parquet")
#     print(df.select(pl.sum("passenger_count")))
# end = datetime.now()
# avg_duration = (end - start).total_seconds() / 5.0
# print(f"Polars average duration: {avg_duration}s")

# start = datetime.now()
# for _ in range(5):
#     table = pq.read_table("/tmp/nyctaxi-tiny.parquet")
#     sum = pc.sum(table.column(0)).as_py()
#     print(f"sum: {sum}")
# end = datetime.now()
# avg_duration = (end - start).total_seconds() / 5.0
# print(f"Parquet average duration: {avg_duration}s")

# with LanceFileWriter("/tmp/tinyrand.lancev2", table.schema) as writer:
#     offset = 0
#     while offset < table.num_rows:
#         next_size = min(1024 * 1024, table.num_rows - offset)
#         print(f"Writing {next_size} rows")
#         batch = table.slice(offset, next_size)
#         offset += next_size
#         writer.write_batch(batch)

print("batch_size, v2_duration, v2_no_py_duration, v1_duration, v1_no_py_duration")

for batch_size in [32, 128, 512, 4096, 16 * 1024, 64 * 1024, 256 * 1024]:
    start = datetime.now()
    reader = LanceFileReader("/tmp/tinyrand.lancev2", schema=table.schema)
    sum = 0
    for batch in reader.read_all(batch_size=batch_size).to_batches():
        sum += pc.sum(batch.column(0)).as_py()
    end = datetime.now()
    v2_duration = (end - start).total_seconds()

    start = datetime.now()
    sum = reader.weston(batch_size)
    end = datetime.now()
    v2_no_py_duration = (end - start).total_seconds()

    lance.write_dataset(
        table, "/tmp/tinyrand.lance", max_rows_per_group=batch_size, mode="overwrite"
    )

    start = datetime.now()
    ds = lance.dataset("/tmp/tinyrand.lance")
    sum = 0
    for batch in ds.scanner(columns=["a"]).to_batches():
        sum += pc.sum(batch.column(0)).as_py()
    end = datetime.now()
    v1_duration = (end - start).total_seconds() / 1.0

    start = datetime.now()
    ds = lance.dataset("/tmp/tinyrand.lance")
    count = ds.count_rows("a % 2 < 2")
    end = datetime.now()
    v1_no_py_duration = (end - start).total_seconds() / 1.0

    print(
        f"{batch_size}, {v2_duration}, {v2_no_py_duration}, {v1_duration}, {v1_no_py_duration}"
    )
