# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from datetime import datetime

import lance
import pyarrow.parquet as pq
from lance.tracing import trace_to_chrome

trace_to_chrome(file="/tmp/foo.trace", level="debug")

# This file compares the performance of lance v1 and v2 on the lineitem dataset,
# specifically for random access scans

tab = pq.read_table("~/lineitemsf1.snappy.parquet")
dsv1 = lance.write_dataset(tab, "/tmp/lineitem.lancev1", use_legacy_format=True)
dsv2 = lance.write_dataset(tab, "/tmp/lineitem.lancev2", use_legacy_format=False)

dsv1 = lance.dataset("/tmp/lineitem.lancev1")
dsv2 = lance.dataset("/tmp/lineitem.lancev2")

start = datetime.now()
dsv1.to_table(filter="l_shipmode = 'FOB'", limit=10000)
duration = (datetime.now() - start).total_seconds()
print(f"V1 query time: {duration}s")

start = datetime.now()
dsv2.to_table(filter="l_shipmode = 'FOB'", limit=10000)
duration = (datetime.now() - start).total_seconds()
print(f"V2 query time: {duration}s")

start = datetime.now()
dsv1.take([1, 40, 100, 130, 200])
duration = (datetime.now() - start).total_seconds()
print(f"V1 query time: {duration}s")

start = datetime.now()
dsv2.take([1, 40, 100, 130, 200])
duration = (datetime.now() - start).total_seconds()
print(f"V2 query time: {duration}s")
