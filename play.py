from datetime import datetime
import glob
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

import lance
from lance.file import LanceFileReader, LanceFileWriter

from lance.tracing import trace_to_chrome

trace_to_chrome(level="trace")

PATH = "/home/x/lineitem_10.parquet"

arr_dataset = ds.dataset(PATH)

schema = ds.dataset(PATH).schema

# for file in glob.glob("/home/pace/dev/data/lineitem_10.parquet.*"):
#     pq_size = os.path.getsize(file)
#     pq_size_gb = pq_size / 1024.0 / 1024.0 / 1024.0
#     print(f"Parquet file size: {pq_size_gb} GiB")

#     pq_metadata = pq.ParquetFile(file).metadata
#     num_row_groups = pq_metadata.num_row_groups
#     print(f"Parquet file num row groups: {num_row_groups}")

#     start = datetime.now()
#     arr_dataset = ds.dataset(PATH)
#     arr_dataset.to_table()
#     end = datetime.now()

#     duration = (end - start).total_seconds()
#     print(f"Parquet (pyarrow) Read Time (lineitem_10): {duration}s")

def get_parquet_file_size_in_memory(file_path):
    # Read the Parquet file into a pyarrow.Table
    table = pq.read_table(file_path)
    
    # Use sys.getsizeof to get the size of the table in memory
    # Note: This is an approximation
    size_in_memory = sys.getsizeof(table)
    
    return size_in_memory

original_size = get_parquet_file_size_in_memory(PATH)
original_size_gb = original_size / 1024.0 / 1024.0 / 1024.0
print(f"original file size: {original_size_gb} GiB")

if not os.path.exists("/tmp/tpch.lancev2"):
    start = datetime.now()
    arr_dataset = ds.dataset(PATH)
    with LanceFileWriter("/tmp/tpch.lancev2", arr_dataset.schema) as writer:
        for batch in arr_dataset.to_batches():
            writer.write_batch(batch)
    end = datetime.now()
    duration = (end - start).total_seconds()
    print(f"V2 Write Time (lineitem_10): {duration}s")

lv2_size = os.path.getsize("/tmp/tpch.lancev2")
lv2_size_gb = lv2_size / 1024.0 / 1024.0 / 1024.0
print(f"Lance v2 file size: {lv2_size_gb} GiB")

start = datetime.now()
reader = LanceFileReader("/tmp/tpch.lancev2")
for batch in reader.read_all(batch_size=8 * 1024).to_batches():
    pass
end = datetime.now()
print(f"V2 Read Time (lineitem_10): {(end - start).total_seconds()}s")
