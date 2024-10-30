import pyarrow as pa
import pyarrow.parquet as pq
import datetime
import os
import numpy as np
from lance.file import LanceFileReader, LanceFileWriter

def format_throughput(value):
    return f"{value:.2f} GiB/s"

parquet_file_path = "/home/admin/tmp/int64_column.parquet"
lance_file_path = "/home/admin/tmp/int64_column.lance"

# Generate 1024 * 1024 * 512 random values modulo 1024 using NumPy
values = np.random.randint(0, 1024, size=1024 * 1024 * 64 + 1, dtype=np.int64)
#values = [8] * 1024
int64_array = pa.array(values, type=pa.int64())
table = pa.Table.from_arrays([int64_array], names=['int64_column'])

values = np.random.randint(0, 1024, size=1024 * 1024 * 4 + 1, dtype=np.int32)
#values = [8] * 1024
int32_array = pa.array(values, type=pa.int32())
table = pa.Table.from_arrays([int32_array], names=['int_column'])

start = datetime.datetime.now()
# Write the table to a Parquet file 
pq.write_table(table, parquet_file_path)
end = datetime.datetime.now()
elapsed_parquet = (end - start).total_seconds()
print(f"Parquet write time: {elapsed_parquet:.2f}s")

start = datetime.datetime.now()
# Write the table to a Lance file
with LanceFileWriter(lance_file_path, version="2.1") as writer:
    writer.write_batch(table)
end = datetime.datetime.now()
elapsed_lance = (end - start).total_seconds()
print(f"Lance write time: {elapsed_lance:.2f}s")

# Flush file from kernel cache
os.system('sync; echo 3 | sudo tee /proc/sys/vm/drop_caches')

# Measure read time for Parquet file with batch size
batch_size = 32 * 1024
start = datetime.datetime.now()
parquet_file = pq.ParquetFile(parquet_file_path)
batches = parquet_file.iter_batches(batch_size=batch_size)
tab_parquet = pa.Table.from_batches(batches)
end = datetime.datetime.now()
elapsed_parquet = (end - start).total_seconds()

# Flush file from kernel cache again before reading Lance file
os.system('sync; echo 3 | sudo tee /proc/sys/vm/drop_caches')

# Measure read time for Lance file with batch size
start = datetime.datetime.now()
tab_lance = LanceFileReader(lance_file_path).read_all(batch_size=batch_size).to_table()
end = datetime.datetime.now()
elapsed_lance = (end - start).total_seconds()

num_rows = tab_parquet['int_column'].length()

print(f"Number of rows: {num_rows}")
print(f"Parquet read time: {elapsed_parquet:.2f}s")
print(f"Lance read time: {elapsed_lance:.2f}s")

# Compute total memory size
parquet_memory_size = tab_parquet.get_total_buffer_size()
lance_memory_size = tab_lance.get_total_buffer_size()

# Convert memory size to GiB
parquet_memory_size_gib = parquet_memory_size / (1024 * 1024 * 1024)
lance_memory_size_gib = lance_memory_size / (1024 * 1024 * 1024)

# Compute read throughput in GiB/sec
throughput_parquet_gib = parquet_memory_size_gib / elapsed_parquet
throughput_lance_gib = lance_memory_size_gib / elapsed_lance

# Format throughput values
formatted_throughput_parquet_gib = format_throughput(throughput_parquet_gib)
formatted_throughput_lance_gib = format_throughput(throughput_lance_gib)

print(f"Parquet read throughput: {formatted_throughput_parquet_gib}")
print(f"Lance read throughput: {formatted_throughput_lance_gib}")

# Check file sizes
lance_file_size = os.path.getsize(lance_file_path)
lance_file_size_mib = lance_file_size // 1048576
parquet_file_size = os.path.getsize(parquet_file_path)
parquet_file_size_mib = parquet_file_size // 1048576

print(f"Parquet file size: {parquet_file_size} bytes ({parquet_file_size_mib:,} MiB)")
print(f"Lance file size: {lance_file_size} bytes ({lance_file_size_mib:,} MiB)")

# Assert that the tables are equal
assert tab_parquet == tab_lance