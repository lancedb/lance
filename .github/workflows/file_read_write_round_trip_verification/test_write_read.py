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

import pyarrow.parquet as pq
from lance.file import LanceFileReader, LanceFileWriter

# Path to the Parquet file
file_path = 'drug-reviews.parquet'
lance_file_path = "drug-reviews.lance"

# Read the Parquet file into a PyArrow table
try:
    parquet_table = pq.read_table(file_path)
    print("Parquet file read into PyArrow table successfully.")
    print(parquet_table)

    with LanceFileWriter(lance_file_path, version="2.1") as writer:
        writer.write_batch(parquet_table)

    tab_lance = LanceFileReader(lance_file_path).read_all().to_table()

    assert tab_lance == parquet_table

    assert 1 == 0

except Exception as e:
    print(f"Error reading Parquet file: {e}")