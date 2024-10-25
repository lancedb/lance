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

import os
import glob
import pyarrow.parquet as pq
from lance.file import LanceFileReader, LanceFileWriter

# Get the current working directory
current_path = os.getcwd()

# Find all Parquet files in the current directory
parquet_files = glob.glob(os.path.join(current_path, '*.parquet'))

# Process each Parquet file
for file_path in parquet_files:
    try:
        # Define the corresponding Lance file path
        lance_file_path = os.path.splitext(file_path)[0] + ".lance"

        # Read the Parquet file into a PyArrow table
        parquet_table = pq.read_table(file_path)

        # Write the table to a Lance file
        with LanceFileWriter(lance_file_path, version="2.0") as writer:
            writer.write_batch(parquet_table)

        # Read the Lance file back into a table
        tab_lance = LanceFileReader(lance_file_path).read_all().to_table()

        # Check if the tables are equal
        assert tab_lance == parquet_table
        print(f"Table read from Lance is the same as table read from Parquet for file: {file_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")