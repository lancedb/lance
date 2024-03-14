import argparse
import os
from main import read_transaction
import pyarrow.fs as pa_fs

parser = argparse.ArgumentParser()
parser.add_argument("uri", type=str, help="Directory to read transactions from")
args = parser.parse_args()

fs, path = pa_fs.FileSystem.from_uri(os.path.join(args.uri, "_transactions"))

txn_files = fs.get_file_info(pa_fs.FileSelector(path))
txn_files = [f.path for f in txn_files if f.path.endswith(".txn")]

for txn_file in txn_files:
    read_transaction("s3://" + txn_file)
