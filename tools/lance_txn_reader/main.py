import pyarrow.fs as pa_fs

from transaction_pb2 import Transaction


def read_transaction(uri: str):
    fs, path = pa_fs.FileSystem.from_uri(uri)

    data = fs.open_input_stream(path).readall()

    # parse transaction protobuf message
    txn_msg = Transaction()
    txn_msg.ParseFromString(data)

    print(txn_msg)
