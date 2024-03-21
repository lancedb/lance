

Build the protos

```bash
protoc -I=../../protos --python_out=. transaction.proto file.proto table.proto
```

List all transactions

```python
python list_transactions.py /path/to/lance/dataset > transactions.log
```