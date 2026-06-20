import lance
import pyarrow as pa
from pydantic import BaseModel


class MyModel(BaseModel):
    name: str
    score: float


data = [MyModel(name="alice", score=0.9), MyModel(name="bob", score=0.8)]

# After fix: Pydantic instances are handled automatically
print("=== After Fix: Direct Pydantic Instance Support ===")
ds = lance.write_dataset(data, "/tmp/test.lance", mode="overwrite")
print(f"Success! Wrote {ds.count_rows()} rows without manual conversion.")
print(ds.to_table())

print()
print("=== Old Workaround (still works) ===")
dicts = [m.model_dump() for m in data]
schema = pa.schema([pa.field("name", pa.string()), pa.field("score", pa.float64())])
table = pa.Table.from_pylist(dicts, schema=schema)
lance.write_dataset(table, "/tmp/test_manual.lance", mode="overwrite")
print("Manual conversion path still works as expected.")
