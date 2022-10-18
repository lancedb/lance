# Lance DuckDB Extension

*Linux Only for now*.

Lance DuckDB extension allows user to run Machine Learning tasks via DuckDB + Lance.

## Functions

List / Array functions

| Function            | Description                        |
|---------------------|------------------------------------|
| `list_argmax(list)` | Run `argmax` over a list of values |

Machine Learning functions

| Function                          | Description                    |
|-----------------------------------|--------------------------------|
| `create_pytorch_model(name, uri)` | Create a Pytorch script model  |
| `predict(model, blob)`            | Run model inference over image |
| `ml_models()`                     | Show all ML models             |

```sql
CALL create_pytorch_model('resnet', './resnet.pth')
SELECT * FROM ml_models();
// Run inference
SELECT predict('resnet', image) as pred FROM images
```

Vector functions

| Function                         | Description                               |
|----------------------------------|-------------------------------------------|
| `l2_distance(list, list)`        | Calculate L2 distance between two vectors |
| `in_rectangle(list, list[list])` | Whether the point is in a bounding box    |


## Development

Currently, Machine Learning functions only works on Linux.

To build the extension, run:

```shell
make release-linux
```

Load extension in Python
```python
import duckdb
duckdb.install_extension("./path/to/lance.duckdb_extension")
duckdb.load_extension("lance")
```
