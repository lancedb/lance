# Lance DuckDB Extension

*Linux and Mac only for now*. Windows support forthcoming

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

Currently, the Lance duckdb extension is compiled against pytorch 1.13

```sql
CALL create_pytorch_model('resnet', './resnet.pth', 'cpu')
SELECT *
FROM ml_models();
// Run inference
SELECT predict('resnet', image) as pred
FROM images
```

Vector functions

| Function                         | Description                               |
|----------------------------------|-------------------------------------------|
| `l2_distance(list, list)`        | Calculate L2 distance between two vectors |
| `in_rectangle(list, list[list])` | Whether the point is in a bounding box    |

Misc functions

| Function                         | Description                                                                         |
|----------------------------------|-------------------------------------------------------------------------------------|
| `dydx(y, x)`                     | Calculate derivative $\frac{dy}{dx}$                                                |
| `is_window(expr, before, after)` | Is the row in a window with size `[-before, start]` <br/> of all `expr` are `true`. |

```sql

SELECT x, is_window(x <= 3, 0, 1) FROM t;
┌───────┬───────────────────────────┐
│   x   │ is_window((x <= 3), 0, 1) │
│ int32 │          boolean          │
├───────┼───────────────────────────┤
│     1 │ true                      │
│     2 │ true                      │
│     3 │ false                     │
│     4 │ false                     │
│     5 │ false                     │
└───────┴───────────────────────────┘

SELECT x, is_window(x <= 3, 2, 0) FROM t;

┌───────┬───────────────────────────┐
│   x   │ is_window((x <= 3), 2, 0) │
│ int32 │          boolean          │
├───────┼───────────────────────────┤
│     1 │ true                      │
│     2 │ true                      │
│     3 │ true                      │
│     4 │ false                     │
│     5 │ false                     │
└───────┴───────────────────────────┘
```

## Development

To build the extension, run:

```shell
make release-linux
```

If you want to use GPU-enabled models, run:

```shell
make release-cuda
```

Load extension in Python

```python
import duckdb

duckdb.install_extension("./path/to/lance.duckdb_extension", force_install=True)
duckdb.load_extension("lance")
```
