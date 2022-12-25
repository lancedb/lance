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

| Function                          | Description                                                                         |
|-----------------------------------|-------------------------------------------------------------------------------------|
| `dydx(y, x)`                      | Calculate derivative $\frac{dy}{dx}$                                                |
| `window_all(expr, before, after)` | Is the row in a window `[-before, start]` <br/> where all `expr` are `true`.        |
| `window_any(expr, before, after)` | Is the row in a window `[-before, start]` <br/> where at least one `expr` is `true` |

```sql

SELECT x, window_all(x <= 3, 2, 0) FROM t;
┌───────┬────────────────────────────┐
│   x   │ window_all((x <= 3), 2, 0) │
│ int32 │          boolean           │
├───────┼────────────────────────────┤
│     1 │ true                       │
│     2 │ true                       │
│     3 │ true                       │
│     4 │ false                      │
│     5 │ false                      │
└───────┴────────────────────────────┘

SELECT x, window_any(x == 3, 2, 0) FROM t;
┌───────┬───────────────────────────┐
│   x   │ window_any((x = 3), 2, 0) │
│ int32 │          boolean          │
├───────┼───────────────────────────┤
│     1 │ false                     │
│     2 │ false                     │
│     3 │ true                      │
│     4 │ true                      │
│     5 │ true                      │
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
