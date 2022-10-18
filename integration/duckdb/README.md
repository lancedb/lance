# Lance DuckDB Extension

*Linux Only*

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

| Function                                        | Description                               |
|-------------------------------------------------|-------------------------------------------|
| `l2_distance(list[float], list[float])`         | Calculate L2 distance between two vectors |
| `rect_contains(list[float], list[list[float]])` | Whether the point is in a bounding box    |


## Development

```shell

make
```
