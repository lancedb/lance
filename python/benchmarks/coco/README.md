# PyTorch Benchmark on MS Coco dataset

## How to generate dataset

```sh

./datagen.py -f lance /path/to/coco
```

## Training

```sh

./train.py -f lance --epoch 10 [--benchmark io] /path/to/coco
```