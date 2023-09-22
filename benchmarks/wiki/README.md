## Cohere wiki text embedding benchmark
a dataset with 35M 768D vectors meant for benchmarking index build

### Generating data
run
```bash
python datagen.py
```
to generate the lance dataset

run
```bash
python index.py --metric L2 --num-partitions 2048 --num-sub-vectors 96
```
to index the data
