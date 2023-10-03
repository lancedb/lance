# Dbpedia-entities-openai dataset

[dbpedia-entities-openai](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M) dataset
contains 1M openai embeddings.

## Prepare Dataset

```sh
# Python 3.10+
python3 -m venv venv
. ./venv/bin/activate

./datagen.py
```

## Run benchmark

`benchmarks.py` run top-k vector query over different combinations of IVF and PQ values,
as well as `refine_factor`.

```sh
./benchmarks.py -k 20
```