# Dbpedia-entities-openai dataset

[dbpedia-entities-openai](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M) dataset
contains 1M openai embeddings.

## Prepare Dataset

```sh
# Generate dataset in lance format.
uv run ./datagen.py
```

## Run benchmark

`benchmarks.py` run top-k vector query over different combinations of IVF and PQ values,
as well as `refine_factor`.

```sh
uv run ./benchmarks.py
```