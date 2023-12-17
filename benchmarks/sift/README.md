# Benchmarks on SIFT/GIST-1M dataset

Dataset URI: http://corpus-texmex.irisa.fr/

The SIFT/GIST-1M benchmarks make use of the [LanceDB](https://github.com/lancedb/lancedb) API to index, manage and query the datasets. Ensure the dependencies are installed. LanceDB is built on top of Lance and stores everything as Lance datasets.

```sh
# Pin the lancedb version to the latest one availale on your own benchmark
pip lancedb==0.3.6
pip install pandas~=2.1.0
pip duckdb~=0.9.0
```

## SIFT-1M

Download `sift.tar.gz` from the [source page](http://corpus-texmex.irisa.fr/) and unzip.

```sh
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
```

### Generate datasets

Generate Lance datasets using the `datagen.py` script:

#### Database vectors

```sh
 ./datagen.py ./sift/sift_base.fvecs sift1m.lance -d 128
```

#### Query vectors

```sh
./datagen.py ./sift/sift_query.fvecs ./.lancedb/sift_query.lance -d 128 -n 1000
```

### Create index

```sh
# -i is ivf partitions and -p is pq subvectors
./index.py ~/.lancedb/sift1m.lance -i 256 -p 16
```

### Run benchmark

```sh
# -k is how many results to fetch and -q is the query vectors
./metrics.py ./.lancedb/sift1m.lance results-sift.csv -i 256 -p 16 -q ./.lancedb/sift_query.lance -k 1
```

The results with mean query time and recall@1 are stored in `results-sift.csv`.

---

## GIST-1M

Download `gist.tar.gz` from the [source page](http://corpus-texmex.irisa.fr/) and unzip.

```sh
wget ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
tar -xzf gist.tar.gz
```

### Generate datasets

Generate Lance datasets using the `datagen.py` script.

#### Database vectors

```sh
./datagen.py ./gist/gist_base.fvecs ./.lancedb/gist1m.lance -g 1024 -m 50000 -d 960
```

#### Query vectors

```sh
./datagen.py ./gist/gist_query.fvecs ./.lancedb/gist_query.lance -g 1024 -m 50000 -d 960 -n 1000
```

### Create index

```sh
# -i is ivf partitions and -p is pq subvectors
./index.py ~/.lancedb/gist1m.lance -i 256 -p 120
```

### Run benchmark

```sh
# -k is how many results to fetch and -q is the query vectors
./metrics.py ./.lancedb/gist1m.lance results-gist.csv -i 256 -p 120 -q ./.lancedb/gist_query.lance -k 1
```

The results with mean query time and recall@1 are stored in `results-gist.csv`.
