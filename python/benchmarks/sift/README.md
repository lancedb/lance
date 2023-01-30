# Benchmark on SIFT-1M dataset

Dataset URI: http://corpus-texmex.irisa.fr/

## Preparation

```sh

wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
```

Convert SIFT dataset to `lance` format.

```
 ./datagen.py ./sift/sift_base.fvecs sift.lance
```