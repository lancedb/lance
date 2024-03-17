# Benchmarks for BigANN dataset

## Prepare dataset

```
python -m venv venv
. ./venv/bin/activate
git clone https://github.com/harsha-simhadri/big-ann-benchmarks.git
cd big-ann-benchmarks
pip install -r requirements_py3.10.txt
```

## Prepare text-to-image 10M dataset

Create `text2image-10m` in lance format, run:

```
python ./big-ann-benchmarks/create_dataset.py --dataset yfcc-10M
./dataset.py -t text2image-10m data/text2image1B
```

After execution, two datasets will be created:

- *text2image-10m.lance* : base dataset
- *text2image-10m-queries.lance* : quries / GT dataset.


