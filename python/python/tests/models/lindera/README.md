# How to build this test language model

Ipadic model is about 45M. so we created a tiny ipadic in zip.

- Download language model

```bash
curl -L -o mecab-ipadic-2.7.0-20070801.tar.gz "https://github.com/lindera-morphology/mecab-ipadic/archive/refs/tags/2.7.0-20070801.tar.gz"
tar xvf mecab-ipadic-2.7.0-20070801.tar.gz
```

- Remove csv files in folder

- Put files in `ipadic/raw` into folder

- Edit matrix.def, reset last column(weight) into zero, except first row.

- build

```bash
lindera build --dictionary-kind=ipadic mecab-ipadic-2.7.0-20070801 main
```

- build user dict

```bash
lindera build --build-user-dictionary --dictionary-kind=ipadic user_dict/userdict.csv user_dict2
```
