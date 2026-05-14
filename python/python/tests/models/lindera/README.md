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

- Create `metadata.json` for the IPADIC schema. Lindera 3.x requires this file
  when building and loading dictionaries.

- build

```bash
lindera build --src mecab-ipadic-2.7.0-20070801 --dest main --metadata metadata.json
```

- build user dict

```bash
lindera build --user --src user_dict/userdic.csv --dest user_dict2 --metadata metadata.json
```

## Version Compatibility

**Important**: The binary user dictionary format (`userdic.bin`) is version-specific and needs to be regenerated when upgrading lindera versions.

- Current version: lindera 3.0.7
- Last regenerated: 2026-05-08
- Binary format changes between versions will cause deserialization errors
