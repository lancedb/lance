import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import pyarrow.fs as fs


def read_pets_metadata(url):
    list_txt = os.path.join(url, "annotations/list.txt")
    df = pd.read_csv(list_txt, delimiter=" ", comment="#", header=None)
    df.columns = ["filename", "class", "species", "breed"]

    species_dtype = pd.CategoricalDtype(["Unknown", "Cat", "Dog"])
    df.species = pd.Categorical.from_codes(df.species, dtype=species_dtype)

    breeds = df.filename.str.rsplit("_", 1).str[0].unique()
    assert len(breeds) == 37

    breeds = np.concatenate([["Unknown"], breeds])
    class_dtype = pd.CategoricalDtype(breeds)
    df["class"] = pd.Categorical.from_codes(df["class"], dtype=class_dtype)

    return df


def get_pets_class_distribution(url):
    # %time df = bench.get_pets_class_distribution(url)
    df = read_pets_metadata(url)
    return df.groupby("class")["class"].count()


def get_pets_filtered_data(url, klass="pug", offset=20, limit=50):
    # %time rs = bench.get_pets_filtered_data(url, "pug", 20, 50)
    df = read_pets_metadata(url)
    filtered = df.loc[df["class"] == klass, ["class", "filename"]]
    limited = filtered[offset : offset + limit]
    pool = mp.Pool(mp.cpu_count() - 1)
    all_bytes = pool.map(get_bytes, limited.filename.values)
    limited["image"] = all_bytes
    return limited


def get_bytes(name):
    s3, key = fs.FileSystem.from_uri(
        f"s3://eto-public/datasets/oxford_pet/images/{name}.jpg"
    )
    return s3.open_input_file(key).read()
