#!/usr/bin/env python3
#

import os

import lance
import pyarrow as pa
import pandas as pd
from gensim.models.word2vec import Word2Vec


def generate_embeddings(col: pd.Series, **hyper_params) -> pd.Series:
    # unique_tracks
    sentences = []
    for s in col:
        if not isinstance(s, str):
            continue
        # We could prob do more cleaning here
        sentences.append(s)
    model = Word2Vec(sentences, **hyper_params)
    print(f"Vector space size: {len(model.wv.index_to_key)}")
    print(model.wv[sentences[0]])


def main():
    df = pd.read_csv("spotify_dataset.csv", on_bad_lines="skip")
    # clean up the col names
    df.columns = df.columns.str.replace('"', "")
    df.columns = df.columns.str.replace("name", "")
    df.columns = df.columns.str.replace(" ", "")
    print(df)

    # Hyper parameters from https://outerbounds.com/docs/recsys-tutorial-L4/
    embeddings = generate_embeddings(
        df["track"],
        min_count=3,
        epochs=30,
        vector_size=48,
        window=10,
        ns_exponent=0.75,
        workers=os.cpu_count(),
    )
    df["embeddings"] = embeddings
    print(embeddings)
    table = pa.Table.from_pandas(df)

    # lance.write_dataset(table, "spotify.lance")


if __name__ == "__main__":
    main()
