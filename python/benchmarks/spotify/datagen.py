#!/usr/bin/env python3
#

import os
from random import choice

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
    embeddings = []
    for query_item in col:
        if query_item not in model.wv:
            query_item = choice(list(model.wv.index_to_key))
        embeding = model.wv[query_item]
        embeddings.append(embeding)

    print(embeddings)
    return embeddings


def main():
    # Get dataset from:
    # https://www.kaggle.com/datasets/andrewmvd/spotify-playlists
    #
    df = pd.read_csv("spotify_dataset.csv", on_bad_lines="skip")
    # clean up the col names
    df.columns = df.columns.str.replace('"', "")
    df.columns = df.columns.str.replace("name", "")
    df.columns = df.columns.str.replace(" ", "")
    print(df)
    print(f"Value counts: {df.nunique(axis=0)}")
    for col in df.columns:
        df[col] = df[col].astype("category")

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

    schema = pa.schema(
        [
            pa.field("user_id", pa.dictionary(pa.uint16(), pa.utf8())),
            pa.field("artist", pa.dictionary(pa.uint32(), pa.utf8())),
            pa.field("track", pa.dictionary(pa.uint32(), pa.utf8())),
            pa.field("playlist", pa.dictionary(pa.uint32(), pa.utf8())),
            pa.field("embeddings", pa.list_(pa.float32(), 48))
        ]
    )
    table = pa.Table.from_pandas(df, schema=schema)

    lance.write_dataset(table, "spotify.lance", max_rows_per_group=8192)


if __name__ == "__main__":
    main()
