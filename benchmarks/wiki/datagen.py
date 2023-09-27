import lance
import pyarrow as pa
import tqdm
from datasets import DownloadConfig, load_dataset


def convert_to_vec(batches, schema):
    for batch in batches:
        arr = batch["emb"]
        vec = pa.FixedSizeListArray.from_arrays(
            arr.values,
            768,
        )
        assert len(vec) == len(batch)
        cols = list(batch.columns[:-1])
        cols.append(vec)
        yield pa.RecordBatch.from_arrays(cols, schema=schema)


def main():
    dataset_en = "Cohere/wikipedia-22-12-en-embeddings"
    docs = load_dataset(
        dataset_en,
        split="train",
        download_config=DownloadConfig(num_proc=8, resume_download=True),
    )

    table: pa.Table = docs.data

    schema = docs.data.schema
    assert schema.get_field_index("emb") == 8
    assert len(schema) == 9
    schema = schema.remove(schema.get_field_index("emb"))
    schema = schema.append(pa.field("emb", pa.list_(pa.float32(), 768)))
    # need this so schema comparison works
    schema = schema.remove_metadata()

    lance.write_dataset(
        tqdm.tqdm(list(convert_to_vec(table.to_batches(), schema))),
        "wiki.lance",
        schema=schema,
        max_rows_per_group=10240,
    )


if __name__ == "__main__":
    main()
