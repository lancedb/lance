import pyarrow as pa
import lance

from lancedb.embeddings import EmbeddingFunctionRegistry

registry = EmbeddingFunctionRegistry.get_instance()
func = registry.get("open-clip").create(name="ViT-L-14-336", pretrained="openai", device="cuda")

def convert_image_to_vector(data_path: str, batch_size: int):
    dataset = lance.dataset(data_path)
    stream = dataset.to_batches(columns=["key", "img"], batch_size=batch_size)

    for batch in stream:
        yield batch.map(lambda x: {"vector": func.generate_image_embedding(x["img"])})
        del batch

if __name__ == "__main__":
    read_uri = "/home/ubuntu/data/yfcc100m_subset/lance.data"
    write_uri = "/home/ubuntu/data/yfcc100m_subset/lance-embedding.data"
    batch_size = 1024 * 1024
    schema=pa.schema(
            [
                pa.field("key", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), func.ndims())),
            ]
        )
    lance.write_dataset(convert_image_to_vector(read_uri, batch_size), schema=schema, 
                        uri=write_uri, max_rows_per_file=1000000)
    
    # verify the data
    dataset = lance.dataset(write_uri)
    print(dataset.count_rows())
    print(dataset.head(5).to_pandas())