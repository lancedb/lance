use lance_index::vector::hnsw::storage::RocksDBGraphStorage;

fn main() {
    let storage = RocksDBGraphStorage::try_new("hnsw.idx").unwrap();
    storage.dump();
}
