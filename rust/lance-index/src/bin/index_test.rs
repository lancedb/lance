use std::io::Read;

use lance_index::vector::hnsw::{index::HNSWIndex, storage::RocksDBGraphStorage};
use npy::NpyData;

fn main() {
    let storage = RocksDBGraphStorage::try_new("hnsw.idx").unwrap();
    let mut index = HNSWIndex::new(storage, 1.25, 7, 12, 400, Some(1000000));
    index.query_mode();

    let mut buf = vec![];
    std::fs::File::open("/home/rmeng/workspace/lance/benchmarks/sift/sift_base.npy")
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();

    let data: Vec<f32> = NpyData::from_bytes(&buf).unwrap().to_vec();

    let mut hits = 0;
    for i in kdam::tqdm!(0..10000) {
        let res = index.search(&data[i * 128..(i + 1) * 128], 1, 400).unwrap();
        if res == vec![i as u64] {
            hits += 1;
        }
    }

    println!("\nhits: {}", hits);
}
