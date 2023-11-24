use std::io::Read;

use lance_index::vector::hnsw::{
    index::{HNSWIndex, VecAndRowId},
    storage::RocksDBGraphStorage,
};
use npy::NpyData;

fn main() {
    let storage = RocksDBGraphStorage::try_new("hnsw.idx").unwrap();
    let mut index = HNSWIndex::new(storage, 0.33, 7, 8, 300, Some(1000000));

    let mut buf = vec![];
    std::fs::File::open("/home/rmeng/workspace/lance/benchmarks/sift/sift_base.npy")
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();

    let data: Vec<f32> = NpyData::from_bytes(&buf).unwrap().to_vec();

    index
        .index(
            kdam::tqdm!(0..1000000)
                .map(|id| VecAndRowId(&data[id * 128..(id + 1) * 128], id as u64)),
        )
        .unwrap();

    println!("{:?}", index.search(&data[0..128], 10, 300).unwrap());
}
