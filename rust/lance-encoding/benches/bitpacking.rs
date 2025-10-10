// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use lance_encoding::buffer::LanceBuffer;
use lance_encoding::compression::{BlockCompressor, BlockDecompressor};
use lance_encoding::data::{BlockInfo, DataBlock, FixedWidthDataBlock};
use lance_encoding::encodings::physical::bitpacking::OutOfLineBitpacking;

const COMPRESSED_BITS: usize = 12;
const UNCOMPRESSED_BITS: usize = 32;

fn generate_values(len: usize) -> Vec<u32> {
    let mask = (1u32 << COMPRESSED_BITS) - 1;
    (0..len)
        .map(|i| ((i as u32).wrapping_mul(2654435761).wrapping_add(12345)) & mask)
        .collect()
}

fn make_block(values: Arc<[u32]>) -> FixedWidthDataBlock {
    let num_values = values.len() as u64;
    FixedWidthDataBlock {
        data: LanceBuffer::reinterpret_slice(values),
        bits_per_value: UNCOMPRESSED_BITS as u64,
        num_values,
        block_info: BlockInfo::new(),
    }
}

fn bench_bitpacking(c: &mut Criterion) {
    let scenarios = [
        ("chunks_2048", 1024 * 2),
        ("tail_1023", 1023),
        ("single_1", 1),
    ];
    let compressor = OutOfLineBitpacking::new(COMPRESSED_BITS as u64, UNCOMPRESSED_BITS as u64);

    for (label, len) in scenarios {
        let values = generate_values(len);
        let arc_values: Arc<[u32]> = values.into();

        let sample_block = make_block(arc_values.clone());
        let compressed = compressor
            .compress(DataBlock::FixedWidth(sample_block))
            .expect("bitpack compress");

        // Sanity check to ensure we measure the intended code paths.
        let decoded = compressor
            .decompress(compressed.clone(), len as u64)
            .expect("bitpack decompress");
        let decoded_block = decoded.as_fixed_width().expect("fixed width block");
        let decoded_values = decoded_block.data.borrow_to_typed_slice::<u32>().to_vec();
        assert_eq!(&*arc_values, decoded_values.as_slice());

        let mut group = c.benchmark_group(format!("bitpacking_{label}"));
        group.throughput(Throughput::Elements(len as u64));

        group.bench_function("compress", |b| {
            b.iter(|| {
                let block = make_block(arc_values.clone());
                black_box(
                    compressor
                        .compress(DataBlock::FixedWidth(block))
                        .expect("bitpack compress"),
                );
            });
        });

        group.bench_function("decompress", |b| {
            b.iter(|| {
                black_box(
                    compressor
                        .decompress(compressed.clone(), len as u64)
                        .expect("bitpack decompress"),
                );
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_bitpacking);
criterion_main!(benches);
