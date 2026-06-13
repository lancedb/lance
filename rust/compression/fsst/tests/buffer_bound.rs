// Regression test for the out-of-bounds write when data compresses better than
// 3:1 (a code byte can expand to an 8-byte symbol, but decompress only required a
// 3x output buffer).
use arrow_array::StringArray;
use fsst::fsst::{FSST_SYMBOL_TABLE_SIZE, compress, decompress};

#[test]
fn decompress_high_ratio_into_documented_minimum_buffer() {
    // >= 32 KiB so the encoder turns on; very repetitive so it compresses ~8:1.
    let row = "a".repeat(64);
    let rows: Vec<&str> = (0..2000).map(|_| row.as_str()).collect();
    let array = StringArray::from(rows);
    let in_data = array.value_data();
    let in_offsets = array.value_offsets();

    let mut comp_buf: Vec<u8> = vec![0; in_data.len()];
    let mut comp_offsets: Vec<i32> = vec![0; in_offsets.len()];
    let mut symbol_table = [0u8; FSST_SYMBOL_TABLE_SIZE];
    compress(
        symbol_table.as_mut(),
        in_data,
        in_offsets,
        &mut comp_buf,
        &mut comp_offsets,
    )
    .unwrap();

    // The input compresses much better than 3:1.
    assert!(
        in_data.len() > 3 * comp_buf.len(),
        "expected >3:1 ratio, got {} -> {}",
        in_data.len(),
        comp_buf.len()
    );

    // Allocate exactly the documented minimum (3x the compressed input).
    let mut dec_buf: Vec<u8> = vec![0; 3 * comp_buf.len()];
    let mut dec_offsets: Vec<i32> = vec![0; comp_offsets.len()];
    decompress(
        &symbol_table,
        &comp_buf,
        &comp_offsets,
        &mut dec_buf,
        &mut dec_offsets,
    )
    .unwrap();

    // Round-trips correctly.
    for i in 1..dec_offsets.len() {
        let got = &dec_buf[dec_offsets[i - 1] as usize..dec_offsets[i] as usize];
        let want = &in_data[in_offsets[i - 1] as usize..in_offsets[i] as usize];
        assert_eq!(got, want, "mismatch on row {i}");
    }
}
