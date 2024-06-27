use rand::Rng;
use fsst::fsst::{compress, decompress};

const TEST_NUM: usize = 10;
const BUFFER_SIZE: usize = 16 * 1024 * 1024;

use std::fs::File;
use std::io::{BufRead, BufReader};
use arrow::array::StringArray; // Add this import statement

fn read_random_16_m_chunk(file_path: &str) -> Result<StringArray, std::io::Error> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let lines: Vec<String> = reader.lines().collect::<std::result::Result<_, _>>()?;
    let num_lines = lines.len();

    let mut rng = rand::thread_rng();
    let mut curr_line = rng.gen_range(0..num_lines);

    let chunk_size = BUFFER_SIZE; // 16MB
    let mut size = 0;
    let mut result_lines = vec![];
    while size + lines[curr_line].len() < chunk_size {
        result_lines.push(lines[curr_line].clone());
        size += lines[curr_line].len();
        curr_line += 1;
        curr_line %= num_lines;
    }

    Ok(StringArray::from(result_lines))
}

fn benchmark(file_path: &str) {
    // Step 1: load data in memory
    let mut inputs: Vec<StringArray> = vec![];
    //let file_path = "/home/x/first_column_fulldocs.tsv";
    for _ in 0..TEST_NUM {
        let this_input = read_random_16_m_chunk(&file_path).unwrap();
        //let this_input = read_file(&file_path).unwrap();
        inputs.push(this_input);
    }

    // Step 2: allocate memory for compression and decompression outputs
    let mut compression_out_bufs = vec![];
    let mut compression_out_offsets_bufs = vec![];
    for _ in 0..TEST_NUM {
        let this_com_out_buf = vec![0u8; BUFFER_SIZE];
        let this_com_out_offsets_buf = vec![0i32; BUFFER_SIZE];
        compression_out_bufs.push(this_com_out_buf);
        compression_out_offsets_bufs.push(this_com_out_offsets_buf);
    }
    let mut decompression_out_bufs = vec![];
    let mut decompression_out_offsets_bufs = vec![];
    for _ in 0..TEST_NUM {
        let this_decom_out_buf = vec![0u8; BUFFER_SIZE * 3];
        let this_decom_out_offsets_buf = vec![0i32; BUFFER_SIZE * 3];
        decompression_out_bufs.push(this_decom_out_buf);
        decompression_out_offsets_bufs.push(this_decom_out_offsets_buf);
    }


    let original_total_size: usize = inputs.iter().map(|input| input.values().len()).sum();

    // Step 3: compress data
    let start = std::time::Instant::now();
    for i in 0..TEST_NUM {
        compress(&inputs[i].values(), inputs[i].value_offsets(), &mut compression_out_bufs[i], &mut compression_out_offsets_bufs[i]).unwrap();
    }
    let compression_finish_time = std::time::Instant::now();

    for i in 0..TEST_NUM {
        decompress(&compression_out_bufs[i], &compression_out_offsets_bufs[i], &mut decompression_out_bufs[i], &mut decompression_out_offsets_bufs[i]).unwrap();
    }
    let decompression_finish_time = std::time::Instant::now();
    let compression_total_size: usize = compression_out_bufs.iter().map(|buf| buf.len()).sum();
    let compression_ratio = original_total_size as f64 / compression_total_size as f64;
    let compress_time = compression_finish_time - start;
    let decompress_time = decompression_finish_time - compression_finish_time;

    let compress_seconds = compress_time.as_secs() as f64
        + compress_time.subsec_nanos() as f64 * 1e-9;

    let decompress_seconds = decompress_time.as_secs() as f64
        + decompress_time.subsec_nanos() as f64 * 1e-9;

    let com_speed = (original_total_size as f64 / compress_seconds) / 1024f64 / 1024f64;

    let d_speed = (original_total_size as f64 / decompress_seconds) / 1024f64 / 1024f64;
    //println!("Compressed form: {:?}", &compression_out_bufs[0][..compression_out_offsets_bufs[0][1] as usize]);
    /* 
    println!("Compressed form: {:?}", &compression_out_bufs[0][compression_out_offsets_bufs[0][0] as usize..compression_out_offsets_bufs[0][1] as usize]);
    println!("input in string: {:?}", std::str::from_utf8(&inputs[0].value_data()[inputs[0].value_offsets()[0] as usize ..inputs[0].value_offsets()[1] as usize]));
    println!("input: {:?}", &inputs[0].value_data()[inputs[0].value_offsets()[0] as usize ..inputs[0].value_offsets()[1] as usize]);
    println!("output: {:?}", &decompression_out_bufs[0][decompression_out_offsets_bufs[0][0] as usize ..decompression_out_offsets_bufs[0][1] as usize]);
    println!("output in string: {:?}", std::str::from_utf8(&decompression_out_bufs[0][decompression_out_offsets_bufs[0][0] as usize ..decompression_out_offsets_bufs[0][1] as usize]));
    */
    for i in 0..TEST_NUM {
        //assert_eq!(inputs[i].value_data().len(), decompression_out_bufs[i].len());
        assert_eq!(inputs[i].value_offsets().len(), decompression_out_offsets_bufs[i].len());
        //assert_eq!(inputs[i].value_data(), decompression_out_bufs[i]);
        //assert_eq!(inputs[i].value_offsets(), decompression_out_offsets_bufs[i]);
    }

    // Print tsv headers
    println!("for file: {}", file_path);
    println!(
        "{}\t{}\t{}",
        "Compression ratio",
        "Compression speed",
        "Decompression speed"
    );
    println!("{:.3}\t\t\t\t{:.2}MB/s\t\t\t{:.2}MB/s", compression_ratio, com_speed, d_speed);
    for i in 0..TEST_NUM {
        assert_eq!(inputs[i].value_data(), decompression_out_bufs[i]);
        assert_eq!(inputs[i].value_offsets(), decompression_out_offsets_bufs[i]);
    }
}

fn main() {
    //let args = Args::parse();
    //let file_path = args.dir.to_str().unwrap();
    // let file_path = "/home/x/first_column_fulldocs.tsv";
    //let file_path = "/home/x/second_column_fulldocs.tsv";
    let file_paths = [
        "/home/x/first_column_fulldocs.tsv",
        "/home/x/second_column_fulldocs.tsv",
        "/home/x/third_column_fulldocs_chunk_0.tsv",
    ];
/* 
    let file_paths = [
        "/home/x/dbtext/c_name",
        "/home/x/dbtext/l_comment",
        "/home/x/dbtext/wikipedia",
        "/home/x/dbtext/chinese"
    ];
*/
    for file_path in file_paths {
        benchmark(file_path);
    } 
}