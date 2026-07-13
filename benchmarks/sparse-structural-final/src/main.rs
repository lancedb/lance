#![allow(clippy::print_stdout)]

use std::{
    collections::HashMap,
    env,
    fs::{self, File},
    path::PathBuf,
    sync::Arc,
    time::Instant,
};

use anyhow::{Context, Result, bail};
use arrow_array::{
    ArrayRef, FixedSizeListArray, Int32Array, ListArray, RecordBatch, RecordBatchReader,
    StructArray, UInt32Array,
};
use arrow_buffer::{BooleanBuffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{ArrowError, DataType, Field, Fields, Schema, SchemaRef};
use futures::{StreamExt, TryStreamExt};
use lance::{
    Dataset,
    dataset::{ProjectionRequest, WriteMode, WriteParams},
    io::ObjectStore,
};
use lance_file::{reader::FileReader, version::LanceFileVersion};
use lance_io::{
    scheduler::{ScanScheduler, SchedulerConfig},
    utils::CachedFileSize,
};
use parquet::{
    arrow::ArrowWriter,
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};
use vortex::{
    VortexSessionDefault,
    array::{ArrayRef as VortexArrayRef, IntoArray, arrays::ChunkedArray, arrow::FromArrowArray},
    dtype::{DType, arrow::FromArrowType},
    file::WriteOptionsSessionExt,
    io::session::RuntimeSessionExt,
    session::VortexSession,
};

const STRUCTURAL_KEY: &str = "lance-encoding:structural-encoding";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CaseKind {
    Hnsw,
    Uniform,
    Deep,
}

impl CaseKind {
    fn name(self) -> &'static str {
        match self {
            Self::Hnsw => "hnsw",
            Self::Uniform => "uniform",
            Self::Deep => "deep",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "hnsw" => Ok(Self::Hnsw),
            "uniform" => Ok(Self::Uniform),
            "deep" => Ok(Self::Deep),
            other => bail!("unknown case {other}"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CaseSpec {
    kind: CaseKind,
    rows: u64,
}

impl CaseSpec {
    fn dense_rows(self) -> u64 {
        match self.kind {
            CaseKind::Hnsw => self.rows.saturating_mul(40_000) / 280_000,
            CaseKind::Uniform | CaseKind::Deep => 0,
        }
    }

    fn non_empty_stride(self) -> u64 {
        match self.kind {
            CaseKind::Hnsw => 1,
            CaseKind::Uniform => 10_000,
            CaseKind::Deep => 4_096,
        }
    }

    fn non_empty_trigger(self) -> u64 {
        match self.kind {
            CaseKind::Hnsw => 0,
            CaseKind::Uniform | CaseKind::Deep => self.non_empty_stride() / 2,
        }
    }

    fn available_non_empty(self) -> u64 {
        match self.kind {
            CaseKind::Hnsw => self.dense_rows(),
            CaseKind::Uniform | CaseKind::Deep => {
                let trigger = self.non_empty_trigger();
                if self.rows <= trigger {
                    0
                } else {
                    (self.rows - trigger).div_ceil(self.non_empty_stride())
                }
            }
        }
    }

    fn non_empty_row(self, ordinal: u64) -> u64 {
        let available = self.available_non_empty().max(1);
        match self.kind {
            CaseKind::Hnsw => ordinal % available,
            CaseKind::Uniform | CaseKind::Deep => {
                self.non_empty_trigger() + (ordinal % available) * self.non_empty_stride()
            }
        }
    }

    fn empty_row(self, ordinal: u64) -> u64 {
        match self.kind {
            CaseKind::Hnsw => {
                let first_empty = self.dense_rows();
                first_empty + ordinal % self.rows.saturating_sub(first_empty).max(1)
            }
            CaseKind::Uniform | CaseKind::Deep => {
                ordinal.saturating_mul(self.non_empty_stride()) % self.rows
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Mode {
    Sparse,
    Miniblock,
    Fullzip,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Sparse => "sparse",
            Self::Miniblock => "miniblock",
            Self::Fullzip => "fullzip",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "sparse" => Ok(Self::Sparse),
            "miniblock" => Ok(Self::Miniblock),
            "fullzip" => Ok(Self::Fullzip),
            other => bail!("unknown mode {other}"),
        }
    }
}

struct SyntheticReader {
    spec: CaseSpec,
    schema: SchemaRef,
    next_row: u64,
    batch_rows: usize,
}

impl SyntheticReader {
    fn new(spec: CaseSpec, mode: Option<Mode>, batch_rows: usize) -> Self {
        Self {
            spec,
            schema: schema_for(spec.kind, mode),
            next_row: 0,
            batch_rows,
        }
    }
}

impl Iterator for SyntheticReader {
    type Item = std::result::Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_row >= self.spec.rows {
            return None;
        }
        let remaining = (self.spec.rows - self.next_row) as usize;
        let len = remaining.min(self.batch_rows);
        let start = self.next_row;
        self.next_row += len as u64;
        if start.is_multiple_of(10_000_000) {
            eprintln!(
                "generate_progress case={} rows_generated={start}",
                self.spec.kind.name()
            );
        }
        Some(make_batch(self.spec, self.schema.clone(), start, len))
    }
}

impl RecordBatchReader for SyntheticReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

fn schema_for(kind: CaseKind, mode: Option<Mode>) -> SchemaRef {
    let data_type = match kind {
        CaseKind::Hnsw | CaseKind::Uniform => {
            DataType::List(Arc::new(Field::new("item", DataType::UInt32, true)))
        }
        CaseKind::Deep => deep_data_type(),
    };
    let mut field = Field::new("c", data_type, true);
    if let Some(mode) = mode {
        field = field.with_metadata(HashMap::from([(
            STRUCTURAL_KEY.to_string(),
            mode.name().to_string(),
        )]));
    }
    Arc::new(Schema::new(vec![field]))
}

fn deep_data_type() -> DataType {
    let tags = DataType::List(Arc::new(Field::new("item", DataType::Int32, true)));
    let pair = DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, true)), 2);
    let event = DataType::Struct(Fields::from(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("tags", tags, true),
        Field::new("pair", pair, true),
    ]));
    let events = DataType::List(Arc::new(Field::new("item", event, true)));
    DataType::Struct(Fields::from(vec![Field::new("events", events, true)]))
}

fn make_batch(
    spec: CaseSpec,
    schema: SchemaRef,
    start: u64,
    len: usize,
) -> std::result::Result<RecordBatch, ArrowError> {
    let array = match spec.kind {
        CaseKind::Hnsw => make_hnsw_batch(spec, start, len)?,
        CaseKind::Uniform => make_uniform_batch(spec, start, len)?,
        CaseKind::Deep => make_deep_batch(spec, start, len)?,
    };
    RecordBatch::try_new(schema, vec![array])
}

fn make_hnsw_batch(
    spec: CaseSpec,
    start: u64,
    len: usize,
) -> std::result::Result<ArrayRef, ArrowError> {
    let dense_rows = spec.dense_rows();
    let mut offsets = Vec::with_capacity(len + 1);
    let dense_rows_in_batch = dense_rows.saturating_sub(start).min(len as u64) as usize;
    let mut values = Vec::with_capacity(dense_rows_in_batch * 32);
    let mut offset = 0_i32;
    offsets.push(offset);
    for local in 0..len {
        let row = start + local as u64;
        if row < dense_rows {
            for neighbor in 0..32_u64 {
                values.push(row.wrapping_mul(32).wrapping_add(neighbor) as u32);
            }
            offset += 32;
        }
        offsets.push(offset);
    }
    let list = ListArray::try_new(
        Arc::new(Field::new("item", DataType::UInt32, true)),
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Arc::new(UInt32Array::from(values)),
        None,
    )?;
    Ok(Arc::new(list))
}

fn make_uniform_batch(
    spec: CaseSpec,
    start: u64,
    len: usize,
) -> std::result::Result<ArrayRef, ArrowError> {
    let stride = spec.non_empty_stride();
    let trigger = spec.non_empty_trigger();
    let mut offsets = Vec::with_capacity(len + 1);
    let mut values = Vec::with_capacity(len / stride as usize + 1);
    let mut offset = 0_i32;
    offsets.push(offset);
    for local in 0..len {
        let row = start + local as u64;
        if row % stride == trigger {
            values.push(row as u32);
            offset += 1;
        }
        offsets.push(offset);
    }
    let list = ListArray::try_new(
        Arc::new(Field::new("item", DataType::UInt32, true)),
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Arc::new(UInt32Array::from(values)),
        None,
    )?;
    Ok(Arc::new(list))
}

fn make_deep_batch(
    spec: CaseSpec,
    start: u64,
    len: usize,
) -> std::result::Result<ArrayRef, ArrowError> {
    let mut event_offsets = Vec::with_capacity(len + 1);
    let mut event_validity = Vec::with_capacity(len);
    let mut top_validity = Vec::with_capacity(len);
    let mut event_count = 0_i32;
    event_offsets.push(event_count);

    let mut event_struct_validity = Vec::new();
    let mut ids = Vec::new();
    let mut tag_offsets = vec![0_i32];
    let mut tag_values = Vec::new();
    let mut tag_validity = Vec::new();
    let mut tag_offset = 0_i32;
    let mut pair_values = Vec::new();
    let mut pair_validity = Vec::new();

    let stride = spec.non_empty_stride();
    let trigger = spec.non_empty_trigger();
    for local in 0..len {
        let row = start + local as u64;
        let is_top_valid = row % (stride * 8) != 3;
        let is_event_list_valid = row % (stride * 4) != 5;
        top_validity.push(is_top_valid);
        event_validity.push(is_event_list_valid);

        if is_top_valid && is_event_list_valid && row % stride == trigger {
            let events_this_row = if (row / stride).is_multiple_of(3) {
                2
            } else {
                1
            };
            for event_idx in 0..events_this_row {
                let event_id = row as i32 + event_idx;
                let is_event_valid = event_idx == 0 || row % (stride * 6) != trigger;
                event_struct_validity.push(is_event_valid);
                ids.push(if row % (stride * 5) == trigger && event_idx == 0 {
                    None
                } else {
                    Some(event_id)
                });

                let tag_count = match (row + event_idx as u64) % 4 {
                    0 => 0,
                    1 => 1,
                    _ => 3,
                };
                let tags_valid = row % (stride * 7) != trigger || event_idx != 0;
                tag_validity.push(tags_valid);
                for tag in 0..tag_count {
                    tag_values.push(event_id + tag);
                    tag_offset += 1;
                }
                tag_offsets.push(tag_offset);

                let pair_valid = row % (stride * 11) != trigger || event_idx != 0;
                pair_validity.push(pair_valid);
                pair_values.push(Some(event_id));
                pair_values.push(if pair_valid {
                    Some(event_id + 1_000)
                } else {
                    None
                });
                event_count += 1;
            }
        }
        event_offsets.push(event_count);
    }

    let ids = Arc::new(Int32Array::from(ids)) as ArrayRef;
    let tags = Arc::new(ListArray::try_new(
        Arc::new(Field::new("item", DataType::Int32, true)),
        OffsetBuffer::new(ScalarBuffer::from(tag_offsets)),
        Arc::new(Int32Array::from(tag_values)),
        Some(NullBuffer::new(BooleanBuffer::from(tag_validity))),
    )?) as ArrayRef;
    let pair = Arc::new(FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Int32, true)),
        2,
        Arc::new(Int32Array::from(pair_values)),
        Some(NullBuffer::new(BooleanBuffer::from(pair_validity))),
    )?) as ArrayRef;
    let event_struct = Arc::new(StructArray::new(
        Fields::from(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("tags", tags.data_type().clone(), true),
            Field::new("pair", pair.data_type().clone(), true),
        ]),
        vec![ids, tags, pair],
        Some(NullBuffer::new(BooleanBuffer::from(event_struct_validity))),
    )) as ArrayRef;
    let events = Arc::new(ListArray::try_new(
        Arc::new(Field::new("item", event_struct.data_type().clone(), true)),
        OffsetBuffer::new(ScalarBuffer::from(event_offsets)),
        event_struct,
        Some(NullBuffer::new(BooleanBuffer::from(event_validity))),
    )?) as ArrayRef;
    let top = StructArray::new(
        Fields::from(vec![Field::new("events", events.data_type().clone(), true)]),
        vec![events],
        Some(NullBuffer::new(BooleanBuffer::from(top_validity))),
    );
    Ok(Arc::new(top))
}

#[derive(Debug, Clone, Copy)]
struct DatasetStats {
    bytes: u64,
    objects: u64,
    data_files: u64,
    pages: u64,
}

async fn dataset_stats(uri: &str) -> Result<DatasetStats> {
    let (store, base) = ObjectStore::from_uri(uri).await?;
    let scheduler = ScanScheduler::new(
        store.clone(),
        SchedulerConfig::max_bandwidth(store.as_ref()),
    );
    let mut stats = DatasetStats {
        bytes: 0,
        objects: 0,
        data_files: 0,
        pages: 0,
    };
    let mut stream = store.list(Some(base));
    while let Some(meta) = stream.try_next().await? {
        stats.bytes += meta.size;
        stats.objects += 1;
        if meta.location.as_ref().ends_with(".lance") {
            stats.data_files += 1;
            let file_scheduler = scheduler
                .open_file(&meta.location, &CachedFileSize::new(meta.size))
                .await?;
            let metadata = FileReader::read_all_metadata(&file_scheduler).await?;
            stats.pages += metadata
                .column_infos
                .iter()
                .map(|column| column.page_infos.len() as u64)
                .sum::<u64>();
        }
    }
    Ok(stats)
}

async fn scan_rows(ds: &Dataset) -> Result<usize> {
    let mut scanner = ds.scan();
    scanner.project(&["c"])?;
    let mut stream = scanner.try_into_stream().await?;
    let mut rows = 0_usize;
    while let Some(batch) = stream.next().await {
        rows += batch?.num_rows();
    }
    Ok(rows)
}

async fn take_rows(ds: &Dataset, indices: &[u64]) -> Result<usize> {
    let projection = ProjectionRequest::from_columns(["c"], ds.schema());
    Ok(ds.take(indices, projection).await?.num_rows())
}

fn random_indices(rows: u64, count: usize) -> Vec<u64> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state % rows
        })
        .collect()
}

fn non_empty_indices(spec: CaseSpec, count: usize) -> Vec<u64> {
    (0..count as u64)
        .map(|ordinal| spec.non_empty_row(ordinal))
        .collect()
}

fn empty_indices(spec: CaseSpec, count: usize) -> Vec<u64> {
    (0..count as u64)
        .map(|ordinal| spec.empty_row(ordinal))
        .collect()
}

async fn measure_scan(uri: &str) -> Result<(f64, f64, usize, usize)> {
    let ds = Dataset::open(uri).await?;
    let start = Instant::now();
    let cold_rows = scan_rows(&ds).await?;
    let cold_ms = start.elapsed().as_secs_f64() * 1_000.0;
    let start = Instant::now();
    let warm_rows = scan_rows(&ds).await?;
    let warm_ms = start.elapsed().as_secs_f64() * 1_000.0;
    Ok((cold_ms, warm_ms, cold_rows, warm_rows))
}

async fn measure_take(uri: &str, indices: &[u64]) -> Result<(f64, f64, usize, usize)> {
    let ds = Dataset::open(uri).await?;
    let start = Instant::now();
    let cold_rows = take_rows(&ds, indices).await?;
    let cold_ms = start.elapsed().as_secs_f64() * 1_000.0;
    let start = Instant::now();
    let warm_rows = take_rows(&ds, indices).await?;
    let warm_ms = start.elapsed().as_secs_f64() * 1_000.0;
    Ok((cold_ms, warm_ms, cold_rows, warm_rows))
}

#[allow(clippy::too_many_arguments)]
fn print_lance_metric(
    case: CaseKind,
    mode: Mode,
    rows: u64,
    stats: DatasetStats,
    op: &str,
    phase: &str,
    ms: f64,
    out_rows: usize,
    uri: &str,
) {
    println!(
        "{},{},{},{},{},{},{},{},{},{:.3},{},{}",
        case.name(),
        mode.name(),
        rows,
        stats.bytes,
        stats.objects,
        stats.data_files,
        stats.pages,
        op,
        phase,
        ms,
        out_rows,
        uri
    );
}

async fn run_lance_case_mode(
    bucket: &str,
    prefix: &str,
    spec: CaseSpec,
    mode: Mode,
    batch_rows: usize,
    take_count: usize,
) -> Result<()> {
    let uri = format!(
        "s3://{bucket}/{prefix}/{}/{}",
        spec.kind.name(),
        mode.name()
    );
    eprintln!(
        "write_start uri={uri} rows={} case={} mode={}",
        spec.rows,
        spec.kind.name(),
        mode.name()
    );
    let reader = SyntheticReader::new(spec, Some(mode), batch_rows);
    let params = WriteParams {
        mode: WriteMode::Create,
        data_storage_version: Some(LanceFileVersion::V2_3),
        max_rows_per_file: 1_000_000,
        max_rows_per_group: batch_rows,
        ..Default::default()
    };
    let start = Instant::now();
    let ds = Dataset::write(reader, uri.as_str(), Some(params))
        .await
        .with_context(|| format!("write failed for {uri}"))?;
    let write_ms = start.elapsed().as_secs_f64() * 1_000.0;
    let count = ds.count_rows(None).await?;
    if count as u64 != spec.rows {
        bail!(
            "row count mismatch for {uri}: expected {}, got {count}",
            spec.rows
        );
    }
    let stats = dataset_stats(&uri).await?;
    print_lance_metric(
        spec.kind, mode, spec.rows, stats, "write", "single", write_ms, count, &uri,
    );

    let (cold, warm, cold_rows, warm_rows) = measure_scan(&uri).await?;
    print_lance_metric(
        spec.kind,
        mode,
        spec.rows,
        stats,
        "scan_full",
        "cold",
        cold,
        cold_rows,
        &uri,
    );
    print_lance_metric(
        spec.kind,
        mode,
        spec.rows,
        stats,
        "scan_full",
        "warm",
        warm,
        warm_rows,
        &uri,
    );

    let operations = [
        ("take_random", random_indices(spec.rows, take_count)),
        ("take_empty", empty_indices(spec, take_count)),
        ("take_non_empty", non_empty_indices(spec, take_count)),
    ];
    for (op, indices) in operations {
        let (cold, warm, cold_rows, warm_rows) = measure_take(&uri, &indices).await?;
        print_lance_metric(
            spec.kind, mode, spec.rows, stats, op, "cold", cold, cold_rows, &uri,
        );
        print_lance_metric(
            spec.kind, mode, spec.rows, stats, op, "warm", warm, warm_rows, &uri,
        );
    }
    eprintln!(
        "case_complete case={} mode={} bytes={} objects={} data_files={} pages={}",
        spec.kind.name(),
        mode.name(),
        stats.bytes,
        stats.objects,
        stats.data_files,
        stats.pages
    );
    Ok(())
}

fn write_parquet(spec: CaseSpec, batch_rows: usize, out_dir: &str) -> Result<(u64, f64, PathBuf)> {
    let schema = schema_for(spec.kind, None);
    let path = PathBuf::from(out_dir).join(format!("{}.parquet", spec.kind.name()));
    if let Err(error) = fs::remove_file(&path)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        return Err(error.into());
    }
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        .set_max_row_group_row_count(Some(batch_rows))
        .build();
    let file = File::create(&path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(properties))?;
    let start = Instant::now();
    for batch in SyntheticReader::new(spec, None, batch_rows) {
        writer.write(&batch?)?;
    }
    writer.close()?;
    let seconds = start.elapsed().as_secs_f64();
    Ok((fs::metadata(&path)?.len(), seconds, path))
}

async fn write_vortex(
    spec: CaseSpec,
    batch_rows: usize,
    out_dir: &str,
) -> Result<(u64, f64, PathBuf)> {
    let schema = schema_for(spec.kind, None);
    let path = PathBuf::from(out_dir).join(format!("{}.vortex", spec.kind.name()));
    if let Err(error) = fs::remove_file(&path)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        return Err(error.into());
    }
    let dtype = DType::from_arrow(schema.as_ref());
    let start = Instant::now();
    let mut chunks = Vec::<VortexArrayRef>::new();
    for batch in SyntheticReader::new(spec, None, batch_rows) {
        chunks.push(VortexArrayRef::from_arrow(batch?, false)?);
    }
    let array = ChunkedArray::try_new(chunks, dtype)?.into_array();
    let session = VortexSession::default().with_tokio();
    let mut file = async_fs::File::create(&path).await?;
    session
        .write_options()
        .write(&mut file, array.to_array_stream())
        .await?;
    drop(file);
    let seconds = start.elapsed().as_secs_f64();
    Ok((fs::metadata(&path)?.len(), seconds, path))
}

fn parse_list<T>(name: &str, default: &str, parse: impl Fn(&str) -> Result<T>) -> Result<Vec<T>> {
    let value = env::var(name).unwrap_or_else(|_| default.to_string());
    value
        .split(',')
        .filter(|part| !part.is_empty())
        .map(parse)
        .collect()
}

fn parse_env<T>(name: &str, default: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .parse::<T>()
        .with_context(|| format!("invalid {name}"))
}

async fn run_lance() -> Result<()> {
    let rows = parse_env("ROWS", "100000000")?;
    if rows == 0 {
        bail!("ROWS must be greater than zero");
    }
    let batch_rows = parse_env("BATCH_ROWS", "65536")?;
    let take_count = parse_env("TAKE_COUNT", "1024")?;
    let bucket =
        env::var("S3_BUCKET").unwrap_or_else(|_| "lance-bench-054483968661-us-east-2".to_string());
    let prefix = env::var("S3_PREFIX").context("S3_PREFIX is required")?;
    let cases = parse_list("CASES", "hnsw,uniform,deep", CaseKind::parse)?;
    let modes = parse_list("MODES", "sparse,miniblock,fullzip", Mode::parse)?;

    println!("case,mode,rows,bytes,objects,data_files,pages,op,phase,ms,out_rows,uri");
    for case in cases {
        let spec = CaseSpec { kind: case, rows };
        if spec.available_non_empty() == 0 {
            bail!("case {} has no non-empty rows at ROWS={rows}", case.name());
        }
        for mode in &modes {
            run_lance_case_mode(&bucket, &prefix, spec, *mode, batch_rows, take_count).await?;
        }
    }
    Ok(())
}

async fn run_formats() -> Result<()> {
    let rows = parse_env("ROWS", "100000000")?;
    if rows == 0 {
        bail!("ROWS must be greater than zero");
    }
    let batch_rows = parse_env("FORMAT_BATCH_ROWS", "1000000")?;
    let out_dir = env::var("OUT_DIR").context("OUT_DIR is required")?;
    fs::create_dir_all(&out_dir)?;
    let cases = parse_list("CASES", "hnsw,uniform,deep", CaseKind::parse)?;

    println!("case,format,rows,bytes,seconds,path");
    for case in cases {
        let spec = CaseSpec { kind: case, rows };
        let (bytes, seconds, path) = write_parquet(spec, batch_rows, &out_dir)?;
        println!(
            "{},parquet_zstd3,{},{},{:.3},{}",
            case.name(),
            rows,
            bytes,
            seconds,
            path.display()
        );
        let (bytes, seconds, path) = write_vortex(spec, batch_rows, &out_dir).await?;
        println!(
            "{},vortex_default,{},{},{:.3},{}",
            case.name(),
            rows,
            bytes,
            seconds,
            path.display()
        );
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    match env::var("ACTION").as_deref() {
        Ok("lance") => run_lance().await,
        Ok("formats") => run_formats().await,
        Ok(other) => bail!("unknown ACTION {other}"),
        Err(_) => bail!("ACTION must be lance or formats"),
    }
}
