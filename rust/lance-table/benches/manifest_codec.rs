// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reproducible process-isolated benchmark for protobuf and Lance manifests.

use std::collections::HashMap;
#[cfg(target_os = "macos")]
use std::ffi::c_void;
use std::io::Write;
use std::num::NonZeroU64;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use bytes::Bytes;
use lance_core::datatypes::Schema;
use lance_core::{Error, Result};
use lance_file::version::LanceFileVersion;
use lance_io::object_store::ObjectStore;
use lance_table::format::{
    DataFile, DataStorageFormat, DeletionFile, DeletionFileType, Fragment, Manifest,
};
use lance_table::io::commit::write_manifest_file_to_path;
use lance_table::io::manifest::{is_columnar_manifest_footer, read_manifest};
use object_store::{ObjectStoreExt, path::Path};
use serde_json::{Value, json};

const SCHEMA_VERSION: u64 = 2;
const DEFAULT_SEED: u64 = 0x4c41_4e43_455f_4d46;
const ROWS_PER_FRAGMENT: usize = 1_024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    S1,
    S2,
}

impl Scenario {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "S1" => Ok(Self::S1),
            "S2" => Ok(Self::S2),
            _ => Err(Error::invalid_input(format!(
                "scenario must be S1 or S2, found '{value}'"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::S1 => "S1",
            Self::S2 => "S2",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ManifestFormat {
    Protobuf,
    Lance,
}

impl ManifestFormat {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "protobuf" => Ok(Self::Protobuf),
            "lance" => Ok(Self::Lance),
            _ => Err(Error::invalid_input(format!(
                "format must be protobuf or lance, found '{value}'"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Protobuf => "protobuf",
            Self::Lance => "lance",
        }
    }

    fn expects_columnar(self) -> bool {
        self == Self::Lance
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Encode,
    Decode,
    Rss,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Warmup {
    Tiny,
    Cold,
}

impl Warmup {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "tiny" => Ok(Self::Tiny),
            "cold" => Ok(Self::Cold),
            _ => Err(Error::invalid_input(format!(
                "warmup must be tiny or cold, found '{value}'"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Tiny => "tiny",
            Self::Cold => "cold",
        }
    }
}

impl Mode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "encode" => Ok(Self::Encode),
            "decode" => Ok(Self::Decode),
            "rss" => Ok(Self::Rss),
            _ => Err(Error::invalid_input(format!(
                "mode must be encode, decode, or rss, found '{value}'"
            ))),
        }
    }
}

struct Args {
    mode: Mode,
    warmup: Warmup,
    scenario: Scenario,
    fragments: usize,
    format: ManifestFormat,
    round: u64,
    seed: u64,
    commit: String,
    host: String,
    fixture: PathBuf,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut mode = None;
        let mut warmup = None;
        let mut scenario = None;
        let mut fragments = None;
        let mut format = None;
        let mut round = None;
        let mut seed = DEFAULT_SEED;
        let mut commit = None;
        let mut host = None;
        let mut fixture = None;
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            let value = args.next().ok_or_else(|| {
                Error::invalid_input(format!("missing value for argument '{flag}'"))
            })?;
            match flag.as_str() {
                "--mode" => mode = Some(Mode::parse(&value)?),
                "--warmup" => warmup = Some(Warmup::parse(&value)?),
                "--scenario" => scenario = Some(Scenario::parse(&value)?),
                "--fragments" => {
                    fragments = Some(value.parse::<usize>().map_err(|error| {
                        Error::invalid_input(format!(
                            "invalid --fragments value '{value}': {error}"
                        ))
                    })?)
                }
                "--format" => format = Some(ManifestFormat::parse(&value)?),
                "--round" => {
                    round = Some(value.parse::<u64>().map_err(|error| {
                        Error::invalid_input(format!("invalid --round value '{value}': {error}"))
                    })?)
                }
                "--seed" => {
                    seed = value.parse::<u64>().map_err(|error| {
                        Error::invalid_input(format!("invalid --seed value '{value}': {error}"))
                    })?
                }
                "--commit" => commit = Some(value),
                "--host" => host = Some(value),
                "--fixture" => fixture = Some(PathBuf::from(value)),
                _ => {
                    return Err(Error::invalid_input(format!(
                        "unknown benchmark argument '{flag}'"
                    )));
                }
            }
        }
        let fragments = fragments.ok_or_else(|| Error::invalid_input("missing --fragments"))?;
        if fragments == 0 {
            return Err(Error::invalid_input(
                "--fragments must be greater than zero",
            ));
        }
        let commit = commit.ok_or_else(|| Error::invalid_input("missing --commit"))?;
        let host = host.ok_or_else(|| Error::invalid_input("missing --host"))?;
        if commit.is_empty() || host.is_empty() {
            return Err(Error::invalid_input(
                "--commit and --host must not be empty",
            ));
        }
        Ok(Self {
            mode: mode.ok_or_else(|| Error::invalid_input("missing --mode"))?,
            warmup: warmup.ok_or_else(|| Error::invalid_input("missing --warmup"))?,
            scenario: scenario.ok_or_else(|| Error::invalid_input("missing --scenario"))?,
            fragments,
            format: format.ok_or_else(|| Error::invalid_input("missing --format"))?,
            round: round.ok_or_else(|| Error::invalid_input("missing --round"))?,
            seed,
            commit,
            host,
            fixture: fixture.ok_or_else(|| Error::invalid_input("missing --fixture"))?,
        })
    }
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn sample(seed: u64, fragment_id: u64, stream: u64) -> u64 {
    mix64(seed ^ fragment_id.wrapping_mul(0xd6e8_feb8_6659_fd93) ^ stream)
}

fn selected(seed: u64, fragment_id: u64, stream: u64, one_in: u64) -> bool {
    let offset = sample(seed, 0, stream) % one_in;
    fragment_id.wrapping_add(offset).is_multiple_of(one_in)
}

fn dataset_schema(num_fields: usize) -> Result<Schema> {
    let fields = (0..num_fields)
        .map(|field_id| Field::new(format!("field_{field_id}"), DataType::Int64, true))
        .collect::<Vec<_>>();
    Schema::try_from(&ArrowSchema::new(fields))
}

fn data_file_template(num_fields: usize, version: LanceFileVersion) -> DataFile {
    let fields = (0..num_fields as i32).collect::<Vec<_>>();
    let (major, minor) = version.to_numbers();
    DataFile::new(
        String::new(),
        fields.clone(),
        fields,
        major,
        minor,
        NonZeroU64::new(1_048_576),
        None,
    )
}

fn short_path(fragment_id: u64, ordinal: usize) -> String {
    let value = fragment_id
        .checked_mul(2)
        .and_then(|value| value.checked_add(ordinal as u64))
        .unwrap_or(fragment_id);
    format!("data/{value:032x}.lance")
}

fn long_path(fragment_id: u64, ordinal: usize, entropy: u64) -> String {
    format!("imports/customer-{fragment_id:016x}/partition-{entropy:016x}/part-{ordinal:02}.lance")
}

fn make_file(
    template: &DataFile,
    fragment_id: u64,
    ordinal: usize,
    is_long_path: bool,
    entropy: u64,
) -> DataFile {
    let mut file = template.clone();
    file.path = if is_long_path {
        long_path(fragment_id, ordinal, entropy)
    } else {
        short_path(fragment_id, ordinal)
    };
    file.file_size_bytes = NonZeroU64::new(1_048_576 + entropy % 65_536).into();
    file.base_id = entropy.is_multiple_of(10).then_some(7);
    file
}

fn make_fragments(
    scenario: Scenario,
    count: usize,
    seed: u64,
    version: LanceFileVersion,
) -> Vec<Fragment> {
    let template_8 = data_file_template(8, version);
    let template_32 = data_file_template(32, version);
    let mut fragments = Vec::with_capacity(count);
    for index in 0..count {
        let id = index as u64;
        let layout = sample(seed, id, 0);
        let physical_rows = ROWS_PER_FRAGMENT + (layout as usize % 17);
        let mut fragment = Fragment::new(id);
        match scenario {
            Scenario::S1 => {
                fragment
                    .files
                    .push(make_file(&template_8, id, 0, false, layout));
                fragment.physical_rows = Some(ROWS_PER_FRAGMENT);
            }
            Scenario::S2 => {
                let template = if layout & 1 == 0 {
                    &template_8
                } else {
                    &template_32
                };
                fragment
                    .files
                    .push(make_file(template, id, 0, layout & 2 != 0, layout));
                if selected(seed, id, 1, 20) {
                    fragment.files.push(make_file(
                        &template_32,
                        id,
                        1,
                        sample(seed, id, 2) & 1 != 0,
                        sample(seed, id, 3),
                    ));
                }
                if selected(seed, id, 4, 5) {
                    fragment.deletion_file = Some(DeletionFile {
                        read_version: 1,
                        id,
                        file_type: if layout & 4 == 0 {
                            DeletionFileType::Array
                        } else {
                            DeletionFileType::Bitmap
                        },
                        num_deleted_rows: Some(1 + layout as usize % 31),
                        base_id: layout.is_multiple_of(10).then_some(7),
                    });
                }
                fragment.physical_rows = if selected(seed, id, 5, 100) {
                    None
                } else {
                    Some(physical_rows)
                };
            }
        }
        fragments.push(fragment);
    }
    fragments
}

fn normalize_missing_rows(manifest: &mut Manifest, scenario: Scenario, seed: u64) -> usize {
    if scenario == Scenario::S1 {
        return 0;
    }
    let mut normalized = 0;
    for fragment in Arc::make_mut(&mut manifest.fragments) {
        if fragment.physical_rows.is_none() {
            let layout = sample(seed, fragment.id, 0);
            fragment.physical_rows = Some(ROWS_PER_FRAGMENT + layout as usize % 17);
            normalized += 1;
        }
    }
    normalized
}

fn storage_version(format: ManifestFormat) -> LanceFileVersion {
    match format {
        ManifestFormat::Protobuf => LanceFileVersion::V2_2,
        ManifestFormat::Lance => LanceFileVersion::V2_3,
    }
}

fn make_manifest(
    scenario: Scenario,
    count: usize,
    seed: u64,
    format: ManifestFormat,
) -> Result<(Manifest, usize)> {
    let num_fields = if scenario == Scenario::S1 { 8 } else { 32 };
    let version = storage_version(format);
    let fragments = make_fragments(scenario, count, seed, version);
    let missing_rows = fragments
        .iter()
        .filter(|fragment| fragment.physical_rows.is_none())
        .count();
    if scenario == Scenario::S2 && missing_rows == 0 {
        return Err(Error::internal(
            "S2 workload did not generate any missing physical_rows",
        ));
    }
    let mut manifest = Manifest::new(
        dataset_schema(num_fields)?,
        Arc::new(fragments),
        DataStorageFormat::new(version),
        HashMap::new(),
    );
    manifest.version = 1;
    manifest.max_fragment_id = count
        .checked_sub(1)
        .map(u32::try_from)
        .transpose()
        .map_err(|_| Error::invalid_input("benchmark fragment count exceeds u32::MAX"))?;
    manifest.next_row_id = (count as u64)
        .checked_mul((ROWS_PER_FRAGMENT + 16) as u64)
        .ok_or_else(|| Error::invalid_input("benchmark next_row_id overflows u64"))?;
    manifest
        .config
        .insert("benchmark.seed".to_string(), seed.to_string());
    manifest.table_metadata.insert(
        "benchmark.scenario".to_string(),
        scenario.as_str().to_string(),
    );
    let normalized = normalize_missing_rows(&mut manifest, scenario, seed);
    if normalized != missing_rows {
        return Err(Error::internal(format!(
            "normalized {normalized} missing row counts, expected {missing_rows}"
        )));
    }
    Ok((manifest, normalized))
}

#[cfg(target_os = "linux")]
fn current_rss_bytes() -> u64 {
    let Ok(statm) = std::fs::read_to_string("/proc/self/statm") else {
        return 0;
    };
    let Some(resident_pages) = statm
        .split_whitespace()
        .nth(1)
        .and_then(|value| value.parse::<u64>().ok())
    else {
        return 0;
    };
    // SAFETY: sysconf reads immutable process configuration and has no pointer arguments.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    u64::try_from(page_size)
        .ok()
        .and_then(|page_size| resident_pages.checked_mul(page_size))
        .unwrap_or(0)
}

#[cfg(target_os = "macos")]
fn current_rss_bytes() -> u64 {
    // SAFETY: proc_pidinfo receives a valid, correctly sized out-parameter for this process.
    unsafe {
        let mut info = std::mem::zeroed::<libc::proc_taskinfo>();
        let size = std::mem::size_of::<libc::proc_taskinfo>() as i32;
        let written = libc::proc_pidinfo(
            libc::getpid(),
            libc::PROC_PIDTASKINFO,
            0,
            (&mut info as *mut libc::proc_taskinfo).cast::<c_void>(),
            size,
        );
        if written == size {
            info.pti_resident_size
        } else {
            0
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn current_rss_bytes() -> u64 {
    0
}

struct RssSampler {
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    baseline: u64,
    handle: Option<JoinHandle<()>>,
}

#[derive(Clone, Copy)]
struct RssMeasurement {
    baseline_bytes: u64,
    process_peak_bytes: u64,
}

impl RssMeasurement {
    const ZERO: Self = Self {
        baseline_bytes: 0,
        process_peak_bytes: 0,
    };

    fn increment_bytes(self) -> u64 {
        self.process_peak_bytes.saturating_sub(self.baseline_bytes)
    }
}

impl RssSampler {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(0));
        let ready = Arc::new(Barrier::new(2));
        let thread_stop = stop.clone();
        let thread_peak = peak.clone();
        let thread_ready = ready.clone();
        let handle = std::thread::spawn(move || {
            thread_ready.wait();
            while !thread_stop.load(Ordering::Relaxed) {
                thread_peak.fetch_max(current_rss_bytes(), Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(1));
            }
        });
        ready.wait();
        let baseline = current_rss_bytes();
        peak.store(baseline, Ordering::Relaxed);
        Self {
            stop,
            peak,
            baseline,
            handle: Some(handle),
        }
    }

    fn stop(mut self) -> RssMeasurement {
        self.peak.fetch_max(current_rss_bytes(), Ordering::Relaxed);
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        RssMeasurement {
            baseline_bytes: self.baseline,
            process_peak_bytes: self.peak.load(Ordering::Relaxed),
        }
    }
}

struct OperationMetrics {
    wall_ns: u64,
    bytes: u64,
    rss: RssMeasurement,
    read_bytes: u64,
    write_bytes: u64,
}

fn record(
    args: &Args,
    operation: &str,
    metrics: OperationMetrics,
    normalized_missing_rows: usize,
) -> Value {
    json!({
        "schema_version": SCHEMA_VERSION,
        "suite": "codec",
        "scenario": args.scenario.as_str(),
        "fragments": args.fragments,
        "format": args.format.as_str(),
        "storage": "memory",
        "operation": operation,
        "round": args.round,
        "wall_ns": metrics.wall_ns,
        "bytes": metrics.bytes,
        "peak_rss_bytes": metrics.rss.increment_bytes(),
        "get_requests": 0,
        "put_requests": 0,
        "read_bytes": metrics.read_bytes,
        "write_bytes": metrics.write_bytes,
        "status": "success",
        "error": null,
        "commit": args.commit,
        "seed": args.seed,
        "host": args.host,
        "normalization": {
            "mode": "deterministic_synthetic_physical_rows",
            "outside_timed_codec": true,
            "normalized_missing_rows": normalized_missing_rows,
        },
        "rss": {
            "metric": "operation_peak_increment",
            "baseline_bytes": metrics.rss.baseline_bytes,
            "process_peak_bytes": metrics.rss.process_peak_bytes,
        },
        "warmup": args.warmup.as_str(),
    })
}

fn duration_ns(duration: Duration) -> Result<u64> {
    u64::try_from(duration.as_nanos())
        .map_err(|_| Error::internal("benchmark duration exceeds u64 nanoseconds"))
}

fn verify_footer_bytes(bytes: &Bytes, expected_columnar: bool) -> Result<()> {
    let is_columnar = is_columnar_manifest_footer(bytes)?;
    if is_columnar != expected_columnar {
        return Err(Error::internal(format!(
            "storage version requested {} manifest but footer identified {}; refusing to record a mislabeled sample",
            if expected_columnar {
                "Lance"
            } else {
                "protobuf"
            },
            if is_columnar { "Lance" } else { "protobuf" },
        )));
    }
    Ok(())
}

async fn verify_stored_footer(
    store: &ObjectStore,
    path: &Path,
    expected_columnar: bool,
) -> Result<Bytes> {
    let reader = store.open(path).await?;
    let bytes = reader.get_all().await?;
    verify_footer_bytes(&bytes, expected_columnar)?;
    Ok(bytes)
}

fn expected_missing_rows(args: &Args) -> usize {
    if args.scenario == Scenario::S1 {
        return 0;
    }
    (0..args.fragments)
        .filter(|index| selected(args.seed, *index as u64, 5, 100))
        .count()
}

fn validate_decoded(args: &Args, manifest: &Manifest) -> Result<()> {
    let expected_fields = if args.scenario == Scenario::S1 { 8 } else { 32 };
    if manifest.schema.fields.len() != expected_fields {
        return Err(Error::internal(format!(
            "decoded schema has {} fields, expected {expected_fields}",
            manifest.schema.fields.len()
        )));
    }
    if manifest.fragments.len() != args.fragments {
        return Err(Error::internal(format!(
            "decoded manifest has {} fragments, expected {}",
            manifest.fragments.len(),
            args.fragments
        )));
    }
    let expected_storage = DataStorageFormat::new(storage_version(args.format));
    if manifest.data_storage_format != expected_storage {
        return Err(Error::internal(format!(
            "decoded storage format {:?} differs from expected {:?}",
            manifest.data_storage_format, expected_storage
        )));
    }
    let (expected_major, expected_minor) = storage_version(args.format).to_numbers();
    for (index, fragment) in manifest.fragments.iter().enumerate() {
        let id = index as u64;
        if fragment.id != id {
            return Err(Error::internal(format!(
                "decoded fragment row {index} has id {}, expected {id}",
                fragment.id
            )));
        }
        let layout = sample(args.seed, id, 0);
        let expected_physical_rows = if args.scenario == Scenario::S1 {
            ROWS_PER_FRAGMENT
        } else {
            ROWS_PER_FRAGMENT + layout as usize % 17
        };
        if fragment.physical_rows != Some(expected_physical_rows) {
            return Err(Error::internal(format!(
                "decoded fragment {id} has physical_rows {:?}, expected {expected_physical_rows}",
                fragment.physical_rows
            )));
        }
        let expected_file_count = if args.scenario == Scenario::S2 && selected(args.seed, id, 1, 20)
        {
            2
        } else {
            1
        };
        if fragment.files.len() != expected_file_count {
            return Err(Error::internal(format!(
                "decoded fragment {id} has {} files, expected {expected_file_count}",
                fragment.files.len()
            )));
        }
        for file in &fragment.files {
            if (file.file_major_version, file.file_minor_version)
                != (expected_major, expected_minor)
            {
                return Err(Error::internal(format!(
                    "decoded fragment {id} has data file version {}.{}, expected {expected_major}.{expected_minor}",
                    file.file_major_version, file.file_minor_version
                )));
            }
        }
        let expected_deletion = args.scenario == Scenario::S2 && selected(args.seed, id, 4, 5);
        if fragment.deletion_file.is_some() != expected_deletion {
            return Err(Error::internal(format!(
                "decoded fragment {id} deletion presence differs from workload"
            )));
        }
    }
    Ok(())
}

fn warmup_fixture(args: &Args) -> PathBuf {
    args.fixture.with_extension("warmup")
}

async fn write_tiny_warmup_fixture(args: &Args) -> Result<()> {
    if args.warmup == Warmup::Cold {
        return Ok(());
    }
    const WARMUP_FRAGMENTS: usize = 16;
    let (mut manifest, _) = make_manifest(Scenario::S1, WARMUP_FRAGMENTS, args.seed, args.format)?;
    let store = ObjectStore::memory();
    let path = Path::from("/manifest-codec-warmup.manifest");
    write_manifest_file_to_path(&store, &mut manifest, None, &path, None).await?;
    let bytes = verify_stored_footer(&store, &path, args.format.expects_columnar()).await?;
    std::fs::write(warmup_fixture(args), bytes)
        .map_err(|error| Error::io_source(Box::new(error)))?;
    Ok(())
}

async fn run_tiny_decode_warmup(args: &Args) -> Result<()> {
    if args.warmup == Warmup::Cold {
        return Ok(());
    }
    const WARMUP_FRAGMENTS: usize = 16;
    let encoded = Bytes::from(
        std::fs::read(warmup_fixture(args)).map_err(|error| Error::io_source(Box::new(error)))?,
    );
    verify_footer_bytes(&encoded, args.format.expects_columnar())?;
    let encoded_bytes = encoded.len() as u64;
    let store = ObjectStore::memory();
    let path = Path::from("/manifest-codec-warmup.manifest");
    store.inner.put(&path, encoded.into()).await?;
    let decoded = read_manifest(&store, &path, Some(encoded_bytes)).await?;
    if decoded.fragments.len() != WARMUP_FRAGMENTS {
        return Err(Error::internal(format!(
            "warm-up decoded {} fragments, expected {WARMUP_FRAGMENTS}",
            decoded.fragments.len()
        )));
    }
    Ok(())
}

async fn run_encode(args: &Args) -> Result<Vec<Value>> {
    let (manifest, normalized_missing_rows) =
        make_manifest(args.scenario, args.fragments, args.seed, args.format)?;
    write_tiny_warmup_fixture(args).await?;
    let store = ObjectStore::memory();
    let path = Path::from(format!(
        "/manifest-{}-{}-{}-{}.manifest",
        args.scenario.as_str(),
        args.fragments,
        args.format.as_str(),
        args.round
    ));

    let mut manifest_to_write = manifest;
    let encode_rss = RssSampler::start();
    let encode_started = Instant::now();
    let write_result =
        write_manifest_file_to_path(&store, &mut manifest_to_write, None, &path, None).await?;
    let encode_wall_ns = duration_ns(encode_started.elapsed())?;
    let encode_peak_rss = encode_rss.stop();
    let encoded = verify_stored_footer(&store, &path, args.format.expects_columnar()).await?;
    let encoded_bytes = u64::try_from(encoded.len())
        .map_err(|_| Error::internal("encoded manifest size exceeds u64"))?;
    if write_result.size as u64 != encoded_bytes {
        return Err(Error::internal(format!(
            "writer reported {} bytes but memory store contains {encoded_bytes}",
            write_result.size
        )));
    }
    std::fs::write(&args.fixture, &encoded).map_err(|error| Error::io_source(Box::new(error)))?;

    Ok(vec![
        record(
            args,
            "encode",
            OperationMetrics {
                wall_ns: encode_wall_ns,
                bytes: encoded_bytes,
                rss: encode_peak_rss,
                read_bytes: 0,
                write_bytes: encoded_bytes,
            },
            normalized_missing_rows,
        ),
        record(
            args,
            "size",
            OperationMetrics {
                wall_ns: 0,
                bytes: encoded_bytes,
                rss: RssMeasurement::ZERO,
                read_bytes: 0,
                write_bytes: 0,
            },
            normalized_missing_rows,
        ),
    ])
}

async fn load_main_fixture(args: &Args, path_prefix: &str) -> Result<(ObjectStore, Path, u64)> {
    let encoded = Bytes::from(
        std::fs::read(&args.fixture).map_err(|error| Error::io_source(Box::new(error)))?,
    );
    verify_footer_bytes(&encoded, args.format.expects_columnar())?;
    let encoded_bytes = u64::try_from(encoded.len())
        .map_err(|_| Error::internal("encoded manifest size exceeds u64"))?;
    let store = ObjectStore::memory();
    let path = Path::from(format!(
        "/{path_prefix}-{}-{}-{}-{}.manifest",
        args.scenario.as_str(),
        args.fragments,
        args.format.as_str(),
        args.round
    ));
    store.inner.put(&path, encoded.into()).await?;
    Ok((store, path, encoded_bytes))
}

async fn run_decode(args: &Args) -> Result<Vec<Value>> {
    run_tiny_decode_warmup(args).await?;
    let (store, path, encoded_bytes) = load_main_fixture(args, "decode").await?;

    let decode_started = Instant::now();
    let decoded = read_manifest(&store, &path, Some(encoded_bytes)).await?;
    let decode_wall_ns = duration_ns(decode_started.elapsed())?;
    validate_decoded(args, &decoded)?;
    let normalized_missing_rows = expected_missing_rows(args);

    Ok(vec![record(
        args,
        "decode",
        OperationMetrics {
            wall_ns: decode_wall_ns,
            bytes: encoded_bytes,
            rss: RssMeasurement::ZERO,
            read_bytes: encoded_bytes,
            write_bytes: 0,
        },
        normalized_missing_rows,
    )])
}

async fn run_decode_rss(args: &Args) -> Result<Vec<Value>> {
    if args.warmup != Warmup::Cold {
        return Err(Error::invalid_input(
            "rss mode requires --warmup cold so retained warm-up allocations cannot reduce the measured increment",
        ));
    }
    let (store, path, encoded_bytes) = load_main_fixture(args, "decode-rss").await?;

    let decode_rss = RssSampler::start();
    let decode_started = Instant::now();
    let decoded = read_manifest(&store, &path, Some(encoded_bytes)).await?;
    let decode_wall_ns = duration_ns(decode_started.elapsed())?;
    let decode_peak_rss = decode_rss.stop();
    validate_decoded(args, &decoded)?;
    let normalized_missing_rows = expected_missing_rows(args);

    Ok(vec![record(
        args,
        "decode_rss",
        OperationMetrics {
            wall_ns: decode_wall_ns,
            bytes: encoded_bytes,
            rss: decode_peak_rss,
            read_bytes: encoded_bytes,
            write_bytes: 0,
        },
        normalized_missing_rows,
    )])
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| Error::io_source(Box::new(error)))?;
    let records = runtime.block_on(async {
        match args.mode {
            Mode::Encode => run_encode(&args).await,
            Mode::Decode => run_decode(&args).await,
            Mode::Rss => run_decode_rss(&args).await,
        }
    })?;
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    for record in records {
        serde_json::to_writer(&mut stdout, &record)?;
        stdout
            .write_all(b"\n")
            .map_err(|error| Error::io_source(Box::new(error)))?;
    }
    Ok(())
}
