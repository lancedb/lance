// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![allow(clippy::print_stdout)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use arrow_array::{RecordBatch, RecordBatchIterator, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::io::{bytes_read_counter, iops_counter};
use lance_core::{Error, Result};
use lance_encoding::predicate::{PrimitivePredicateStats, primitive_predicate_stats};

struct CountingAllocator;

static ALLOCATION_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const PAYLOAD_COLUMNS: [&str; 8] = [
    "payload_0",
    "payload_1",
    "payload_2",
    "payload_3",
    "payload_4",
    "payload_5",
    "payload_6",
    "payload_7",
];

#[tokio::main]
async fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    match args.as_slice() {
        [_, command, uri, rows, rows_per_batch] if command == "prepare" => {
            prepare(
                uri,
                parse_u64(rows, "rows")?,
                parse_usize(rows_per_batch, "rows_per_batch")?,
            )
            .await
        }
        [_, command, uri, column, operator, literal, mode, iterations] if command == "run" => {
            run(
                uri,
                column,
                operator,
                parse_u32(literal, "literal")?,
                mode,
                parse_usize(iterations, "iterations")?,
            )
            .await
        }
        _ => Err(Error::invalid_input(
            "usage: encoded_predicate_bench prepare <uri> <rows> <rows_per_batch> | run <uri> <column> <operator> <literal> <baseline|encoded> <iterations>",
        )),
    }
}

async fn prepare(uri: &str, rows: u64, rows_per_batch: usize) -> Result<()> {
    let schema = Arc::new(Schema::new(
        [
            Field::new("bitpack", DataType::UInt32, false),
            Field::new("rle", DataType::UInt32, false),
            Field::new("constant", DataType::UInt32, false),
        ]
        .into_iter()
        .chain(
            PAYLOAD_COLUMNS
                .iter()
                .map(|name| Field::new(*name, DataType::UInt64, false)),
        )
        .collect::<Vec<_>>(),
    ));
    let batches = (0..rows).step_by(rows_per_batch).map({
        let schema = schema.clone();
        move |start| {
            let end = rows.min(start + rows_per_batch as u64);
            let bitpack =
                UInt32Array::from_iter_values((start..end).map(|row| (row % 1024) as u32));
            let rle =
                UInt32Array::from_iter_values((start..end).map(|row| ((row / 4096) % 1024) as u32));
            let constant = UInt32Array::from_iter_values((start..end).map(|_| 7));
            let mut columns = vec![
                Arc::new(bitpack) as _,
                Arc::new(rle) as _,
                Arc::new(constant) as _,
            ];
            columns.extend((0..PAYLOAD_COLUMNS.len()).map(|payload| {
                Arc::new(UInt64Array::from_iter_values(
                    (start..end).map(move |row| row.wrapping_mul(31 + payload as u64)),
                )) as _
            }));
            RecordBatch::try_new(schema.clone(), columns)
        }
    });
    let reader = RecordBatchIterator::new(batches, schema);
    Dataset::write(
        reader,
        uri,
        Some(WriteParams {
            max_rows_per_file: 1_000_000,
            ..Default::default()
        }),
    )
    .await?;
    println!("prepared\turi={uri}\trows={rows}\trows_per_batch={rows_per_batch}");
    Ok(())
}

async fn run(
    uri: &str,
    column: &str,
    operator: &str,
    literal: u32,
    mode: &str,
    iterations: usize,
) -> Result<()> {
    match mode {
        "baseline" => unsafe { std::env::set_var("LANCE_ENCODED_PRIMITIVE_PREDICATE", "0") },
        "encoded" => unsafe { std::env::set_var("LANCE_ENCODED_PRIMITIVE_PREDICATE", "1") },
        _ => return Err(Error::invalid_input("mode must be baseline or encoded")),
    }
    if !matches!(operator, "=" | "!=" | "<" | "<=" | ">" | ">=") {
        return Err(Error::invalid_input("unsupported comparison operator"));
    }
    let dataset = Dataset::open(uri).await?;
    for iteration in 0..iterations {
        let allocations_before = ALLOCATION_CALLS.load(Ordering::Relaxed);
        let allocated_bytes_before = ALLOCATED_BYTES.load(Ordering::Relaxed);
        let io_bytes_before = bytes_read_counter();
        let iops_before = iops_counter();
        let predicate_before = primitive_predicate_stats();
        let cpu_before = cpu_micros();
        let started = Instant::now();

        let mut scanner = dataset.scan();
        scanner.project(&PAYLOAD_COLUMNS)?;
        scanner.filter(&format!("{column} {operator} {literal}"))?;
        let mut stream = scanner.try_into_stream().await?;
        let mut output_rows = 0_u64;
        while let Some(batch) = stream.try_next().await? {
            output_rows += batch.num_rows() as u64;
        }

        let elapsed = started.elapsed();
        let cpu_micros = cpu_micros().saturating_sub(cpu_before);
        let predicate = stats_delta(primitive_predicate_stats(), predicate_before);
        println!(
            "result\turi={uri}\tcolumn={column}\toperator={operator}\tliteral={literal}\tmode={mode}\titeration={iteration}\toutput_rows={output_rows}\twall_micros={}\tcpu_micros={cpu_micros}\tio_bytes={}\tiops={}\tallocation_calls={}\tallocated_bytes={}\tdirect_values={}\tfallback_values={}\tpredicate_output_bytes={}",
            elapsed.as_micros(),
            bytes_read_counter().saturating_sub(io_bytes_before),
            iops_counter().saturating_sub(iops_before),
            ALLOCATION_CALLS
                .load(Ordering::Relaxed)
                .saturating_sub(allocations_before),
            ALLOCATED_BYTES
                .load(Ordering::Relaxed)
                .saturating_sub(allocated_bytes_before),
            predicate.direct_values,
            predicate.fallback_values,
            predicate.output_bytes,
        );
    }
    Ok(())
}

fn stats_delta(
    after: PrimitivePredicateStats,
    before: PrimitivePredicateStats,
) -> PrimitivePredicateStats {
    PrimitivePredicateStats {
        direct_values: after.direct_values.saturating_sub(before.direct_values),
        fallback_values: after.fallback_values.saturating_sub(before.fallback_values),
        output_bytes: after.output_bytes.saturating_sub(before.output_bytes),
    }
}

fn cpu_micros() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return 0;
    }
    let usage = unsafe { usage.assume_init() };
    timeval_micros(usage.ru_utime).saturating_add(timeval_micros(usage.ru_stime))
}

fn timeval_micros(value: libc::timeval) -> u64 {
    (value.tv_sec as u64)
        .saturating_mul(1_000_000)
        .saturating_add(value.tv_usec as u64)
}

fn parse_u64(value: &str, name: &str) -> Result<u64> {
    value
        .parse()
        .map_err(|error| Error::invalid_input(format!("invalid {name}: {error}")))
}

fn parse_u32(value: &str, name: &str) -> Result<u32> {
    value
        .parse()
        .map_err(|error| Error::invalid_input(format!("invalid {name}: {error}")))
}

fn parse_usize(value: &str, name: &str) -> Result<usize> {
    value
        .parse()
        .map_err(|error| Error::invalid_input(format!("invalid {name}: {error}")))
}
