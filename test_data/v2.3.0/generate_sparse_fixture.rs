// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Generates the Lance 2.3 sparse-reader compatibility fixture.
//!
//! This test-only generator must be run from Lance prototype commit
//! 1aca0a6d4fbb1010adb3b8fc2d0b6951a11e736b.

use std::{collections::HashMap, fs::File, path::PathBuf, sync::Arc};

use arrow_array::{
    ArrayRef, FixedSizeListArray, Int32Array, Int64Array, ListArray, RecordBatch, StructArray,
    builder::{
        FixedSizeListBuilder, Int32Builder, Int64Builder, LargeListBuilder, ListBuilder, MapBuilder,
    },
};
use arrow_buffer::{BooleanBuffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_ipc::writer::FileWriter as IpcFileWriter;
use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema};
use lance_core::datatypes::Schema as LanceSchema;
use lance_encoding::{
    decoder::{ColumnInfo, PageEncoding, PageInfo},
    encoder::{EncodingOptions, default_encoding_strategy, encode_batch},
    format::{ProtobufUtils21, pb21},
    version::LanceFileVersion,
};
use lance_file::writer::{EncodedBatchWriteExt, FileWriter, FileWriterOptions};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;

const ROWS: usize = 4096;
const STRUCTURAL_ENCODING_KEY: &str = "lance-encoding:structural-encoding";

fn sparse_metadata() -> HashMap<String, String> {
    HashMap::from([(STRUCTURAL_ENCODING_KEY.to_string(), "sparse".to_string())])
}

fn validity(values: Vec<bool>) -> Option<NullBuffer> {
    Some(NullBuffer::new(BooleanBuffer::from(values)))
}

fn valid_island_primitive() -> ArrayRef {
    Arc::new(Int32Array::from_iter(
        (0..ROWS).map(|row| (100..116).contains(&row).then_some(row as i32)),
    ))
}

fn null_range_primitive() -> ArrayRef {
    Arc::new(Int64Array::from_iter((0..ROWS).map(|row| {
        (!(200..216).contains(&row)).then_some((row as i64) * 10)
    })))
}

fn all_non_empty_list() -> ArrayRef {
    let mut builder = ListBuilder::new(Int32Builder::new());
    for row in 0..ROWS {
        builder.values().append_value(row as i32);
        builder.values().append_value(row as i32 + 1);
        builder.append(true);
    }
    Arc::new(builder.finish())
}

fn mixed_list() -> ArrayRef {
    let mut builder = ListBuilder::new(Int32Builder::new());
    for row in 0..ROWS {
        match row % 8 {
            0 | 7 => builder.append_null(),
            1 | 5 => builder.append(true),
            2 | 6 => {
                builder.values().append_value(row as i32);
                builder.append(true);
            }
            3 => {
                builder.values().append_value(row as i32);
                builder.values().append_value(row as i32 + 10);
                builder.append(true);
            }
            4 => {
                builder.values().append_value(row as i32);
                builder.values().append_value(row as i32 + 10);
                builder.values().append_value(row as i32 + 20);
                builder.append(true);
            }
            _ => unreachable!(),
        }
    }
    Arc::new(builder.finish())
}

fn no_value_list() -> ArrayRef {
    let mut builder = ListBuilder::new(Int32Builder::new());
    for _ in 0..ROWS {
        builder.append(true);
    }
    Arc::new(builder.finish())
}

fn sparse_large_list() -> ArrayRef {
    let mut builder = LargeListBuilder::new(Int64Builder::new());
    for row in 0..ROWS {
        if row.is_multiple_of(256) {
            if row.is_multiple_of(512) {
                builder.values().append_value(row as i64);
            }
            builder.append(true);
        } else {
            builder.append_null();
        }
    }
    Arc::new(builder.finish())
}

fn mixed_map() -> ArrayRef {
    let mut builder = MapBuilder::new(None, Int32Builder::new(), Int64Builder::new());
    for row in 0..ROWS {
        match row % 6 {
            0 => builder.append(false).unwrap(),
            1 | 4 => builder.append(true).unwrap(),
            2 | 5 => {
                builder.keys().append_value(row as i32);
                builder.values().append_value(row as i64 * 100);
                builder.append(true).unwrap();
            }
            3 => {
                builder.keys().append_value(row as i32);
                builder.values().append_value(row as i64 * 100);
                builder.keys().append_value(row as i32 + 1);
                builder.values().append_value(row as i64 * 100 + 1);
                builder.append(true).unwrap();
            }
            _ => unreachable!(),
        }
    }
    Arc::new(builder.finish())
}

fn nullable_fixed_size_list() -> ArrayRef {
    let mut builder = FixedSizeListBuilder::new(Int32Builder::new(), 3);
    for row in 0..ROWS {
        for lane in 0..3 {
            if (row + lane).is_multiple_of(11) {
                builder.values().append_null();
            } else {
                builder.values().append_value((row * 3 + lane) as i32);
            }
        }
        builder.append((300..332).contains(&row));
    }
    Arc::new(builder.finish())
}

fn fixed_size_list_struct() -> ArrayRef {
    let child_len = ROWS * 2;
    let child = Arc::new(StructArray::new(
        Fields::from(vec![
            ArrowField::new("left", DataType::Int32, true),
            ArrowField::new("right", DataType::Int64, true),
        ]),
        vec![
            Arc::new(Int32Array::from_iter((0..child_len).map(|index| {
                (!index.is_multiple_of(7)).then_some(index as i32)
            }))) as ArrayRef,
            Arc::new(Int64Array::from_iter((0..child_len).map(|index| {
                (!index.is_multiple_of(11)).then_some((index as i64) * 10)
            }))) as ArrayRef,
        ],
        validity(
            (0..child_len)
                .map(|index| !index.is_multiple_of(13))
                .collect(),
        ),
    )) as ArrayRef;

    Arc::new(
        FixedSizeListArray::try_new(
            Arc::new(ArrowField::new("item", child.data_type().clone(), true)),
            2,
            child,
            validity((0..ROWS).map(|row| !row.is_multiple_of(17)).collect()),
        )
        .unwrap(),
    )
}

fn deeply_nested() -> ArrayRef {
    let mut event_offsets = Vec::with_capacity(ROWS + 1);
    let mut event_list_validity = Vec::with_capacity(ROWS);
    let mut event_rows = Vec::new();
    event_offsets.push(0_i32);

    for row in 0..ROWS {
        match row % 8 {
            0 | 7 => event_list_validity.push(false),
            1 | 5 => event_list_validity.push(true),
            2 | 6 => {
                event_rows.push(row as i32);
                event_list_validity.push(true);
            }
            3 => {
                event_rows.push(row as i32);
                event_rows.push(row as i32 + 100_000);
                event_list_validity.push(true);
            }
            4 => {
                event_rows.push(row as i32);
                event_rows.push(row as i32 + 100_000);
                event_rows.push(row as i32 + 200_000);
                event_list_validity.push(true);
            }
            _ => unreachable!(),
        }
        event_offsets.push(event_rows.len() as i32);
    }

    let ids =
        Arc::new(Int32Array::from_iter(event_rows.iter().enumerate().map(
            |(index, value)| (!index.is_multiple_of(13)).then_some(*value),
        ))) as ArrayRef;

    let pair_values = Arc::new(Int32Array::from_iter(
        event_rows.iter().enumerate().flat_map(|(index, value)| {
            [
                Some(*value),
                (!index.is_multiple_of(17)).then_some(*value + 1),
            ]
        }),
    )) as ArrayRef;
    let pair = Arc::new(
        FixedSizeListArray::try_new(
            Arc::new(ArrowField::new("item", DataType::Int32, true)),
            2,
            pair_values,
            validity(
                (0..event_rows.len())
                    .map(|index| !index.is_multiple_of(19))
                    .collect(),
            ),
        )
        .unwrap(),
    ) as ArrayRef;

    let events = Arc::new(StructArray::new(
        Fields::from(vec![
            ArrowField::new("id", DataType::Int32, true),
            ArrowField::new("pair", pair.data_type().clone(), true),
        ]),
        vec![ids, pair],
        validity(
            (0..event_rows.len())
                .map(|index| !index.is_multiple_of(23))
                .collect(),
        ),
    )) as ArrayRef;

    let event_lists = Arc::new(
        ListArray::try_new(
            Arc::new(ArrowField::new("item", events.data_type().clone(), true)),
            OffsetBuffer::new(ScalarBuffer::from(event_offsets)),
            events,
            validity(event_list_validity),
        )
        .unwrap(),
    ) as ArrayRef;

    Arc::new(StructArray::new(
        Fields::from(vec![ArrowField::new(
            "events",
            event_lists.data_type().clone(),
            true,
        )]),
        vec![event_lists],
        validity((0..ROWS).map(|row| !row.is_multiple_of(29)).collect()),
    ))
}

fn fixture_batch() -> RecordBatch {
    let arrays = vec![
        valid_island_primitive(),
        null_range_primitive(),
        all_non_empty_list(),
        mixed_list(),
        sparse_large_list(),
        mixed_map(),
        nullable_fixed_size_list(),
        fixed_size_list_struct(),
        deeply_nested(),
    ];
    let names = [
        "valid_island_i32",
        "null_range_i64",
        "all_non_empty_list",
        "mixed_list",
        "large_list",
        "map",
        "fixed_size_list",
        "fixed_size_list_struct",
        "deep",
    ];
    let fields = names
        .into_iter()
        .zip(&arrays)
        .map(|(name, array)| {
            ArrowField::new(name, array.data_type().clone(), true).with_metadata(sparse_metadata())
        })
        .collect::<Vec<_>>();
    let schema = Arc::new(ArrowSchema::new(fields));
    RecordBatch::try_new(schema, arrays).unwrap()
}

fn empty_fixture_batch() -> RecordBatch {
    let values = no_value_list();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("no_value_list", values.data_type().clone(), true)
            .with_metadata(sparse_metadata()),
    ]));
    RecordBatch::try_new(schema, vec![values]).unwrap()
}

fn empty_position_set() -> Option<pb21::SparsePositionSet> {
    Some(pb21::SparsePositionSet {
        positions: Some(pb21::sparse_position_set::Positions::Empty(
            pb21::SparsePositionEmpty {},
        )),
        num_positions: 0,
    })
}

fn empty_sparse_layout() -> pb21::PageLayout {
    pb21::PageLayout {
        layout: Some(pb21::page_layout::Layout::SparseLayout(
            pb21::SparseLayout {
                value_compression: Some(ProtobufUtils21::flat(32, None)),
                num_buffers: 1,
                num_items: ROWS as u64,
                num_visible_items: 0,
                has_large_chunk: false,
                structural_layers: vec![
                    pb21::SparseStructuralLayer {
                        kind: pb21::sparse_structural_layer::Kind::SparseLayerList as i32,
                        num_slots: ROWS as u64,
                        num_child_slots: 0,
                        non_empty_positions: empty_position_set(),
                        counts: Some(pb21::SparseCountSet {
                            counts: Some(pb21::sparse_count_set::Counts::Empty(
                                pb21::SparseCountEmpty {},
                            )),
                        }),
                        fixed_size_list_dimension: 0,
                        validity: Some(pb21::SparseValiditySet {
                            meaning:
                                pb21::sparse_validity_set::Meaning::SparseValidityNullPositions
                                    as i32,
                            positions: empty_position_set(),
                        }),
                    },
                    pb21::SparseStructuralLayer {
                        kind: pb21::sparse_structural_layer::Kind::SparseLayerValidity as i32,
                        num_slots: 0,
                        num_child_slots: 0,
                        non_empty_positions: None,
                        counts: None,
                        fixed_size_list_dimension: 0,
                        validity: Some(pb21::SparseValiditySet {
                            meaning:
                                pb21::sparse_validity_set::Meaning::SparseValidityNullPositions
                                    as i32,
                            positions: empty_position_set(),
                        }),
                    },
                ],
            },
        )),
    }
}

fn write_ipc(path: PathBuf, batch: &RecordBatch) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = IpcFileWriter::try_new(File::create(path)?, batch.schema().as_ref())?;
    writer.write(batch)?;
    writer.finish()?;
    Ok(())
}

async fn write_empty_sparse_fixture(
    path: PathBuf,
    batch: &RecordBatch,
) -> Result<(), Box<dyn std::error::Error>> {
    let strategy = default_encoding_strategy(LanceFileVersion::V2_3);
    let options = EncodingOptions {
        buffer_alignment: 4096,
        version: LanceFileVersion::V2_3,
        ..Default::default()
    };
    let schema = Arc::new(LanceSchema::try_from(batch.schema().as_ref())?);
    let mut encoded = encode_batch(batch, schema, strategy.as_ref(), &options).await?;
    let original = encoded
        .page_table
        .first()
        .ok_or("empty fixture has no column")?;
    let original_page = original
        .page_infos
        .first()
        .ok_or("empty fixture has no page")?;
    encoded.page_table[0] = Arc::new(ColumnInfo {
        index: original.index,
        page_infos: Arc::from([PageInfo {
            buffer_offsets_and_sizes: Arc::from([(0, 0), (0, 0)]),
            encoding: PageEncoding::Structural(empty_sparse_layout()),
            num_rows: ROWS as u64,
            priority: original_page.priority,
        }]),
        buffer_offsets_and_sizes: original.buffer_offsets_and_sizes.clone(),
        encoding: original.encoding.clone(),
    });
    let aligned_len = encoded.data.len().next_multiple_of(4096);
    let mut aligned_data = encoded.data.to_vec();
    aligned_data.resize(aligned_len, 0);
    encoded.data = aligned_data.into();
    std::fs::write(
        path,
        encoded.try_to_self_described_lance(LanceFileVersion::V2_3)?,
    )?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: generate_sparse_fixture <output-directory>")?;
    std::fs::create_dir_all(&output_dir)?;

    let lance_path = output_dir.join("sparse_reader.lance");
    let expected_path = output_dir.join("sparse_reader.arrow");
    let empty_lance_path = output_dir.join("empty_sparse_reader.lance");
    let empty_expected_path = output_dir.join("empty_sparse_reader.arrow");
    if lance_path.exists() {
        std::fs::remove_file(&lance_path)?;
    }
    if expected_path.exists() {
        std::fs::remove_file(&expected_path)?;
    }
    if empty_lance_path.exists() {
        std::fs::remove_file(&empty_lance_path)?;
    }
    if empty_expected_path.exists() {
        std::fs::remove_file(&empty_expected_path)?;
    }

    let batch = fixture_batch();
    write_ipc(expected_path, &batch)?;

    File::create(&lance_path)?;
    let store = ObjectStore::local();
    let object_path = Path::from_filesystem_path(&lance_path)?;
    let object_writer = store.create(&object_path).await?;
    let lance_schema = LanceSchema::try_from(batch.schema().as_ref())?;
    let mut writer = FileWriter::try_new(
        object_writer,
        lance_schema,
        FileWriterOptions {
            format_version: Some(LanceFileVersion::V2_3),
            ..Default::default()
        },
    )?;
    writer.write_batch(&batch).await?;
    writer.finish().await?;

    let empty_batch = empty_fixture_batch();
    write_ipc(empty_expected_path, &empty_batch)?;
    write_empty_sparse_fixture(empty_lance_path, &empty_batch).await?;

    Ok(())
}
