// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, ops::Range, sync::Arc};

use arrow::{array::AsArray, datatypes::UInt64Type};
use arrow_array::{new_null_array, RecordBatch};
use arrow_schema::Schema as ArrowSchema;
use datafusion::{
    execution::SendableRecordBatchStream, physical_plan::stream::RecordBatchStreamAdapter,
};
use futures::{StreamExt, TryStreamExt};
use lance_core::{datatypes::Schema, DATASET_OFFSET, ROW_ID};
use serde::{Deserialize, Serialize};
use snafu::{location, Location};

use crate::{
    dataset::transaction::{Operation, Transaction},
    Error, Result,
};
use lance_table::format::{DataFile, Fragment};

use super::{write::open_writer, CommitBuilder, Dataset};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FragmentTakeRange {
    /// The source fragment id
    fragment_id: u32,
    /// The range of offsets to take from the fragment
    offset_range: Range<u32>,
    /// Offset into the target that this range begins
    target_offset: u32,
}

/// An item in an [`AlignFragmentsTask`]
#[derive(Clone, Debug, Serialize, Deserialize)]
enum AlignmentItem {
    /// Insert rows from the source fragment
    SourceFragmentRange(FragmentTakeRange),
    /// Insert nulls
    Nulls(u32),
}

#[derive(Clone, Debug)]
struct AlignmentConfig {
    source: Arc<Dataset>,
    target: Arc<Dataset>,
}

impl AlignmentConfig {
    fn join_key(&self) -> &str {
        &self.source.schema().fields.first().unwrap().name
    }

    // Returns the schema of the target dataset, minus the join key
    fn new_columns_schema(&self) -> Schema {
        self.source
            .schema()
            .exclude(self.source.schema().project(&[self.join_key()]).unwrap())
            .unwrap()
    }

    // Returns the combined schema of the two datasets
    fn combined_schema(&self) -> Schema {
        let new_columns = self.new_columns_schema();
        let target_columns = self.target.schema();
        target_columns.merge(&new_columns).unwrap()
    }
}

impl AlignmentItem {
    fn target_offset(&self) -> u32 {
        match self {
            AlignmentItem::SourceFragmentRange(range) => range.target_offset,
            _ => unreachable!(),
        }
    }

    fn num_rows(&self) -> u32 {
        match self {
            AlignmentItem::SourceFragmentRange(range) => {
                range.offset_range.end - range.offset_range.start
            }
            AlignmentItem::Nulls(count) => *count,
        }
    }

    async fn into_stream(
        self,
        src_dataset: Arc<Dataset>,
        schema: Arc<Schema>,
    ) -> Result<SendableRecordBatchStream> {
        match self {
            Self::Nulls(count) => {
                let arrow_schema = Arc::new(ArrowSchema::from(schema.as_ref()));
                let arrays = arrow_schema
                    .fields
                    .iter()
                    .map(|f| new_null_array(f.data_type(), count as usize))
                    .collect::<Vec<_>>();
                let batch = RecordBatch::try_new(arrow_schema.clone(), arrays).unwrap();
                Ok(Box::pin(RecordBatchStreamAdapter::new(
                    arrow_schema,
                    futures::stream::iter(vec![Ok(batch)]),
                )))
            }
            Self::SourceFragmentRange(frag_range) => {
                let columns = src_dataset
                    .schema()
                    .fields
                    .iter()
                    .skip(1)
                    .map(|f| &f.name)
                    .collect::<Vec<_>>();
                let limit = frag_range.offset_range.end - frag_range.offset_range.start;
                let offset = frag_range.offset_range.start;
                let stream = src_dataset
                    .scan()
                    .with_fragments(vec![
                        src_dataset
                            .get_fragment(frag_range.fragment_id as usize)
                            .unwrap()
                            .metadata,
                    ])
                    .project(&columns)
                    .unwrap()
                    .limit(Some(limit as i64), Some(offset as i64))
                    .unwrap()
                    .try_into_stream()
                    .await?;
                Ok(SendableRecordBatchStream::from(stream))
            }
        }
    }
}

/// A task of work to align one or more source fragments into a target fragment
///
/// A collection of these tasks make up an [`AlignFragmentsPlan`]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlignFragmentsTask {
    /// The source fragments (and nulls) that correspond to the target
    source: Vec<AlignmentItem>,
    /// The id of the target fragment
    pub target_id: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlignFragmentsTaskResult {
    /// The newly created data file
    pub data_file: DataFile,
    /// The fragment the data file will be added to
    pub target_id: u32,
}

impl AlignFragmentsTask {
    /// Executes the task
    ///
    /// This reads from the source fragments and writes a new target fragment
    pub async fn execute(
        &self,
        source: Arc<Dataset>,
        target: Arc<Dataset>,
    ) -> Result<AlignFragmentsTaskResult> {
        let config = AlignmentConfig { source, target };
        let schema = Arc::new(config.new_columns_schema());
        let src_dataset = config.source.clone();
        let capture_schema = schema.clone();
        let mut stream = futures::stream::iter(self.source.clone())
            .map(move |item| item.into_stream(src_dataset.clone(), capture_schema.clone()))
            .buffered(1)
            .try_flatten()
            .boxed();
        let obj_store = config.target.object_store.clone();

        let mut writer = open_writer(
            &obj_store,
            &schema,
            &config.target.base,
            config
                .target
                .manifest
                .data_storage_format
                .lance_file_version()
                .unwrap(),
        )
        .await?;

        while let Some(batch) = stream.try_next().await? {
            writer.write(&[batch]).await?;
        }
        let (num_rows, data_file) = writer.finish().await?;

        assert_eq!(
            num_rows as usize,
            config
                .target
                .get_fragment(self.target_id as usize)
                .unwrap()
                .physical_rows()
                .await
                .unwrap()
        );

        Ok(AlignFragmentsTaskResult {
            data_file,
            target_id: self.target_id,
        })
    }

    /// Called when building the plan and should only be called when building a plan
    /// with dataset offsets which only uses FragmentTakeRange
    fn sort_and_fill_nulls(&mut self, target_fragment: &Fragment) -> Result<()> {
        let mut source = std::mem::take(&mut self.source);
        source.sort_by_key(|item| item.target_offset());
        let mut new_source = Vec::with_capacity(source.len() * 2);
        let mut target_offset = 0;
        for item in source {
            let gap = item.target_offset() - target_offset;
            if gap > 0 {
                new_source.push(AlignmentItem::Nulls(gap as u32));
            }
            target_offset = item.target_offset() + item.num_rows();
            new_source.push(item);
        }
        let frag_num_rows = target_fragment.physical_rows.expect("Alignment cannot be performed on older datasets which do not contain physical row counts") as u32;
        if target_offset < frag_num_rows {
            new_source.push(AlignmentItem::Nulls(frag_num_rows - target_offset));
        }
        self.source = new_source;
        Ok(())
    }
}

/// Aligns fragments to prepare for a Merge operation
///
/// This task aligns fragments from a _source_ dataset to be inserted into a _target_ dataset
/// using a Merge operation.
///
/// Creating a plan requires calculating an alignment.  The alignment is determined by figuring
/// out which rows in the target dataset are the same as the rows in the source dataset.  This
/// is done utilizing some kind of join key.
///
/// The simplest approach is when the join key is a _dataset_offset.  It is simple because we
/// can determine the destination address by looking at target fragment metadata.  However, it
/// is also brittle because it requires that the target dataset has not been modified (beyond
/// appends) since the source dataset was created.
///
/// A more robust approach is to use the _rowid as the join key along with the stable row ids
/// feature.  If stable row ids are enabled then this we can ignore compaction (and later also
/// updates).
///
/// Another robust, but slower, approach is to use an a custom join key.  This requires us to
/// run a hash-join operation to determine the alignment.
#[derive(Serialize, Deserialize)]
pub struct AlignFragmentsPlan {
    /// The tasks that need to be executed to create the target fragments
    tasks: Vec<AlignFragmentsTask>,
    read_version: u64,
}

impl AlignFragmentsPlan {
    /// Creates a new plan by calculating the alignment
    pub async fn new(
        source: Arc<Dataset>,
        target: Arc<Dataset>,
        join_key: &str,
    ) -> Result<AlignFragmentsPlan> {
        let config = AlignmentConfig { source, target };
        match join_key {
            DATASET_OFFSET => Self::create_from_dataset_offset(config).await,
            ROW_ID => Self::create_from_rowid(config).await,
            _ => Self::create_from_custom_key(config).await,
        }
    }

    pub async fn commit(
        &self,
        results: Vec<AlignFragmentsTaskResult>,
        source: Arc<Dataset>,
        target: Arc<Dataset>,
    ) -> Result<Dataset> {
        let config = AlignmentConfig { source, target };
        let frag_id_to_new_file = results
            .into_iter()
            .map(|res| (res.target_id, res.data_file))
            .collect::<HashMap<_, _>>();

        let fragments = config
            .target
            .fragments()
            .iter()
            .map(|frag| {
                if let Some(new_file) = frag_id_to_new_file.get(&(frag.id as u32)) {
                    let mut all_files = frag.files.clone();
                    all_files.push(new_file.clone());
                    Fragment {
                        deletion_file: frag.deletion_file.clone(),
                        id: frag.id,
                        files: all_files,
                        physical_rows: frag.physical_rows.clone(),
                        row_id_meta: frag.row_id_meta.clone(),
                    }
                } else {
                    frag.clone()
                }
            })
            .collect::<Vec<_>>();

        let mut schema = config.combined_schema();
        schema.set_field_id(Some(config.target.manifest.max_field_id()));

        let op = Operation::Merge { fragments, schema };

        let tx = Transaction::new(self.read_version, op, None, None);

        CommitBuilder::new(config.target.clone()).execute(tx).await
    }

    async fn dataset_offset_range_from_fragment(
        fragment: &Fragment,
        dataset: &Dataset,
    ) -> Result<Range<u64>> {
        let mut min = u64::MAX;
        let mut max = u64::MIN;
        let mut batches = dataset
            .scan()
            .with_fragments(vec![fragment.clone()])
            .project(&[DATASET_OFFSET])?
            .try_into_stream()
            .await?;
        while let Some(batch) = batches.next().await {
            let batch = batch?;
            let offsets = batch.column(0).as_primitive::<UInt64Type>();
            if offsets.is_empty() {
                continue;
            }
            let start = offsets.value(0);
            if min == u64::MAX {
                // First batch, set the min
                min = start;
            } else {
                // Not the first batch, ensure it follows the previous
                if start != max + 1 {
                    return Err(Error::InvalidInput {
                        source: "Dataset offsets are not sorted and contiguous.  A batch did not follow the previous batch.".into(),
                        location: location!(),
                    });
                }
            }
            max = offsets.value(offsets.len() - 1);
            for offs in offsets.values().windows(2) {
                if offs[0] + 1 != offs[1] {
                    return Err(Error::InvalidInput {
                        source: "Dataset offsets are not sorted and contiguous".into(),
                        location: location!(),
                    });
                }
            }
        }
        Ok(min..max + 1)
    }

    async fn create_from_dataset_offset(config: AlignmentConfig) -> Result<AlignFragmentsPlan> {
        let target_fragments = config.target.fragments();
        let mut tasks = target_fragments
            .iter()
            .map(|frag| AlignFragmentsTask {
                source: vec![],
                target_id: frag.id as u32,
            })
            .collect::<Vec<_>>();

        // For each new fragment, find which target fragments it overlaps with, and add it to
        // the list of tasks for that target fragment.
        for src_fragment in config.source.fragments().iter() {
            let src_frag_num_rows = src_fragment.num_rows().expect("Alignment cannot be performed on older datasets which do not contain physical row counts") as u32;
            let off_range =
                Self::dataset_offset_range_from_fragment(src_fragment, config.source.as_ref())
                    .await?;
            let mut rows_to_skip = off_range.start;
            let mut rows_to_take = off_range.end - off_range.start;
            let mut src_frag_offset = 0;
            for (target, task) in target_fragments.iter().zip(tasks.iter_mut()) {
                let target_num_rows = target.num_rows().expect("Alignment cannot be performed on older datasets which do not contain physical row counts") as u32;
                let skip_in_this_frag = if rows_to_skip < target_num_rows as u64 {
                    let skip_in_this_frag = rows_to_skip as u32;
                    rows_to_skip = 0;
                    skip_in_this_frag
                } else {
                    rows_to_skip -= target_num_rows as u64;
                    continue;
                };
                let remaining_in_target = (target_num_rows - skip_in_this_frag) as u64;
                let take_in_this_frag = rows_to_take.min(remaining_in_target) as u32;
                task.source
                    .push(AlignmentItem::SourceFragmentRange(FragmentTakeRange {
                        fragment_id: src_fragment.id as u32,
                        offset_range: src_frag_offset..src_frag_offset + take_in_this_frag,
                        target_offset: skip_in_this_frag,
                    }));
                src_frag_offset += take_in_this_frag;
                rows_to_take -= take_in_this_frag as u64;
                if src_frag_offset == src_frag_num_rows {
                    debug_assert!(rows_to_take == 0);
                    break;
                }
            }
            if rows_to_take > 0 {
                log::warn!("Source data in alignment plan has more rows than target data.  Excess rows will be ignored.");
            }
        }

        for (task, frag) in tasks.iter_mut().zip(target_fragments.iter()) {
            task.sort_and_fill_nulls(frag)?;
        }

        Ok(Self {
            tasks,
            read_version: config.target.version().version,
        })
    }

    async fn create_from_rowid(_config: AlignmentConfig) -> Result<AlignFragmentsPlan> {
        todo!()
    }

    async fn create_from_custom_key(_config: AlignmentConfig) -> Result<AlignFragmentsPlan> {
        Err(Error::NotSupported { source: "Creating an alignment plan from a custom join key is not yet supported.  Only _rowid and _dataset_offset work today".into(), location: location!() })
    }

    pub fn tasks(&self) -> &[AlignFragmentsTask] {
        &self.tasks
    }
}

#[cfg(test)]
pub mod tests {
    use std::sync::Arc;

    use arrow_array::{
        types::{Int32Type, Int64Type, UInt64Type},
        RecordBatchIterator,
    };
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::{datatypes::Schema, DATASET_OFFSET};
    use lance_datagen::{array, gen, ArrayGeneratorExt, RowCount};
    use tempfile::tempdir;

    use crate::{
        dataset::align::{AlignFragmentsPlan, AlignmentItem, FragmentTakeRange},
        utils::test::{DatagenExt, FragmentCount, FragmentRowCount},
        Dataset,
    };

    #[tokio::test]
    async fn test_create_plan_from_dataset_offset() {
        let test_dir = tempdir().unwrap();
        let target_uri = test_dir.path().join("target");
        let target_uri = target_uri.to_str().unwrap();
        let source_uri = test_dir.path().join("source");
        let source_uri = source_uri.to_str().unwrap();

        // Target dataset: 4 fragments of 10 rows each
        let target_dataset = gen()
            .col("a", array::step::<Int32Type>())
            .into_dataset(
                target_uri,
                FragmentCount::from(4),
                FragmentRowCount::from(10),
            )
            .await
            .unwrap();

        // Source dataset: 2 fragments, with 6 rows each, straddling the target fragments
        let new_schema = Schema::try_from(&ArrowSchema::new(vec![
            ArrowField::new(DATASET_OFFSET, DataType::UInt64, false),
            ArrowField::new("b", DataType::Int64, true),
        ]))
        .unwrap();
        let mut source_dataset = Dataset::write_empty(new_schema, source_uri, None)
            .await
            .unwrap();

        let batch1 = gen()
            .col(
                DATASET_OFFSET,
                array::cycle::<UInt64Type>(vec![7, 8, 9, 10, 11, 12]),
            )
            .col("b", array::step::<Int64Type>())
            .into_batch_rows(RowCount::from(6))
            .unwrap();
        let batch2 = gen()
            .col(
                DATASET_OFFSET,
                array::cycle::<UInt64Type>(vec![37, 38, 39, 40, 41, 42]),
            )
            .col("b", array::step_custom::<Int64Type>(6, 1))
            .into_batch_rows(RowCount::from(6))
            .unwrap();

        let schema = batch1.schema();
        let source_data = RecordBatchIterator::new(vec![Ok(batch1)], schema.clone());
        source_dataset.append(source_data, None).await.unwrap();
        let source_data = RecordBatchIterator::new(vec![Ok(batch2)], schema);
        source_dataset.append(source_data, None).await.unwrap();

        let source_dataset = Arc::new(source_dataset);
        let target_dataset = Arc::new(target_dataset);

        let plan = AlignFragmentsPlan::new(
            source_dataset.clone(),
            target_dataset.clone(),
            DATASET_OFFSET,
        )
        .await
        .unwrap();

        assert_eq!(plan.tasks.len(), 4);
        let task = &plan.tasks[0];
        assert_eq!(task.source.len(), 2);
        assert!(matches!(&task.source[0], AlignmentItem::Nulls(7)));
        assert!(matches!(
            &task.source[1],
            AlignmentItem::SourceFragmentRange(FragmentTakeRange {
                fragment_id: 0,
                offset_range: std::ops::Range { start: 0, end: 3 },
                target_offset: 7
            })
        ));
        let task = &plan.tasks[1];
        assert_eq!(task.source.len(), 2);
        assert!(matches!(
            &task.source[0],
            AlignmentItem::SourceFragmentRange(FragmentTakeRange {
                fragment_id: 0,
                offset_range: std::ops::Range { start: 3, end: 6 },
                target_offset: 0
            })
        ));
        assert!(matches!(&task.source[1], AlignmentItem::Nulls(7)));

        let mut task_results = Vec::with_capacity(plan.tasks.len());
        for task in plan.tasks.iter().rev() {
            task_results.push(
                task.execute(source_dataset.clone(), target_dataset.clone())
                    .await
                    .unwrap(),
            );
        }

        plan.commit(task_results, source_dataset.clone(), target_dataset.clone())
            .await
            .unwrap();

        let target_dataset = Dataset::open(target_uri).await.unwrap();
        let data = target_dataset.scan().try_into_batch().await.unwrap();

        let expected = gen()
            .col("a", array::step::<Int32Type>())
            .col(
                "b",
                array::cycle::<Int64Type>(
                    [
                        vec![0; 7],
                        vec![0, 1, 2, 3, 4, 5],
                        vec![0; 24],
                        vec![6, 7, 8],
                    ]
                    .concat(),
                )
                .with_nulls(
                    &[
                        vec![true; 7],
                        vec![false; 6],
                        vec![true; 24],
                        vec![false; 3],
                    ]
                    .concat(),
                ),
            )
            .into_batch_rows(RowCount::from(40))
            .unwrap();

        assert_eq!(data, expected);
    }
}
