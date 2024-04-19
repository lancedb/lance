// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchIterator};
use arrow_schema::Schema as ArrowSchema;
use lance_arrow::RecordBatchExt;
use lance_core::datatypes::Schema;
use lance_table::format::Fragment;
use rand::prelude::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::dataset::fragment::write::FragmentCreateBuilder;
use crate::dataset::transaction::Operation;
use crate::Dataset;

pub struct TestDatasetGenerator {
    seed: Option<u64>,
    data: Vec<RecordBatch>,
}

impl TestDatasetGenerator {
    /// Create a new dataset generator with the given data.
    ///
    /// Each batch will become a separate fragment in the dataset.
    pub fn new(data: Vec<RecordBatch>) -> Self {
        assert!(!data.is_empty());
        Self { data, seed: None }
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Make a new dataset that has a "hostile" layout.
    ///
    /// By "hostile", we mean that:
    /// 1. Top-level columns are randomly split into different files. If there
    ///    are multiple fragments, they do not have the same arrangement of
    ///    fields in data files.
    /// 2. The field ids are not in sorted order, and have holes.
    /// 3. The order of fields across the data files is random, and not
    ///    consistent across fragments.
    ///
    pub async fn make_hostile(&self, uri: &str) -> Dataset {
        let seed = self.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let schema = self.make_schema(&mut rng);

        // If we only have one fragment, we should split it into two files. But
        // if we have multiple fragments, we can allow one of them to have a single
        // file. This prevents an infinite loop.
        let min_num_files = if self.data.len() > 1 { 1 } else { 2 };

        let mut fragments = Vec::with_capacity(self.data.len());
        let mut id = 0;

        for batch in &self.data {
            loop {
                let mut fragment = self
                    .make_fragment(uri, batch, &schema, &mut rng, min_num_files)
                    .await;

                let fields = field_structure(&fragment);
                let first_fields = fragments.first().map(field_structure);
                if let Some(first_fields) = first_fields {
                    if fields == first_fields {
                        // The layout is the same as the first fragment, try again
                        continue;
                    }
                }

                fragment.id = id;
                id += 1;
                fragments.push(fragment);
                break;
            }
        }

        let operation = Operation::Overwrite { fragments, schema };

        Dataset::commit(uri, operation, None, Default::default(), None)
            .await
            .unwrap()
    }

    fn make_schema(&self, rng: &mut impl Rng) -> Schema {
        let arrow_schema = self.data[0].schema();
        let mut schema = Schema::try_from(arrow_schema.as_ref()).unwrap();

        let field_ids = schema.fields_pre_order().map(|f| f.id).collect::<Vec<_>>();
        let mut new_ids = field_ids.clone();
        // Add a hole
        if new_ids.len() > 2 {
            let hole_pos = rng.gen_range(1..new_ids.len() - 1);
            for id in new_ids.iter_mut().skip(hole_pos) {
                *id += 1;
            }
        }
        // Randomize the order of ids
        loop {
            new_ids.shuffle(rng);
            // In case we accidentally shuffled to the same order
            if new_ids.len() == 1 || new_ids != field_ids {
                break;
            }
        }
        for (old_id, new_id) in field_ids.iter().zip(new_ids.iter()) {
            let field = schema.mut_field_by_id(*old_id).unwrap();
            field.id = *new_id;
        }

        schema
    }

    async fn make_fragment(
        &self,
        uri: &str,
        batch: &RecordBatch,
        schema: &Schema,
        rng: &mut impl Rng,
        min_num_files: usize,
    ) -> Fragment {
        // Choose a random number of files.
        let num_files = if batch.num_columns() == 1 {
            1
        } else {
            rng.gen_range(min_num_files..=batch.num_columns())
        };

        // Randomly assign top level fields to files.
        let column_names = batch
            .schema()
            .fields
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>();
        let mut file_assignments = (0..num_files)
            .cycle()
            .take(column_names.len())
            .collect::<Vec<_>>();
        file_assignments.shuffle(rng);

        // Write each as own fragment.
        let mut sub_fragments = Vec::with_capacity(num_files);
        for file_id in 0..num_files {
            let columns = column_names
                .iter()
                .zip(file_assignments.iter())
                .filter_map(|(name, &file)| {
                    if file == file_id {
                        Some(name.clone())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            let file_schema = schema.project(&columns).unwrap();
            let file_arrow_schema = Arc::new(ArrowSchema::from(&file_schema));
            let data = batch.project_by_schema(file_arrow_schema.as_ref()).unwrap();
            let reader = RecordBatchIterator::new(vec![Ok(data)], file_arrow_schema.clone());
            let sub_frag = FragmentCreateBuilder::new(uri)
                .schema(&file_schema)
                .write(reader, None)
                .await
                .unwrap();

            // The sub_fragment has it's own schema, with field ids that are local to
            // it. We need to remap the field ids to the global schema.

            sub_fragments.push(sub_frag);
        }

        // Combine the fragments into a single one.
        let mut files = sub_fragments
            .into_iter()
            .flat_map(|frag| frag.files.into_iter())
            .collect::<Vec<_>>();

        // Make sure the field id order is distinct from the schema.
        let schema_field_ids = schema.fields_pre_order().map(|f| f.id).collect::<Vec<_>>();
        if files
            .iter()
            .flat_map(|file| file.fields.iter().cloned())
            .collect::<Vec<_>>()
            == schema_field_ids
            && files.len() > 1
        {
            // Swap first two files
            files.swap(0, 1);
        }

        Fragment {
            id: 0,
            files,
            deletion_file: None,
            physical_rows: Some(batch.num_rows()),
        }
    }
}

fn get_field_structure(dataset: &Dataset) -> Vec<Vec<Vec<i32>>> {
    dataset
        .get_fragments()
        .into_iter()
        .map(|frag| field_structure(frag.metadata()))
        .collect::<Vec<_>>()
}

fn field_structure(fragment: &Fragment) -> Vec<Vec<i32>> {
    fragment
        .files
        .iter()
        .map(|file| file.fields.clone())
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use arrow_array::{ArrayRef, BooleanArray, Float64Array, Int32Array, StringArray, StructArray};
    use arrow_schema::{DataType, Field as ArrowField, Fields as ArrowFields};

    #[test]
    fn test_make_schema() {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new(
                "b",
                DataType::Struct(
                    vec![
                        ArrowField::new("f1", DataType::Utf8, true),
                        ArrowField::new("f2", DataType::Boolean, false),
                    ]
                    .into(),
                ),
                true,
            ),
            ArrowField::new("c", DataType::Float64, false),
        ]));
        let data = vec![RecordBatch::new_empty(arrow_schema.clone())];

        let generator = TestDatasetGenerator::new(data);
        let schema = generator.make_schema(&mut rand::thread_rng());

        let roundtripped_schema = ArrowSchema::try_from(&schema).unwrap();
        assert_eq!(&roundtripped_schema, arrow_schema.as_ref());

        let field_ids = schema.fields_pre_order().map(|f| f.id).collect::<Vec<_>>();
        let mut sorted_ids = field_ids.clone();
        sorted_ids.sort_unstable();
        assert_ne!(field_ids, sorted_ids);

        let mut num_holes = 0;
        for w in sorted_ids.windows(2) {
            let prev = w[0];
            let next = w[1];
            if next - prev > 1 {
                num_holes += 1;
            }
        }
        assert!(num_holes > 0, "Expected at least one hole in the field ids");
    }

    #[tokio::test]
    async fn test_make_fragment() {
        let tmp_dir = tempfile::tempdir().unwrap();

        let struct_fields: ArrowFields = vec![
            ArrowField::new("f1", DataType::Utf8, true),
            ArrowField::new("f2", DataType::Boolean, false),
        ]
        .into();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new("b", DataType::Struct(struct_fields.clone()), true),
            ArrowField::new("c", DataType::Float64, false),
        ]));
        let data = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StructArray::new(
                    struct_fields,
                    vec![
                        Arc::new(StringArray::from(vec!["foo", "bar", "baz"])) as ArrayRef,
                        Arc::new(BooleanArray::from(vec![true, false, true])),
                    ],
                    None,
                )),
                Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3])),
            ],
        )
        .unwrap();

        let generator = TestDatasetGenerator::new(vec![data.clone()]);
        let mut rng = rand::thread_rng();
        for _ in 1..50 {
            let schema = generator.make_schema(&mut rng);
            let fragment = generator
                .make_fragment(
                    tmp_dir.path().to_str().unwrap(),
                    &data,
                    &schema,
                    &mut rng,
                    2,
                )
                .await;

            assert!(fragment.files.len() > 1, "Expected multiple files");

            let mut field_ids_frags = fragment
                .files
                .iter()
                .flat_map(|file| file.fields.iter())
                .cloned()
                .collect::<Vec<_>>();
            let mut field_ids = schema.fields_pre_order().map(|f| f.id).collect::<Vec<_>>();
            assert_ne!(field_ids_frags, field_ids);
            field_ids_frags.sort_unstable();
            field_ids.sort_unstable();
            assert_eq!(field_ids_frags, field_ids);
        }
    }

    #[tokio::test]
    async fn test_make_hostile() {
        let tmp_dir = tempfile::tempdir().unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new("c", DataType::Float64, false),
        ]));
        let data = vec![
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(vec![1, 2, 3])),
                    Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3])),
                ],
            )
            .unwrap(),
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(vec![4, 5, 6])),
                    Arc::new(Float64Array::from(vec![4.4, 5.5, 6.6])),
                ],
            )
            .unwrap(),
        ];

        let seed = 42;
        let generator = TestDatasetGenerator::new(data.clone()).seed(seed);

        let path = tmp_dir.path().join("ds1");
        let dataset = generator.make_hostile(path.to_str().unwrap()).await;

        let path2 = tmp_dir.path().join("ds2");
        let dataset2 = generator.make_hostile(path2.to_str().unwrap()).await;

        // Given the same seed, should produce the same layout.
        assert_eq!(dataset.schema(), dataset2.schema());
        let field_structure_1 = get_field_structure(&dataset);
        let field_structure_2 = get_field_structure(&dataset2);
        assert_eq!(field_structure_1, field_structure_2);

        // Now make it random
        for i in 1..20 {
            let generator = TestDatasetGenerator::new(data.clone());
            let path = tmp_dir.path().join(format!("test_ds{}", i));
            let dataset = generator.make_hostile(path.to_str().unwrap()).await;

            let field_structure = get_field_structure(&dataset);

            // The two fragments should have different layout.
            assert_eq!(field_structure.len(), 2);
            assert_ne!(field_structure[0], field_structure[1]);
        }
    }
}
