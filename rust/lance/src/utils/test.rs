// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors


struct TestDatasetGenerator {
    uri: Path,
    seed: Option<u64>,
    data: Vec<RecordBatch>,
}

impl TestDatasetGenerator {
    /// Create a new dataset generator with the given data.
    /// 
    /// Each batch will become a separate fragment in the dataset.
    pub fn new(data: Vec<RecordBatch>, uri: Path) -> Self {
        assert!(!data.is_empty());
        Self {
            uri,
            data,
            seed: None,
        }
    }

    pub fn seed(self, seed: u64)  -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn make_hostile(&self) -> Dataset {
        todo!()
    }

    fn make_schema(&self) -> Schema {
        let arrow_schema = self.data[0].schema();
        let mut schema = Schema::try_from(&arrow_schema).unwrap();
        // Randomize the field ids

        schema
    }

    fn make_fragment(&self, batch: &RecordBatch, schema: &Schema) -> FragmentMetadata {
        // Split each column into it's own batch.

        // Write each as own fragment.

        // Combine the fragments in random order.

        todo!()
    }
}

