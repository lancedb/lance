// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! tests with fixtures -- nothing in this file is should be public interface

#[cfg(test)]
mod test {
    use std::{
        any::Any,
        cell::OnceCell,
        collections::HashMap,
        sync::{Arc, Mutex},
    };

    use approx::assert_relative_eq;
    use arrow::array::AsArray;
    use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use async_trait::async_trait;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::{vector::Query, Index, IndexType};
    use lance_io::{local::LocalObjectReader, traits::Reader};
    use lance_linalg::distance::MetricType;
    use roaring::RoaringBitmap;
    use uuid::Uuid;

    use super::super::VectorIndex;
    use crate::{
        index::{
            prefilter::PreFilter,
            vector::ivf::{IVFIndex, Ivf},
        },
        session::Session,
        Result,
    };

    #[derive(Clone, Debug)]
    struct ResidualCheckMockIndex {
        use_residual: bool,
        metric_type: MetricType,

        assert_query_value: Vec<f32>,

        ret_val: RecordBatch,
    }

    #[async_trait]
    impl Index for ResidualCheckMockIndex {
        /// Cast to [Any].
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_mut_any(&mut self) -> &mut dyn Any {
            self
        }

        /// Cast to [Index]
        fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
            self
        }

        /// Retrieve index statistics as a JSON Value
        fn statistics(&self) -> Result<serde_json::Value> {
            Ok(serde_json::Value::Null)
        }

        /// Get the type of the index
        fn index_type(&self) -> IndexType {
            IndexType::Vector
        }

        async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
            Ok(RoaringBitmap::new())
        }
    }

    #[async_trait]
    impl VectorIndex for ResidualCheckMockIndex {
        async fn search(&self, query: &Query, _pre_filter: Arc<PreFilter>) -> Result<RecordBatch> {
            let key: &Float32Array = query.key.as_primitive();
            assert_eq!(key.len(), self.assert_query_value.len());
            for (i, &v) in key.iter().zip(self.assert_query_value.iter()) {
                assert_relative_eq!(v, i.unwrap());
            }
            Ok(self.ret_val.clone())
        }

        fn is_loadable(&self) -> bool {
            true
        }

        fn use_residual(&self) -> bool {
            self.use_residual
        }

        fn check_can_remap(&self) -> Result<()> {
            Ok(())
        }

        async fn load(
            &self,
            _reader: Arc<dyn Reader>,
            _offset: usize,
            _length: usize,
        ) -> Result<Box<dyn VectorIndex>> {
            Ok(Box::new(self.clone()))
        }

        fn remap(&mut self, _mapping: &HashMap<u64, Option<u64>>) -> Result<()> {
            Ok(())
        }

        fn cast(&mut self, _to: &Field) -> Result<()> {
            todo!()
        }

        /// The metric type of this vector index.
        fn metric_type(&self) -> MetricType {
            self.metric_type
        }
    }

    #[tokio::test]
    async fn test_ivf_residual_handling() {
        let centroids = Float32Array::from_iter(vec![1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0]);
        let centroids = FixedSizeListArray::try_new_from_values(centroids, 2).unwrap();
        let mut ivf = Ivf::new(Arc::new(centroids));
        // Add 4 partitions
        for _ in 0..4 {
            ivf.add_partition(0, 0);
        }
        // hold on to this pointer, because the index only holds a weak reference
        let session = Arc::new(Session::default());

        let make_idx = move |assert_query: Vec<f32>, metric: MetricType| async move {
            let f = tempfile::NamedTempFile::new().unwrap();

            let reader = LocalObjectReader::open_local_path(f.path(), 64)
                .await
                .unwrap();

            let mock_sub_index = Arc::new(ResidualCheckMockIndex {
                use_residual: true,
                // always L2
                metric_type: MetricType::L2,
                assert_query_value: assert_query,
                ret_val: RecordBatch::new_empty(Arc::new(Schema::new(vec![
                    Field::new("id", DataType::UInt64, false),
                    Field::new("_distance", DataType::Float32, false),
                ]))),
            });
            IVFIndex::try_new(
                session.clone(),
                &Uuid::new_v4().to_string(),
                ivf,
                reader.into(),
                mock_sub_index,
                metric,
            )
            .unwrap()
        };

        struct TestCase {
            query: Vec<f32>,
            metric: MetricType,
            expected_query_at_subindex: Vec<f32>,
        }

        for TestCase {
            query,
            metric,
            expected_query_at_subindex,
        } in [
            // L2 should residualize with the correct centroid
            TestCase {
                query: vec![1.0; 2],
                metric: MetricType::L2,
                expected_query_at_subindex: vec![0.0; 2],
            },
            // Cosine should normalize and residualize
            TestCase {
                query: vec![1.0; 2],
                metric: MetricType::Cosine,
                expected_query_at_subindex: vec![1.0 / 2.0_f32.sqrt() - 1.0; 2],
            },
            TestCase {
                query: vec![2.0; 2],
                metric: MetricType::Cosine,
                expected_query_at_subindex: vec![2.0 / 8.0_f32.sqrt() - 1.0; 2],
            },
        ] {
            let q = Query {
                column: "test".to_string(),
                key: Arc::new(Float32Array::from(query)),
                k: 1,
                nprobes: 1,
                refine_factor: None,
                metric_type: metric,
                use_index: true,
            };
            let idx = make_idx.clone()(expected_query_at_subindex, metric).await;
            idx.search(
                &q,
                Arc::new(PreFilter {
                    deleted_ids: None,
                    filtered_ids: None,
                    final_mask: Mutex::new(OnceCell::new()),
                }),
            )
            .await
            .unwrap();
        }
    }
}
