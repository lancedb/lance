/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.lance.index;

import org.lance.CommitBuilder;
import org.lance.Dataset;
import org.lance.Fragment;
import org.lance.TestVectorDataset;
import org.lance.Transaction;
import org.lance.index.vector.IvfBuildParams;
import org.lance.index.vector.PQBuildParams;
import org.lance.index.vector.RQBuildParams;
import org.lance.index.vector.SQBuildParams;
import org.lance.index.vector.VectorIndexParams;
import org.lance.index.vector.VectorTrainer;
import org.lance.operation.CreateIndex;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class VectorIndexTest {

  @Test
  public void testCreateIvfFlatIndexDistributively(@TempDir Path tempDir) throws Exception {
    try (TestVectorDataset testVectorDataset =
        new TestVectorDataset(tempDir.resolve("merge_ivfflat_index_metadata"))) {
      try (Dataset dataset = testVectorDataset.create()) {
        List<Fragment> fragments = dataset.getFragments();
        assertTrue(
            fragments.size() >= 2,
            "Expected dataset to have at least two fragments for distributed indexing");

        int numPartitions = 2;

        IvfBuildParams ivfTrainParams =
            new IvfBuildParams.Builder().setNumPartitions(numPartitions).setMaxIters(1).build();

        float[] centroids =
            VectorTrainer.trainIvfCentroids(
                dataset, TestVectorDataset.vectorColumnName, ivfTrainParams);

        IvfBuildParams ivfParams =
            new IvfBuildParams.Builder()
                .setNumPartitions(numPartitions)
                .setMaxIters(1)
                .setCentroids(centroids)
                .build();

        VectorIndexParams vectorIndexParams =
            new VectorIndexParams.Builder(ivfParams).setDistanceType(DistanceType.L2).build();

        IndexParams indexParams =
            IndexParams.builder().setVectorIndexParams(vectorIndexParams).build();

        UUID indexUUID = UUID.randomUUID();

        // Partially create index on the first fragment
        dataset.createIndex(
            IndexOptions.builder(
                    Collections.singletonList(TestVectorDataset.vectorColumnName),
                    IndexType.IVF_FLAT,
                    indexParams)
                .withIndexName(TestVectorDataset.indexName)
                .withIndexUUID(indexUUID.toString())
                .withFragmentIds(Collections.singletonList(fragments.get(0).getId()))
                .build());

        // Partially create index on the second fragment with the same UUID
        dataset.createIndex(
            IndexOptions.builder(
                    Collections.singletonList(TestVectorDataset.vectorColumnName),
                    IndexType.IVF_FLAT,
                    indexParams)
                .withIndexName(TestVectorDataset.indexName)
                .withIndexUUID(indexUUID.toString())
                .withFragmentIds(Collections.singletonList(fragments.get(1).getId()))
                .build());

        // The index should not be visible before metadata merge & commit
        assertFalse(
            dataset.listIndexes().contains(TestVectorDataset.indexName),
            "Partially created IVF_FLAT index should not present before commit");

        // Merge index metadata for all fragment-level pieces
        dataset.mergeIndexMetadata(indexUUID.toString(), IndexType.IVF_FLAT, Optional.empty());

        int fieldId =
            dataset.getLanceSchema().fields().stream()
                .filter(f -> f.getName().equals(TestVectorDataset.vectorColumnName))
                .findAny()
                .orElseThrow(
                    () -> new RuntimeException("Cannot find vector field for TestVectorDataset"))
                .getId();

        long datasetVersion = dataset.version();

        Index index =
            Index.builder()
                .uuid(indexUUID)
                .name(TestVectorDataset.indexName)
                .fields(Collections.singletonList(fieldId))
                .datasetVersion(datasetVersion)
                .indexVersion(0)
                .fragments(
                    fragments.stream().limit(2).map(Fragment::getId).collect(Collectors.toList()))
                .build();

        CreateIndex createIndexOp =
            CreateIndex.builder().withNewIndices(Collections.singletonList(index)).build();

        try (Transaction createIndexTx =
            new Transaction.Builder()
                .readVersion(dataset.version())
                .operation(createIndexOp)
                .build()) {
          try (Dataset newDataset = new CommitBuilder(dataset).execute(createIndexTx)) {
            assertEquals(datasetVersion + 1, newDataset.version());
            assertTrue(newDataset.listIndexes().contains(TestVectorDataset.indexName));
          }
        }
      }
    }
  }

  @Test
  public void testCreateIvfPqIndexDistributively(@TempDir Path tempDir) throws Exception {
    try (TestVectorDataset testVectorDataset =
        new TestVectorDataset(tempDir.resolve("merge_ivfpq_index_metadata"))) {
      try (Dataset dataset = testVectorDataset.create()) {
        List<Fragment> fragments = dataset.getFragments();
        assertTrue(
            fragments.size() >= 2,
            "Expected dataset to have at least two fragments for distributed indexing");

        int numPartitions = 2;
        int numSubVectors = 2;
        int numBits = 8;

        IvfBuildParams ivfTrainParams =
            new IvfBuildParams.Builder().setNumPartitions(numPartitions).setMaxIters(1).build();

        PQBuildParams pqTrainParams =
            new PQBuildParams.Builder()
                .setNumSubVectors(numSubVectors)
                .setNumBits(numBits)
                .setMaxIters(2)
                .setSampleRate(256)
                .build();

        float[] centroids =
            VectorTrainer.trainIvfCentroids(
                dataset, TestVectorDataset.vectorColumnName, ivfTrainParams);

        float[] codebook =
            VectorTrainer.trainPqCodebook(
                dataset, TestVectorDataset.vectorColumnName, pqTrainParams);

        IvfBuildParams ivfParams =
            new IvfBuildParams.Builder()
                .setNumPartitions(numPartitions)
                .setMaxIters(1)
                .setCentroids(centroids)
                .build();

        PQBuildParams pqParams =
            new PQBuildParams.Builder()
                .setNumSubVectors(numSubVectors)
                .setNumBits(numBits)
                .setMaxIters(2)
                .setSampleRate(256)
                .setCodebook(codebook)
                .build();

        VectorIndexParams vectorIndexParams =
            VectorIndexParams.withIvfPqParams(DistanceType.L2, ivfParams, pqParams);

        IndexParams indexParams =
            IndexParams.builder().setVectorIndexParams(vectorIndexParams).build();

        UUID indexUUID = UUID.randomUUID();

        dataset.createIndex(
            IndexOptions.builder(
                    Collections.singletonList(TestVectorDataset.vectorColumnName),
                    IndexType.IVF_PQ,
                    indexParams)
                .withIndexName(TestVectorDataset.indexName)
                .withIndexUUID(indexUUID.toString())
                .withFragmentIds(Collections.singletonList(fragments.get(0).getId()))
                .build());

        dataset.createIndex(
            IndexOptions.builder(
                    Collections.singletonList(TestVectorDataset.vectorColumnName),
                    IndexType.IVF_PQ,
                    indexParams)
                .withIndexName(TestVectorDataset.indexName)
                .withIndexUUID(indexUUID.toString())
                .withFragmentIds(Collections.singletonList(fragments.get(1).getId()))
                .build());

        assertFalse(
            dataset.listIndexes().contains(TestVectorDataset.indexName),
            "Partially created IVF_PQ index should not present before commit");

        dataset.mergeIndexMetadata(indexUUID.toString(), IndexType.IVF_PQ, Optional.empty());

        int fieldId =
            dataset.getLanceSchema().fields().stream()
                .filter(f -> f.getName().equals(TestVectorDataset.vectorColumnName))
                .findAny()
                .orElseThrow(
                    () -> new RuntimeException("Cannot find vector field for TestVectorDataset"))
                .getId();

        long datasetVersion = dataset.version();

        Index index =
            Index.builder()
                .uuid(indexUUID)
                .name(TestVectorDataset.indexName)
                .fields(Collections.singletonList(fieldId))
                .datasetVersion(datasetVersion)
                .indexVersion(0)
                .fragments(
                    fragments.stream().limit(2).map(Fragment::getId).collect(Collectors.toList()))
                .build();

        CreateIndex createIndexOp =
            CreateIndex.builder().withNewIndices(Collections.singletonList(index)).build();

        try (Transaction createIndexTx =
            new Transaction.Builder()
                .readVersion(dataset.version())
                .operation(createIndexOp)
                .build()) {
          try (Dataset newDataset = new CommitBuilder(dataset).execute(createIndexTx)) {
            assertEquals(datasetVersion + 1, newDataset.version());
            assertTrue(newDataset.listIndexes().contains(TestVectorDataset.indexName));
          }
        }
      }
    }
  }

  @Test
  public void testCreateIvfSqIndexDistributively(@TempDir Path tempDir) throws Exception {
    try (TestVectorDataset testVectorDataset =
        new TestVectorDataset(tempDir.resolve("merge_ivfsq_index_metadata"))) {
      try (Dataset dataset = testVectorDataset.create()) {
        List<Fragment> fragments = dataset.getFragments();
        assertTrue(
            fragments.size() >= 2,
            "Expected dataset to have at least two fragments for distributed indexing");

        int numPartitions = 2;
        short numBits = 8;

        IvfBuildParams ivfTrainParams =
            new IvfBuildParams.Builder().setNumPartitions(numPartitions).setMaxIters(1).build();

        SQBuildParams sqParams =
            new SQBuildParams.Builder().setNumBits(numBits).setSampleRate(256).build();

        float[] centroids =
            VectorTrainer.trainIvfCentroids(
                dataset, TestVectorDataset.vectorColumnName, ivfTrainParams);

        IvfBuildParams ivfParams =
            new IvfBuildParams.Builder()
                .setNumPartitions(numPartitions)
                .setMaxIters(1)
                .setCentroids(centroids)
                .build();

        VectorIndexParams vectorIndexParams =
            new VectorIndexParams.Builder(ivfParams)
                .setDistanceType(DistanceType.L2)
                .setSqParams(sqParams)
                .build();

        IndexParams indexParams =
            IndexParams.builder().setVectorIndexParams(vectorIndexParams).build();

        UUID indexUUID = UUID.randomUUID();

        dataset.createIndex(
            IndexOptions.builder(
                    Collections.singletonList(TestVectorDataset.vectorColumnName),
                    IndexType.IVF_SQ,
                    indexParams)
                .withIndexName(TestVectorDataset.indexName)
                .withIndexUUID(indexUUID.toString())
                .withFragmentIds(Collections.singletonList(fragments.get(0).getId()))
                .build());

        dataset.createIndex(
            IndexOptions.builder(
                    Collections.singletonList(TestVectorDataset.vectorColumnName),
                    IndexType.IVF_SQ,
                    indexParams)
                .withIndexName(TestVectorDataset.indexName)
                .withIndexUUID(indexUUID.toString())
                .withFragmentIds(Collections.singletonList(fragments.get(1).getId()))
                .build());

        assertFalse(
            dataset.listIndexes().contains(TestVectorDataset.indexName),
            "Partially created IVF_SQ index should not present before commit");

        dataset.mergeIndexMetadata(indexUUID.toString(), IndexType.IVF_SQ, Optional.empty());

        int fieldId =
            dataset.getLanceSchema().fields().stream()
                .filter(f -> f.getName().equals(TestVectorDataset.vectorColumnName))
                .findAny()
                .orElseThrow(
                    () -> new RuntimeException("Cannot find vector field for TestVectorDataset"))
                .getId();

        long datasetVersion = dataset.version();

        Index index =
            Index.builder()
                .uuid(indexUUID)
                .name(TestVectorDataset.indexName)
                .fields(Collections.singletonList(fieldId))
                .datasetVersion(datasetVersion)
                .indexVersion(0)
                .fragments(
                    fragments.stream().limit(2).map(Fragment::getId).collect(Collectors.toList()))
                .build();

        CreateIndex createIndexOp =
            CreateIndex.builder().withNewIndices(Collections.singletonList(index)).build();

        try (Transaction createIndexTx =
            new Transaction.Builder()
                .readVersion(dataset.version())
                .operation(createIndexOp)
                .build()) {
          try (Dataset newDataset = new CommitBuilder(dataset).execute(createIndexTx)) {
            assertEquals(datasetVersion + 1, newDataset.version());
            assertTrue(newDataset.listIndexes().contains(TestVectorDataset.indexName));
          }
        }
      }
    }
  }

  @Test
  public void testCreateIvfRqIndex(@TempDir Path tempDir) throws Exception {
    Path datasetPath = tempDir.resolve("ivf_rq_index");

    try (TestVectorDataset testVectorDataset = new TestVectorDataset(datasetPath)) {
      try (Dataset dataset = testVectorDataset.create()) {
        IvfBuildParams ivf = new IvfBuildParams.Builder().setNumPartitions(2).build();
        RQBuildParams rq = new RQBuildParams.Builder().setNumBits((byte) 1).build();

        VectorIndexParams vectorIndexParams =
            VectorIndexParams.withIvfRqParams(DistanceType.L2, ivf, rq);
        IndexParams indexParams =
            IndexParams.builder().setVectorIndexParams(vectorIndexParams).build();

        dataset.createIndex(
            IndexOptions.builder(
                    Collections.singletonList(TestVectorDataset.vectorColumnName),
                    IndexType.IVF_RQ,
                    indexParams)
                .withIndexName(TestVectorDataset.indexName)
                .build());

        List<Index> indexes = dataset.getIndexes();
        Index rqIndex =
            indexes.stream()
                .filter(idx -> TestVectorDataset.indexName.equals(idx.name()))
                .findFirst()
                .orElse(null);

        assertNotNull(rqIndex, "Expected IVF_RQ index to be present");

        IndexType indexType = rqIndex.indexType();
        assertNotNull(indexType, "IndexType should be set for IVF_RQ index");

        // Today all vector indices share the same VectorIndexDetails type and map to VECTOR.
        // This assertion allows both VECTOR and IVF_RQ so it remains valid if the mapping
        // is refined in the future.
        assertTrue(
            indexType == IndexType.VECTOR || indexType == IndexType.IVF_RQ,
            "IndexType for IVF_RQ index should be VECTOR or IVF_RQ but was " + indexType);
      }
    }
  }
}
