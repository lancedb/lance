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

import org.lance.Dataset;
import org.lance.Fragment;
import org.lance.FragmentMetadata;
import org.lance.FragmentOperation;
import org.lance.TestVectorDataset;
import org.lance.WriteParams;
import org.lance.index.vector.IvfBuildParams;
import org.lance.index.vector.PQBuildParams;
import org.lance.index.vector.RQBuildParams;
import org.lance.index.vector.SQBuildParams;
import org.lance.index.vector.VectorIndexParams;
import org.lance.index.vector.VectorTrainer;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.FixedSizeListVector;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

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

        Index firstSegment =
            dataset.createIndex(
                IndexOptions.builder(
                        Collections.singletonList(TestVectorDataset.vectorColumnName),
                        IndexType.IVF_FLAT,
                        indexParams)
                    .withIndexName(TestVectorDataset.indexName)
                    .withFragmentIds(Collections.singletonList(fragments.get(0).getId()))
                    .build());

        Index secondSegment =
            dataset.createIndex(
                IndexOptions.builder(
                        Collections.singletonList(TestVectorDataset.vectorColumnName),
                        IndexType.IVF_FLAT,
                        indexParams)
                    .withIndexName(TestVectorDataset.indexName)
                    .withFragmentIds(Collections.singletonList(fragments.get(1).getId()))
                    .build());

        // The index should not be visible before metadata merge & commit
        assertFalse(
            dataset.listIndexes().contains(TestVectorDataset.indexName),
            "Partially created IVF_FLAT index should not present before commit");

        List<Index> builtSegments =
            dataset.buildIndexSegments(List.of(firstSegment, secondSegment), Optional.empty());
        assertEquals(2, builtSegments.size());

        List<Index> committed =
            dataset.commitExistingIndexSegments(
                TestVectorDataset.indexName, TestVectorDataset.vectorColumnName, builtSegments);
        assertEquals(2, committed.size());
        assertTrue(dataset.listIndexes().contains(TestVectorDataset.indexName));
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

        Index firstSegment =
            dataset.createIndex(
                IndexOptions.builder(
                        Collections.singletonList(TestVectorDataset.vectorColumnName),
                        IndexType.IVF_PQ,
                        indexParams)
                    .withIndexName(TestVectorDataset.indexName)
                    .withFragmentIds(Collections.singletonList(fragments.get(0).getId()))
                    .build());

        Index secondSegment =
            dataset.createIndex(
                IndexOptions.builder(
                        Collections.singletonList(TestVectorDataset.vectorColumnName),
                        IndexType.IVF_PQ,
                        indexParams)
                    .withIndexName(TestVectorDataset.indexName)
                    .withFragmentIds(Collections.singletonList(fragments.get(1).getId()))
                    .build());

        assertFalse(
            dataset.listIndexes().contains(TestVectorDataset.indexName),
            "Partially created IVF_PQ index should not present before commit");

        List<Index> builtSegments =
            dataset.buildIndexSegments(List.of(firstSegment, secondSegment), Optional.empty());
        assertEquals(2, builtSegments.size());

        List<Index> committed =
            dataset.commitExistingIndexSegments(
                TestVectorDataset.indexName, TestVectorDataset.vectorColumnName, builtSegments);
        assertEquals(2, committed.size());
        assertTrue(dataset.listIndexes().contains(TestVectorDataset.indexName));
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

        Index firstSegment =
            dataset.createIndex(
                IndexOptions.builder(
                        Collections.singletonList(TestVectorDataset.vectorColumnName),
                        IndexType.IVF_SQ,
                        indexParams)
                    .withIndexName(TestVectorDataset.indexName)
                    .withFragmentIds(Collections.singletonList(fragments.get(0).getId()))
                    .build());

        Index secondSegment =
            dataset.createIndex(
                IndexOptions.builder(
                        Collections.singletonList(TestVectorDataset.vectorColumnName),
                        IndexType.IVF_SQ,
                        indexParams)
                    .withIndexName(TestVectorDataset.indexName)
                    .withFragmentIds(Collections.singletonList(fragments.get(1).getId()))
                    .build());

        assertFalse(
            dataset.listIndexes().contains(TestVectorDataset.indexName),
            "Partially created IVF_SQ index should not present before commit");

        List<Index> builtSegments =
            dataset.buildIndexSegments(List.of(firstSegment, secondSegment), Optional.empty());
        assertEquals(2, builtSegments.size());

        List<Index> committed =
            dataset.commitExistingIndexSegments(
                TestVectorDataset.indexName, TestVectorDataset.vectorColumnName, builtSegments);
        assertEquals(2, committed.size());
        assertTrue(dataset.listIndexes().contains(TestVectorDataset.indexName));
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

  @Test
  public void testIvfCentroidsWithFragmentIds(@TempDir Path tempDir) throws Exception {
    int dim = 8;
    int rowsPerFragment = 32;
    String column = "vec";

    Schema schema =
        new Schema(
            Collections.singletonList(
                new Field(
                    column,
                    FieldType.nullable(new ArrowType.FixedSizeList(dim)),
                    Collections.singletonList(
                        new Field(
                            "item",
                            FieldType.nullable(
                                new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)),
                            null)))));

    try (BufferAllocator allocator = new RootAllocator()) {
      Path datasetPath = tempDir.resolve("fragment_ivf");
      WriteParams emptyParams =
          new WriteParams.Builder().withMaxRowsPerFile(rowsPerFragment).build();
      Dataset.create(allocator, datasetPath.toString(), schema, emptyParams).close();

      // Fragment 0: all zeros
      FragmentMetadata frag0;
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        root.allocateNew();
        FixedSizeListVector vecVector = (FixedSizeListVector) root.getVector(column);
        Float4Vector items = (Float4Vector) vecVector.getDataVector();
        for (int i = 0; i < rowsPerFragment; i++) {
          for (int j = 0; j < dim; j++) {
            items.setSafe(i * dim + j, 0.0f);
          }
          vecVector.setNotNull(i);
        }
        root.setRowCount(rowsPerFragment);
        frag0 =
            Fragment.create(
                    datasetPath.toString(), allocator, root, new WriteParams.Builder().build())
                .get(0);
      }

      // Fragment 1: all 10.0
      FragmentMetadata frag1;
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        root.allocateNew();
        FixedSizeListVector vecVector = (FixedSizeListVector) root.getVector(column);
        Float4Vector items = (Float4Vector) vecVector.getDataVector();
        for (int i = 0; i < rowsPerFragment; i++) {
          for (int j = 0; j < dim; j++) {
            items.setSafe(i * dim + j, 10.0f);
          }
          vecVector.setNotNull(i);
        }
        root.setRowCount(rowsPerFragment);
        frag1 =
            Fragment.create(
                    datasetPath.toString(), allocator, root, new WriteParams.Builder().build())
                .get(0);
      }

      List<FragmentMetadata> fragments = new ArrayList<>();
      fragments.add(frag0);
      fragments.add(frag1);
      FragmentOperation.Append appendOp = new FragmentOperation.Append(fragments);
      try (Dataset dataset =
          Dataset.commit(allocator, datasetPath.toString(), appendOp, Optional.of(1L))) {

        List<Fragment> dsFragments = dataset.getFragments();
        assertEquals(2, dsFragments.size());

        IvfBuildParams ivfParams =
            new IvfBuildParams.Builder()
                .setNumPartitions(1)
                .setMaxIters(1)
                .setSampleRate(2)
                .build();

        // Train IVF on fragment 0 (zeros) only
        float[] firstCentroids =
            VectorTrainer.trainIvfCentroids(
                dataset, column, ivfParams, Collections.singletonList(dsFragments.get(0).getId()));

        // Train IVF on fragment 1 (10.0s) only
        float[] secondCentroids =
            VectorTrainer.trainIvfCentroids(
                dataset, column, ivfParams, Collections.singletonList(dsFragments.get(1).getId()));

        assertEquals(dim, firstCentroids.length);
        assertEquals(dim, secondCentroids.length);

        for (int j = 0; j < dim; j++) {
          assertEquals(0.0f, firstCentroids[j], 1e-4f, "first centroid[" + j + "] should be ~0.0");
          assertEquals(
              10.0f, secondCentroids[j], 1e-4f, "second centroid[" + j + "] should be ~10.0");
        }
      }
    }
  }

  @Test
  public void testPqCodebookWithFragmentIds(@TempDir Path tempDir) throws Exception {
    try (TestVectorDataset testVectorDataset =
        new TestVectorDataset(tempDir.resolve("pq_fragment_ids"))) {
      try (Dataset dataset = testVectorDataset.create()) {
        List<Fragment> fragments = dataset.getFragments();
        assertTrue(fragments.size() >= 4, "Expected at least four fragments");
        // Use 4 fragments (320 rows total) to meet PQ sample requirements
        List<Integer> fragmentIds =
            List.of(
                fragments.get(0).getId(),
                fragments.get(1).getId(),
                fragments.get(2).getId(),
                fragments.get(3).getId());

        IvfBuildParams ivfParams =
            new IvfBuildParams.Builder()
                .setNumPartitions(4)
                .setMaxIters(1)
                .setSampleRate(16)
                .build();

        float[] centroids =
            VectorTrainer.trainIvfCentroids(
                dataset, TestVectorDataset.vectorColumnName, ivfParams, fragmentIds);
        assertNotNull(centroids);
        assertTrue(centroids.length > 0, "IVF centroids should not be empty");

        PQBuildParams pqParams =
            new PQBuildParams.Builder()
                .setNumSubVectors(2)
                .setNumBits(8)
                .setMaxIters(2)
                .setSampleRate(1)
                .build();

        float[] codebook =
            VectorTrainer.trainPqCodebook(
                dataset, TestVectorDataset.vectorColumnName, pqParams, centroids, fragmentIds);
        assertNotNull(codebook);
        assertTrue(codebook.length > 0, "PQ codebook should not be empty");
      }
    }
  }
}
