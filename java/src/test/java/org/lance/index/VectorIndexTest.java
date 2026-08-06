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
import org.lance.TestVectorDataset;
import org.lance.index.vector.IvfBuildParams;
import org.lance.index.vector.PQBuildParams;
import org.lance.index.vector.RQBuildParams;
import org.lance.index.vector.SQBuildParams;
import org.lance.index.vector.VectorIndexParams;
import org.lance.index.vector.VectorTrainer;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
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

        List<Index> committed =
            dataset.commitExistingIndexSegments(
                TestVectorDataset.indexName,
                TestVectorDataset.vectorColumnName,
                List.of(firstSegment, secondSegment));
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

        List<Index> committed =
            dataset.commitExistingIndexSegments(
                TestVectorDataset.indexName,
                TestVectorDataset.vectorColumnName,
                List.of(firstSegment, secondSegment));
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

        List<Index> committed =
            dataset.commitExistingIndexSegments(
                TestVectorDataset.indexName,
                TestVectorDataset.vectorColumnName,
                List.of(firstSegment, secondSegment));
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

  /**
   * {@code includeColumns} must be a defensive, unmodifiable copy: mutating the list the caller
   * passed to the builder (or the list the getter returns) must not change the params, so the JNI
   * read at build time sees exactly what was requested.
   */
  @Test
  public void testIncludeColumnsIsDefensivelyCopied() {
    IvfBuildParams ivf = new IvfBuildParams.Builder().setNumPartitions(2).build();
    List<String> cols = new ArrayList<>();
    cols.add("i");
    VectorIndexParams params = new VectorIndexParams.Builder(ivf).setIncludeColumns(cols).build();

    // Mutating the caller's list after build must not affect the params.
    cols.add("s");
    cols.clear();
    assertEquals(
        Collections.singletonList("i"),
        params.getIncludeColumns(),
        "include columns must be a defensive copy, unaffected by caller mutation");

    // The returned list must be unmodifiable.
    assertThrows(UnsupportedOperationException.class, () -> params.getIncludeColumns().add("x"));
  }

  /**
   * {@code Index.includedFields} must be a defensive, unmodifiable copy, matching {@code
   * VectorIndexParams.includeColumns}: mutating the caller's list (or the list the getter returns)
   * must not change the value object -- equals/hashCode would silently drift, and a later JNI
   * commit would read mutated covering metadata while the index files still carry the payload.
   */
  @Test
  public void testIndexIncludedFieldsIsDefensivelyCopied() {
    List<Integer> ids = new ArrayList<>();
    ids.add(3);
    Index index =
        Index.builder()
            .uuid(UUID.randomUUID())
            .fields(Collections.singletonList(0))
            .name("covered_idx")
            .datasetVersion(1L)
            .indexVersion(0)
            .includedFields(ids)
            .build();

    // Mutating the caller's list after build must not affect the index.
    ids.add(7);
    ids.clear();
    assertEquals(
        Collections.singletonList(3),
        index.includedFields(),
        "included fields must be a defensive copy, unaffected by caller mutation");

    // The returned list must be unmodifiable.
    assertThrows(UnsupportedOperationException.class, () -> index.includedFields().add(9));
  }

  /**
   * A covered ("included") column set on {@link VectorIndexParams} must be threaded through the JNI
   * create path into the built index's metadata, so the committed index reports the covered field
   * id. Without the wiring the index would build but silently carry no covering columns.
   */
  @Test
  public void testCreateIvfFlatIndexWithCoveringColumns(@TempDir Path tempDir) throws Exception {
    try (TestVectorDataset testVectorDataset =
        new TestVectorDataset(tempDir.resolve("ivf_flat_covering"))) {
      try (Dataset dataset = testVectorDataset.create()) {
        IvfBuildParams ivf = new IvfBuildParams.Builder().setNumPartitions(2).build();

        // Cover the non-vector "i" column so a projection of it is answered from the index.
        VectorIndexParams vectorIndexParams =
            new VectorIndexParams.Builder(ivf)
                .setDistanceType(DistanceType.L2)
                .setIncludeColumns(Collections.singletonList("i"))
                .build();
        IndexParams indexParams =
            IndexParams.builder().setVectorIndexParams(vectorIndexParams).build();

        dataset.createIndex(
            IndexOptions.builder(
                    Collections.singletonList(TestVectorDataset.vectorColumnName),
                    IndexType.IVF_FLAT,
                    indexParams)
                .withIndexName(TestVectorDataset.indexName)
                .build());

        Index covered =
            dataset.getIndexes().stream()
                .filter(idx -> TestVectorDataset.indexName.equals(idx.name()))
                .findFirst()
                .orElse(null);
        assertNotNull(covered, "Expected covered IVF_FLAT index to be present");
        assertEquals(
            1,
            covered.includedFields().size(),
            "index should report exactly the one covering column that was requested; was "
                + covered.includedFields());
      }
    }
  }

  /**
   * Regression test for the metric_type passthrough in the JNI VectorTrainer.
   *
   * <p>Before this fix, {@code trainIvfCentroids} hardcoded {@code MetricType::L2} on the Rust
   * side, so users requesting a non-L2 metric got centroids clustered on L2 geometry while
   * per-fragment encoders later quantized using the requested metric — silently degraded recall.
   *
   * <p>This test trains the same dataset twice with L2 and Cosine and asserts the centroid arrays
   * differ. With the bug present, the two arrays were identical because both paths fell through to
   * L2.
   */
  @Test
  public void testTrainIvfCentroidsHonorsDistanceType(@TempDir Path tempDir) throws Exception {
    try (TestVectorDataset testVectorDataset =
        new TestVectorDataset(tempDir.resolve("ivf_centroids_metric"))) {
      try (Dataset dataset = testVectorDataset.create()) {
        IvfBuildParams params =
            new IvfBuildParams.Builder().setNumPartitions(2).setMaxIters(10).build();

        float[] l2Centroids =
            VectorTrainer.trainIvfCentroids(
                dataset, TestVectorDataset.vectorColumnName, params, DistanceType.L2);
        float[] cosineCentroids =
            VectorTrainer.trainIvfCentroids(
                dataset, TestVectorDataset.vectorColumnName, params, DistanceType.Cosine);

        assertEquals(l2Centroids.length, cosineCentroids.length);
        assertFalse(
            arraysApproximatelyEqual(l2Centroids, cosineCentroids),
            "L2 and Cosine centroids should differ — Cosine normalizes input before clustering."
                + " If they are equal, the metric is not being threaded into the trainer.");
      }
    }
  }

  /**
   * Regression test for the metric_type passthrough in PQ codebook training.
   *
   * <p>Cosine training normalizes the input vectors before k-means; L2 does not. So the resulting
   * codebooks must differ when the metric is honored.
   */
  @Test
  public void testTrainPqCodebookHonorsDistanceType(@TempDir Path tempDir) throws Exception {
    try (TestVectorDataset testVectorDataset =
        new TestVectorDataset(tempDir.resolve("pq_codebook_metric"))) {
      try (Dataset dataset = testVectorDataset.create()) {
        PQBuildParams params =
            new PQBuildParams.Builder()
                .setNumSubVectors(2)
                .setNumBits(4)
                .setMaxIters(5)
                .setSampleRate(64)
                .build();

        float[] l2Codebook =
            VectorTrainer.trainPqCodebook(
                dataset, TestVectorDataset.vectorColumnName, params, DistanceType.L2);
        float[] cosineCodebook =
            VectorTrainer.trainPqCodebook(
                dataset, TestVectorDataset.vectorColumnName, params, DistanceType.Cosine);

        assertEquals(l2Codebook.length, cosineCodebook.length);
        assertFalse(
            arraysApproximatelyEqual(l2Codebook, cosineCodebook),
            "L2 and Cosine PQ codebooks should differ — Cosine normalizes the training data."
                + " If they are equal, the metric is not being threaded into the trainer.");
      }
    }
  }

  private static boolean arraysApproximatelyEqual(float[] a, float[] b) {
    if (a.length != b.length) {
      return false;
    }
    final float epsilon = 1e-6f;
    for (int i = 0; i < a.length; i++) {
      if (Math.abs(a[i] - b[i]) > epsilon) {
        return false;
      }
    }
    return true;
  }
}
