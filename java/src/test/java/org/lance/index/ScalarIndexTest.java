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
import org.lance.TestUtils;
import org.lance.WriteParams;
import org.lance.index.scalar.ScalarIndexParams;
import org.lance.ipc.LanceScanner;
import org.lance.ipc.ScanOptions;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ScalarIndexTest {

  private static final class RecordingIndexBuildProgress implements IndexBuildProgress {
    private final List<String> events = Collections.synchronizedList(new ArrayList<>());

    @Override
    public void stageStart(String stage, Optional<Long> total, String unit) {
      events.add(
          "start:" + stage + ":" + total.map(String::valueOf).orElse("unknown") + ":" + unit);
    }

    @Override
    public void stageProgress(String stage, long completed) {
      events.add("progress:" + stage + ":" + completed);
    }

    @Override
    public void stageComplete(String stage) {
      events.add("complete:" + stage);
    }

    private List<String> snapshot() {
      synchronized (events) {
        return new ArrayList<>(events);
      }
    }
  }

  private static final class FailingProgressIndexBuildProgress implements IndexBuildProgress {
    private final RecordingIndexBuildProgress recorder = new RecordingIndexBuildProgress();

    @Override
    public void stageStart(String stage, Optional<Long> total, String unit) {
      recorder.stageStart(stage, total, unit);
    }

    @Override
    public void stageProgress(String stage, long completed) {
      recorder.stageProgress(stage, completed);
      throw new IllegalStateException("progress callback failure");
    }

    @Override
    public void stageComplete(String stage) {
      recorder.stageComplete(stage);
    }
  }

  private static final class FailingCompleteIndexBuildProgress implements IndexBuildProgress {
    private final RecordingIndexBuildProgress recorder = new RecordingIndexBuildProgress();

    @Override
    public void stageStart(String stage, Optional<Long> total, String unit) {
      recorder.stageStart(stage, total, unit);
    }

    @Override
    public void stageProgress(String stage, long completed) {
      recorder.stageProgress(stage, completed);
    }

    @Override
    public void stageComplete(String stage) {
      recorder.stageComplete(stage);
      throw new IllegalStateException("complete callback failure");
    }
  }

  /**
   * Progress callback that re-enters the same Dataset via JNI. Without releasing the native field
   * lock before an index operation starts, these calls would deadlock.
   */
  private static final class ReentrantDatasetIndexBuildProgress implements IndexBuildProgress {
    private final Dataset dataset;
    private final RecordingIndexBuildProgress recorder = new RecordingIndexBuildProgress();
    private final AtomicInteger reentries = new AtomicInteger();

    private ReentrantDatasetIndexBuildProgress(Dataset dataset) {
      this.dataset = dataset;
    }

    @Override
    public void stageStart(String stage, Optional<Long> total, String unit) {
      recorder.stageStart(stage, total, unit);
      touchDataset();
    }

    @Override
    public void stageProgress(String stage, long completed) {
      recorder.stageProgress(stage, completed);
      touchDataset();
    }

    @Override
    public void stageComplete(String stage) {
      recorder.stageComplete(stage);
      touchDataset();
    }

    private void touchDataset() {
      assertNotNull(dataset.uri());
      assertTrue(dataset.version() > 0);
      assertTrue(dataset.latestVersion() > 0);
      assertTrue(dataset.countRows() > 0);
      assertFalse(dataset.getFragments().isEmpty());
      assertFalse(dataset.memWalIndexDetails().isPresent());
      reentries.incrementAndGet();
    }
  }

  private static final class WriteReentrantIndexBuildProgress implements IndexBuildProgress {
    private final Dataset dataset;
    private final RecordingIndexBuildProgress recorder = new RecordingIndexBuildProgress();
    private final AtomicInteger reentries = new AtomicInteger();
    private final AtomicReference<RuntimeException> writeFailure = new AtomicReference<>();

    private WriteReentrantIndexBuildProgress(Dataset dataset) {
      this.dataset = dataset;
    }

    @Override
    public void stageStart(String stage, Optional<Long> total, String unit) {
      recorder.stageStart(stage, total, unit);
      if (reentries.getAndIncrement() == 0) {
        try {
          dataset.checkoutLatest();
        } catch (RuntimeException failure) {
          writeFailure.set(failure);
        }
      }
    }

    @Override
    public void stageProgress(String stage, long completed) {
      recorder.stageProgress(stage, completed);
    }

    @Override
    public void stageComplete(String stage) {
      recorder.stageComplete(stage);
    }
  }

  private static final class NestedCreateIndexBuildProgress implements IndexBuildProgress {
    private final Dataset dataset;
    private final IndexOptions segmentOptions;
    private final RecordingIndexBuildProgress recorder = new RecordingIndexBuildProgress();
    private final AtomicReference<RuntimeException> nestedFailure = new AtomicReference<>();
    private boolean attempted;

    private NestedCreateIndexBuildProgress(Dataset dataset, IndexOptions segmentOptions) {
      this.dataset = dataset;
      this.segmentOptions = segmentOptions;
    }

    @Override
    public void stageStart(String stage, Optional<Long> total, String unit) {
      recorder.stageStart(stage, total, unit);
      if (!attempted) {
        attempted = true;
        try {
          dataset.createIndex(segmentOptions);
        } catch (RuntimeException failure) {
          nestedFailure.set(failure);
        }
      }
    }

    @Override
    public void stageProgress(String stage, long completed) {
      recorder.stageProgress(stage, completed);
    }

    @Override
    public void stageComplete(String stage) {
      recorder.stageComplete(stage);
    }
  }

  private static final class BlockingIndexBuildProgress implements IndexBuildProgress {
    private final CountDownLatch started = new CountDownLatch(1);
    private final CountDownLatch release = new CountDownLatch(1);
    private final RecordingIndexBuildProgress recorder = new RecordingIndexBuildProgress();

    @Override
    public void stageStart(String stage, Optional<Long> total, String unit) {
      recorder.stageStart(stage, total, unit);
      started.countDown();
      try {
        assertTrue(release.await(5, TimeUnit.SECONDS), "Callback release latch timed out");
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new IllegalStateException("Interrupted while blocking progress callback", e);
      }
    }

    @Override
    public void stageProgress(String stage, long completed) {
      recorder.stageProgress(stage, completed);
    }

    @Override
    public void stageComplete(String stage) {
      recorder.stageComplete(stage);
    }
  }

  @Test
  public void testCreateBTreeIndex(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("btree_test").toString();
    Schema schema =
        new Schema(
            Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true)),
                Field.nullable("name", new ArrowType.Utf8())),
            null);

    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset dataset =
          Dataset.create(allocator, datasetPath, schema, new WriteParams.Builder().build())) {

        // Create BTree scalar index parameters
        ScalarIndexParams scalarParams = ScalarIndexParams.create("btree", "{\"zone_size\": 2048}");

        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();

        // Create BTree index on 'id' column
        Index index =
            dataset.createIndex(
                Collections.singletonList("id"),
                IndexType.BTREE,
                Optional.of("btree_id_index"),
                indexParams,
                true);

        // Verify the returned Index object
        assertEquals("btree_id_index", index.name());
        assertNotNull(index.uuid());
        assertFalse(index.fields().isEmpty());

        // Verify index was created and is in the list
        assertTrue(
            dataset.listIndexes().contains("btree_id_index"),
            "Expected 'btree_id_index' to be in the list of indexes: " + dataset.listIndexes());

        // TODO: Verify zone_size parameter was applied
        // Currently the Java API doesn't expose index configuration details,
        // but we could add a getIndexDetails() method in the future to verify
        // that the zone_size parameter was correctly set to 2048
      }
    }
  }

  @Test
  public void testCreateBTreeIndexDistributively(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("build_index_distributedly").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      // 1. write two fragments
      testDataset.write(1, 10).close();
      try (Dataset dataset = testDataset.write(2, 10)) {
        List<Fragment> fragments = dataset.getFragments();
        assertEquals(2, fragments.size());

        ScalarIndexParams scalarParams = ScalarIndexParams.create("btree", "{\"zone_size\": 2048}");
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();
        String indexName = "test_index";

        List<Index> segments = new ArrayList<>();
        for (Fragment fragment : fragments) {
          segments.add(
              dataset.createIndex(
                  IndexOptions.builder(
                          Collections.singletonList("name"), IndexType.BTREE, indexParams)
                      .withIndexName(indexName)
                      .withFragmentIds(Collections.singletonList(fragment.getId()))
                      .build()));
        }

        assertFalse(
            dataset.listIndexes().contains(indexName),
            "Partially created index should not present");

        List<Index> committed = dataset.commitExistingIndexSegments(indexName, "name", segments);
        assertEquals(2, committed.size());
        assertTrue(dataset.listIndexes().contains(indexName));

        assertEquals(2, dataset.countIndexedRows(indexName, "name = 'Person 5'", Optional.empty()));
        assertEquals(
            10,
            dataset.countIndexedRows(
                indexName, "name >= 'Person 3' AND name < 'Person 8'", Optional.empty()));
      }
    }
  }

  @Test
  public void testRangedBTreeIndex(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("ranged_btree_map").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      testDataset.write(1, 100).close();
      try (Dataset dataset = testDataset.write(2, 100)) {
        List<Fragment> fragments = dataset.getFragments();
        assertEquals(2, fragments.size());

        List<Index> segments = new ArrayList<>();
        for (Fragment fragment : fragments) {
          List<long[]> data = new ArrayList<>();
          try (LanceScanner scanner =
                  dataset.newScan(
                      new ScanOptions.Builder()
                          .fragmentIds(Collections.singletonList(fragment.getId()))
                          .withRowId(true)
                          .columns(Collections.singletonList("id"))
                          .build());
              ArrowReader arrowReader = scanner.scanBatches(); ) {
            while (arrowReader.loadNextBatch()) {
              VectorSchemaRoot root = arrowReader.getVectorSchemaRoot();
              UInt8Vector rowIdVec = (UInt8Vector) root.getVector("_rowid");
              IntVector idVec = (IntVector) root.getVector("id");
              for (int i = 0; i < root.getRowCount(); i++) {
                data.add(new long[] {idVec.get(i), rowIdVec.get(i)});
              }
            }
          }

          data.sort((d1, d2) -> Long.compare(d1[0], d2[0]));
          segments.add(createBtreeIndexFromPreprocessedData(dataset, data, fragment, allocator));
        }

        String indexName = "test_index";
        List<Index> committed = dataset.commitExistingIndexSegments(indexName, "id", segments);
        assertEquals(2, committed.size());
        assertTrue(dataset.listIndexes().contains(indexName));

        assertEquals(
            6, dataset.countIndexedRows(indexName, "id in (10, 20, 30)", Optional.empty()));
        assertEquals(
            20, dataset.countIndexedRows(indexName, "id >= 50 AND id < 60", Optional.empty()));
      }
    }
  }

  private Index createBtreeIndexFromPreprocessedData(
      Dataset dataset,
      List<long[]> preprocessedData,
      Fragment fragment,
      BufferAllocator allocator) {
    Schema schema =
        new Schema(
            Arrays.asList(
                Field.nullable("value", new ArrowType.Int(32, true)),
                Field.nullable("_rowid", new ArrowType.Int(64, false))),
            null);
    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      root.allocateNew();
      IntVector idVec = (IntVector) root.getVector("value");
      UInt8Vector rowIdVec = (UInt8Vector) root.getVector("_rowid");
      for (int i = 0; i < preprocessedData.size(); i++) {
        long[] dataPair = preprocessedData.get(i);
        idVec.setSafe(i, (int) dataPair[0]);
        rowIdVec.setSafe(i, dataPair[1]);
      }
      root.setRowCount(preprocessedData.size());

      ByteArrayOutputStream out = new ByteArrayOutputStream();
      try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, out)) {
        writer.start();
        writer.writeBatch();
        writer.end();
      } catch (IOException e) {
        throw new RuntimeException("Cannot write schema root", e);
      }

      byte[] arrowData = out.toByteArray();
      ByteArrayInputStream in = new ByteArrayInputStream(arrowData);

      try (ArrowStreamReader reader = new ArrowStreamReader(in, allocator);
          ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
        Data.exportArrayStream(allocator, reader, stream);

        ScalarIndexParams scalarParams = ScalarIndexParams.create("btree", "{\"zone_size\": 64}");
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();
        return dataset.createIndex(
            IndexOptions.builder(Collections.singletonList("id"), IndexType.BTREE, indexParams)
                .withIndexName("test_index")
                .withFragmentIds(Collections.singletonList(fragment.getId()))
                .withPreprocessedData(stream)
                .build());
      } catch (Exception e) {
        throw new RuntimeException("Cannot read arrow stream.", e);
      }
    }
  }

  @Test
  public void testBtreeMergeIndexMetadataSoftBreak(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("btree_merge_metadata_soft_break").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      testDataset.write(1, 10).close();
      try (Dataset dataset = testDataset.write(2, 10)) {
        Exception ex =
            Assertions.assertThrows(
                Exception.class,
                () ->
                    dataset.mergeIndexMetadata(
                        UUID.randomUUID().toString(), IndexType.BTREE, Optional.empty()));
        assertTrue(
            ex.getMessage() != null
                && ex.getMessage().contains("no longer supports merge_index_metadata"),
            "expected BTree merge_index_metadata soft-break error, got: " + ex.getMessage());
      }
    }
  }

  @Test
  public void testCreateInvertedIndexReportsProgress(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("inverted_create_progress").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 20)) {
        Fragment fragment = dataset.getFragments().get(0);
        RecordingIndexBuildProgress progress = new RecordingIndexBuildProgress();

        Index segment =
            dataset.createIndex(createInvertedSegmentOptions(fragment.getId()), progress);

        assertNotNull(segment);
        List<String> events = progress.snapshot();
        assertEventsInOrder(
            events,
            "start:load_data:unknown:rows",
            "complete:load_data",
            "start:tokenize_docs:unknown:rows",
            "progress:tokenize_docs:",
            "complete:tokenize_docs",
            "start:copy_partitions:",
            "complete:copy_partitions",
            "start:write_metadata:",
            "progress:write_metadata:",
            "complete:write_metadata");
        assertTrue(
            events.stream()
                .anyMatch(event -> event.matches("start:copy_partitions:[0-9]+:partitions")),
            "Expected a known partition total, got: " + events);
        assertTrue(
            events.stream().anyMatch(event -> event.matches("start:write_metadata:[0-9]+:files")),
            "Expected a known file total, got: " + events);
      }
    }
  }

  @Test
  public void testCreateInvertedIndexPropagatesProgressFailure(@TempDir Path tempDir)
      throws Exception {
    String datasetPath = tempDir.resolve("inverted_create_progress_failure").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 20)) {
        Fragment fragment = dataset.getFragments().get(0);

        RuntimeException failure =
            Assertions.assertThrows(
                RuntimeException.class,
                () ->
                    dataset.createIndex(
                        createInvertedSegmentOptions(fragment.getId()),
                        new FailingProgressIndexBuildProgress()));

        assertFalse(
            failure instanceof IllegalArgumentException,
            "Progress callback failures should not be reported as invalid input: " + failure);
        assertTrue(
            causeChainContains(failure, "stageProgress")
                && causeChainContains(failure, "java.lang.IllegalStateException")
                && causeChainContains(failure, "progress callback failure"),
            "Expected callback context and original Java exception details, got: " + failure);
      }
    }
  }

  @Test
  public void testCreateInvertedIndexIgnoresCompleteFailure(@TempDir Path tempDir)
      throws Exception {
    String datasetPath = tempDir.resolve("inverted_create_complete_failure").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 20)) {
        Fragment fragment = dataset.getFragments().get(0);
        FailingCompleteIndexBuildProgress progress = new FailingCompleteIndexBuildProgress();

        Index segment =
            dataset.createIndex(createInvertedSegmentOptions(fragment.getId()), progress);

        assertNotNull(segment);
        assertTrue(
            progress.recorder.snapshot().contains("complete:write_metadata"),
            "Expected create to continue after stageComplete callback failures");
      }
    }
  }

  @Test
  @Timeout(value = 60, unit = TimeUnit.SECONDS)
  public void testCreateInvertedIndexAllowsReentrantDatasetAccess(@TempDir Path tempDir)
      throws Exception {
    String datasetPath = tempDir.resolve("inverted_create_reentrant_dataset").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 20)) {
        Fragment fragment = dataset.getFragments().get(0);
        ReentrantDatasetIndexBuildProgress progress =
            new ReentrantDatasetIndexBuildProgress(dataset);

        Index segment =
            dataset.createIndex(createInvertedSegmentOptions(fragment.getId()), progress);

        assertNotNull(segment);
        assertTrue(
            progress.reentries.get() > 0,
            "Expected progress callbacks to re-enter Dataset JNI methods");
        assertTrue(
            progress.recorder.snapshot().contains("complete:write_metadata"),
            "Expected create to finish after re-entrant Dataset access, got: "
                + progress.recorder.snapshot());
      }
    }
  }

  @Test
  @Timeout(value = 5, unit = TimeUnit.SECONDS)
  public void testCreateInvertedIndexRejectsWriteReentryPromptly(@TempDir Path tempDir)
      throws Exception {
    String datasetPath = tempDir.resolve("inverted_create_write_reentry").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 20)) {
        Fragment fragment = dataset.getFragments().get(0);
        WriteReentrantIndexBuildProgress progress = new WriteReentrantIndexBuildProgress(dataset);

        Index segment =
            dataset.createIndex(createInvertedSegmentOptions(fragment.getId()), progress);

        assertNotNull(segment);
        RuntimeException failure = progress.writeFailure.get();
        assertNotNull(failure, "Expected write re-entry to be rejected");
        assertTrue(
            failure.getMessage().contains("busy in an index progress callback"),
            "Unexpected write re-entry failure: " + failure.getMessage());
        assertTrue(progress.reentries.get() > 0, "Expected callback to attempt a write lock");
      }
    }
  }

  @Test
  @Timeout(value = 5, unit = TimeUnit.SECONDS)
  public void testCreateInvertedIndexRejectsNestedCreatePromptly(@TempDir Path tempDir)
      throws Exception {
    String datasetPath = tempDir.resolve("inverted_create_nested_reentry").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 20)) {
        Fragment fragment = dataset.getFragments().get(0);
        IndexOptions segmentOptions = createInvertedSegmentOptions(fragment.getId());
        NestedCreateIndexBuildProgress progress =
            new NestedCreateIndexBuildProgress(dataset, segmentOptions);

        Index segment = dataset.createIndex(segmentOptions, progress);

        assertNotNull(segment);
        RuntimeException failure = progress.nestedFailure.get();
        assertNotNull(failure, "Expected nested create to be rejected");
        assertTrue(
            failure.getMessage().contains("busy in an index progress callback"),
            "Unexpected nested create failure: " + failure.getMessage());
      }
    }
  }

  @Test
  @Timeout(value = 5, unit = TimeUnit.SECONDS)
  public void testCreateInvertedIndexAllowsConcurrentCreate(@TempDir Path tempDir)
      throws Exception {
    String datasetPath = tempDir.resolve("inverted_concurrent_create").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 20)) {
        Fragment fragment = dataset.getFragments().get(0);
        IndexOptions segmentOptions = createInvertedSegmentOptions(fragment.getId());
        BlockingIndexBuildProgress progress = new BlockingIndexBuildProgress();
        AtomicReference<Index> concurrentResult = new AtomicReference<>();
        AtomicReference<RuntimeException> outerFailure = new AtomicReference<>();

        Thread outerCreate =
            new Thread(
                () -> {
                  try {
                    dataset.createIndex(segmentOptions, progress);
                  } catch (RuntimeException failure) {
                    outerFailure.set(failure);
                  }
                });
        outerCreate.start();
        assertTrue(progress.started.await(5, TimeUnit.SECONDS), "Progress callback never started");

        Thread concurrentCreate =
            new Thread(() -> concurrentResult.set(dataset.createIndex(segmentOptions)));
        concurrentCreate.start();
        concurrentCreate.join(100);
        assertTrue(concurrentCreate.isAlive(), "Concurrent create should wait for active create");

        progress.release.countDown();
        outerCreate.join(5000);
        concurrentCreate.join(5000);
        assertFalse(outerCreate.isAlive(), "Outer create timed out");
        assertFalse(concurrentCreate.isAlive(), "Concurrent create timed out");
        assertNull(outerFailure.get(), "Outer create failed");
        assertNotNull(concurrentResult.get(), "Concurrent create did not return an index");
      }
    }
  }

  @Test
  @Timeout(value = 5, unit = TimeUnit.SECONDS)
  public void testCreateInvertedIndexAllowsConcurrentClose(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("inverted_concurrent_close").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 20)) {
        Fragment fragment = dataset.getFragments().get(0);
        BlockingIndexBuildProgress progress = new BlockingIndexBuildProgress();
        AtomicReference<RuntimeException> closeFailure = new AtomicReference<>();
        AtomicReference<RuntimeException> outerFailure = new AtomicReference<>();

        Thread outerCreate =
            new Thread(
                () -> {
                  try {
                    dataset.createIndex(createInvertedSegmentOptions(fragment.getId()), progress);
                  } catch (RuntimeException failure) {
                    outerFailure.set(failure);
                  }
                });
        outerCreate.start();
        assertTrue(progress.started.await(5, TimeUnit.SECONDS), "Progress callback never started");

        Thread concurrentClose =
            new Thread(
                () -> {
                  try {
                    dataset.close();
                  } catch (RuntimeException failure) {
                    closeFailure.set(failure);
                  }
                });
        concurrentClose.start();
        concurrentClose.join(100);
        assertTrue(concurrentClose.isAlive(), "Concurrent close should wait for active create");

        progress.release.countDown();
        outerCreate.join(5000);
        concurrentClose.join(5000);
        assertFalse(outerCreate.isAlive(), "Outer create timed out");
        assertFalse(concurrentClose.isAlive(), "Concurrent close timed out");
        assertNull(outerFailure.get(), "Outer create failed");
        assertNull(closeFailure.get(), "Concurrent close should not fail");
      }
    }
  }

  @Test
  public void testCreateInvertedIndexWithProgressUpdatesDatasetState(@TempDir Path tempDir)
      throws Exception {
    String datasetPath = tempDir.resolve("inverted_create_committed_progress").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 20)) {
        long previousVersion = dataset.version();
        RecordingIndexBuildProgress progress = new RecordingIndexBuildProgress();
        IndexOptions options =
            IndexOptions.builder(
                    Collections.singletonList("name"),
                    IndexType.INVERTED,
                    createInvertedIndexParams())
                .withIndexName("committed_inverted_progress_idx")
                .replace(true)
                .build();

        Index index = dataset.createIndex(options, progress);

        assertEquals(previousVersion + 1, dataset.version());
        assertEquals("committed_inverted_progress_idx", index.name());
        assertTrue(dataset.listIndexes().contains("committed_inverted_progress_idx"));
        assertTrue(
            progress.snapshot().stream().anyMatch(event -> event.startsWith("progress:")),
            "Expected committed create to report progress");
      }
    }
  }

  @Test
  public void testCreateIndexRejectsNullProgress(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("create_null_progress").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 1)) {
        Fragment fragment = dataset.getFragments().get(0);

        NullPointerException failure =
            Assertions.assertThrows(
                NullPointerException.class,
                () -> dataset.createIndex(createInvertedSegmentOptions(fragment.getId()), null));

        assertTrue(failure.getMessage().contains("progress cannot be null"));
      }
    }
  }

  @Test
  public void testMergeInvertedIndexMetadataReportsProgress(@TempDir Path tempDir)
      throws Exception {
    String datasetPath = tempDir.resolve("inverted_merge_progress").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      testDataset.write(1, 10).close();
      try (Dataset dataset = testDataset.write(2, 10)) {
        String indexUuid = createDistributedInvertedIndex(dataset);
        RecordingIndexBuildProgress progress = new RecordingIndexBuildProgress();

        dataset.mergeIndexMetadata(indexUuid, IndexType.INVERTED, Optional.empty(), progress);

        List<String> events = progress.snapshot();
        assertEventsInOrder(
            events,
            "start:read_partition_metadata:",
            "complete:read_partition_metadata",
            "start:remap_partition_files:",
            "complete:remap_partition_files",
            "start:write_merged_metadata:",
            "complete:write_merged_metadata");
        assertTrue(
            events.contains("progress:read_partition_metadata:2"),
            "Expected metadata progress to reach both fragments, got: " + events);
        assertTrue(
            events.stream().anyMatch(event -> event.startsWith("progress:remap_partition_files:")),
            "Expected remap progress, got: " + events);
        assertTrue(
            events.contains("progress:write_merged_metadata:1"),
            "Expected merged metadata write progress, got: " + events);
      }
    }
  }

  @Test
  public void testMergeInvertedIndexMetadataPropagatesProgressFailure(@TempDir Path tempDir)
      throws Exception {
    String datasetPath = tempDir.resolve("inverted_merge_progress_failure").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      testDataset.write(1, 10).close();
      try (Dataset dataset = testDataset.write(2, 10)) {
        String indexUuid = createDistributedInvertedIndex(dataset);

        RuntimeException failure =
            Assertions.assertThrows(
                RuntimeException.class,
                () ->
                    dataset.mergeIndexMetadata(
                        indexUuid,
                        IndexType.INVERTED,
                        Optional.empty(),
                        new FailingProgressIndexBuildProgress()));

        assertFalse(
            failure instanceof IllegalArgumentException,
            "Progress callback failures should not be reported as invalid input: " + failure);
        assertTrue(
            causeChainContains(failure, "stageProgress")
                && causeChainContains(failure, "read_partition_metadata")
                && causeChainContains(failure, "java.lang.IllegalStateException")
                && causeChainContains(failure, "progress callback failure"),
            "Expected callback context and original Java exception details, got: " + failure);
      }
    }
  }

  @Test
  public void testMergeInvertedIndexMetadataIgnoresCompleteFailure(@TempDir Path tempDir)
      throws Exception {
    String datasetPath = tempDir.resolve("inverted_merge_complete_failure").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      testDataset.write(1, 10).close();
      try (Dataset dataset = testDataset.write(2, 10)) {
        String indexUuid = createDistributedInvertedIndex(dataset);
        FailingCompleteIndexBuildProgress progress = new FailingCompleteIndexBuildProgress();

        dataset.mergeIndexMetadata(indexUuid, IndexType.INVERTED, Optional.empty(), progress);

        assertTrue(
            progress.recorder.snapshot().contains("complete:write_merged_metadata"),
            "Expected merge to continue after stageComplete callback failures");
      }
    }
  }

  @Test
  @Timeout(value = 60, unit = TimeUnit.SECONDS)
  public void testMergeInvertedIndexMetadataAllowsReentrantDatasetAccess(@TempDir Path tempDir)
      throws Exception {
    String datasetPath = tempDir.resolve("inverted_merge_reentrant_dataset").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      testDataset.write(1, 10).close();
      try (Dataset dataset = testDataset.write(2, 10)) {
        String indexUuid = createDistributedInvertedIndex(dataset);
        ReentrantDatasetIndexBuildProgress progress =
            new ReentrantDatasetIndexBuildProgress(dataset);

        dataset.mergeIndexMetadata(indexUuid, IndexType.INVERTED, Optional.empty(), progress);

        assertTrue(
            progress.reentries.get() > 0,
            "Expected progress callbacks to re-enter Dataset JNI methods");
        assertTrue(
            progress.recorder.snapshot().contains("complete:write_merged_metadata"),
            "Expected merge to finish after re-entrant Dataset access, got: "
                + progress.recorder.snapshot());
      }
    }
  }

  private static String createDistributedInvertedIndex(Dataset dataset) {
    String indexUuid = UUID.randomUUID().toString();
    for (Fragment fragment : dataset.getFragments()) {
      dataset.createIndex(
          IndexOptions.builder(
                  Collections.singletonList("name"),
                  IndexType.INVERTED,
                  createInvertedIndexParams())
              .replace(true)
              .withIndexName("inverted_progress_idx")
              .withIndexUUID(indexUuid)
              .withFragmentIds(Collections.singletonList(fragment.getId()))
              .build());
    }
    return indexUuid;
  }

  private static IndexOptions createInvertedSegmentOptions(int fragmentId) {
    return IndexOptions.builder(
            Collections.singletonList("name"), IndexType.INVERTED, createInvertedIndexParams())
        .withFragmentIds(Collections.singletonList(fragmentId))
        .build();
  }

  private static IndexParams createInvertedIndexParams() {
    ScalarIndexParams scalarParams =
        ScalarIndexParams.create(
            "inverted",
            "{\"base_tokenizer\":\"simple\",\"language\":\"English\","
                + "\"max_token_length\":40,\"lower_case\":true,\"stem\":false,"
                + "\"remove_stop_words\":false}");
    return IndexParams.builder().setScalarIndexParams(scalarParams).build();
  }

  private static void assertEventsInOrder(List<String> events, String... prefixes) {
    int previous = -1;
    for (String prefix : prefixes) {
      int current = -1;
      for (int i = previous + 1; i < events.size(); i++) {
        if (events.get(i).startsWith(prefix)) {
          current = i;
          break;
        }
      }
      assertTrue(
          current >= 0,
          "Missing event '" + prefix + "' after position " + previous + ": " + events);
      previous = current;
    }
  }

  private static boolean causeChainContains(Throwable failure, String expected) {
    for (Throwable current = failure; current != null; current = current.getCause()) {
      if (current.getMessage() != null && current.getMessage().contains(expected)) {
        return true;
      }
    }
    return false;
  }

  @Test
  public void testCreateZonemapIndex(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("zonemap_test").toString();
    Schema schema =
        new Schema(
            Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true)),
                Field.nullable("value", new ArrowType.Utf8())),
            null);

    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset dataset =
          Dataset.create(allocator, datasetPath, schema, new WriteParams.Builder().build())) {

        // Create Zonemap scalar index parameters with rows_per_zone setting
        ScalarIndexParams scalarParams =
            ScalarIndexParams.create("zonemap", "{\"rows_per_zone\": 1024}");

        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();

        // Create Zonemap index on 'value' column
        Index index =
            dataset.createIndex(
                Collections.singletonList("value"),
                IndexType.ZONEMAP,
                Optional.of("zonemap_value_index"),
                indexParams,
                true);

        // Verify the returned Index object
        assertEquals("zonemap_value_index", index.name());
        assertNotNull(index.uuid());

        // Verify index was created
        assertTrue(
            dataset.listIndexes().contains("zonemap_value_index"),
            "Expected 'zonemap_value_index' to be in the list of indexes: "
                + dataset.listIndexes());

        // TODO: Verify rows_per_zone parameter was applied
        // Currently the Java API doesn't expose index configuration details,
        // but we could add a getIndexDetails() method in the future to verify
        // that the rows_per_zone parameter was correctly set to 1024
      }
    }
  }

  @Test
  public void testCreateRTreeIndex(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("rtree_test").toString();
    ArrowType f64 = new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE);
    Field geometryField =
        new Field(
            "geometry",
            new FieldType(
                true,
                new ArrowType.Struct(),
                null,
                Collections.singletonMap("ARROW:extension:name", "geoarrow.point")),
            Arrays.asList(Field.notNullable("x", f64), Field.notNullable("y", f64)));
    Schema schema = new Schema(Collections.singletonList(geometryField), null);

    int rowCount = 3;
    try (RootAllocator allocator = new RootAllocator();
        VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      root.allocateNew();
      StructVector geometry = (StructVector) root.getVector("geometry");
      Float8Vector x = (Float8Vector) geometry.getChild("x");
      Float8Vector y = (Float8Vector) geometry.getChild("y");
      for (int i = 0; i < rowCount; i++) {
        geometry.setIndexDefined(i);
        x.setSafe(i, (double) i);
        y.setSafe(i, i * 2.0);
      }
      geometry.setValueCount(rowCount);
      root.setRowCount(rowCount);

      ByteArrayOutputStream out = new ByteArrayOutputStream();
      try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, out)) {
        writer.start();
        writer.writeBatch();
        writer.end();
      }

      try (ArrowStreamReader reader =
              new ArrowStreamReader(new ByteArrayInputStream(out.toByteArray()), allocator);
          Dataset dataset =
              Dataset.write()
                  .reader(reader)
                  .uri(datasetPath)
                  .allocator(allocator)
                  .mode(WriteParams.WriteMode.CREATE)
                  .execute()) {
        // The point data round-trips through Lance.
        assertEquals(rowCount, dataset.countRows());
        try (ArrowReader scan = dataset.newScan(new ScanOptions.Builder().build()).scanBatches()) {
          assertTrue(scan.loadNextBatch());
          StructVector readGeometry =
              (StructVector) scan.getVectorSchemaRoot().getVector("geometry");
          assertEquals(2.0, ((Float8Vector) readGeometry.getChild("x")).get(2));
          assertEquals(4.0, ((Float8Vector) readGeometry.getChild("y")).get(2));
        }

        // Creating and listing an RTree index via the typed IndexType works end-to-end.
        Index index =
            dataset.createIndex(
                Collections.singletonList("geometry"),
                IndexType.RTREE,
                Optional.of("rtree_geometry_index"),
                IndexParams.builder()
                    .setScalarIndexParams(ScalarIndexParams.create("rtree"))
                    .build(),
                true);
        assertEquals(IndexType.RTREE, index.indexType());
        assertTrue(
            dataset.listIndexes().contains("rtree_geometry_index"),
            "Expected 'rtree_geometry_index' in: " + dataset.listIndexes());
      }
    }
  }
}
