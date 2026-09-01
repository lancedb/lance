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
package org.lance;

import org.lance.index.Index;
import org.lance.index.IndexFile;
import org.lance.index.IndexOptions;
import org.lance.index.IndexParams;
import org.lance.index.IndexType;
import org.lance.index.scalar.ScalarIndexParams;
import org.lance.memwal.CompactedSsTable;
import org.lance.memwal.InitializeMemWalParams;
import org.lance.operation.Append;
import org.lance.operation.Clone;
import org.lance.operation.CreateIndex;
import org.lance.operation.DataOverlay;
import org.lance.operation.KeyExistenceFilter;
import org.lance.operation.Overwrite;
import org.lance.operation.Rewrite;
import org.lance.operation.RewrittenIndex;
import org.lance.operation.Update;
import org.lance.operation.UpdateBases;
import org.lance.operation.UpdateMemWalState;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TransactionTest {

  private static final byte[] OFFSETS_1_3_5 =
      new byte[] {0x3A, 0x30, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0x10, 0, 0, 0, 1, 0, 3, 0, 5, 0};

  @Test
  public void testReadTransactionCreateIndex(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("read_transaction_create_index").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      try (Dataset dataset = testDataset.createEmptyDataset()) {
        assertEquals(1, dataset.version());
      }

      try (Dataset dataset = testDataset.write(1, 10)) {
        ScalarIndexParams scalarParams = ScalarIndexParams.create("btree", "{\"zone_size\": 2048}");
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();

        dataset.createIndex(
            IndexOptions.builder(Collections.singletonList("id"), IndexType.BTREE, indexParams)
                .withIndexName("btree_id_index")
                .build());

        assertTrue(
            dataset.listIndexes().contains("btree_id_index"),
            "Expected 'btree_id_index' to be created");

        Transaction readTx = dataset.readTransaction().orElse(null);
        assertNotNull(readTx, "readTransaction() should return a transaction for CreateIndex");
        assertEquals("CreateIndex", readTx.operation().name());

        assertInstanceOf(CreateIndex.class, readTx.operation());
        CreateIndex op = (CreateIndex) readTx.operation();
        assertFalse(op.getNewIndices().isEmpty(), "newIndices should not be empty for CreateIndex");
        assertTrue(
            op.getRemovedIndices().isEmpty(), "removedIndices should be empty for CreateIndex");
        assertEquals("btree_id_index", (op.getNewIndices().get(0).name()));
      }
    }
  }

  @Test
  public void testCommitToUri(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testCommitToUri").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      Schema schema = testDataset.getSchema();

      // Create fragments at the dataset path
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(20);

      // Build a transaction targeting a URI (no existing dataset)
      try (Transaction txn =
          new Transaction.Builder()
              .operation(
                  Overwrite.builder()
                      .fragments(Collections.singletonList(fragmentMeta))
                      .schema(schema)
                      .build())
              .build()) {
        try (Dataset committedDataset = new CommitBuilder(datasetPath, allocator).execute(txn)) {
          assertEquals(1, committedDataset.version());
          assertEquals(20, committedDataset.countRows());
        }
      }
    }
  }

  @Test
  public void testTagRoundTrip(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testTagRoundTrip").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        FragmentMetadata fragmentMeta = testDataset.createNewFragment(10);

        try (Transaction txn =
            new Transaction.Builder()
                .readVersion(dataset.version())
                .tag("v1.0")
                .operation(
                    Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
                .build()) {
          assertEquals("v1.0", txn.tag().orElse(null));

          try (Dataset committed = new CommitBuilder(dataset).execute(txn)) {
            Transaction readTx = committed.readTransaction().orElse(null);
            assertNotNull(readTx);
            assertEquals("v1.0", readTx.tag().orElse(null));
          }
        }
      }
    }
  }

  @Test
  public void testTransactionPropertiesRoundTrip(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testTransactionPropertiesRoundTrip").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        FragmentMetadata fragmentMeta = testDataset.createNewFragment(10);

        Map<String, String> properties = new HashMap<>();
        properties.put("source", "ingestion-pipeline");
        properties.put("batchId", "42");

        try (Transaction txn =
            new Transaction.Builder()
                .readVersion(dataset.version())
                .transactionProperties(properties)
                .operation(
                    Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
                .build()) {
          try (Dataset committed = new CommitBuilder(dataset).execute(txn)) {
            Transaction readTx = committed.readTransaction().orElse(null);
            assertNotNull(readTx);
            Map<String, String> readProps = readTx.transactionProperties().orElse(null);
            assertNotNull(readProps);
            assertEquals("ingestion-pipeline", readProps.get("source"));
            assertEquals("42", readProps.get("batchId"));
          }
        }
      }
    }
  }

  @Test
  public void testCustomUuid(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testCustomUuid").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        FragmentMetadata fragmentMeta = testDataset.createNewFragment(10);

        String customUuid = "custom-uuid-12345";
        try (Transaction txn =
            new Transaction.Builder()
                .readVersion(dataset.version())
                .uuid(customUuid)
                .operation(
                    Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
                .build()) {
          assertEquals(customUuid, txn.uuid());

          try (Dataset committed = new CommitBuilder(dataset).execute(txn)) {
            Transaction readTx = committed.readTransaction().orElse(null);
            assertNotNull(readTx);
            assertEquals(customUuid, readTx.uuid());
          }
        }
      }
    }
  }

  @Test
  public void testCloneRoundTrip(@TempDir Path tempDir) {
    String sourcePath = tempDir.resolve("clone_source").toString();
    String targetPath = tempDir.resolve("clone_target").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset sourceDataset =
          new TestUtils.SimpleTestDataset(allocator, sourcePath);
      sourceDataset.createEmptyDataset().close();
      try (Dataset source = sourceDataset.write(1, 10);
          Transaction transaction =
              new Transaction.Builder()
                  .readVersion(source.version())
                  .operation(
                      Clone.builder()
                          .shallow(false)
                          .refVersion(source.version())
                          .refPath(sourcePath)
                          .build())
                  .build();
          Dataset cloned = new CommitBuilder(targetPath, allocator).execute(transaction);
          Transaction read = cloned.readTransaction().orElseThrow()) {
        assertInstanceOf(Clone.class, read.operation());
        Clone clone = (Clone) read.operation();
        assertFalse(clone.isShallow());
        assertEquals(sourcePath, clone.getRefPath());
        assertEquals(source.version(), clone.getRefVersion());
        assertTrue(clone.getRefName().isEmpty());
        assertTrue(clone.getBranchName().isEmpty());
      }
    }
  }

  @Test
  public void testUpdateBasesRoundTrip(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("update_bases").toString();
    String basePath = tempDir.resolve("external_base").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset();
          Transaction transaction =
              new Transaction.Builder()
                  .readVersion(dataset.version())
                  .operation(
                      UpdateBases.builder()
                          .newBases(
                              Collections.singletonList(
                                  new BasePath(0, Optional.of("external"), basePath, false)))
                          .build())
                  .build();
          Dataset committed = new CommitBuilder(dataset).execute(transaction);
          Transaction read = committed.readTransaction().orElseThrow()) {
        UpdateBases operation = assertInstanceOf(UpdateBases.class, read.operation());
        assertEquals(1, operation.getNewBases().size());
        assertEquals("external", operation.getNewBases().get(0).getName().orElseThrow());
        assertEquals(basePath, operation.getNewBases().get(0).getPath());
      }
    }
  }

  @Test
  public void testUpdateMemWalStateRoundTrip(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("update_memwal").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        dataset.initializeMemWal(new InitializeMemWalParams().withUnsharded());
        CompactedSsTable sstable = new CompactedSsTable(UUID.randomUUID().toString(), 7);
        try (Transaction transaction =
                new Transaction.Builder()
                    .readVersion(dataset.version())
                    .operation(
                        UpdateMemWalState.builder()
                            .compactedSstables(Collections.singletonList(sstable))
                            .build())
                    .build();
            Dataset committed = new CommitBuilder(dataset).execute(transaction);
            Transaction read = committed.readTransaction().orElseThrow()) {
          UpdateMemWalState operation = assertInstanceOf(UpdateMemWalState.class, read.operation());
          assertEquals(1, operation.getCompactedSstables().size());
          assertEquals(sstable.getShardId(), operation.getCompactedSstables().get(0).getShardId());
          assertEquals(7, operation.getCompactedSstables().get(0).getGeneration());
        }
      }
    }
  }

  @Test
  public void testDataOverlayRoundTrip(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("data_overlay").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 10)) {
        FragmentMetadata fragment = dataset.getFragments().get(0).metadata();
        DataOverlay.DataOverlayFile overlay =
            new DataOverlay.DataOverlayFile(
                fragment.getFiles().get(0), DataOverlay.OverlayCoverage.shared(OFFSETS_1_3_5), 0);
        DataOverlay.DataOverlayGroup group =
            new DataOverlay.DataOverlayGroup(fragment.getId(), Collections.singletonList(overlay));
        try (Transaction transaction =
                new Transaction.Builder()
                    .readVersion(dataset.version())
                    .operation(
                        DataOverlay.builder().groups(Collections.singletonList(group)).build())
                    .build();
            Dataset committed = new CommitBuilder(dataset).execute(transaction);
            Transaction read = committed.readTransaction().orElseThrow()) {
          DataOverlay operation = assertInstanceOf(DataOverlay.class, read.operation());
          DataOverlay.DataOverlayFile readOverlay =
              operation.getGroups().get(0).getOverlays().get(0);
          assertEquals(fragment.getId(), operation.getGroups().get(0).getFragmentId());
          assertTrue(readOverlay.getCoverage().isShared());
          assertArrayEquals(OFFSETS_1_3_5, readOverlay.getCoverage().getBitmaps().get(0));
          assertEquals(0, readOverlay.getCommittedVersion());
        }
      }
    }
  }

  @Test
  public void testOverwriteInitialBasesAndUpdateFilterRoundTrip(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("overwrite_initial_bases").toString();
    String basePath = tempDir.resolve("initial_base").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Transaction overwrite =
              new Transaction.Builder()
                  .operation(
                      Overwrite.builder()
                          .fragments(Collections.emptyList())
                          .schema(testDataset.getSchema())
                          .initialBases(
                              Collections.singletonList(
                                  new BasePath(0, Optional.of("initial"), basePath, false)))
                          .build())
                  .build();
          Dataset dataset = new CommitBuilder(datasetPath, allocator).execute(overwrite);
          Transaction readOverwrite = dataset.readTransaction().orElseThrow()) {
        Overwrite operation = assertInstanceOf(Overwrite.class, readOverwrite.operation());
        assertEquals(basePath, operation.initialBases().orElseThrow().get(0).getPath());

        KeyExistenceFilter filter = KeyExistenceFilter.exact(new int[] {0}, new long[] {42});
        try (Transaction update =
                new Transaction.Builder()
                    .readVersion(dataset.version())
                    .operation(Update.builder().insertedRowsFilter(filter).build())
                    .build();
            Dataset committed = new CommitBuilder(dataset).execute(update);
            Transaction readUpdate = committed.readTransaction().orElseThrow()) {
          Update updateOperation = assertInstanceOf(Update.class, readUpdate.operation());
          KeyExistenceFilter readFilter = updateOperation.insertedRowsFilter().orElseThrow();
          assertArrayEquals(new int[] {0}, readFilter.getFieldIds());
          assertArrayEquals(new long[] {42}, readFilter.getExactKeyHashes());
          assertTrue(updateOperation.compactedSstables().isEmpty());
        }
      }
    }
  }

  @Test
  public void testUnknownOperationReturnsDiagnosticError(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("unknown_operation").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset();
          Transaction transaction =
              new Transaction.Builder()
                  .readVersion(dataset.version())
                  .operation(() -> "FutureOperation")
                  .build()) {
        IllegalArgumentException error =
            assertThrows(
                IllegalArgumentException.class,
                () -> new CommitBuilder(dataset).execute(transaction));
        assertTrue(error.getMessage().contains("FutureOperation"));
      }
    }
  }

  @Test
  public void testRewrittenIndexFilesModelPreservesAbsentAndEmpty() {
    UUID oldId = UUID.randomUUID();
    UUID newId = UUID.randomUUID();
    RewrittenIndex absent =
        RewrittenIndex.builder()
            .oldId(oldId)
            .newId(newId)
            .newIndexDetailsTypeUrl("type.googleapis.com/lance.index.BTreeIndexDetails")
            .newIndexDetailsValue(new byte[0])
            .newIndexVersion(1)
            .build();
    RewrittenIndex empty =
        RewrittenIndex.builder()
            .oldId(oldId)
            .newId(newId)
            .newIndexDetailsTypeUrl("type.googleapis.com/lance.index.BTreeIndexDetails")
            .newIndexDetailsValue(new byte[0])
            .newIndexVersion(1)
            .newIndexFiles(Collections.emptyList())
            .build();
    RewrittenIndex populated =
        RewrittenIndex.builder()
            .oldId(oldId)
            .newId(newId)
            .newIndexDetailsTypeUrl("type.googleapis.com/lance.index.BTreeIndexDetails")
            .newIndexDetailsValue(new byte[0])
            .newIndexVersion(1)
            .newIndexFiles(Collections.singletonList(new IndexFile("index.idx", 123)))
            .build();

    assertTrue(absent.getNewIndexFiles().isEmpty());
    assertTrue(empty.getNewIndexFiles().orElseThrow().isEmpty());
    assertEquals(123, populated.getNewIndexFiles().orElseThrow().get(0).getSizeBytes());
  }

  @Test
  public void testRewrittenIndexFilesRoundTrip(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("rewritten_index_files").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 10)) {
        ScalarIndexParams scalarParams = ScalarIndexParams.create("btree", "{}");
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();
        dataset.createIndex(
            IndexOptions.builder(Collections.singletonList("id"), IndexType.BTREE, indexParams)
                .withIndexName("btree_id")
                .build());
        Index oldIndex = dataset.getIndexes().get(0);
        IndexFile indexFile = new IndexFile("index.idx", 123);
        RewrittenIndex rewrittenIndex =
            RewrittenIndex.builder()
                .oldId(oldIndex.uuid())
                .newId(UUID.randomUUID())
                .newIndexDetailsTypeUrl("type.googleapis.com/lance.index.BTreeIndexDetails")
                .newIndexDetailsValue(new byte[0])
                .newIndexVersion(oldIndex.indexVersion())
                .newIndexFiles(Collections.singletonList(indexFile))
                .build();
        try (Transaction transaction =
                new Transaction.Builder()
                    .readVersion(dataset.version())
                    .operation(
                        Rewrite.builder()
                            .rewrittenIndices(Collections.singletonList(rewrittenIndex))
                            .build())
                    .build();
            Dataset committed = new CommitBuilder(dataset).execute(transaction);
            Transaction read = committed.readTransaction().orElseThrow()) {
          Rewrite operation = assertInstanceOf(Rewrite.class, read.operation());
          IndexFile readFile =
              operation.rewrittenIndices().get(0).getNewIndexFiles().orElseThrow().get(0);
          assertEquals("index.idx", readFile.getPath());
          assertEquals(123, readFile.getSizeBytes());
        }
      }
    }
  }
}
