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

import org.lance.memwal.CompactedSsTable;
import org.lance.memwal.InitializeMemWalParams;
import org.lance.merge.MergeInsertParams;
import org.lance.merge.MergeInsertResult;
import org.lance.merge.UncommittedMergeInsertResult;
import org.lance.operation.Update;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.UUID;

public class MergeInsertTest {
  @TempDir private Path tempDir;
  private RootAllocator allocator;
  private TestUtils.SimpleTestDataset testDataset;
  private Dataset dataset;

  @BeforeEach
  public void setup() {
    String datasetPath = tempDir.resolve(UUID.randomUUID().toString()).toString();
    allocator = new RootAllocator(Long.MAX_VALUE);
    testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
    testDataset.createEmptyDataset().close();
    dataset = testDataset.write(1, 5);
  }

  @AfterEach
  public void tearDown() {
    dataset.close();
    allocator.close();
  }

  @Test
  public void testWhenNotMatchedInsertAll() throws Exception {
    // Test insert all unmatched source rows

    try (VectorSchemaRoot source = buildSource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        MergeInsertResult result =
            dataset.mergeInsert(
                new MergeInsertParams(Collections.singletonList("id")), sourceStream);

        Assertions.assertEquals(
            "{0=Person 0, 1=Person 1, 2=Person 2, 3=Person 3, 4=Person 4, 7=Source 7, 8=Source 8, 9=Source 9}",
            readAll(result.dataset()).toString());
      }
    }
  }

  @Test
  public void testWhenNotMatchedDoNothing() throws Exception {
    // Test ignore unmatched source rows

    try (VectorSchemaRoot source = buildSource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        MergeInsertResult result =
            dataset.mergeInsert(
                new MergeInsertParams(Collections.singletonList("id"))
                    .withMatchedUpdateAll()
                    .withNotMatched(MergeInsertParams.WhenNotMatched.DoNothing),
                sourceStream);

        Assertions.assertEquals(
            "{0=Source 0, 1=Source 1, 2=Source 2, 3=Person 3, 4=Person 4}",
            readAll(result.dataset()).toString());
      }
    }
  }

  @Test
  public void testWhenMatchedUpdateIf() throws Exception {
    // Test update matched rows if expression is true

    try (VectorSchemaRoot source = buildSource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        MergeInsertResult result =
            dataset.mergeInsert(
                new MergeInsertParams(Collections.singletonList("id"))
                    .withMatchedUpdateIf("target.name = 'Person 0' or target.name = 'Person 1'")
                    .withNotMatched(MergeInsertParams.WhenNotMatched.DoNothing),
                sourceStream);

        Assertions.assertEquals(
            "{0=Source 0, 1=Source 1, 2=Person 2, 3=Person 3, 4=Person 4}",
            readAll(result.dataset()).toString());
      }
    }
  }

  @Test
  public void testWhenNotMatchedBySourceDelete() throws Exception {
    // Test delete target rows which are not matched with source.

    try (VectorSchemaRoot source = buildSource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        MergeInsertResult result =
            dataset.mergeInsert(
                new MergeInsertParams(Collections.singletonList("id"))
                    .withNotMatchedBySourceDelete()
                    .withNotMatched(MergeInsertParams.WhenNotMatched.DoNothing),
                sourceStream);

        Assertions.assertEquals(
            "{0=Person 0, 1=Person 1, 2=Person 2}", readAll(result.dataset()).toString());
      }
    }
  }

  @Test
  public void testWhenNotMatchedBySourceDeleteIf() throws Exception {
    // Test delete target rows which are not matched with source if expression is true

    try (VectorSchemaRoot source = buildSource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        MergeInsertResult result =
            dataset.mergeInsert(
                new MergeInsertParams(Collections.singletonList("id"))
                    .withNotMatchedBySourceDeleteIf("name = 'Person 3'")
                    .withNotMatched(MergeInsertParams.WhenNotMatched.DoNothing),
                sourceStream);

        Assertions.assertEquals(
            "{0=Person 0, 1=Person 1, 2=Person 2, 4=Person 4}",
            readAll(result.dataset()).toString());
      }
    }
  }

  @Test
  public void testWhenMatchedFailWithMatches() throws Exception {
    // Test fail when there are matched rows

    try (VectorSchemaRoot source = buildSource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        String originalDataset = readAll(dataset).toString();

        Assertions.assertThrows(
            Exception.class,
            () ->
                dataset.mergeInsert(
                    new MergeInsertParams(Collections.singletonList("id")).withMatchedFail(),
                    sourceStream));

        // Verify dataset remains unchanged
        Assertions.assertEquals(
            originalDataset,
            readAll(dataset).toString(),
            "Dataset should remain unchanged after failed mergeInsert");
      }
    }
  }

  @Test
  public void testWhenMatchedFailWithoutMatches() throws Exception {
    // Test success when there are no matched rows

    try (VectorSchemaRoot root = VectorSchemaRoot.create(testDataset.getSchema(), allocator)) {
      root.allocateNew();

      IntVector idVector = (IntVector) root.getVector("id");
      VarCharVector nameVector = (VarCharVector) root.getVector("name");

      List<Integer> sourceIds = Arrays.asList(100, 101, 102);
      for (int i = 0; i < sourceIds.size(); i++) {
        idVector.setSafe(i, sourceIds.get(i));
        String name = "New Data " + sourceIds.get(i);
        nameVector.setSafe(i, name.getBytes(StandardCharsets.UTF_8));
      }

      root.setRowCount(sourceIds.size());

      try (ArrowArrayStream sourceStream = convertToStream(root, allocator)) {
        MergeInsertResult result =
            dataset.mergeInsert(
                new MergeInsertParams(Collections.singletonList("id")).withMatchedFail(),
                sourceStream);

        // Verify new data is inserted
        Map<Integer, String> resultMap = readAll(result.dataset());
        for (int id : sourceIds) {
          Assertions.assertTrue(resultMap.containsKey(id));
          Assertions.assertEquals("New Data " + id, resultMap.get(id));
        }
      }
    }
  }

  @Test
  public void testWhenMatchedDelete() throws Exception {
    // Test delete matched target rows if expression is true

    try (VectorSchemaRoot source = buildSource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        MergeInsertResult result =
            dataset.mergeInsert(
                new MergeInsertParams(Collections.singletonList("id"))
                    .withMatchedDelete()
                    .withNotMatched(MergeInsertParams.WhenNotMatched.DoNothing),
                sourceStream);

        Assertions.assertEquals("{3=Person 3, 4=Person 4}", readAll(result.dataset()).toString());
      }
    }
  }

  private VectorSchemaRoot buildSource(Schema schema, RootAllocator allocator) {
    List<Integer> sourceIds = Arrays.asList(0, 1, 2, 7, 8, 9);

    VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
    root.allocateNew();

    IntVector idVector = (IntVector) root.getVector("id");
    VarCharVector nameVector = (VarCharVector) root.getVector("name");

    for (int i = 0; i < sourceIds.size(); i++) {
      idVector.setSafe(i, sourceIds.get(i));
      String name = "Source " + sourceIds.get(i);
      nameVector.setSafe(i, name.getBytes(StandardCharsets.UTF_8));
    }

    root.setRowCount(sourceIds.size());

    return root;
  }

  private ArrowArrayStream convertToStream(VectorSchemaRoot root, RootAllocator allocator)
      throws Exception {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, out)) {
      writer.start();
      writer.writeBatch();
      writer.end();
    }

    ByteArrayInputStream in = new ByteArrayInputStream(out.toByteArray());
    ArrowStreamReader reader = new ArrowStreamReader(in, allocator);

    ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator);
    Data.exportArrayStream(allocator, reader, stream);

    return stream;
  }

  @Test
  public void testMergeInsertWithoutIndex() throws Exception {
    // Verify that merge insert with useIndex=false still completes and
    // produces results consistent with the default (useIndex=true).

    try (VectorSchemaRoot source = buildSource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        MergeInsertResult result =
            dataset.mergeInsert(
                new MergeInsertParams(Collections.singletonList("id"))
                    .withMatchedUpdateAll()
                    .withNotMatched(MergeInsertParams.WhenNotMatched.InsertAll)
                    .withUseIndex(false),
                sourceStream);

        Assertions.assertEquals(
            "{0=Source 0, 1=Source 1, 2=Source 2, 3=Person 3, 4=Person 4, 7=Source 7, 8=Source 8, 9=Source 9}",
            readAll(result.dataset()).toString(),
            "merge insert with useIndex=false should produce correct upsert results");
      }
    }
  }

  @Test
  public void testMergeInsertUncommitted_InsertAllAndCommit() throws Exception {
    try (VectorSchemaRoot source = buildSource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        long originalVersion = dataset.version();
        UncommittedMergeInsertResult result =
            dataset.mergeInsertUncommitted(
                new MergeInsertParams(Collections.singletonList("id")), sourceStream);

        // Verify result getters
        Assertions.assertNotNull(result.dataset());
        Assertions.assertNotNull(result.getDataset());
        Assertions.assertNotNull(result.transaction());
        Assertions.assertNotNull(result.getTransaction());
        Assertions.assertNotNull(result.stats());
        Assertions.assertNotNull(result.getStats());

        // Verify stats
        Assertions.assertEquals(3, result.stats().numInsertedRows());
        Assertions.assertEquals(0, result.stats().numUpdatedRows());
        Assertions.assertEquals(0, result.stats().numDeletedRows());
        Assertions.assertEquals(1, result.stats().numAttempts());
        Assertions.assertTrue(result.stats().bytesWritten() > 0);
        Assertions.assertTrue(result.stats().numFilesWritten() > 0);

        // Original dataset should not be modified
        Assertions.assertEquals(originalVersion, dataset.version());
        Assertions.assertEquals(
            "{0=Person 0, 1=Person 1, 2=Person 2, 3=Person 3, 4=Person 4}",
            readAll(dataset).toString());

        // Transaction can be inspected and committed via CommitBuilder
        try (Transaction txn = result.transaction()) {
          Assertions.assertEquals(originalVersion, txn.readVersion());
          Assertions.assertNotNull(txn.uuid());
          Assertions.assertNotNull(txn.operation());

          try (Dataset committed = new CommitBuilder(result.dataset()).execute(txn)) {
            Assertions.assertEquals(originalVersion + 1, committed.version());
            Assertions.assertEquals(
                "{0=Person 0, 1=Person 1, 2=Person 2, 3=Person 3, 4=Person 4, 7=Source 7,"
                    + " 8=Source 8, 9=Source 9}",
                readAll(committed).toString());
          }
        }
      }
    }
  }

  @Test
  public void testMergeInsertUncommitted_MatchedUpdateAndCommit() throws Exception {
    try (VectorSchemaRoot source = buildSource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        long originalVersion = dataset.version();
        UncommittedMergeInsertResult result =
            dataset.mergeInsertUncommitted(
                new MergeInsertParams(Collections.singletonList("id"))
                    .withMatchedUpdateAll()
                    .withNotMatched(MergeInsertParams.WhenNotMatched.DoNothing),
                sourceStream);

        Assertions.assertEquals(0, result.stats().numInsertedRows());
        Assertions.assertEquals(3, result.stats().numUpdatedRows());
        Assertions.assertEquals(0, result.stats().numDeletedRows());

        // Commit via dataset.commitTransaction
        try (Transaction txn = result.transaction()) {
          try (Dataset committed = dataset.commitTransaction(txn)) {
            Assertions.assertEquals(originalVersion + 1, committed.version());
            Assertions.assertEquals(
                "{0=Source 0, 1=Source 1, 2=Source 2, 3=Person 3, 4=Person 4}",
                readAll(committed).toString());
          }
        }
      }
    }
  }

  @Test
  public void testMergeInsertUncommitted_AutoCloseable() throws Exception {
    try (VectorSchemaRoot source = buildSource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        try (UncommittedMergeInsertResult result =
            dataset.mergeInsertUncommitted(
                new MergeInsertParams(Collections.singletonList("id")), sourceStream)) {
          Assertions.assertNotNull(result.transaction());
          Assertions.assertEquals(3, result.stats().numInsertedRows());
        }
      }
    }
  }

  @Test
  public void testMergeInsertUncommitted_PrimaryKeyConflict() throws Exception {
    Map<String, String> pkMeta =
        Collections.singletonMap("lance-schema:unenforced-primary-key", "true");
    Schema pkSchema =
        new Schema(
            Arrays.asList(
                new Field(
                    "id", new FieldType(false, new ArrowType.Int(32, true), null, pkMeta), null),
                Field.nullable("name", new ArrowType.Utf8())));

    String pkDatasetPath = tempDir.resolve("pk_conflict_" + UUID.randomUUID()).toString();
    try (VectorSchemaRoot baseRoot =
            buildSourceWithIds(pkSchema, allocator, Arrays.asList(1, 2, 3));
        ArrowArrayStream baseStream = convertToStream(baseRoot, allocator);
        Dataset pkDataset =
            Dataset.write().allocator(allocator).stream(baseStream).uri(pkDatasetPath).execute()) {

      // Prepare two concurrent uncommitted merge_insert operations inserting id=100
      try (VectorSchemaRoot root1 =
              buildSourceWithIds(pkSchema, allocator, Collections.singletonList(100));
          ArrowArrayStream stream1 = convertToStream(root1, allocator);
          VectorSchemaRoot root2 =
              buildSourceWithIds(pkSchema, allocator, Collections.singletonList(100));
          ArrowArrayStream stream2 = convertToStream(root2, allocator)) {

        MergeInsertParams params =
            new MergeInsertParams(Collections.singletonList("id"))
                .withMatchedUpdateAll()
                .withNotMatched(MergeInsertParams.WhenNotMatched.InsertAll);

        try (UncommittedMergeInsertResult uncommitted1 =
                pkDataset.mergeInsertUncommitted(params, stream1);
            UncommittedMergeInsertResult uncommitted2 =
                pkDataset.mergeInsertUncommitted(params, stream2)) {

          // Verify insertedRowsFilter is populated on the Update operation
          Assertions.assertInstanceOf(Update.class, uncommitted1.transaction().operation());
          Update update1 = (Update) uncommitted1.transaction().operation();
          Assertions.assertNotNull(update1.insertedRowsFilter());
          Assertions.assertTrue(update1.insertedRowsFilter().length > 0);

          // Commit transaction 1 successfully
          try (Dataset committed1 = new CommitBuilder(pkDataset).execute(uncommitted1)) {
            Assertions.assertEquals(4, committed1.countRows());
            Assertions.assertEquals(1, committed1.countRows("id = 100"));

            // Attempting to commit transaction 2 should either fail due to PK conflict
            // or resolve without duplicating id=100 in the dataset
            try (Dataset committed2 = new CommitBuilder(committed1).execute(uncommitted2)) {
              Assertions.assertEquals(1, committed2.countRows("id = 100"));
            } catch (Exception e) {
              Assertions.assertTrue(
                  e.getMessage().toLowerCase().contains("conflict")
                      || e.getMessage().toLowerCase().contains("transaction")
                      || e.getMessage().toLowerCase().contains("primary key"));
            }
          }
        }
      }
    }
  }

  @Test
  public void testMergeInsertUncommitted_MemWalCompactedSstablesPreserved() throws Exception {
    String shardId = UUID.randomUUID().toString();
    Map<String, String> pkMeta =
        Collections.singletonMap("lance-schema:unenforced-primary-key", "true");
    Schema pkSchema =
        new Schema(
            Arrays.asList(
                new Field(
                    "id", new FieldType(false, new ArrowType.Int(32, true), null, pkMeta), null),
                Field.nullable("name", new ArrowType.Utf8())));

    String memwalPath = tempDir.resolve("memwal_" + UUID.randomUUID()).toString();
    try (VectorSchemaRoot baseRoot =
            buildSourceWithIds(pkSchema, allocator, Arrays.asList(1, 2, 3));
        ArrowArrayStream baseStream = convertToStream(baseRoot, allocator);
        Dataset baseDataset =
            Dataset.write().allocator(allocator).stream(baseStream).uri(memwalPath).execute()) {

      baseDataset.initializeMemWal(new InitializeMemWalParams());

      MergeInsertParams params =
          new MergeInsertParams(Collections.singletonList("id"))
              .withMatchedUpdateAll()
              .withNotMatched(MergeInsertParams.WhenNotMatched.InsertAll)
              .markSstablesAsCompacted(Collections.singletonList(new CompactedSsTable(shardId, 1)));

      try (VectorSchemaRoot root = buildSourceWithIds(pkSchema, allocator, Arrays.asList(2, 4));
          ArrowArrayStream stream = convertToStream(root, allocator);
          UncommittedMergeInsertResult uncommitted =
              baseDataset.mergeInsertUncommitted(params, stream)) {

        // Verify Update operation preserved compactedSstables and insertedRowsFilter
        Assertions.assertInstanceOf(Update.class, uncommitted.transaction().operation());
        Update update = (Update) uncommitted.transaction().operation();
        Assertions.assertEquals(1, update.compactedSstables().size());
        Assertions.assertEquals(shardId, update.compactedSstables().get(0).shardId());
        Assertions.assertEquals(1L, update.compactedSstables().get(0).generation());
        Assertions.assertNotNull(update.insertedRowsFilter());

        try (Dataset committed = new CommitBuilder(baseDataset).execute(uncommitted)) {
          Assertions.assertEquals(4, committed.countRows());
        }
      }
    }
  }

  private VectorSchemaRoot buildSourceWithIds(
      Schema schema, RootAllocator allocator, List<Integer> sourceIds) {
    VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
    root.allocateNew();

    IntVector idVector = (IntVector) root.getVector("id");
    VarCharVector nameVector = (VarCharVector) root.getVector("name");

    for (int i = 0; i < sourceIds.size(); i++) {
      idVector.setSafe(i, sourceIds.get(i));
      String name = "Source " + sourceIds.get(i);
      nameVector.setSafe(i, name.getBytes(StandardCharsets.UTF_8));
    }

    root.setRowCount(sourceIds.size());
    return root;
  }

  private TreeMap<Integer, String> readAll(Dataset dataset) throws Exception {
    try (ArrowReader reader = dataset.newScan().scanBatches()) {
      TreeMap<Integer, String> map = new TreeMap<>();

      while (reader.loadNextBatch()) {
        VectorSchemaRoot batch = reader.getVectorSchemaRoot();
        for (int i = 0; i < batch.getRowCount(); i++) {
          IntVector idVector = (IntVector) batch.getVector("id");
          VarCharVector nameVector = (VarCharVector) batch.getVector("name");
          map.put(idVector.get(i), new String(nameVector.get(i)));
        }
      }

      return map;
    }
  }
}
