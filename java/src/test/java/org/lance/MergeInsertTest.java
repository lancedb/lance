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

import org.lance.merge.MergeInsertParams;
import org.lance.merge.MergeInsertResult;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.pojo.Field;
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

  /**
   * Build a source whose join key {@code id=0} appears twice ("First 0", then "Second 0"), so the
   * source-dedupe behavior is exercised. Remaining ids (3, 4) are unique matches.
   */
  private VectorSchemaRoot buildDuplicateKeySource(Schema schema, RootAllocator allocator) {
    List<Integer> sourceIds = Arrays.asList(0, 0, 3, 4);
    List<String> sourceNames = Arrays.asList("First 0", "Second 0", "Source 3", "Source 4");

    VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
    root.allocateNew();

    IntVector idVector = (IntVector) root.getVector("id");
    VarCharVector nameVector = (VarCharVector) root.getVector("name");

    for (int i = 0; i < sourceIds.size(); i++) {
      idVector.setSafe(i, sourceIds.get(i));
      nameVector.setSafe(i, sourceNames.get(i).getBytes(StandardCharsets.UTF_8));
    }

    root.setRowCount(sourceIds.size());

    return root;
  }

  /**
   * Build a key-only source (only the join key column) whose key {@code id=0} appears twice. A pure
   * delete over such a source routes through the delete-only plan, exercising source dedupe there.
   */
  private VectorSchemaRoot buildDuplicateKeyOnlySource(Schema schema, RootAllocator allocator) {
    Field idField = schema.findField("id");
    Schema keyOnlySchema = new Schema(Collections.singletonList(idField));

    List<Integer> sourceIds = Arrays.asList(0, 0);

    VectorSchemaRoot root = VectorSchemaRoot.create(keyOnlySchema, allocator);
    root.allocateNew();

    IntVector idVector = (IntVector) root.getVector("id");
    for (int i = 0; i < sourceIds.size(); i++) {
      idVector.setSafe(i, sourceIds.get(i));
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
  public void testSourceDedupeFirstSeenKeepsFirst() throws Exception {
    // Source has two rows for id=0 ("First 0" then "Second 0"). FirstSeen keeps
    // the first encountered row and skips the duplicate.

    try (VectorSchemaRoot source = buildDuplicateKeySource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        MergeInsertResult result =
            dataset.mergeInsert(
                new MergeInsertParams(Collections.singletonList("id"))
                    .withMatchedUpdateAll()
                    .withNotMatched(MergeInsertParams.WhenNotMatched.InsertAll)
                    .withSourceDedupeBehavior(MergeInsertParams.SourceDedupeBehavior.FirstSeen),
                sourceStream);

        Assertions.assertEquals(
            "{0=First 0, 1=Person 1, 2=Person 2, 3=Source 3, 4=Source 4}",
            readAll(result.dataset()).toString(),
            "FirstSeen should keep the first duplicate source row (id=0) and update unique matches");
      }
    }
  }

  @Test
  public void testSourceDedupeFailWithDuplicates() throws Exception {
    // Default behavior (Fail) must error when the source contains duplicate join keys.

    try (VectorSchemaRoot source = buildDuplicateKeySource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        String originalDataset = readAll(dataset).toString();

        Exception ex =
            Assertions.assertThrows(
                Exception.class,
                () ->
                    dataset.mergeInsert(
                        new MergeInsertParams(Collections.singletonList("id"))
                            .withMatchedUpdateAll()
                            .withNotMatched(MergeInsertParams.WhenNotMatched.InsertAll)
                            .withSourceDedupeBehavior(MergeInsertParams.SourceDedupeBehavior.Fail),
                        sourceStream));

        Assertions.assertNotNull(ex.getMessage(), "exception should carry a message");
        Assertions.assertTrue(
            ex.getMessage().contains("Ambiguous merge inserts are prohibited"),
            "Fail should report the ambiguous-merge cause, got: " + ex.getMessage());

        Assertions.assertEquals(
            originalDataset,
            readAll(dataset).toString(),
            "Dataset should remain unchanged after a failed mergeInsert");
      }
    }
  }

  @Test
  public void testSourceDedupeDeleteFullSchemaFirstSeen() throws Exception {
    // Full-schema delete: source id=0 is duplicated ("First 0", then "Second 0").
    // FirstSeen deletes the matched target row once and skips the duplicate; ids
    // 3 and 4 are unique matches that are also deleted.

    try (VectorSchemaRoot source = buildDuplicateKeySource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        MergeInsertResult result =
            dataset.mergeInsert(
                new MergeInsertParams(Collections.singletonList("id"))
                    .withMatchedDelete()
                    .withNotMatched(MergeInsertParams.WhenNotMatched.InsertAll)
                    .withSourceDedupeBehavior(MergeInsertParams.SourceDedupeBehavior.FirstSeen),
                sourceStream);

        Assertions.assertEquals(
            "{1=Person 1, 2=Person 2}",
            readAll(result.dataset()).toString(),
            "FirstSeen should delete each matched target row once, skipping the duplicate id=0");
      }
    }
  }

  @Test
  public void testSourceDedupeDeleteFullSchemaFailWithDuplicates() throws Exception {
    // Full-schema delete with Fail must reject the ambiguous duplicate source key
    // and leave the target unchanged.

    try (VectorSchemaRoot source = buildDuplicateKeySource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        String originalDataset = readAll(dataset).toString();

        Exception ex =
            Assertions.assertThrows(
                Exception.class,
                () ->
                    dataset.mergeInsert(
                        new MergeInsertParams(Collections.singletonList("id"))
                            .withMatchedDelete()
                            .withNotMatched(MergeInsertParams.WhenNotMatched.InsertAll)
                            .withSourceDedupeBehavior(MergeInsertParams.SourceDedupeBehavior.Fail),
                        sourceStream));

        Assertions.assertNotNull(ex.getMessage(), "exception should carry a message");
        Assertions.assertTrue(
            ex.getMessage().contains("Ambiguous merge inserts are prohibited"),
            "Fail should report the ambiguous-merge cause, got: " + ex.getMessage());

        Assertions.assertEquals(
            originalDataset,
            readAll(dataset).toString(),
            "Dataset should remain unchanged after a failed delete mergeInsert");
      }
    }
  }

  @Test
  public void testSourceDedupeDeleteOnlyKeyOnlyFirstSeen() throws Exception {
    // Key-only delete-only plan: the source carries just the join key and id=0 is
    // duplicated. FirstSeen deletes the matched target row once and skips the
    // duplicate; no other rows are touched.

    try (VectorSchemaRoot source =
        buildDuplicateKeyOnlySource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        MergeInsertResult result =
            dataset.mergeInsert(
                new MergeInsertParams(Collections.singletonList("id"))
                    .withMatchedDelete()
                    .withNotMatched(MergeInsertParams.WhenNotMatched.DoNothing)
                    .withSourceDedupeBehavior(MergeInsertParams.SourceDedupeBehavior.FirstSeen),
                sourceStream);

        Assertions.assertEquals(
            "{1=Person 1, 2=Person 2, 3=Person 3, 4=Person 4}",
            readAll(result.dataset()).toString(),
            "FirstSeen should delete the matched id=0 once and skip the duplicate");
      }
    }
  }

  @Test
  public void testSourceDedupeDeleteOnlyKeyOnlyFailWithDuplicates() throws Exception {
    // Key-only delete-only plan with Fail must reject the ambiguous duplicate
    // source key and leave the target unchanged.

    try (VectorSchemaRoot source =
        buildDuplicateKeyOnlySource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        String originalDataset = readAll(dataset).toString();

        Exception ex =
            Assertions.assertThrows(
                Exception.class,
                () ->
                    dataset.mergeInsert(
                        new MergeInsertParams(Collections.singletonList("id"))
                            .withMatchedDelete()
                            .withNotMatched(MergeInsertParams.WhenNotMatched.DoNothing)
                            .withSourceDedupeBehavior(MergeInsertParams.SourceDedupeBehavior.Fail),
                        sourceStream));

        Assertions.assertNotNull(ex.getMessage(), "exception should carry a message");
        Assertions.assertTrue(
            ex.getMessage().contains("Ambiguous merge inserts are prohibited"),
            "Fail should report the ambiguous-merge cause, got: " + ex.getMessage());

        Assertions.assertEquals(
            originalDataset,
            readAll(dataset).toString(),
            "Dataset should remain unchanged after a failed delete-only mergeInsert");
      }
    }
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
