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
package com.lancedb.lance;

import com.lancedb.lance.merge.MergeInsertParams;
import com.lancedb.lance.merge.MergeInsertResult;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
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
