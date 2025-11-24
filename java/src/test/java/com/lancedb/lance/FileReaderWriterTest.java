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

import com.lancedb.lance.file.LanceFileReader;
import com.lancedb.lance.file.LanceFileWriter;
import com.lancedb.lance.util.Range;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.arrow.vector.util.Text;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

public class FileReaderWriterTest {

  private VectorSchemaRoot createBatch(BufferAllocator allocator) throws IOException {
    Schema schema =
        new Schema(
            Arrays.asList(
                Field.nullable("x", new ArrowType.Int(64, true)),
                Field.nullable("y", new ArrowType.Utf8())),
            null);
    VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
    root.allocateNew();
    BigIntVector iVector = (BigIntVector) root.getVector("x");
    VarCharVector sVector = (VarCharVector) root.getVector("y");

    for (int i = 0; i < 100; i++) {
      iVector.setSafe(i, i);
      sVector.setSafe(i, new Text("s-" + i));
    }
    root.setRowCount(100);

    return root;
  }

  void createSimpleFile(String filePath) throws Exception {
    BufferAllocator allocator = new RootAllocator();
    try (LanceFileWriter writer = LanceFileWriter.open(filePath, allocator, null)) {
      try (VectorSchemaRoot batch = createBatch(allocator)) {
        writer.write(batch);
      }
    }
  }

  @Test
  void testBasicRead(@TempDir Path tempDir) throws Exception {
    BufferAllocator allocator = new RootAllocator();
    String filePath = tempDir.resolve("basic_read.lance").toString();
    createSimpleFile(filePath);
    LanceFileReader reader = LanceFileReader.open(filePath, allocator);

    Schema expectedSchema =
        new Schema(
            Arrays.asList(
                Field.nullable("x", new ArrowType.Int(64, true)),
                Field.nullable("y", new ArrowType.Utf8())),
            null);

    assertEquals(100, reader.numRows());
    assertEquals(expectedSchema, reader.schema());

    try (ArrowReader batches = reader.readAll(null, null, 100)) {
      assertTrue(batches.loadNextBatch());
      VectorSchemaRoot batch = batches.getVectorSchemaRoot();
      assertEquals(100, batch.getRowCount());
      assertEquals(2, batch.getSchema().getFields().size());
      assertFalse(batches.loadNextBatch());
    }

    try (ArrowReader batches = reader.readAll(null, null, 15)) {
      for (int i = 0; i < 100; i += 15) {
        int expected = Math.min(15, 100 - i);
        assertTrue(batches.loadNextBatch());
        VectorSchemaRoot batch = batches.getVectorSchemaRoot();
        assertEquals(expected, batch.getRowCount());
        assertEquals(2, batch.getSchema().getFields().size());
      }
      assertFalse(batches.loadNextBatch());
    }

    reader.close();
    try {
      reader.numRows();
      fail("Expected LanceException to be thrown");
    } catch (IOException e) {
      assertEquals("FileReader has already been closed", e.getMessage());
    }

    // Ok to call schema after close
    assertEquals(expectedSchema, reader.schema());

    // close should be idempotent
    reader.close();
  }

  @Test
  void testReadWithProjection(@TempDir Path tempDir) throws Exception {
    BufferAllocator allocator = new RootAllocator();
    String filePath = tempDir.resolve("basic_read.lance").toString();
    createSimpleFile(filePath);
    LanceFileReader reader = LanceFileReader.open(filePath, allocator);

    Schema expectedSchema =
        new Schema(
            Arrays.asList(
                Field.nullable("x", new ArrowType.Int(64, true)),
                Field.nullable("y", new ArrowType.Utf8())),
            null);

    assertEquals(100, reader.numRows());
    assertEquals(expectedSchema, reader.schema());

    try (ArrowReader batches = reader.readAll(Collections.singletonList("x"), null, 100)) {
      assertTrue(batches.loadNextBatch());
      VectorSchemaRoot batch = batches.getVectorSchemaRoot();
      assertEquals(100, batch.getRowCount());
      assertEquals(1, batch.getSchema().getFields().size());
      assertEquals("x", batch.getSchema().getFields().get(0).getName());
      assertFalse(batches.loadNextBatch());
    }

    try (ArrowReader batches = reader.readAll(Collections.singletonList("y"), null, 100)) {
      assertTrue(batches.loadNextBatch());
      VectorSchemaRoot batch = batches.getVectorSchemaRoot();
      assertEquals(100, batch.getRowCount());
      assertEquals(1, batch.getSchema().getFields().size());
      assertEquals("y", batch.getSchema().getFields().get(0).getName());
      assertFalse(batches.loadNextBatch());
    }

    try (ArrowReader batches =
        reader.readAll(
            null, Arrays.asList(Range.of(1, 11), Range.of(14, 19), Range.of(20, 21)), 100)) {
      assertTrue(batches.loadNextBatch());
      VectorSchemaRoot batch = batches.getVectorSchemaRoot();
      assertEquals(16, batch.getRowCount());
      assertEquals(2, batch.getSchema().getFields().size());
      assertFalse(batches.loadNextBatch());
    }

    try (ArrowReader batches =
        reader.readAll(
            Collections.singletonList("x"),
            Arrays.asList(Range.of(23, 25), Range.of(27, 29)),
            100)) {
      assertTrue(batches.loadNextBatch());
      VectorSchemaRoot batch = batches.getVectorSchemaRoot();
      assertEquals(4, batch.getRowCount());
      assertEquals(1, batch.getSchema().getFields().size());
      assertFalse(batches.loadNextBatch());
    }

    try (ArrowReader batches =
        reader.readAll(
            Collections.singletonList("y"),
            Arrays.asList(Range.of(23, 25), Range.of(27, 29)),
            100)) {
      assertTrue(batches.loadNextBatch());
      VectorSchemaRoot batch = batches.getVectorSchemaRoot();
      assertEquals(4, batch.getRowCount());
      assertEquals(1, batch.getSchema().getFields().size());
      assertFalse(batches.loadNextBatch());
    }

    reader.close();
  }

  @Test
  void testBasicWrite(@TempDir Path tempDir) throws Exception {
    String filePath = tempDir.resolve("basic_write.lance").toString();
    createSimpleFile(filePath);
  }

  @Test
  void testWriteNoData(@TempDir Path tempDir) throws Exception {
    String filePath = tempDir.resolve("no_data.lance").toString();
    BufferAllocator allocator = new RootAllocator();

    LanceFileWriter writer = LanceFileWriter.open(filePath, allocator, null);

    try {
      writer.close();
      fail("Expected IllegalArgumentException to be thrown");
    } catch (IllegalArgumentException e) {
      assertTrue(e.getMessage().contains("no data provided"));
    }
  }

  @Test
  void testWriteWithStorage(@TempDir Path tempDir) throws IOException {
    String filePath = "az://fail_bucket" + tempDir.resolve("test_write_with_storage");
    BufferAllocator allocator = new RootAllocator();
    Map<String, String> storageOptions = new HashMap<>();
    try {
      LanceFileWriter.open(filePath, allocator, null, storageOptions);
    } catch (IllegalArgumentException e) {
      assertTrue(
          e.getMessage()
              .contains(
                  "Unable to find object store prefix: no Azure account "
                      + "name in URI, and no storage account configured."));
    }

    storageOptions.put("account_name", "some_account");
    storageOptions.put("account_key", "some_key");
    try {
      // Verify the config in storage options is worked. The message will change.
      LanceFileWriter.open(filePath, allocator, null, storageOptions);
    } catch (IOException e) {
      assertTrue(e.getMessage().contains("Invalid Access Key"));
    }
  }

  @Test
  void testInvalidPath() {
    BufferAllocator allocator = new RootAllocator();
    try {
      LanceFileReader.open("/tmp/does_not_exist.lance", allocator);
      fail("Expected LanceException to be thrown");
    } catch (IOException e) {
      assertTrue(e.getMessage().contains("Object at location /tmp/does_not_exist.lance not found"));
    }
    try {
      LanceFileReader.open("", allocator);
      fail("Expected LanceException to be thrown");
    } catch (IOException e) {
    }
  }
}
