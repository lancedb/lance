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

import com.lancedb.lance.ipc.LanceScanner;
import com.lancedb.lance.schema.ColumnAlteration;
import com.lancedb.lance.schema.SqlExpressions;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.channels.ClosedChannelException;
import java.nio.file.Path;
import java.util.*;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Optional;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

public class DatasetTest {
  @TempDir static Path tempDir; // Temporary directory for the tests
  private static Dataset dataset;

  @BeforeAll
  static void setup() {}

  @AfterAll
  static void tearDown() {
    // Cleanup resources used by the tests
    if (dataset != null) {
      dataset.close();
    }
  }

  @Test
  void testWriteStreamAndOpenPath() throws IOException, URISyntaxException {
    String datasetPath = tempDir.resolve("write_stream").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.RandomAccessDataset testDataset =
          new TestUtils.RandomAccessDataset(allocator, datasetPath);
      testDataset.createDatasetAndValidate();
      testDataset.openDatasetAndValidate();
    }
  }

  @Test
  void testCreateEmptyDataset() {
    String datasetPath = tempDir.resolve("new_empty_dataset").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
    }
  }

  @Test
  void testCreateDirNotExist() throws IOException, URISyntaxException {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
    }
  }

  @Test
  void testOpenInvalidPath() {
    String validPath = tempDir.resolve("Invalid_dataset").toString();
    assertThrows(
        RuntimeException.class,
        () -> {
          dataset = Dataset.open(validPath, new RootAllocator(Long.MAX_VALUE));
        });
  }

  @Test
  void testDatasetVersion() {
    String datasetPath = tempDir.resolve("dataset_version").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        assertEquals(1, dataset.version());
        assertEquals(1, dataset.latestVersion());

        // Write first batch of data
        try (Dataset dataset2 = testDataset.write(1, 5)) {
          assertEquals(1, dataset.version());
          assertEquals(2, dataset.latestVersion());
          assertEquals(2, dataset2.version());
          assertEquals(2, dataset2.latestVersion());

          // Open dataset with version 1
          ReadOptions options1 = new ReadOptions.Builder().setVersion(1).build();
          try (Dataset datasetV1 = Dataset.open(allocator, datasetPath, options1)) {
            assertEquals(1, datasetV1.version());
            assertEquals(2, datasetV1.latestVersion());
          }

          // Write second batch of data
          try (Dataset dataset3 = testDataset.write(2, 3)) {
            assertEquals(1, dataset.version());
            assertEquals(3, dataset.latestVersion());
            assertEquals(2, dataset2.version());
            assertEquals(3, dataset2.latestVersion());
            assertEquals(3, dataset3.version());
            assertEquals(3, dataset3.latestVersion());

            // Open dataset with version 2
            ReadOptions options2 = new ReadOptions.Builder().setVersion(2).build();
            try (Dataset datasetV2 = Dataset.open(allocator, datasetPath, options2)) {
              assertEquals(2, datasetV2.version());
              assertEquals(3, datasetV2.latestVersion());
            }

            // Open dataset with latest version (3)
            try (Dataset datasetLatest = Dataset.open(datasetPath, allocator)) {
              assertEquals(3, datasetLatest.version());
              assertEquals(3, datasetLatest.latestVersion());
            }
          }
        }
      }
    }
  }

  @Test
  void testDatasetUri() {
    String datasetPath = tempDir.resolve("dataset_uri").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        assertEquals(datasetPath, dataset.uri());
      }
    }
  }

  @Test
  void testOpenNonExist() throws IOException, URISyntaxException {
    String datasetPath = tempDir.resolve("non_exist").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      assertThrows(
          IllegalArgumentException.class,
          () -> {
            Dataset.open(datasetPath, allocator);
          });
    }
  }

  @Test
  void testCreateExist() throws IOException, URISyntaxException {
    String datasetPath = tempDir.resolve("create_exist").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      assertThrows(
          IllegalArgumentException.class,
          () -> {
            testDataset.createEmptyDataset();
          });
    }
  }

  @Test
  void testCommitConflict() {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        assertEquals(1, dataset.version());
        assertEquals(1, dataset.latestVersion());
        assertThrows(
            IllegalArgumentException.class,
            () -> {
              testDataset.write(0, 5);
            });
      }
    }
  }

  @Test
  void testGetSchemaWithClosedDataset() {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      Dataset dataset = testDataset.createEmptyDataset();
      dataset.close();
      assertThrows(RuntimeException.class, dataset::getSchema);
    }
  }

  @Test
  void testDropColumns() {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      assertEquals(testDataset.getSchema(), dataset.getSchema());
      dataset.dropColumns(Collections.singletonList("name"));

      Schema changedSchema =
          new Schema(
              Collections.singletonList(Field.nullable("id", new ArrowType.Int(32, true))), null);

      assertEquals(changedSchema.getFields().size(), dataset.getSchema().getFields().size());
      assertEquals(
          changedSchema.getFields().stream().map(Field::getName).collect(Collectors.toList()),
          dataset.getSchema().getFields().stream()
              .map(Field::getName)
              .collect(Collectors.toList()));
    }
  }

  @Test
  void testAlterColumns() {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      assertEquals(testDataset.getSchema(), dataset.getSchema());

      ColumnAlteration nameColumnAlteration =
          new ColumnAlteration.Builder("name")
              .rename("new_name")
              .nullable(true)
              .castTo(new ArrowType.Utf8())
              .build();

      dataset.alterColumns(Collections.singletonList(nameColumnAlteration));

      Schema changedSchema =
          new Schema(
              Arrays.asList(
                  Field.nullable("id", new ArrowType.Int(32, true)),
                  Field.notNullable("new_name", new ArrowType.Utf8())),
              null);

      assertEquals(changedSchema.getFields().size(), dataset.getSchema().getFields().size());
      assertEquals(
          changedSchema.getFields().stream().map(Field::getName).collect(Collectors.toList()),
          dataset.getSchema().getFields().stream()
              .map(Field::getName)
              .collect(Collectors.toList()));

      nameColumnAlteration =
          new ColumnAlteration.Builder("new_name")
              .rename("new_name_2")
              .castTo(new ArrowType.LargeUtf8())
              .build();

      dataset.alterColumns(Collections.singletonList(nameColumnAlteration));
      changedSchema =
          new Schema(
              Arrays.asList(
                  Field.nullable("id", new ArrowType.Int(32, true)),
                  Field.notNullable("new_name_2", new ArrowType.LargeUtf8())),
              null);

      assertEquals(changedSchema.getFields().size(), dataset.getSchema().getFields().size());
      assertEquals(
          changedSchema.getFields().stream().map(Field::getName).collect(Collectors.toList()),
          dataset.getSchema().getFields().stream()
              .map(Field::getName)
              .collect(Collectors.toList()));

      nameColumnAlteration = new ColumnAlteration.Builder("new_name_2").build();
      dataset.alterColumns(Collections.singletonList(nameColumnAlteration));
      assertNotNull(dataset.getSchema().findField("new_name_2"));
    }
  }

  @Test
  void testAddColumnBySqlExpressions() {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      SqlExpressions sqlExpressions =
          new SqlExpressions.Builder().withExpression("double_id", "id * 2").build();
      dataset.addColumns(sqlExpressions, Optional.empty());

      Schema changedSchema =
          new Schema(
              Arrays.asList(
                  Field.nullable("id", new ArrowType.Int(32, true)),
                  Field.nullable("name", new ArrowType.Utf8()),
                  Field.nullable("double_id", new ArrowType.Int(32, true))),
              null);

      assertEquals(changedSchema.getFields().size(), dataset.getSchema().getFields().size());
      assertEquals(
          changedSchema.getFields().stream().map(Field::getName).collect(Collectors.toList()),
          dataset.getSchema().getFields().stream()
              .map(Field::getName)
              .collect(Collectors.toList()));

      sqlExpressions = new SqlExpressions.Builder().withExpression("triple_id", "id * 3").build();
      dataset.addColumns(sqlExpressions, Optional.empty());
      changedSchema =
          new Schema(
              Arrays.asList(
                  Field.nullable("id", new ArrowType.Int(32, true)),
                  Field.nullable("name", new ArrowType.Utf8()),
                  Field.nullable("double_id", new ArrowType.Int(32, true)),
                  Field.nullable("triple_id", new ArrowType.Int(32, true))),
              null);
      assertEquals(changedSchema.getFields().size(), dataset.getSchema().getFields().size());
      assertEquals(
          changedSchema.getFields().stream().map(Field::getName).collect(Collectors.toList()),
          dataset.getSchema().getFields().stream()
              .map(Field::getName)
              .collect(Collectors.toList()));
    }
  }

  @Test
  void testAddColumnsByStream() throws IOException {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      try (Dataset initialDataset = testDataset.createEmptyDataset()) {
        try (Dataset datasetV1 = testDataset.write(1, 3)) {
          assertEquals(3, datasetV1.countRows());
        }
      }

      dataset = Dataset.open(datasetPath, allocator);

      Schema newColumnSchema =
          new Schema(
              Collections.singletonList(Field.nullable("age", new ArrowType.Int(32, true))), null);

      try (VectorSchemaRoot vector = VectorSchemaRoot.create(newColumnSchema, allocator);
          ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {

        IntVector ageVector = (IntVector) vector.getVector("age");
        ageVector.allocateNew(3);
        ageVector.set(0, 25);
        ageVector.set(1, 30);
        ageVector.set(2, 35);
        vector.setRowCount(3);

        class SimpleVectorReader extends ArrowReader {
          private boolean batchLoaded = false;

          protected SimpleVectorReader(BufferAllocator allocator) {
            super(allocator);
          }

          @Override
          public boolean loadNextBatch() {
            if (!batchLoaded) {
              batchLoaded = true;
              return true;
            }
            return false;
          }

          @Override
          public VectorSchemaRoot getVectorSchemaRoot() {
            return vector;
          }

          @Override
          public long bytesRead() {
            return vector.getFieldVectors().stream().mapToLong(FieldVector::getBufferSize).sum();
          }

          @Override
          protected void closeReadSource() {}

          @Override
          protected Schema readSchema() {
            return newColumnSchema;
          }
        }

        try (ArrowReader reader = new SimpleVectorReader(allocator)) {
          Data.exportArrayStream(allocator, reader, stream);

          dataset.addColumns(stream, Optional.of(3L));

          Schema expectedSchema =
              new Schema(
                  Arrays.asList(
                      Field.nullable("id", new ArrowType.Int(32, true)),
                      Field.nullable("name", new ArrowType.Utf8()),
                      Field.nullable("age", new ArrowType.Int(32, true))),
                  null);
          Schema actualSchema = dataset.getSchema();
          assertEquals(expectedSchema.getFields(), actualSchema.getFields());

          try (LanceScanner scanner = dataset.newScan()) {
            try (ArrowReader resultReader = scanner.scanBatches()) {
              assertTrue(resultReader.loadNextBatch());
              VectorSchemaRoot root = resultReader.getVectorSchemaRoot();
              assertEquals(3, root.getRowCount());

              IntVector idVector = (IntVector) root.getVector("id");
              IntVector ageVectorResult = (IntVector) root.getVector("age");
              for (int i = 0; i < 3; i++) {
                assertEquals(i, idVector.get(i));
                assertEquals(25 + i * 5, ageVectorResult.get(i));
              }
            }
          }
        }
      }
    } catch (Exception e) {
      fail("Exception occurred during test: " + e.getMessage(), e);
    }
  }

  @Test
  void testDropPath() {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      Dataset.drop(datasetPath, new HashMap<>());
    }
  }

  @Test
  void testTake() throws IOException, ClosedChannelException {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      try (Dataset dataset2 = testDataset.write(1, 5)) {
        List<Long> indices = Arrays.asList(1L, 4L);
        List<String> columns = Arrays.asList("id", "name");
        try (ArrowReader reader = dataset2.take(indices, columns)) {
          while (reader.loadNextBatch()) {
            VectorSchemaRoot result = reader.getVectorSchemaRoot();
            assertNotNull(result);
            assertEquals(indices.size(), result.getRowCount());

            for (int i = 0; i < indices.size(); i++) {
              assertEquals(indices.get(i).intValue(), result.getVector("id").getObject(i));
              assertNotNull(result.getVector("name").getObject(i));
            }
          }
        }
      }
    }
  }

  @Test
  void testCountRows() {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      try (Dataset dataset2 = testDataset.write(1, 5)) {
        assertEquals(5, dataset2.countRows());
        // get id = 3 and 4
        assertEquals(2, dataset2.countRows("id > 2"));

        assertThrows(IllegalArgumentException.class, () -> dataset2.countRows(null));
        assertThrows(IllegalArgumentException.class, () -> dataset2.countRows(""));
      }
    }
  }

  @Test
  void testCalculateDataSize() {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      try (Dataset dataset2 = testDataset.write(1, 5)) {
        assertEquals(100, dataset2.calculateDataSize());
      }
    }
  }
}
