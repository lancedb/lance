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
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.channels.ClosedChannelException;
import java.nio.file.Path;
import java.time.Clock;
import java.time.ZonedDateTime;
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
      ZonedDateTime before = ZonedDateTime.now();
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        ZonedDateTime time1 = dataset.getVersion().getDataTime();
        assertEquals(1, dataset.version());
        assertTrue(time1.isEqual(before) || time1.isAfter(before));
        assertTrue(time1.isEqual(ZonedDateTime.now()) || time1.isBefore(ZonedDateTime.now()));
        assertEquals(time1.getZone(), Clock.systemUTC().getZone());
        assertEquals(1, dataset.latestVersion());

        // Write first batch of data
        try (Dataset dataset2 = testDataset.write(1, 5)) {
          ZonedDateTime time2 = dataset2.getVersion().getDataTime();
          assertEquals(1, dataset.version());
          assertEquals(2, dataset.latestVersion());
          assertEquals(2, dataset2.version());
          assertEquals(2, dataset2.latestVersion());
          assertTrue(time2.isEqual(before) || time2.isAfter(before));
          assertTrue(time2.isEqual(time1) || time2.isAfter(time1));
          assertTrue(time1.isEqual(dataset.getVersion().getDataTime()));

          // Open dataset with version 1
          ReadOptions options1 = new ReadOptions.Builder().setVersion(1).build();
          try (Dataset datasetV1 = Dataset.open(allocator, datasetPath, options1)) {
            assertEquals(1, datasetV1.version());
            assertTrue(time1.isEqual(dataset.getVersion().getDataTime()));
            assertEquals(2, datasetV1.latestVersion());
          }

          // Write second batch of data
          try (Dataset dataset3 = testDataset.write(2, 3)) {
            ZonedDateTime time3 = dataset3.getVersion().getDataTime();
            assertEquals(1, dataset.version());
            assertTrue(time1.isEqual(dataset.getVersion().getDataTime()));
            assertEquals(3, dataset.latestVersion());
            assertEquals(2, dataset2.version());
            assertTrue(time2.isEqual(dataset2.getVersion().getDataTime()));
            assertEquals(3, dataset2.latestVersion());
            assertTrue(time3.isEqual(before) || time3.isAfter(before));
            assertEquals(3, dataset3.version());
            assertEquals(3, dataset3.latestVersion());

            // Open dataset with version 2
            ReadOptions options2 = new ReadOptions.Builder().setVersion(2).build();
            try (Dataset datasetV2 = Dataset.open(allocator, datasetPath, options2)) {
              assertEquals(2, datasetV2.version());
              assertTrue(time2.isEqual(datasetV2.getVersion().getDataTime()));
              assertEquals(3, datasetV2.latestVersion());
            }

            // Open dataset with latest version (3)
            try (Dataset datasetLatest = Dataset.open(datasetPath, allocator)) {
              assertEquals(3, datasetLatest.version());
              assertTrue(time3.isEqual(datasetLatest.getVersion().getDataTime()));
              assertEquals(3, datasetLatest.latestVersion());
            }

            List<Version> versions = dataset.listVersions();
            assertEquals(3, versions.size());
            assertEquals(1, versions.get(0).getId());
            assertEquals(2, versions.get(1).getId());
            assertEquals(3, versions.get(2).getId());
            assertTrue(time1.isEqual(versions.get(0).getDataTime()));
            assertTrue(time2.isEqual(versions.get(1).getDataTime()));
            assertTrue(time3.isEqual(versions.get(2).getDataTime()));
            assertArrayEquals(versions.toArray(), dataset2.listVersions().toArray());
            assertArrayEquals(versions.toArray(), dataset3.listVersions().toArray());
            dataset.checkoutLatest();
            assertEquals(3, dataset.version());
            assertTrue(time3.isEqual(dataset.getVersion().getDataTime()));
            assertEquals(3, dataset.latestVersion());
          }
        }
      }
    }
  }

  @Test
  void testDatasetCheckoutVersion() {
    String datasetPath = tempDir.resolve("dataset_checkout_version").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      // version 1, empty dataset
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        assertEquals(1, dataset.version());
        assertEquals(1, dataset.latestVersion());
        assertEquals(0, dataset.countRows());
      }

      // write first batch of data, version 2
      try (Dataset dataset2 = testDataset.write(1, 5)) {
        assertEquals(2, dataset2.version());
        assertEquals(2, dataset2.latestVersion());
        assertEquals(5, dataset2.countRows());

        // checkout the dataset at version 1
        try (Dataset checkoutV1 = dataset2.checkoutVersion(1)) {
          assertEquals(1, checkoutV1.version());
          assertEquals(2, checkoutV1.latestVersion());
          assertEquals(0, checkoutV1.countRows());
        }
      }
    }
  }

  @Test
  void testDatasetTags() {
    String datasetPath = tempDir.resolve("dataset_tags").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      // version 1, empty dataset
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        assertEquals(1, dataset.version());
        dataset.tags().create("tag1", 1);
        assertEquals(1, dataset.tags().list().size());
        assertEquals(1, dataset.tags().list().get(0).getVersion());
        assertEquals(1, dataset.tags().getVersion("tag1"));
      }

      // write first batch of data, version 2
      try (Dataset dataset2 = testDataset.write(1, 5)) {
        assertEquals(2, dataset2.version());
        assertEquals(1, dataset2.tags().list().size());
        assertEquals(1, dataset2.tags().list().get(0).getVersion());
        assertEquals(1, dataset2.tags().getVersion("tag1"));
        dataset2.tags().create("tag2", 2);
        assertEquals(2, dataset2.tags().list().size());
        assertEquals(1, dataset2.tags().getVersion("tag1"));
        assertEquals(2, dataset2.tags().getVersion("tag2"));
        dataset2.tags().update("tag2", 1);
        assertEquals(2, dataset2.tags().list().size());
        assertEquals(1, dataset2.tags().list().get(0).getVersion());
        assertEquals(1, dataset2.tags().list().get(1).getVersion());
        assertEquals(1, dataset2.tags().getVersion("tag1"));
        assertEquals(1, dataset2.tags().getVersion("tag2"));
        dataset2.tags().delete("tag2");
        assertEquals(1, dataset2.tags().list().size());
        assertEquals(1, dataset2.tags().list().get(0).getVersion());
        assertEquals(1, dataset2.tags().getVersion("tag1"));
        assertThrows(RuntimeException.class, () -> dataset2.tags().getVersion("tag2"));

        // checkout the dataset at version 1
        try (Dataset checkoutV1 = dataset2.checkoutTag("tag1")) {
          assertEquals(1, checkoutV1.version());
          assertEquals(2, checkoutV1.latestVersion());
          assertEquals(0, checkoutV1.countRows());
          assertEquals(1, checkoutV1.tags().list().size());
          assertEquals(1, checkoutV1.tags().list().get(0).getVersion());
          assertEquals(1, checkoutV1.tags().getVersion("tag1"));
        }
      }
    }
  }

  @Test
  void testDatasetRestore() {
    String datasetPath = tempDir.resolve("dataset_restore").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      // version 1, empty dataset
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        assertEquals(1, dataset.version());
        assertEquals(1, dataset.latestVersion());
        assertEquals(0, dataset.countRows());
      }
      // write first batch of data, version 2
      try (Dataset dataset2 = testDataset.write(1, 5)) {
        assertEquals(2, dataset2.version());
        assertEquals(2, dataset2.latestVersion());
        assertEquals(5, dataset2.countRows());
        dataset2.tags().create("tag1", 2);
        try (Dataset dataset3 = dataset2.checkoutVersion(1)) {
          assertEquals(1, dataset3.version());
          assertEquals(2, dataset3.latestVersion());
          assertEquals(0, dataset3.countRows());
          dataset3.restore();
          assertEquals(3, dataset3.version());
          assertEquals(3, dataset3.latestVersion());
          assertEquals(0, dataset3.countRows());

          try (Dataset dataset4 = dataset3.checkoutTag("tag1")) {
            assertEquals(2, dataset4.version());
            assertEquals(3, dataset4.latestVersion());
            assertEquals(5, dataset4.countRows());
            dataset4.restore();
            assertEquals(4, dataset4.version());
            assertEquals(4, dataset4.latestVersion());
            assertEquals(5, dataset4.countRows());
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
  void testAddColumnByFieldsOrSchema() {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      Field newColumnField = Field.nullable("age", new ArrowType.Int(32, true));
      dataset.addColumns(Collections.singletonList(newColumnField));
      Schema expectedSchema =
          new Schema(
              Arrays.asList(
                  Field.nullable("id", new ArrowType.Int(32, true)),
                  Field.nullable("name", new ArrowType.Utf8()),
                  Field.nullable("age", new ArrowType.Int(32, true))));
      assertEquals(expectedSchema.getFields().size(), dataset.getSchema().getFields().size());
      assertEquals(
          expectedSchema.getFields().stream().map(Field::getName).collect(Collectors.toList()),
          dataset.getSchema().getFields().stream()
              .map(Field::getName)
              .collect(Collectors.toList()));

      Field complexField =
          new Field(
              "extra",
              FieldType.nullable(new ArrowType.Struct()),
              Arrays.asList(
                  Field.nullable("tag1", new ArrowType.Int(64, true)),
                  Field.nullable("tag2", new ArrowType.Utf8())));

      Schema addedColumns =
          new Schema(
              Arrays.asList(
                  Field.nullable("height", new ArrowType.Int(64, true)),
                  Field.nullable("desc", new ArrowType.Utf8()),
                  complexField));
      dataset.addColumns(addedColumns);

      expectedSchema =
          new Schema(
              Arrays.asList(
                  Field.nullable("id", new ArrowType.Int(32, true)),
                  Field.nullable("name", new ArrowType.Utf8()),
                  Field.nullable("age", new ArrowType.Int(32, true)),
                  Field.nullable("height", new ArrowType.Int(64, true)),
                  Field.nullable("desc", new ArrowType.Utf8()),
                  complexField),
              null);

      assertEquals(expectedSchema.getFields().size(), dataset.getSchema().getFields().size());
      assertEquals(
          expectedSchema.getFields().stream().map(Field::getName).collect(Collectors.toList()),
          dataset.getSchema().getFields().stream()
              .map(Field::getName)
              .collect(Collectors.toList()));
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

  @Test
  void testDeleteRows() {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      try (Dataset dataset2 = testDataset.write(1, 5)) {
        // Initially there are 5 rows
        assertEquals(5, dataset2.countRows());

        // Delete rows where id > 2 (should delete id=3, id=4)
        dataset2.delete("id > 2");

        // Now verify we have 3 rows left (id=0, id=1, id=2)
        assertEquals(3, dataset2.countRows());

        // Verify the rows that remain
        assertEquals(0, dataset2.countRows("id > 2"));
        assertEquals(3, dataset2.countRows("id <= 2"));

        // Delete another row
        dataset2.delete("id = 1");

        // Now verify we have 2 rows left (id=0, id=2)
        assertEquals(2, dataset2.countRows());
        assertEquals(1, dataset2.countRows("id = 0"));
        assertEquals(1, dataset2.countRows("id = 2"));
        assertEquals(0, dataset2.countRows("id = 1"));
      }
    }
  }
}
