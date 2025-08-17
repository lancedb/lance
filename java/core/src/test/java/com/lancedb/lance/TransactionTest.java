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

import com.lancedb.lance.file.LanceFileWriter;
import com.lancedb.lance.fragment.DataFile;
import com.lancedb.lance.ipc.LanceScanner;
import com.lancedb.lance.operation.Append;
import com.lancedb.lance.operation.DataReplacement;
import com.lancedb.lance.operation.Delete;
import com.lancedb.lance.operation.Merge;
import com.lancedb.lance.operation.Overwrite;
import com.lancedb.lance.operation.Project;
import com.lancedb.lance.operation.ReserveFragments;
import com.lancedb.lance.operation.Restore;
import com.lancedb.lance.operation.Rewrite;
import com.lancedb.lance.operation.RewriteGroup;
import com.lancedb.lance.operation.Update;
import com.lancedb.lance.operation.UpdateConfig;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TransactionTest {

  public static final int TEST_FILE_FORMAT_MAJOR_VERSION = 2;
  public static final int TEST_FILE_FORMAT_MINOR_VERSION = 0;
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
  void testProjection(@TempDir Path tempDir) {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      assertEquals(testDataset.getSchema(), dataset.getSchema());
      List<Field> fieldList = new ArrayList<>(testDataset.getSchema().getFields());
      Collections.reverse(fieldList);
      Transaction txn1 =
          dataset
              .newTransactionBuilder()
              .operation(Project.builder().schema(new Schema(fieldList)).build())
              .build();
      try (Dataset committedDataset = txn1.commit()) {
        assertEquals(1, txn1.readVersion());
        assertEquals(1, dataset.version());
        assertEquals(2, committedDataset.version());
        assertEquals(new Schema(fieldList), committedDataset.getSchema());
        fieldList.remove(1);
        Transaction txn2 =
            committedDataset
                .newTransactionBuilder()
                .operation(Project.builder().schema(new Schema(fieldList)).build())
                .build();
        try (Dataset committedDataset2 = txn2.commit()) {
          assertEquals(2, txn2.readVersion());
          assertEquals(2, committedDataset.version());
          assertEquals(3, committedDataset2.version());
          assertEquals(new Schema(fieldList), committedDataset2.getSchema());
          assertEquals(txn1, committedDataset.readTransaction().orElse(null));
          assertEquals(txn2, committedDataset2.readTransaction().orElse(null));
        }
      }
    }
  }

  @Test
  void testAppend(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testAppend").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      // Commit fragment
      int rowCount = 20;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      Transaction transaction =
          dataset
              .newTransactionBuilder()
              .operation(
                  Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
              .transactionProperties(Collections.singletonMap("key", "value"))
              .build();
      assertEquals("value", transaction.transactionProperties().get("key"));
      try (Dataset dataset = transaction.commit()) {
        assertEquals(2, dataset.version());
        assertEquals(2, dataset.latestVersion());
        assertEquals(rowCount, dataset.countRows());
        assertThrows(
            IllegalArgumentException.class,
            () ->
                dataset
                    .newTransactionBuilder()
                    .operation(Append.builder().fragments(new ArrayList<>()).build())
                    .build()
                    .commit()
                    .close());
        assertEquals(transaction, dataset.readTransaction().orElse(null));
      }
    }
  }

  @Test
  void testDelete(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testDelete").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      // Commit fragment
      int rowCount = 20;
      FragmentMetadata fragmentMeta0 = testDataset.createNewFragment(rowCount);
      FragmentMetadata fragmentMeta1 = testDataset.createNewFragment(rowCount);
      Transaction transaction =
          dataset
              .newTransactionBuilder()
              .operation(
                  Append.builder().fragments(Arrays.asList(fragmentMeta0, fragmentMeta1)).build())
              .build();
      try (Dataset dataset = transaction.commit()) {
        assertEquals(2, dataset.version());
        assertEquals(2, dataset.latestVersion());
      }

      dataset = Dataset.open(datasetPath, allocator);

      List<Long> deletedFragmentIds =
          dataset.getFragments().stream()
              .map(t -> Long.valueOf(t.getId()))
              .collect(Collectors.toList());

      Transaction delete =
          dataset
              .newTransactionBuilder()
              .operation(
                  Delete.builder().deletedFragmentIds(deletedFragmentIds).predicate("1=1").build())
              .build();
      try (Dataset dataset = delete.commit()) {
        Transaction txn = dataset.readTransaction().get();
        Delete execDelete = (Delete) txn.operation();
        assertEquals(delete.operation(), execDelete);
        assertEquals(0, dataset.countRows());
      }
    }
  }

  @Test
  void testOverwrite(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testOverwrite").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Commit fragment
      int rowCount = 20;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      Transaction transaction =
          dataset
              .newTransactionBuilder()
              .operation(
                  Overwrite.builder()
                      .fragments(Collections.singletonList(fragmentMeta))
                      .schema(testDataset.getSchema())
                      .build())
              .build();
      try (Dataset dataset = transaction.commit()) {
        assertEquals(2, dataset.version());
        assertEquals(2, dataset.latestVersion());
        assertEquals(rowCount, dataset.countRows());
        Fragment fragment = dataset.getFragments().get(0);

        try (LanceScanner scanner = fragment.newScan()) {
          Schema schemaRes = scanner.schema();
          assertEquals(testDataset.getSchema(), schemaRes);
        }
      }

      // Commit fragment again
      rowCount = 40;
      fragmentMeta = testDataset.createNewFragment(rowCount);
      transaction =
          dataset
              .newTransactionBuilder()
              .operation(
                  Overwrite.builder()
                      .fragments(Collections.singletonList(fragmentMeta))
                      .schema(testDataset.getSchema())
                      .configUpsertValues(Collections.singletonMap("config_key", "config_value"))
                      .build())
              .transactionProperties(Collections.singletonMap("key", "value"))
              .build();
      assertEquals("value", transaction.transactionProperties().get("key"));
      try (Dataset dataset = transaction.commit()) {
        assertEquals(3, dataset.version());
        assertEquals(3, dataset.latestVersion());
        assertEquals(rowCount, dataset.countRows());
        assertEquals("config_value", dataset.getConfig().get("config_key"));
        Fragment fragment = dataset.getFragments().get(0);

        try (LanceScanner scanner = fragment.newScan()) {
          Schema schemaRes = scanner.schema();
          assertEquals(testDataset.getSchema(), schemaRes);
        }
        assertEquals(transaction, dataset.readTransaction().orElse(null));
      }
    }
  }

  @Test
  void testUpdate(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testUpdate").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Commit fragment
      int rowCount = 20;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      Transaction transaction =
          dataset
              .newTransactionBuilder()
              .operation(
                  Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
              .build();

      try (Dataset dataset = transaction.commit()) {
        assertEquals(2, dataset.version());
        assertEquals(2, dataset.latestVersion());
        assertEquals(rowCount, dataset.countRows());
        assertThrows(
            IllegalArgumentException.class,
            () ->
                dataset
                    .newTransactionBuilder()
                    .operation(Append.builder().fragments(new ArrayList<>()).build())
                    .build()
                    .commit()
                    .close());
      }

      dataset = Dataset.open(datasetPath, allocator);
      // Update fragments
      rowCount = 40;
      FragmentMetadata newFragment = testDataset.createNewFragment(rowCount);
      transaction =
          dataset
              .newTransactionBuilder()
              .operation(
                  Update.builder()
                      .removedFragmentIds(
                          Collections.singletonList(
                              Long.valueOf(dataset.getFragments().get(0).getId())))
                      .newFragments(Collections.singletonList(newFragment))
                      .build())
              .build();

      try (Dataset dataset = transaction.commit()) {
        assertEquals(3, dataset.version());
        assertEquals(3, dataset.latestVersion());
        assertEquals(rowCount, dataset.countRows());

        Transaction txn = dataset.readTransaction().orElse(null);
        assertEquals(transaction, txn);
      }
    }
  }

  @Test
  void testUpdateConfig(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testUpdateConfig").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Test 1: Update configuration values using upsertValues
      Map<String, String> upsertValues = new HashMap<>();
      upsertValues.put("key1", "value1");
      upsertValues.put("key2", "value2");

      Transaction transaction =
          dataset
              .newTransactionBuilder()
              .operation(UpdateConfig.builder().upsertValues(upsertValues).build())
              .build();
      try (Dataset updatedDataset = transaction.commit()) {
        assertEquals(2, updatedDataset.version());
        assertEquals("value1", updatedDataset.getConfig().get("key1"));
        assertEquals("value2", updatedDataset.getConfig().get("key2"));

        // Test 2: Delete configuration keys using deleteKeys
        List<String> deleteKeys = Collections.singletonList("key1");
        transaction =
            updatedDataset
                .newTransactionBuilder()
                .operation(UpdateConfig.builder().deleteKeys(deleteKeys).build())
                .build();
        try (Dataset updatedDataset2 = transaction.commit()) {
          assertEquals(3, updatedDataset2.version());
          assertNull(updatedDataset2.getConfig().get("key1"));
          assertEquals("value2", updatedDataset2.getConfig().get("key2"));

          // Test 3: Update schema metadata using schemaMetadata
          Map<String, String> schemaMetadata = new HashMap<>();
          schemaMetadata.put("schema_key1", "schema_value1");
          schemaMetadata.put("schema_key2", "schema_value2");

          transaction =
              updatedDataset2
                  .newTransactionBuilder()
                  .operation(UpdateConfig.builder().schemaMetadata(schemaMetadata).build())
                  .build();
          try (Dataset updatedDataset3 = transaction.commit()) {
            assertEquals(4, updatedDataset3.version());
            assertEquals(
                "schema_value1", updatedDataset3.getLanceSchema().metadata().get("schema_key1"));
            assertEquals(
                "schema_value2", updatedDataset3.getLanceSchema().metadata().get("schema_key2"));

            // Test 4: Update field metadata using fieldMetadata
            Map<Integer, Map<String, String>> fieldMetadata = new HashMap<>();
            Map<String, String> field0Metadata = new HashMap<>();
            field0Metadata.put("field0_key1", "field0_value1");

            Map<String, String> field1Metadata = new HashMap<>();
            field1Metadata.put("field1_key1", "field1_value1");
            field1Metadata.put("field1_key2", "field1_value2");

            fieldMetadata.put(0, field0Metadata);
            fieldMetadata.put(1, field1Metadata);

            transaction =
                updatedDataset3
                    .newTransactionBuilder()
                    .operation(UpdateConfig.builder().fieldMetadata(fieldMetadata).build())
                    .build();
            try (Dataset updatedDataset4 = transaction.commit()) {
              assertEquals(5, updatedDataset4.version());

              // Verify field metadata for field 0
              Map<String, String> fieldMetadata0 =
                  updatedDataset4.getLanceSchema().fields().get(0).getMetadata();
              assertEquals("field0_value1", fieldMetadata0.get("field0_key1"));

              // Verify field metadata for field 1
              Map<String, String> field1Result =
                  updatedDataset4.getLanceSchema().fields().get(1).getMetadata();
              assertEquals("field1_value1", field1Result.get("field1_key1"));
              assertEquals("field1_value2", field1Result.get("field1_key2"));
            }
          }
        }
      }
    }
  }

  @Test
  void testMerge(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testMerge").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      int rowCount = 20;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      Transaction transaction =
          dataset
              .newTransactionBuilder()
              .operation(
                  Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
              .build();
      try (Dataset dataset = transaction.commit()) {
        assertEquals(2, dataset.version());
        assertEquals(rowCount, dataset.countRows());

        Schema newSchema = testDataset.getSchema();
        FragmentMetadata newFragmentMeta = testDataset.createNewFragment(rowCount);

        Transaction mergeTransaction =
            dataset
                .newTransactionBuilder()
                .operation(
                    Merge.builder()
                        .fragments(Collections.singletonList(newFragmentMeta))
                        .schema(newSchema)
                        .build())
                .transactionProperties(Collections.singletonMap("key", "value"))
                .build();
        assertEquals("value", mergeTransaction.transactionProperties().get("key"));
        try (Dataset mergedDataset = mergeTransaction.commit()) {
          assertEquals(3, mergedDataset.version());
          assertEquals(3, mergedDataset.latestVersion());
          assertEquals(rowCount, mergedDataset.countRows());

          assertEquals(newSchema, mergedDataset.getSchema());

          Fragment fragment = mergedDataset.getFragments().get(0);
          try (LanceScanner scanner = fragment.newScan()) {
            Schema schemaRes = scanner.schema();
            assertEquals(newSchema, schemaRes);
          }

          assertEquals(mergeTransaction, mergedDataset.readTransaction().orElse(null));
        }
      }
    }
  }

  @Test
  void testRewrite(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testRewrite").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // First, append some data
      int rowCount = 20;
      FragmentMetadata fragmentMeta1 = testDataset.createNewFragment(rowCount);
      FragmentMetadata fragmentMeta2 = testDataset.createNewFragment(rowCount);

      Transaction appendTx =
          dataset
              .newTransactionBuilder()
              .operation(
                  Append.builder().fragments(Arrays.asList(fragmentMeta1, fragmentMeta2)).build())
              .build();

      try (Dataset datasetWithData = appendTx.commit()) {
        assertEquals(2, datasetWithData.version());
        assertEquals(rowCount * 2, datasetWithData.countRows());

        // Now create a rewrite operation
        List<RewriteGroup> groups = new ArrayList<>();

        // Create a rewrite group with old fragments and new fragments
        List<FragmentMetadata> oldFragments = new ArrayList<>();
        oldFragments.add(fragmentMeta1);

        List<FragmentMetadata> newFragments = new ArrayList<>();
        FragmentMetadata newFragmentMeta = testDataset.createNewFragment(rowCount);
        newFragments.add(newFragmentMeta);

        RewriteGroup group =
            RewriteGroup.builder().oldFragments(oldFragments).newFragments(newFragments).build();

        groups.add(group);

        // Create and commit the rewrite transaction
        Transaction rewriteTx =
            datasetWithData
                .newTransactionBuilder()
                .operation(Rewrite.builder().groups(groups).build())
                .build();

        try (Dataset rewrittenDataset = rewriteTx.commit()) {
          assertEquals(3, rewrittenDataset.version());
          // The row count should remain the same since we're just rewriting
          assertEquals(rowCount * 2, rewrittenDataset.countRows());

          // Verify that the transaction was recorded
          assertEquals(rewriteTx, rewrittenDataset.readTransaction().orElse(null));
        }
      }
    }
  }

  @Test
  void testRestore(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testRestore").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Record the initial version
      long initialVersion = dataset.version();

      // Append data to create a new version
      int rowCount = 20;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      Transaction transaction =
          dataset
              .newTransactionBuilder()
              .operation(
                  Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
              .build();
      try (Dataset modifiedDataset = transaction.commit()) {
        // Verify the dataset was modified
        long newVersion = modifiedDataset.version();
        assertEquals(initialVersion + 1, newVersion);
        assertEquals(rowCount, modifiedDataset.countRows());

        // Restore to the initial version
        Transaction restoreTransaction =
            modifiedDataset
                .newTransactionBuilder()
                .operation(new Restore.Builder().version(initialVersion).build())
                .build();
        try (Dataset restoredDataset = restoreTransaction.commit()) {
          // Verify the dataset was restored to the initial version, but the version increases
          assertEquals(initialVersion + 2, restoredDataset.version());
          // Initial dataset had 0 rows
          assertEquals(0, restoredDataset.countRows());
          assertEquals(restoreTransaction, restoredDataset.readTransaction().orElse(null));
        }
      }
    }
  }

  @Test
  void testReserveFragments(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testReserveFragments").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Create an initial fragment to establish a baseline fragment ID
      FragmentMetadata initialFragmentMeta = testDataset.createNewFragment(10);
      Transaction appendTransaction =
          dataset
              .newTransactionBuilder()
              .operation(
                  Append.builder()
                      .fragments(Collections.singletonList(initialFragmentMeta))
                      .build())
              .build();
      try (Dataset datasetWithFragment = appendTransaction.commit()) {
        // Reserve fragment IDs
        int numFragmentsToReserve = 5;
        Transaction reserveTransaction =
            datasetWithFragment
                .newTransactionBuilder()
                .operation(
                    new ReserveFragments.Builder().numFragments(numFragmentsToReserve).build())
                .build();
        try (Dataset datasetWithReservedFragments = reserveTransaction.commit()) {
          // Create a new fragment and verify its ID reflects the reservation
          FragmentMetadata newFragmentMeta = testDataset.createNewFragment(10);
          Transaction appendTransaction2 =
              datasetWithReservedFragments
                  .newTransactionBuilder()
                  .operation(
                      Append.builder()
                          .fragments(Collections.singletonList(newFragmentMeta))
                          .build())
                  .build();
          try (Dataset finalDataset = appendTransaction2.commit()) {
            // Verify the fragment IDs were properly reserved
            // The new fragment should have an ID that's at least numFragmentsToReserve higher
            // than it would have been without the reservation
            List<Fragment> fragments = finalDataset.getFragments();
            assertEquals(2, fragments.size());

            // The first fragment ID is typically 0, and the second would normally be 1
            // But after reserving 5 fragments, the second fragment ID should be at least 6
            Fragment firstFragment = fragments.get(0);
            Fragment secondFragment = fragments.get(1);

            // Check that the second fragment has a significantly higher ID than the first
            // This is an indirect way to verify that fragment IDs were reserved
            Assertions.assertNotEquals(
                firstFragment.metadata().getId() + 1, secondFragment.getId());

            // Verify the transaction is recorded
            assertEquals(
                reserveTransaction, datasetWithReservedFragments.readTransaction().orElse(null));
          }
        }
      }
    }
  }

  @Test
  void testDataReplacement(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testDataReplacement").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {

      // step 1. create a dataset with schema: id: int, name: varchar
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // step 2. create a new VectorSchemaRoot with only id values and append it to the dataset
      int rowCount = 20;
      Schema idOnlySchema =
          new Schema(
              Collections.singletonList(Field.nullable("id", new ArrowType.Int(32, true))), null);

      try (VectorSchemaRoot idRoot = VectorSchemaRoot.create(idOnlySchema, allocator)) {
        idRoot.allocateNew();
        IntVector idVector = (IntVector) idRoot.getVector("id");
        for (int i = 0; i < rowCount; i++) {
          idVector.setSafe(i, i);
        }
        idRoot.setRowCount(rowCount);

        List<FragmentMetadata> fragmentMetas =
            Fragment.create(datasetPath, allocator, idRoot, new WriteParams.Builder().build());

        Transaction appendTxn =
            dataset
                .newTransactionBuilder()
                .operation(Append.builder().fragments(fragmentMetas).build())
                .build();

        try (Dataset initDataset = appendTxn.commit()) {
          assertEquals(2, initDataset.version());
          assertEquals(rowCount, initDataset.countRows());

          // step 3. use dataset.addColumn to add a new column named as address with all null values
          Field addressField = Field.nullable("address", new ArrowType.Utf8());
          Schema addressSchema = new Schema(Collections.singletonList(addressField), null);
          initDataset.addColumns(addressSchema);

          try (LanceScanner scanner = initDataset.newScan()) {
            try (ArrowReader resultReader = scanner.scanBatches()) {
              assertTrue(resultReader.loadNextBatch());
              VectorSchemaRoot batch = resultReader.getVectorSchemaRoot();
              assertEquals(rowCount, initDataset.countRows());
              assertEquals(rowCount, batch.getRowCount());

              // verify all null values
              VarCharVector resultNameVector = (VarCharVector) batch.getVector("address");
              for (int i = 0; i < rowCount; i++) {
                Assertions.assertTrue(resultNameVector.isNull(i));
              }
            }
          }

          // step 4. use DataReplacement transaction to replace null values
          try (VectorSchemaRoot replaceVectorRoot =
              VectorSchemaRoot.create(addressSchema, allocator)) {
            replaceVectorRoot.allocateNew();
            VarCharVector addressVector = (VarCharVector) replaceVectorRoot.getVector("address");

            for (int i = 0; i < rowCount; i++) {
              String name = "District " + i;
              addressVector.setSafe(i, name.getBytes(StandardCharsets.UTF_8));
            }
            replaceVectorRoot.setRowCount(rowCount);

            DataFile datafile =
                createDataFile(dataset.allocator(), datasetPath, replaceVectorRoot, 2);
            List<DataReplacement.DataReplacementGroup> replacementGroups =
                Collections.singletonList(
                    new DataReplacement.DataReplacementGroup(
                        fragmentMetas.get(0).getId(), datafile));
            Transaction replaceTxn =
                initDataset
                    .newTransactionBuilder()
                    .operation(DataReplacement.builder().replacements(replacementGroups).build())
                    .build();

            try (Dataset datasetWithAddress = replaceTxn.commit()) {
              assertEquals(4, datasetWithAddress.version());
              assertEquals(rowCount, datasetWithAddress.countRows());

              try (LanceScanner scanner = datasetWithAddress.newScan()) {
                try (ArrowReader resultReader = scanner.scanBatches()) {
                  assertTrue(resultReader.loadNextBatch());
                  VectorSchemaRoot batch = resultReader.getVectorSchemaRoot();
                  assertEquals(rowCount, datasetWithAddress.countRows());
                  assertEquals(rowCount, batch.getRowCount());

                  // verify all address values not null
                  VarCharVector resultNameVector = (VarCharVector) batch.getVector("address");
                  for (int i = 0; i < rowCount; i++) {
                    Assertions.assertFalse(resultNameVector.isNull(i));
                    String expectedName = "District " + i;
                    String actualName = new String(resultNameVector.get(i), StandardCharsets.UTF_8);
                    assertEquals(expectedName, actualName);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  /**
   * Helper method to create a DataFile from a VectorSchemaRoot. This implementation uses
   * LanceFileWriter to ensure compatibility with Lance format.
   */
  private DataFile createDataFile(
      BufferAllocator allocator, String basePath, VectorSchemaRoot root, int fieldIndex) {
    // Create a unique file path for the data file
    String fileName = UUID.randomUUID() + ".lance";
    String filePath = basePath + "/data/" + fileName;

    // Create parent directories if they don't exist
    File file = new File(filePath);

    // Use LanceFileWriter to write the data
    try (LanceFileWriter writer = LanceFileWriter.open(filePath, allocator, null)) {
      writer.write(root);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    // Create a DataFile object with the field index
    // The fields array contains the indices of the fields in the schema
    // The columnIndices array contains the indices of the columns in the file
    // Use a stable file format version
    return new DataFile(
        fileName,
        new int[] {fieldIndex}, // Field index in the schema
        new int[] {0}, // Column index in the file (always 0 for single column)
        TEST_FILE_FORMAT_MAJOR_VERSION, // File major version
        TEST_FILE_FORMAT_MINOR_VERSION, // File minor version
        file.length() // File size in bytes (now contains actual data)
        );
  }
}
