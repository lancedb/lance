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
package com.lancedb.lance.operation;

import com.lancedb.lance.Dataset;
import com.lancedb.lance.FragmentMetadata;
import com.lancedb.lance.TestUtils;
import com.lancedb.lance.Transaction;
import com.lancedb.lance.fragment.DataFile;
import com.lancedb.lance.ipc.LanceScanner;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;

public class MergeTest extends OperationTestBase {

  @Test
  void testMergeNewColumn(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testMergeNewColumn").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      int rowCount = 15;
      try (Dataset initialDataset = createAndAppendRows(testDataset, 15)) {
        // Add a new column with different data type
        Field ageField = Field.nullable("age", new ArrowType.Int(32, true));
        Schema evolvedSchema =
            new Schema(
                Arrays.asList(
                    Field.nullable("id", new ArrowType.Int(32, true)),
                    Field.nullable("name", new ArrowType.Utf8()),
                    Field.nullable("age", new ArrowType.Int(32, true))),
                null);

        try (VectorSchemaRoot ageRoot =
            VectorSchemaRoot.create(
                new Schema(Collections.singletonList(ageField), null), allocator)) {
          ageRoot.allocateNew();
          IntVector ageVector = (IntVector) ageRoot.getVector("age");

          for (int i = 0; i < rowCount; i++) {
            ageVector.setSafe(i, 20 + i);
          }
          ageRoot.setRowCount(rowCount);

          DataFile ageDataFile =
              writeLanceDataFile(
                  dataset.allocator(), datasetPath, ageRoot, 2 // field index for age column
                  );

          FragmentMetadata fragmentMeta = initialDataset.getFragment(0).metadata();
          FragmentMetadata evolvedFragment =
              new FragmentMetadata(
                  fragmentMeta.getId(),
                  Collections.singletonList(ageDataFile),
                  fragmentMeta.getPhysicalRows(),
                  fragmentMeta.getDeletionFile(),
                  fragmentMeta.getRowIdMeta());

          Transaction mergeTransaction =
              initialDataset
                  .newTransactionBuilder()
                  .operation(
                      Merge.builder()
                          .fragments(Collections.singletonList(evolvedFragment))
                          .schema(evolvedSchema)
                          .build())
                  .build();

          try (Dataset evolvedDataset = mergeTransaction.commit()) {
            Assertions.assertEquals(3, evolvedDataset.version());
            Assertions.assertEquals(rowCount, evolvedDataset.countRows());
            Assertions.assertEquals(evolvedSchema, evolvedDataset.getSchema());
            Assertions.assertEquals(3, evolvedDataset.getSchema().getFields().size());
            // Verify merged data
            try (LanceScanner scanner = evolvedDataset.newScan()) {
              try (ArrowReader resultReader = scanner.scanBatches()) {
                Assertions.assertTrue(resultReader.loadNextBatch());
                VectorSchemaRoot batch = resultReader.getVectorSchemaRoot();
                Assertions.assertEquals(rowCount, batch.getRowCount());
                Assertions.assertEquals(3, batch.getSchema().getFields().size());
                // Verify age column
                IntVector ageResultVector = (IntVector) batch.getVector("age");
                for (int i = 0; i < rowCount; i++) {
                  Assertions.assertEquals(20 + i, ageResultVector.get(i));
                }
              }
            }
          }
        }
      }
    }
  }

  @Test
  void testMergeExistingColumn(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testMergeExistingColumn").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      // Test merging with existing column updates
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      int rowCount = 10;
      try (Dataset initialDataset = createAndAppendRows(testDataset, rowCount)) {
        // Create updated name column data
        Field nameField = Field.nullable("name", new ArrowType.Utf8());
        Schema nameSchema = new Schema(Collections.singletonList(nameField), null);

        try (VectorSchemaRoot updatedNameRoot = VectorSchemaRoot.create(nameSchema, allocator)) {
          updatedNameRoot.allocateNew();
          VarCharVector nameVector = (VarCharVector) updatedNameRoot.getVector("name");

          for (int i = 0; i < rowCount; i++) {
            String updatedName = "UpdatedName_" + i;
            nameVector.setSafe(i, updatedName.getBytes(StandardCharsets.UTF_8));
          }
          updatedNameRoot.setRowCount(rowCount);

          // Create DataFile for updated column
          DataFile updatedNameDataFile =
              writeLanceDataFile(
                  dataset.allocator(),
                  datasetPath,
                  updatedNameRoot,
                  1 // field index for name column
                  );

          // Perform merge with updated column
          FragmentMetadata fragmentMeta = initialDataset.getFragment(0).metadata();
          FragmentMetadata updatedFragment =
              new FragmentMetadata(
                  fragmentMeta.getId(),
                  Collections.singletonList(updatedNameDataFile),
                  fragmentMeta.getPhysicalRows(),
                  fragmentMeta.getDeletionFile(),
                  fragmentMeta.getRowIdMeta());

          Transaction mergeTransaction =
              initialDataset
                  .newTransactionBuilder()
                  .operation(
                      Merge.builder()
                          .fragments(Collections.singletonList(updatedFragment))
                          .schema(testDataset.getSchema())
                          .build())
                  .build();

          try (Dataset mergedDataset = mergeTransaction.commit()) {
            Assertions.assertEquals(3, mergedDataset.version());
            Assertions.assertEquals(rowCount, mergedDataset.countRows());

            // Verify updated data
            try (LanceScanner scanner = mergedDataset.newScan()) {
              try (ArrowReader resultReader = scanner.scanBatches()) {
                Assertions.assertTrue(resultReader.loadNextBatch());
                VectorSchemaRoot batch = resultReader.getVectorSchemaRoot();

                VarCharVector nameResultVector = (VarCharVector) batch.getVector("name");
                for (int i = 0; i < rowCount; i++) {
                  String expectedName = "UpdatedName_" + i;
                  String actualName = new String(nameResultVector.get(i), StandardCharsets.UTF_8);
                  Assertions.assertEquals(expectedName, actualName);
                }
              }
            }
          }
        }
      }
    }
  }
}
