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
import com.lancedb.lance.Fragment;
import com.lancedb.lance.FragmentMetadata;
import com.lancedb.lance.TestUtils;
import com.lancedb.lance.Transaction;
import com.lancedb.lance.WriteParams;
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
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class DataReplacementTest extends OperationTestBase {

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
                writeLanceDataFile(
                    dataset.allocator(),
                    datasetPath,
                    replaceVectorRoot,
                    new int[] {2},
                    new int[] {0});
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
}
