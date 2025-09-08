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
import com.lancedb.lance.TestUtils;
import com.lancedb.lance.Transaction;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ProjectTest extends OperationTestBase {

  @Test
  void testProjection(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testProjection").toString();
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
}
