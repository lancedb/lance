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

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;

public class SqlQueryTest {
  private static final String NAME = "sqlquery_test_dataset";
  private static BufferAllocator allocator;
  private static Dataset dataset;

  @BeforeAll
  static void setup(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve(NAME).toString();
    allocator = new RootAllocator();
    TestUtils.SimpleTestDataset testDataset =
        new TestUtils.SimpleTestDataset(allocator, datasetPath);
    testDataset.createEmptyDataset().close();
    // write id with value from 0 to 39
    dataset = testDataset.write(1, 40);
  }

  @AfterAll
  static void tearDown() {
    // Cleanup resources used by the tests
    if (dataset != null) {
      dataset.close();
    }
    if (allocator != null) {
      allocator.close();
    }
  }

  @Test
  public void testToRecordBatches() throws IOException {
    // Test normal query
    ArrowReader reader = dataset.sql("select * from " + NAME).tableName(NAME).intoBatchRecords();
    Assertions.assertEquals(
        "Schema<id: Int(32, true), name: Utf8>",
        reader.getVectorSchemaRoot().getSchema().toString());
    int rowCount = 0;
    int totalSum = 0;
    while (reader.loadNextBatch()) {
      rowCount += reader.getVectorSchemaRoot().getRowCount();
      for (int index = 0; index < reader.getVectorSchemaRoot().getRowCount(); index++) {
        int id = (Integer) reader.getVectorSchemaRoot().getVector(0).getObject(index);
        totalSum += id;
      }
    }
    Assertions.assertEquals(40, rowCount);
    Assertions.assertEquals(780, totalSum);
    reader.close();

    // Test agg query
    reader = dataset.sql("select sum(id) from " + NAME).tableName(NAME).intoBatchRecords();
    Assertions.assertEquals(
        "Schema<sum(sqlquery_test_dataset.id): Int(64, true)>",
        reader.getVectorSchemaRoot().getSchema().toString());
    Assertions.assertTrue(reader.loadNextBatch());
    long sum = (Long) reader.getVectorSchemaRoot().getVector(0).getObject(0);
    Assertions.assertEquals(780, sum);
    reader.close();

    // Test empty result
    reader =
        dataset.sql("select * from " + NAME + " where id < 0").tableName(NAME).intoBatchRecords();
    Assertions.assertEquals(
        "Schema<id: Int(32, true), name: Utf8>",
        reader.getVectorSchemaRoot().getSchema().toString());
    rowCount = 0;
    while (reader.loadNextBatch()) {
      rowCount += reader.getVectorSchemaRoot().getRowCount();
    }
    Assertions.assertEquals(0, rowCount);
    reader.close();

    // Test withRowId and rowAddr
    reader =
        dataset
            .sql("select id, name, _rowid, _rowaddr from " + NAME)
            .tableName(NAME)
            .withRowId(true)
            .withRowAddr(true)
            .intoBatchRecords();
    Assertions.assertEquals(
        "Schema<id: Int(32, true), name: Utf8, _rowid: Int(64, false), _rowaddr: Int(64, false)>",
        reader.getVectorSchemaRoot().getSchema().toString());
    reader.close();
  }
}
