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

package com.lancedb.lance.spark.read;

import com.lancedb.lance.spark.TestUtils;
import com.lancedb.lance.spark.internal.LanceFragmentColumnarBatchScanner;

import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.vectorized.ColumnarBatch;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class LanceFragmentColumnarBatchScannerTest {

  @Test
  public void scanner() throws IOException {
    List<List<Long>> expectedValues = TestUtils.TestTable1Config.expectedValues;
    int rowIndex = 0;
    int fragmentId = 0;
    while (fragmentId <= 1) {
      try (LanceFragmentColumnarBatchScanner scanner =
          LanceFragmentColumnarBatchScanner.create(
              fragmentId, TestUtils.TestTable1Config.inputPartition)) {
        while (scanner.loadNextBatch()) {
          try (ColumnarBatch batch = scanner.getCurrentBatch()) {
            Iterator<InternalRow> rows = batch.rowIterator();
            while (rows.hasNext()) {
              InternalRow row = rows.next();
              assertNotNull(row);
              for (int colIndex = 0; colIndex < row.numFields(); colIndex++) {
                long actualValue = row.getLong(colIndex);
                long expectedValue = expectedValues.get(rowIndex).get(colIndex);
                assertEquals(
                    expectedValue,
                    actualValue,
                    "Mismatch at row " + rowIndex + " column " + colIndex);
              }
              rowIndex++;
            }
          }
        }
      }
      fragmentId++;
    }
    assertEquals(4, rowIndex);
  }
}
