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

package com.lancedb.lance.spark.write;

import com.lancedb.lance.Dataset;
import com.lancedb.lance.WriteParams;
import com.lancedb.lance.spark.LanceConfig;
import org.apache.arrow.dataset.scanner.Scanner;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow;
import org.apache.spark.sql.connector.write.DataWriter;
import org.apache.spark.sql.connector.write.DataWriterFactory;
import org.apache.spark.sql.connector.write.PhysicalWriteInfo;
import org.apache.spark.sql.connector.write.WriterCommitMessage;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.ArrowUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInfo;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class BatchAppendTest {
  @TempDir
  static Path tempDir;

  @Test
  public void testLanceDataWriter(TestInfo testInfo) throws Exception {
    String tableName = testInfo.getTestMethod().get().getName();
    String tablePath = LanceConfig.getTablePath(tempDir.toString(), tableName);
    try (BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      // Create lance dataset
      Field field = new Field("column1", FieldType.nullable(new ArrowType.Int(32, true)), null);
      Schema schema = new Schema(Collections.singletonList(field));
      Dataset.create(allocator, tablePath, schema, new WriteParams.Builder().build()).close();

      // Append data to lance dataset
      LanceConfig config = LanceConfig.from(tablePath);
      StructType sparkSchema = ArrowUtils.fromArrowSchema(schema);
      BatchAppend batchAppend = new BatchAppend(sparkSchema, config);
      DataWriterFactory factor = batchAppend.createBatchWriterFactory(() -> 1);

      int rows = 132;
      WriterCommitMessage message;
      try (DataWriter<InternalRow> writer = factor.createWriter(0, 0)) {
        for (int i = 0; i < rows; i++) {
          InternalRow row = new GenericInternalRow(new Object[]{i});
          writer.write(row);
        }
        message = writer.commit();
      }
      batchAppend.commit(new WriterCommitMessage[]{message});

      // Validate lance dataset data
      try (Dataset dataset = Dataset.open(tablePath, allocator)) {
        try (Scanner scanner = dataset.newScan()) {
          try (ArrowReader reader = scanner.scanBatches()) {
            VectorSchemaRoot readerRoot = reader.getVectorSchemaRoot();
            int totalRowsRead = 0;
            while (reader.loadNextBatch()) {
              int batchRows = readerRoot.getRowCount();
              for (int i = 0; i < batchRows; i++) {
                int value = (int) readerRoot.getVector("column1").getObject(i);
                assertEquals(totalRowsRead + i, value);
              }
              totalRowsRead += batchRows;
            }
            assertEquals(rows, totalRowsRead);
          }
        }
      }
    }
  }
}
