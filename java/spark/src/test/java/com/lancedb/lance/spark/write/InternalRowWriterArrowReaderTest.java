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

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.VectorUnloader;
import org.apache.arrow.vector.ipc.message.ArrowRecordBatch;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.Collections;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class InternalRowWriterArrowReaderTest {

  @Test
  public void testInternalRowWriterArrowReader() throws InterruptedException, IOException {
    try (BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      Field field = new Field("column1", FieldType.nullable(org.apache.arrow.vector.types.Types.MinorType.INT.getType()), null);
      Schema schema = new Schema(Collections.singletonList(field));

      final int totalRows = 125;
      final int batchSize = 34;
      final InternalRowWriterArrowReader arrowReader = new InternalRowWriterArrowReader(allocator, schema, batchSize);

      AtomicInteger rowsWritten = new AtomicInteger(0);
      AtomicInteger rowsRead = new AtomicInteger(0);
      AtomicLong expectedBytesRead = new AtomicLong(0);

      Thread writerThread = new Thread(() -> {
        try {
          for (int i = 0; i < totalRows; i++) {
            InternalRow row = new GenericInternalRow(new Object[]{rowsWritten.incrementAndGet()});
            arrowReader.write(row);
            Thread.sleep(1);
          }
          arrowReader.setFinished();
        } catch (Exception e) {
          e.printStackTrace();
        }
      });

      Thread readerThread = new Thread(() -> {
        try {
          while (arrowReader.loadNextBatch()) {
            VectorSchemaRoot root = arrowReader.getVectorSchemaRoot();
            int rowCount = root.getRowCount();
            rowsRead.addAndGet(rowCount);
            try (ArrowRecordBatch recordBatch = new VectorUnloader(root).getRecordBatch()) {
              expectedBytesRead.addAndGet(recordBatch.computeBodyLength());
            }
            for (int i = 0; i < rowCount; i++) {
              int value = (int) root.getVector("column1").getObject(i);
              assertEquals(value, rowsRead.get() - rowCount + i + 1);
            }
          }
        } catch (Exception e) {
          e.printStackTrace();
        }
      });

      writerThread.start();
      readerThread.start();

      writerThread.join();
      readerThread.join();
      
      int expectedRowsWritten = rowsWritten.get();
      int expectedRowsRead = rowsRead.get();

      assertEquals(totalRows, expectedRowsWritten);
      assertEquals(totalRows, expectedRowsRead);
      assertEquals(expectedBytesRead.get(), arrowReader.bytesRead());
      arrowReader.close();
    }
  }
}
