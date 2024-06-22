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

import com.lancedb.lance.FragmentMetadata;
import com.lancedb.lance.spark.LanceConfig;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.ArrowUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInfo;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class LanceDataWriterTest {
  @TempDir
  static Path tempDir;

  @Test
  public void testLanceDataWriter(TestInfo testInfo) throws IOException {
    String tableName = testInfo.getTestMethod().get().getName();
    try (BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      Field field = new Field("column1", FieldType.nullable(new ArrowType.Int(32, true)), null);
      Schema schema = new Schema(Collections.singletonList(field));
      LanceConfig config = LanceConfig.from(tempDir.resolve(tableName + LanceConfig.LANCE_FILE_SUFFIX).toString());
      StructType sparkSchema = ArrowUtils.fromArrowSchema(schema);
      LanceDataWriter.WriterFactory writerFactory = new LanceDataWriter.WriterFactory(sparkSchema, config);
      LanceDataWriter dataWriter = (LanceDataWriter) writerFactory.createWriter(0, 0);

      int rows = 132;
      for (int i = 0; i < rows; i++) {
        InternalRow row = new GenericInternalRow(new Object[]{i});
        dataWriter.write(row);
      }

      BatchAppend.TaskCommit commitMessage = (BatchAppend.TaskCommit) dataWriter.commit();
      dataWriter.close();
      List<FragmentMetadata> fragments = commitMessage.getFragments();
      assertEquals(1, fragments.size());
      assertEquals(rows, fragments.get(0).getPhysicalRows());
    }
  }
}
