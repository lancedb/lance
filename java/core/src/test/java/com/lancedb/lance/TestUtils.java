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

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowFileReader;
import org.apache.arrow.vector.ipc.SeekableReadChannel;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.arrow.vector.util.ByteArrayReadableSeekableByteChannel;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class TestUtils {
  public static class SimpleTestDataset {
    private final Schema schema = new Schema(Arrays.asList(
        new Field("name", FieldType.nullable(new ArrowType.Utf8()), null),
        new Field("age", FieldType.nullable(new ArrowType.Int(32, true)), null)
    ));
    private final BufferAllocator allocator;
    private final String datasetPath;

    public SimpleTestDataset(BufferAllocator allocator, String datasetPath) {
      this.allocator = allocator;
      this.datasetPath = datasetPath;
    }
    
    public Schema getSchema() {
      return schema;
    }
    
    public void createEmptyDataset() {
      try (Dataset dataset = Dataset.create(allocator, datasetPath,
          schema, new WriteParams.Builder().build())) {
        assertEquals(0, dataset.countRows());
        assertEquals(schema, dataset.getSchema());
        var fragments = dataset.getFragments();
        assertEquals(0, fragments.size());
      }
    }

    public FragmentMetadata createNewFragment(int fragmentId, int rowCount) {
      FragmentMetadata fragmentMeta;
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        root.allocateNew();
        // Fill data
        VarCharVector nameVector = (VarCharVector) root.getVector("name");
        IntVector ageVector = (IntVector) root.getVector("age");
        for (int i = 0; i < rowCount; i++) {
          String name = "Person " + i;
          int age = 20 + i;  // Example age increment
          nameVector.setSafe(i, name.getBytes(StandardCharsets.UTF_8));
          ageVector.setSafe(i, age);
        }
        root.setRowCount(rowCount);

        fragmentMeta = Fragment.create(datasetPath,
            allocator, root, Optional.of(fragmentId), new WriteParams.Builder().build());
        assertEquals(fragmentId, fragmentMeta.getId());
        assertEquals(rowCount, fragmentMeta.getPhysicalRows());
      }
      return fragmentMeta;
    }
  }

  public static class RandomAccessDataset {
    private static final String DATA_FILE = "/random_access.arrow";
    private static final int ROW_COUNT = 9;
    private final BufferAllocator allocator;
    private final String datasetPath;
    private Schema schema;
    
    public RandomAccessDataset(BufferAllocator allocator, String datasetPath) {
      this.allocator = allocator;
      this.datasetPath = datasetPath;
    }
    
    public void createDatasetAndValidate() throws IOException, URISyntaxException {
      Path path = Paths.get(DatasetTest.class.getResource(DATA_FILE).toURI());
      try (BufferAllocator allocator = new RootAllocator();
           ArrowFileReader reader =
               new ArrowFileReader(
                   new SeekableReadChannel(
                       new ByteArrayReadableSeekableByteChannel(Files.readAllBytes(path))),
                   allocator);
           ArrowArrayStream arrowStream = ArrowArrayStream.allocateNew(allocator)) {
        Data.exportArrayStream(allocator, reader, arrowStream);
        try (Dataset dataset = Dataset.write(
            allocator,
            arrowStream,
            datasetPath,
            new WriteParams.Builder()
                .withMaxRowsPerFile(10)
                .withMaxRowsPerGroup(20)
                .withMode(WriteParams.WriteMode.CREATE)
                .build())) {
          assertEquals(ROW_COUNT, dataset.countRows());
          schema = reader.getVectorSchemaRoot().getSchema();
          validateFragments(dataset);
        }
      }
    }

    public void openDatasetAndValidate() throws IOException {
      try (Dataset datasetRead = Dataset.open(datasetPath, allocator)) {
        assertEquals(ROW_COUNT, datasetRead.countRows());
        validateFragments(datasetRead);
      }
    }

    public Schema getSchema() {
      assertNotNull(schema);
      return schema;
    }

    private void validateFragments(Dataset dataset) {
      assertNotNull(schema);
      assertNotNull(dataset);
      var fragments = dataset.getFragments();
      assertEquals(1, fragments.size());
      assertEquals(0, fragments.get(0).getId());
      assertEquals(9, fragments.get(0).countRows());
      assertEquals(schema, dataset.getSchema());
    }
  }
}
