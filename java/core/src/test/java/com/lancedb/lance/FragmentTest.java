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

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.dataset.scanner.ScanOptions;
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
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

public class FragmentTest {
  @TempDir private static Path tempDir; // Temporary directory for the tests

  @Test
  void testFragmentScannerSchema() throws IOException, URISyntaxException {
    Path path = Paths.get(DatasetTest.class.getResource("/random_access.arrow").toURI());
    try (BufferAllocator allocator = new RootAllocator();
        ArrowFileReader reader =
            new ArrowFileReader(
                new SeekableReadChannel(
                    new ByteArrayReadableSeekableByteChannel(Files.readAllBytes(path))),
                allocator);
        ArrowArrayStream arrowStream = ArrowArrayStream.allocateNew(allocator)) {
      Data.exportArrayStream(allocator, reader, arrowStream);
      Path datasetPath = tempDir.resolve("fragment_scheme");
      Dataset.write(allocator, arrowStream, datasetPath.toString(),
          new WriteParams.Builder()
              .withMaxRowsPerFile(10)
              .withMaxRowsPerGroup(20)
              .withMode(WriteParams.WriteMode.CREATE)
              .build()).close();

      try (var dataset = Dataset.open(datasetPath.toString(), allocator)) {
        assertEquals(9, dataset.countRows());
        var fragment = dataset.getFragments().get(0);

        var scanner = fragment.newScan(new ScanOptions(1024));
        var schema = scanner.schema();
        assertEquals(schema, reader.getVectorSchemaRoot().getSchema());

        try (var fragmentReader = scanner.scanBatches()) {
          var batchCount = 0;
          while (fragmentReader.loadNextBatch()) {
            fragmentReader.getVectorSchemaRoot();
            batchCount++;
          }
          assert (batchCount > 0);
        }
      }
    }
  }

  @Test
  void testFragmentCreateFfiArray() throws IOException, URISyntaxException {
    Schema schema = new Schema(Arrays.asList(
        new Field("name", FieldType.nullable(new ArrowType.Utf8()), null),
        new Field("age", FieldType.nullable(new ArrowType.Int(32, true)), null)
    ));

    try (BufferAllocator allocator = new RootAllocator();
         VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      root.allocateNew();
      // Fill data
      VarCharVector nameVector = (VarCharVector) root.getVector("name");
      IntVector ageVector = (IntVector) root.getVector("age");
      nameVector.setSafe(0, "John Doe".getBytes(StandardCharsets.UTF_8));
      ageVector.setSafe(0, 30);
      nameVector.setSafe(1, "Jane Doe".getBytes(StandardCharsets.UTF_8));
      ageVector.setSafe(1, 25);
      root.setRowCount(2);

      Path datasetPath = tempDir.resolve("new_fragment_array");
      int fragmentId = 1;
      FragmentMetadata fragmentMeta = Fragment.create(datasetPath.toString(),
          allocator, root, Optional.of(fragmentId), new WriteParams.Builder().build());
      assertEquals(fragmentId, fragmentMeta.getId());
    }
  }

  @Test
  void testFragmentCreate() {
    String datasetPath = tempDir.resolve("new_fragment").toString();
    Schema schema = new Schema(Arrays.asList(
      new Field("name", FieldType.nullable(new ArrowType.Utf8()), null),
      new Field("age", FieldType.nullable(new ArrowType.Int(32, true)), null)
    ));
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      // Create empty dataset
      try (Dataset dataset = Dataset.create(allocator, datasetPath,
          schema, new WriteParams.Builder().build())) {
        assertEquals(0, dataset.countRows());
        assertEquals(schema, dataset.getSchema());
        var fragments = dataset.getFragments();
        assertEquals(0, fragments.size());
      }
      // Create fragment
      int fragmentId = 1;
      FragmentMetadata fragmentMeta;
      int rowCount = 2;
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        root.allocateNew();
        // Fill data
        VarCharVector nameVector = (VarCharVector) root.getVector("name");
        IntVector ageVector = (IntVector) root.getVector("age");
        nameVector.setSafe(0, "John Doe".getBytes(StandardCharsets.UTF_8));
        ageVector.setSafe(0, 30);
        nameVector.setSafe(1, "Jane Doe".getBytes(StandardCharsets.UTF_8));
        ageVector.setSafe(1, 25);
        root.setRowCount(rowCount);

         fragmentMeta = Fragment.create(datasetPath,
            allocator, root, Optional.of(fragmentId), new WriteParams.Builder().build());
        assertEquals(fragmentId, fragmentMeta.getId());
      }
      
      // Commit fragment
      FragmentOperation.Append appendOp = new FragmentOperation.Append(List.of(fragmentMeta));
      try (Dataset dataset = Dataset.commit(allocator, datasetPath, appendOp, Optional.of(1))) {
        assertEquals(rowCount, dataset.countRows());
        var fragment = dataset.getFragments().get(0);
        assertEquals(fragmentId, fragment.getId());
  
        var scanner = fragment.newScan(new ScanOptions(1024));
        var schemaRes = scanner.schema();
        assertEquals(schema, schemaRes);
      }
    }
  }
}
