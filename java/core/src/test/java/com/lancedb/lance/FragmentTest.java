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
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.dataset.scanner.ScanOptions;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.ipc.ArrowFileReader;
import org.apache.arrow.vector.ipc.SeekableReadChannel;
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
      Dataset.write(
          arrowStream,
          datasetPath.toString(),
          new WriteParams.Builder()
              .withMaxRowsPerFile(10)
              .withMaxRowsPerGroup(20)
              .withMode(WriteParams.WriteMode.CREATE)
              .build());

      var dataset = Dataset.open(datasetPath.toString(), allocator);
      assertEquals(9, dataset.countRows());
      var fragment = dataset.getFragments().get(0);

      var scanner = fragment.newScan(new ScanOptions(1024));
      var schema = scanner.schema();
      assertEquals(schema, reader.getVectorSchemaRoot().getSchema());

      try (var fragmentReader = scanner.scanBatches()) {
        var batchCount = 0;
        while (fragmentReader.loadNextBatch()) {
          fragmentReader.getVectorSchemaRoot();
          batchCount ++;
        }
        assert(batchCount > 0);
      }

    }
  }

  @Test
  void testFragementCreate() throws IOException, URISyntaxException {
    Path path = Paths.get(DatasetTest.class.getResource("/random_access.arrow").toURI());
    try (BufferAllocator allocator = new RootAllocator();
        ArrowFileReader reader =
            new ArrowFileReader(
                new SeekableReadChannel(
                    new ByteArrayReadableSeekableByteChannel(Files.readAllBytes(path))),
                allocator);
        ArrowArrayStream arrowStream = ArrowArrayStream.allocateNew(allocator)) {
      Data.exportArrayStream(allocator, reader, arrowStream);
      Path datasetPath = tempDir.resolve("new_fragment");
      int fragmentId = 1;
      FragmentMetadata fragmentMeta = Fragment.create(datasetPath.toString(), arrowStream, Optional.of(fragmentId), new WriteParams.Builder().build());
      assertEquals(fragmentId, fragmentMeta.getFragementId());
    }
  }
}
