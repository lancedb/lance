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

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.lancedb.lance.WriteParams.WriteMode;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.ipc.ArrowFileReader;
import org.apache.arrow.vector.ipc.SeekableReadChannel;
import org.apache.arrow.vector.util.ByteArrayReadableSeekableByteChannel;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

public class DatasetTest {

  @TempDir static Path tempDir; // Temporary directory for the tests
  private static Dataset dataset;

  @BeforeAll
  static void setup() {}

  @AfterAll
  static void tearDown() {
    // Cleanup resources used by the tests
    if (dataset != null) {
      dataset.close();
    }
  }

  @Test
  void testWriteStreamAndOpenPath() throws URISyntaxException, IOException {
    Path path = Paths.get(DatasetTest.class.getResource("/random_access.arrow").toURI());
    try (BufferAllocator allocator = new RootAllocator();
        ArrowFileReader reader =
            new ArrowFileReader(
                new SeekableReadChannel(
                    new ByteArrayReadableSeekableByteChannel(Files.readAllBytes(path))),
                allocator);
        ArrowArrayStream arrowStream = ArrowArrayStream.allocateNew(allocator)) {
      Data.exportArrayStream(allocator, reader, arrowStream);
      Path datasetPath = tempDir.resolve("new_dataset");
      assertDoesNotThrow(
          () -> {
            dataset =
                Dataset.write(
                    arrowStream,
                    datasetPath.toString(),
                    new WriteParams.Builder()
                        .withMaxRowsPerFile(10)
                        .withMaxRowsPerGroup(20)
                        .withMode(WriteMode.CREATE)
                        .build());
            assertEquals(9, dataset.countRows());
            Dataset datasetRead = Dataset.open(datasetPath.toString(), allocator);
            assertEquals(9, datasetRead.countRows());
          });

      var fragments = dataset.getFragments();
      assertEquals(1, fragments.size());
      assertEquals(0, fragments.get(0).getFragmentId());
      assertEquals(9, fragments.get(0).countRows());
    }
  }

  @Test
  void testOpenInvalidPath() {
    String validPath = tempDir.resolve("Invalid_dataset").toString();
    assertThrows(
        RuntimeException.class,
        () -> {
          dataset = Dataset.open(validPath, new RootAllocator());
        });
  }
}
