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
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

import com.lancedb.lance.ipc.DatasetScanner;
import com.lancedb.lance.ipc.FragmentScanner;
import org.apache.arrow.dataset.scanner.ScanOptions;
import org.apache.arrow.dataset.scanner.Scanner;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

public class FragmentTest {
  @TempDir private static Path tempDir; // Temporary directory for the tests

  @Test
  void testFragmentScannerSchema() throws IOException, URISyntaxException {
    String datasetPath = tempDir.resolve("fragment_scheme").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.RandomAccessDataset testDataset = new TestUtils.RandomAccessDataset(allocator, datasetPath);
      testDataset.createDatasetAndValidate();

      try (var dataset = Dataset.open(datasetPath, allocator)) {
        var fragment = dataset.getFragments().get(0);
        var scanner = fragment.newScan(new ScanOptions(1024));
        var schema = scanner.schema();
        assertEquals(testDataset.getSchema(), schema);

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
  @Disabled
  void testFragmentScannerSubstrait() throws Exception {
    String datasetPath = tempDir.resolve("fragment_substrait").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 40)) {
        var fragment = dataset.getFragments().get(0);
        // TODO(lu): Create a valid substrait
        ByteBuffer substraitExpressionFilter = TestUtils.getSubstraitByteBuffer("");
        ScanOptions options = new ScanOptions.Builder(/*batchSize*/ 1024)
            .columns(Optional.empty())
            .substraitFilter(substraitExpressionFilter)
            .build();
        try (Scanner scanner = fragment.newScan(options)) {
          try (ArrowReader reader = scanner.scanBatches()) {
            assertEquals(dataset.getSchema().getFields(), reader.getVectorSchemaRoot().getSchema().getFields());
            int rowcount = 0;
            while (reader.loadNextBatch()) {
              rowcount += reader.getVectorSchemaRoot().getRowCount();
            }
            assertEquals(40, rowcount);
          }
        }
      }
    }
  }

  @Test
  void testFragmentCreateFfiArray() {
    String datasetPath = tempDir.resolve("new_fragment_array").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset();
      testDataset.createNewFragment(123, 20);
    }
  }

  @Test
  void testFragmentCreate() {
    String datasetPath = tempDir.resolve("new_fragment").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset();
      int fragmentId = 312;
      int rowCount = 21;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(fragmentId, rowCount);

      // Commit fragment
      FragmentOperation.Append appendOp = new FragmentOperation.Append(List.of(fragmentMeta));
      try (Dataset dataset = Dataset.commit(allocator, datasetPath, appendOp, Optional.of(1L))) {
        assertEquals(2, dataset.version());
        assertEquals(2, dataset.latestVersion());
        assertEquals(rowCount, dataset.countRows());
        var fragment = dataset.getFragments().get(0);
        assertEquals(fragmentId, fragment.getId());
  
        var scanner = fragment.newScan(new ScanOptions(1024));
        var schemaRes = scanner.schema();
        assertEquals(testDataset.getSchema(), schemaRes);
      }
    }
  }
}
