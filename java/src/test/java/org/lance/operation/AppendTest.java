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
package org.lance.operation;

import org.lance.CommitBuilder;
import org.lance.Dataset;
import org.lance.FragmentMetadata;
import org.lance.TestUtils;
import org.lance.Transaction;

import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class AppendTest extends OperationTestBase {

  @Test
  void testAppendSingleFragment(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testAppendSingleFragment").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      int rowCount = 10;
      try (Dataset result = createAndAppendRows(testDataset, rowCount)) {
        assertEquals(2, result.version());
        assertEquals(rowCount, result.countRows());
      }
    }
  }

  @Test
  void testAppendMultipleFragments(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testAppendMultipleFragments").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      int rowCount = 10;
      List<FragmentMetadata> fragments =
          Arrays.asList(
              testDataset.createNewFragment(rowCount),
              testDataset.createNewFragment(rowCount),
              testDataset.createNewFragment(rowCount));

      try (Transaction txn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(Append.builder().fragments(fragments).build())
              .build()) {
        try (Dataset dataset = new CommitBuilder(this.dataset).execute(txn)) {
          assertEquals(2, dataset.version());
          assertEquals(rowCount * 3, dataset.countRows());
          assertEquals(3, dataset.getFragments().size());
        }
      }
    }
  }

  @Test
  void testAppendEmptyFragmentList(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testAppendEmptyFragmentList").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      try (Dataset dataset = testDataset.createEmptyDataset()) {
        assertThrows(
            IllegalArgumentException.class,
            () -> {
              try (Transaction txn =
                  new Transaction.Builder()
                      .readVersion(dataset.version())
                      .operation(Append.builder().fragments(new ArrayList<>()).build())
                      .build()) {
                new CommitBuilder(dataset).execute(txn).close();
              }
            });
      }
    }
  }
}
