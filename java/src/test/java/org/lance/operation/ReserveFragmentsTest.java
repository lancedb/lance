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
import org.lance.Fragment;
import org.lance.FragmentMetadata;
import org.lance.TestUtils;
import org.lance.Transaction;

import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ReserveFragmentsTest extends OperationTestBase {

  @Test
  void testReserveFragments(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testReserveFragments").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Create an initial fragment to establish a baseline fragment ID
      FragmentMetadata initialFragmentMeta = testDataset.createNewFragment(10);
      try (Transaction appendTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Append.builder()
                      .fragments(Collections.singletonList(initialFragmentMeta))
                      .build())
              .build()) {
        try (Dataset datasetWithFragment = new CommitBuilder(dataset).execute(appendTxn)) {
          // Reserve fragment IDs
          int numFragmentsToReserve = 5;
          try (Transaction reserveTxn =
              new Transaction.Builder()
                  .readVersion(datasetWithFragment.version())
                  .operation(
                      new ReserveFragments.Builder().numFragments(numFragmentsToReserve).build())
                  .build()) {
            try (Dataset datasetWithReservedFragments =
                new CommitBuilder(datasetWithFragment).execute(reserveTxn)) {
              // Create a new fragment and verify its ID reflects the reservation
              FragmentMetadata newFragmentMeta = testDataset.createNewFragment(10);
              try (Transaction appendTxn2 =
                  new Transaction.Builder()
                      .readVersion(datasetWithReservedFragments.version())
                      .operation(
                          Append.builder()
                              .fragments(Collections.singletonList(newFragmentMeta))
                              .build())
                      .build()) {
                try (Dataset finalDataset =
                    new CommitBuilder(datasetWithReservedFragments).execute(appendTxn2)) {
                  // Verify the fragment IDs were properly reserved
                  // The new fragment should have an ID that's at least numFragmentsToReserve
                  // higher than it would have been without the reservation
                  List<Fragment> fragments = finalDataset.getFragments();
                  assertEquals(2, fragments.size());

                  // The first fragment ID is typically 0, and the second would normally be 1
                  // But after reserving 5 fragments, the second fragment ID should be at least 6
                  Fragment firstFragment = fragments.get(0);
                  Fragment secondFragment = fragments.get(1);

                  // Check that the second fragment has a significantly higher ID than the first
                  // This is an indirect way to verify that fragment IDs were reserved
                  Assertions.assertNotEquals(
                      firstFragment.metadata().getId() + 1, secondFragment.getId());
                }
              }
            }
          }
        }
      }
    }
  }
}
