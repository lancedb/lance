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

public class RewriteTest extends OperationTestBase {

  @Test
  void testRewrite(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testRewrite").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // First, append some data
      int rowCount = 20;
      FragmentMetadata fragmentMeta1 = testDataset.createNewFragment(rowCount);
      FragmentMetadata fragmentMeta2 = testDataset.createNewFragment(rowCount);

      try (Transaction appendTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Append.builder().fragments(Arrays.asList(fragmentMeta1, fragmentMeta2)).build())
              .build()) {
        try (Dataset datasetWithData = new CommitBuilder(dataset).execute(appendTxn)) {
          assertEquals(2, datasetWithData.version());
          assertEquals(rowCount * 2, datasetWithData.countRows());

          // Now create a rewrite operation
          List<RewriteGroup> groups = new ArrayList<>();

          // Create a rewrite group with old fragments and new fragments
          List<FragmentMetadata> oldFragments = new ArrayList<>();
          oldFragments.add(fragmentMeta1);

          List<FragmentMetadata> newFragments = new ArrayList<>();
          FragmentMetadata newFragmentMeta = testDataset.createNewFragment(rowCount);
          newFragments.add(newFragmentMeta);

          RewriteGroup group =
              RewriteGroup.builder().oldFragments(oldFragments).newFragments(newFragments).build();

          groups.add(group);

          // Create and commit the rewrite transaction
          try (Transaction rewriteTxn =
              new Transaction.Builder()
                  .readVersion(datasetWithData.version())
                  .operation(Rewrite.builder().groups(groups).build())
                  .build()) {
            try (Dataset rewrittenDataset =
                new CommitBuilder(datasetWithData).execute(rewriteTxn)) {
              assertEquals(3, rewrittenDataset.version());
              // The row count should remain the same since we're just rewriting
              assertEquals(rowCount * 2, rewrittenDataset.countRows());
            }
          }
        }
      }
    }
  }
}
