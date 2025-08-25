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
package com.lancedb.lance.operation;

import com.lancedb.lance.Dataset;
import com.lancedb.lance.FragmentMetadata;
import com.lancedb.lance.TestUtils;
import com.lancedb.lance.Transaction;

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

      Transaction appendTx =
          dataset
              .newTransactionBuilder()
              .operation(
                  Append.builder().fragments(Arrays.asList(fragmentMeta1, fragmentMeta2)).build())
              .build();

      try (Dataset datasetWithData = appendTx.commit()) {
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
        Transaction rewriteTx =
            datasetWithData
                .newTransactionBuilder()
                .operation(Rewrite.builder().groups(groups).build())
                .build();

        try (Dataset rewrittenDataset = rewriteTx.commit()) {
          assertEquals(3, rewrittenDataset.version());
          // The row count should remain the same since we're just rewriting
          assertEquals(rowCount * 2, rewrittenDataset.countRows());

          // Verify that the transaction was recorded
          assertEquals(rewriteTx, rewrittenDataset.readTransaction().orElse(null));
        }
      }
    }
  }
}
