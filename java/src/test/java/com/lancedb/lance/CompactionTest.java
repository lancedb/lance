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

import com.lancedb.lance.compaction.Compaction;
import com.lancedb.lance.compaction.CompactionMetrics;
import com.lancedb.lance.compaction.CompactionOptions;
import com.lancedb.lance.compaction.CompactionPlan;
import com.lancedb.lance.compaction.CompactionTask;
import com.lancedb.lance.compaction.RewriteResult;

import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Path;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Add test for distributed compaction. */
public class CompactionTest {
  @Test
  public void testBasicCompaction(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("test_dataset_for_compaction").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      testDataset.createEmptyDataset().close();

      // Step-1: write two fragments
      testDataset.write(1, 10).close();
      try (Dataset dataset = testDataset.write(2, 10)) {
        CompactionOptions compactionOptions =
            CompactionOptions.builder().withTargetRowsPerFragment(100).withNumThreads(1).build();
        CompactionPlan compactionPlan = Compaction.planCompaction(dataset, compactionOptions);

        // will plan to compact two fragments into one.
        assertEquals(1, compactionPlan.getCompactionTasks().size());
        CompactionTask task = compactionPlan.getCompactionTasks().get(0);
        assertEquals(2, task.getTaskData().getFragments().size());

        // Step-2: individually execute single task

        // mock network transferring
        task = serializeAndDeserialize(task);
        RewriteResult result = task.execute(dataset);
        CompactionMetrics metrics = result.getMetrics();
        // remove previous fragments and add new single fragment
        assertEquals(2, metrics.getFragmentsRemoved());
        assertEquals(1, metrics.getFragmentsAdded());

        // Step-3: commit the RewriteResults

        // mock network transferring
        result = serializeAndDeserialize(result);
        CompactionMetrics ignored =
            Compaction.commitCompaction(
                dataset, Collections.singletonList(result), compactionPlan.getCompactionOptions());

        // checkout to the latest snapshot and verify row num and fragment num.
        dataset.checkoutLatest();
        assertEquals(1, dataset.getFragments().size());
        assertEquals(20, dataset.getFragments().get(0).countRows());
      }
    }
  }

  @Test
  public void testDeletionCompaction(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("test_dataset_for_compaction").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      // Step-1: write two fragments
      testDataset.write(1, 10).close();
      try (Dataset dataset = testDataset.write(2, 10)) {
        dataset.delete("_rowid <= 8");

        dataset.checkoutLatest();
        // still 2 fragments
        assertEquals(2, dataset.getFragments().size());

        CompactionOptions compactionOptions =
            CompactionOptions.builder()
                .withMaterializeDeletions(true)
                .withMaterializeDeletionsThreshold(0.5f)
                .withNumThreads(1)
                .build();
        CompactionPlan compactionPlan = Compaction.planCompaction(dataset, compactionOptions);

        assertEquals(1, compactionPlan.getCompactionTasks().size());

        CompactionTask task = compactionPlan.getCompactionTasks().get(0);

        task = serializeAndDeserialize(task);
        RewriteResult result = task.execute(dataset);
        assertEquals(2, result.getMetrics().getFragmentsRemoved());
        assertEquals(1, result.getMetrics().getFragmentsAdded());

        result = serializeAndDeserialize(result);
        CompactionMetrics ignored =
            Compaction.commitCompaction(
                dataset, Collections.singletonList(result), compactionPlan.getCompactionOptions());

        // checkout to the latest snapshot and verify row num and fragment num.
        dataset.checkoutLatest();
        assertEquals(1, dataset.getFragments().size());
        assertEquals(11, dataset.getFragments().get(0).countRows());
      }
    }
  }

  private static <T> T serializeAndDeserialize(T object)
      throws IOException, ClassNotFoundException {
    ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
    try (ObjectOutputStream out = new ObjectOutputStream(outputStream)) {
      out.writeObject(object);
    }
    byte[] serialized = outputStream.toByteArray();
    ByteArrayInputStream inputStream = new ByteArrayInputStream(serialized);
    try (ObjectInputStream in = new ObjectInputStream(inputStream)) {
      @SuppressWarnings("unchecked")
      T deserialized = (T) in.readObject();
      return deserialized;
    }
  }
}
