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
package org.lance;

import org.lance.ipc.FilteredRead;
import org.lance.ipc.LanceScanner;
import org.lance.ipc.ScanOptions;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** End-to-end tests for the distributed filtered read workflow. */
public class FilteredReadTest {

  @Test
  void testBasicPlanAndExecute(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("basic_plan_execute").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      FragmentMetadata meta0 = testDataset.createNewFragment(20);
      FragmentMetadata meta1 = testDataset.createNewFragment(30);
      FragmentOperation.Append appendOp = new FragmentOperation.Append(Arrays.asList(meta0, meta1));
      try (Dataset dataset = Dataset.commit(allocator, datasetPath, appendOp, Optional.of(1L))) {
        FilteredRead plan;
        try (LanceScanner scanner = dataset.newScan()) {
          plan = FilteredRead.planFilteredRead(scanner);
        }
        assertNotNull(plan);
        assertTrue(plan.getFilteredReadExecProto().length > 0);

        List<byte[]> tasks = plan.getTasks();
        assertNotNull(tasks);
        assertTrue(tasks.size() > 0);

        int totalRows = 0;
        for (byte[] task : tasks) {
          try (ArrowReader reader = FilteredRead.executeFilteredRead(dataset, task, allocator)) {
            while (reader.loadNextBatch()) {
              totalRows += reader.getVectorSchemaRoot().getRowCount();
            }
          }
        }
        assertEquals(50, totalRows);
      }
    }
  }

  @Test
  void testPlanMetadata(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("plan_metadata").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      FragmentMetadata meta0 = testDataset.createNewFragment(15);
      FragmentMetadata meta1 = testDataset.createNewFragment(25);
      FragmentMetadata meta2 = testDataset.createNewFragment(10);
      FragmentOperation.Append appendOp =
          new FragmentOperation.Append(Arrays.asList(meta0, meta1, meta2));
      try (Dataset dataset = Dataset.commit(allocator, datasetPath, appendOp, Optional.of(1L))) {
        FilteredRead plan;
        try (LanceScanner scanner = dataset.newScan()) {
          plan = FilteredRead.planFilteredRead(scanner);
        }
        assertEquals(3, plan.getNumFragments());

        int[] fragmentIds = plan.getFragmentIds();
        assertEquals(3, fragmentIds.length);

        long[] rowsPerFragment = plan.getRowsPerFragment();
        assertEquals(3, rowsPerFragment.length);
      }
    }
  }

  @Test
  void testDistributedSplitAndExecute(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("distributed_split").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      FragmentMetadata meta0 = testDataset.createNewFragment(10);
      FragmentMetadata meta1 = testDataset.createNewFragment(20);
      FragmentMetadata meta2 = testDataset.createNewFragment(30);
      FragmentOperation.Append appendOp =
          new FragmentOperation.Append(Arrays.asList(meta0, meta1, meta2));
      try (Dataset dataset = Dataset.commit(allocator, datasetPath, appendOp, Optional.of(1L))) {
        // Plan the read and close the scanner (simulating coordinator node)
        FilteredRead plan;
        try (LanceScanner scanner = dataset.newScan()) {
          plan = FilteredRead.planFilteredRead(scanner);
        }

        // Split into per-fragment tasks (coordinator distributes to workers)
        List<byte[]> tasks = plan.getTasks();
        assertEquals(3, tasks.size());

        // Execute each task separately (simulating worker nodes)
        int distributedTotal = 0;
        for (byte[] task : tasks) {
          try (ArrowReader reader = FilteredRead.executeFilteredRead(dataset, task, allocator)) {
            while (reader.loadNextBatch()) {
              distributedTotal += reader.getVectorSchemaRoot().getRowCount();
            }
          }
        }

        // Compare with direct scan total
        int directTotal = 0;
        try (LanceScanner scanner = dataset.newScan()) {
          try (ArrowReader reader = scanner.scanBatches()) {
            while (reader.loadNextBatch()) {
              directTotal += reader.getVectorSchemaRoot().getRowCount();
            }
          }
        }

        assertEquals(60, directTotal);
        assertEquals(directTotal, distributedTotal);
      }
    }
  }

  @Test
  void testPlanWithFilter(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("plan_with_filter").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      // Write a single fragment with ids 0..39
      try (Dataset dataset = testDataset.write(1, 40)) {
        FilteredRead plan;
        try (LanceScanner scanner =
            dataset.newScan(new ScanOptions.Builder().filter("id > 10").build())) {
          plan = FilteredRead.planFilteredRead(scanner);
        }

        List<byte[]> tasks = plan.getTasks();
        assertNotNull(tasks);
        assertTrue(tasks.size() > 0);

        int totalRows = 0;
        for (byte[] task : tasks) {
          try (ArrowReader reader = FilteredRead.executeFilteredRead(dataset, task, allocator)) {
            while (reader.loadNextBatch()) {
              totalRows += reader.getVectorSchemaRoot().getRowCount();
            }
          }
        }
        // ids 11..39 = 29 rows
        assertEquals(29, totalRows);
      }
    }
  }

  @Test
  void testSerializableRoundtrip(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("serializable_roundtrip").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 40)) {
        FilteredRead plan;
        try (LanceScanner scanner = dataset.newScan()) {
          plan = FilteredRead.planFilteredRead(scanner);
        }

        List<byte[]> tasks = plan.getTasks();
        assertTrue(tasks.size() > 0);
        byte[] originalTask = tasks.get(0);

        // Serialize
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
          oos.writeObject(originalTask);
        }

        // Deserialize
        ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
        byte[] deserializedTask;
        try (ObjectInputStream ois = new ObjectInputStream(bais)) {
          deserializedTask = (byte[]) ois.readObject();
        }

        // Execute the deserialized task
        int originalRows = 0;
        try (ArrowReader reader =
            FilteredRead.executeFilteredRead(dataset, originalTask, allocator)) {
          while (reader.loadNextBatch()) {
            originalRows += reader.getVectorSchemaRoot().getRowCount();
          }
        }

        int deserializedRows = 0;
        try (ArrowReader reader =
            FilteredRead.executeFilteredRead(dataset, deserializedTask, allocator)) {
          while (reader.loadNextBatch()) {
            deserializedRows += reader.getVectorSchemaRoot().getRowCount();
          }
        }

        assertTrue(originalRows > 0);
        assertEquals(originalRows, deserializedRows);
      }
    }
  }
}
