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

import org.lance.ipc.AsyncScanner;
import org.lance.ipc.ScanOptions;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/**
 * Helper loaded by the isolated {@link java.net.URLClassLoader} in the forked JVM.
 *
 * <p>This class has normal lance imports. It is <b>not</b> on the system classpath of the forked
 * JVM — the system classloader cannot find it (or any other {@code org.lance.*} class). Only the
 * isolated classloader can.
 *
 * <p>If the JNI dispatcher thread tries to {@code FindClass("org/lance/ipc/AsyncScanner")} after
 * {@code attach_current_thread_permanently()}, it will use the system classloader and fail —
 * reproducing the bug. With the fix, the class is pre-resolved in {@code JNI_OnLoad} (which runs on
 * a thread with the correct classloader) and passed as a {@code GlobalRef}.
 */
public class ClassloaderBugHelper {

  /**
   * Entry point invoked reflectively from {@link ClassloaderBugBootstrap}.
   *
   * @param tempDir temporary directory for the lance dataset
   */
  public static void run(String tempDir) throws Exception {
    String datasetPath = tempDir + "/classloader_bug_test";

    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 40;

      try (Dataset dataset = testDataset.write(1, totalRows)) {
        ScanOptions options = new ScanOptions.Builder().batchSize(20L).build();

        try (AsyncScanner scanner = AsyncScanner.create(dataset, options, allocator)) {
          CompletableFuture<ArrowReader> future = scanner.scanBatchesAsync();

          // Use a generous timeout — the bug causes the dispatcher thread to panic,
          // so the future will never complete.
          ArrowReader reader = future.get(30, TimeUnit.SECONDS);

          int rowCount = 0;
          while (reader.loadNextBatch()) {
            VectorSchemaRoot root = reader.getVectorSchemaRoot();
            rowCount += root.getRowCount();
          }
          reader.close();

          if (rowCount != totalRows) {
            throw new AssertionError("Expected " + totalRows + " rows but got " + rowCount);
          }
        }
      }
    }

    // Signal success to the parent process
    System.out.println("SUCCESS");
  }
}
