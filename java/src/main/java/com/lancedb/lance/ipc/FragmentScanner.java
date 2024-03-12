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

package com.lancedb.lance.ipc;

import com.lancedb.lance.Dataset;
import org.apache.arrow.dataset.scanner.ScanOptions;
import org.apache.arrow.dataset.scanner.ScanTask;
import org.apache.arrow.dataset.scanner.Scanner;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;

/** Scanner over a Fragment. */
public class FragmentScanner implements Scanner {
  private final Dataset dataset;
  private final int fragmentId;

  private final ScanOptions options;

  private final BufferAllocator allocator;

  /** Create FragmentScanner. */
  public FragmentScanner(
      Dataset dataset, int fragmentId, ScanOptions options, BufferAllocator allocator) {
    this.dataset = dataset;
    this.fragmentId = fragmentId;
    this.options = options;
    this.allocator = allocator;
  }

  private static native long getSchema(Dataset dataset, int fragmentId);

  @Override
  public ArrowReader scanBatches() {
    return new FragmentArrowReader(allocator);
  }

  @Override
  public Iterable<? extends ScanTask> scan() {
    // Marked as deprecated in Scanner
    throw new UnsupportedOperationException();
  }

  /** Get the schema of the Scanner. */
  @Override
  public Schema schema() {
    long address = getSchema(dataset, fragmentId);
    System.out.println("Schema address: " + address);
    //    var schema = ArrowSchema.wrap(address);
    //    return Data.importSchema(allocator, schema, null);
    return null;
  }

  @Override
  public void close() throws Exception {}
}
