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

import com.lancedb.lance.ipc.FragmentScanner;
import com.lancedb.lance.ipc.ScanOptions;
import org.apache.arrow.dataset.scanner.Scanner;

/**
 * Dataset format.
 * Matching to Lance Rust FileFragment.
 * */
public class DatasetFragment {
  /** Pointer to the {@link Dataset} instance in Java. */
  private final Dataset dataset;
  private final FragmentMetadata metadata;

  /** Private constructor, calling from JNI. */
  DatasetFragment(Dataset dataset, FragmentMetadata metadata) {
    this.dataset = dataset;
    this.metadata = metadata;
  }

  /**
   * Create a new Dataset Scanner.
   *
   * @return a dataset scanner
   */
  public Scanner newScan() {
    return newScan(new com.lancedb.lance.ipc.ScanOptions.Builder().build());
  }

  /**
   * Create a new Dataset Scanner.
   *
   * @param batchSize the scan options with batch size, columns filter, and substrait
   * @return a dataset scanner
   */
  public Scanner newScan(long batchSize) {
    return newScan(new com.lancedb.lance.ipc.ScanOptions.Builder().batchSize(batchSize).build());
  }

  /**
   * Create a new Dataset Scanner.
   *
   * @param options the scan options
   * @return a dataset scanner
   */
  public Scanner newScan(ScanOptions options) {
    return new FragmentScanner(dataset, metadata.getId(), options, dataset.allocator);
  }

  private native int countRowsNative(Dataset dataset, long fragmentId);

  public int getId() {
    return metadata.getId();
  }

  /** Count rows in this Fragment. */
  public int countRows() {
    return countRowsNative(dataset, metadata.getId());
  }

  public String toString() {
    return String.format("Fragment(%s)", metadata.getJsonMetadata());
  }
}
