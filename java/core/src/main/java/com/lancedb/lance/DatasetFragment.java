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

import com.lancedb.lance.ipc.LanceScanner;
import com.lancedb.lance.ipc.ScanOptions;
import java.util.List;
import org.apache.arrow.util.Preconditions;

/**
 * Dataset format.
 * Matching to Lance Rust FileFragment.
 */
public class DatasetFragment {
  /** Pointer to the {@link Dataset} instance in Java. */
  private final Dataset dataset;
  private final FragmentMetadata metadata;

  /** Private constructor, calling from JNI. */
  DatasetFragment(Dataset dataset, FragmentMetadata metadata) {
    Preconditions.checkNotNull(dataset);
    Preconditions.checkNotNull(metadata);
    this.dataset = dataset;
    this.metadata = metadata;
  }

  /**
   * Create a new Dataset Scanner.
   *
   * @return a dataset scanner
   */
  public LanceScanner newScan() {
    Preconditions.checkState(!dataset.closed(), "Dataset is closed");
    return LanceScanner.create(dataset, new ScanOptions.Builder()
        .fragmentIds(List.of(metadata.getId())).build(), dataset.allocator);
  }

  /**
   * Create a new Dataset Scanner.
   *
   * @param batchSize scan batch size
   * @return a dataset scanner
   */
  public LanceScanner newScan(long batchSize) {
    return LanceScanner.create(dataset,
        new ScanOptions.Builder()
            .fragmentIds(List.of(metadata.getId())).batchSize(batchSize).build(),
        dataset.allocator);
  }

  /**
   * Create a new Dataset Scanner.
   *
   * @param options the scan options
   * @return a dataset scanner
   */
  public LanceScanner newScan(ScanOptions options) {
    Preconditions.checkNotNull(options);
    return LanceScanner.create(dataset,
        new ScanOptions.Builder(options).fragmentIds(List.of(metadata.getId())).build(),
        dataset.allocator);
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
