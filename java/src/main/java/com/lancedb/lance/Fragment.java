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
import java.util.Optional;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.dataset.scanner.ScanOptions;
import org.apache.arrow.dataset.scanner.Scanner;

/** Data Fragment. */
public class Fragment {
  // Only keep fragmentId for reference, so we don't need to make this
  // object to be {@link Closable} to track Rust native object.
  private final int fragmentId;

  /** Pointer to the {@link Dataset} instance in Java. */
  private final Dataset dataset;

  /** Private constructor, calling from JNI. */
  Fragment(Dataset dataset, int fragmentId) {
    this.dataset = dataset;
    this.fragmentId = fragmentId;
  }

  /** Create a fragment from the given data. */
  public static FragmentMetadata create(String datasetUri, ArrowArrayStream stream,
      Optional<Integer> fragementId, WriteParams params) {
    return new FragmentMetadata(createNative(datasetUri, stream.memoryAddress(), fragementId,
        params.getMaxRowsPerFile(), params.getMaxRowsPerGroup(),
        params.getMaxBytesPerFile(), params.getMode()));
  }

  private static native int createNative(String datasetUri, long arrowStreamMemoryAddress,
      Optional<Integer> fragmentId, Optional<Integer> maxRowsPerFile,
      Optional<Integer> maxRowsPerGroup, Optional<Long> maxBytesPerFile,
      Optional<String> mode);

  private native int countRowsNative(Dataset dataset, long fragmentId);

  public int getFragmentId() {
    return fragmentId;
  }

  public String toString() {
    return String.format("Fragment(id=%d)", fragmentId);
  }

  /** Count rows in this Fragment. */
  public int countRows() {
    return countRowsNative(dataset, fragmentId);
  }

  /** Create a new Fragment Scanner. */
  public Scanner newScan(ScanOptions options) {
    return new FragmentScanner(dataset, fragmentId, options, dataset.allocator);
  }
}
