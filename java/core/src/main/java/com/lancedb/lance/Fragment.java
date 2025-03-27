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

import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/** Fragment operations. */
public class Fragment {
  static {
    JniLoader.ensureLoaded();
  }

  /** Pointer to the {@link Dataset} instance in Java. */
  private final Dataset dataset;

  private final FragmentMetadata fragment;

  public Fragment(Dataset dataset, int fragmentId) {
    Preconditions.checkNotNull(dataset);
    this.dataset = dataset;
    this.fragment = dataset.getFragment(fragmentId).fragment;
  }

  public Fragment(Dataset dataset, FragmentMetadata fragment) {
    Preconditions.checkNotNull(dataset);
    Preconditions.checkNotNull(fragment);
    this.dataset = dataset;
    this.fragment = fragment;
  }

  /**
   * Create a new Dataset Scanner.
   *
   * @return a dataset scanner
   */
  public LanceScanner newScan() {
    return LanceScanner.create(
        dataset,
        new ScanOptions.Builder().fragmentIds(Arrays.asList(fragment.getId())).build(),
        dataset.allocator);
  }

  /**
   * Create a new Dataset Scanner.
   *
   * @param batchSize scan batch size
   * @return a dataset scanner
   */
  public LanceScanner newScan(long batchSize) {
    return LanceScanner.create(
        dataset,
        new ScanOptions.Builder()
            .fragmentIds(Arrays.asList(fragment.getId()))
            .batchSize(batchSize)
            .build(),
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
    return LanceScanner.create(
        dataset,
        new ScanOptions.Builder(options).fragmentIds(Arrays.asList(fragment.getId())).build(),
        dataset.allocator);
  }

  private native int countRowsNative(Dataset dataset, long fragmentId);

  public int getId() {
    return fragment.getId();
  }

  /** @return row counts in this Fragment */
  public int countRows() {
    return countRowsNative(dataset, fragment.getId());
  }

  public FragmentMetadata getFragment() {
    return fragment;
  }

  public Dataset getDataset() {
    return dataset;
  }

  public Pair<FragmentMetadata, Schema> merge(
      BufferAllocator allocator, ArrowReader reader, String leftOn, String rightOn) {
    try (ArrowSchema ffiArrowSchema = ArrowSchema.allocateNew(allocator);
        ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
      if (rightOn == null) {
        rightOn = leftOn;
      }
      int maxFieldId = this.dataset.getMaxFieldId();
      Data.exportArrayStream(allocator, reader, stream);
      FragmentMetadata fragmentMetadata =
          mergeNative(
              ffiArrowSchema.memoryAddress(), stream.memoryAddress(), leftOn, rightOn, maxFieldId);
      return ImmutablePair.of(fragmentMetadata, Data.importSchema(allocator, ffiArrowSchema, null));
    }
  }

  public native FragmentMetadata mergeNative(
      long arrowSchemaMemoryAddress,
      long arrowStreamMemoryAddress,
      String leftOn,
      String rightOn,
      long maxFieldId);
  /**
   * Create a fragment from the given data.
   *
   * @param datasetUri the dataset uri
   * @param allocator the buffer allocator
   * @param root the vector schema root
   * @param params the write params
   * @return the fragment metadata
   */
  public static List<FragmentMetadata> create(
      String datasetUri, BufferAllocator allocator, VectorSchemaRoot root, WriteParams params) {
    Preconditions.checkNotNull(datasetUri);
    Preconditions.checkNotNull(allocator);
    Preconditions.checkNotNull(root);
    Preconditions.checkNotNull(params);
    try (ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator);
        ArrowArray arrowArray = ArrowArray.allocateNew(allocator)) {
      Data.exportVectorSchemaRoot(allocator, root, null, arrowArray, arrowSchema);
      return createWithFfiArray(
          datasetUri,
          arrowArray.memoryAddress(),
          arrowSchema.memoryAddress(),
          params.getMaxRowsPerFile(),
          params.getMaxRowsPerGroup(),
          params.getMaxBytesPerFile(),
          params.getMode(),
          params.getStorageOptions());
    }
  }

  /**
   * Create a fragment from the given arrow stream.
   *
   * @param datasetUri the dataset uri
   * @param stream the arrow stream
   * @param params the write params
   * @return the fragment metadata
   */
  public static List<FragmentMetadata> create(
      String datasetUri, ArrowArrayStream stream, WriteParams params) {
    Preconditions.checkNotNull(datasetUri);
    Preconditions.checkNotNull(stream);
    Preconditions.checkNotNull(params);
    return createWithFfiStream(
        datasetUri,
        stream.memoryAddress(),
        params.getMaxRowsPerFile(),
        params.getMaxRowsPerGroup(),
        params.getMaxBytesPerFile(),
        params.getMode(),
        params.getStorageOptions());
  }

  /**
   * Create a fragment from the given arrow array and schema.
   *
   * @return the fragment metadata
   */
  private static native List<FragmentMetadata> createWithFfiArray(
      String datasetUri,
      long arrowArrayMemoryAddress,
      long arrowSchemaMemoryAddress,
      Optional<Integer> maxRowsPerFile,
      Optional<Integer> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile,
      Optional<String> mode,
      Map<String, String> storageOptions);

  /**
   * Create a fragment from the given arrow stream.
   *
   * @return the fragment metadata
   */
  private static native List<FragmentMetadata> createWithFfiStream(
      String datasetUri,
      long arrowStreamMemoryAddress,
      Optional<Integer> maxRowsPerFile,
      Optional<Integer> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile,
      Optional<String> mode,
      Map<String, String> storageOptions);
}
