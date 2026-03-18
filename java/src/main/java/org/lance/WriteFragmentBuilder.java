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

import org.lance.io.StorageOptionsProvider;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.VectorSchemaRoot;

import java.util.List;
import java.util.Map;

/**
 * Builder for writing fragments.
 *
 * <p>This builder provides a fluent API for creating fragments with various configuration options.
 * It supports both VectorSchemaRoot and ArrowArrayStream as data sources.
 *
 * <p>Example usage:
 *
 * <pre>{@code
 * List<FragmentMetadata> fragments = Fragment.write()
 *     .datasetUri("s3://bucket/dataset.lance")
 *     .allocator(allocator)
 *     .data(vectorSchemaRoot)
 *     .storageOptions(storageOptions)
 *     .execute();
 * }</pre>
 */
public class WriteFragmentBuilder {
  private String datasetUri;
  private BufferAllocator allocator;
  private VectorSchemaRoot vectorSchemaRoot;
  private ArrowArrayStream arrowArrayStream;
  private WriteParams writeParams;
  private WriteParams.Builder writeParamsBuilder;
  private StorageOptionsProvider storageOptionsProvider;

  WriteFragmentBuilder() {}

  /**
   * Set the dataset URI where fragments will be written.
   *
   * @param datasetUri the dataset URI
   * @return this builder
   */
  public WriteFragmentBuilder datasetUri(String datasetUri) {
    this.datasetUri = datasetUri;
    return this;
  }

  /**
   * Set the buffer allocator for Arrow operations.
   *
   * @param allocator the buffer allocator
   * @return this builder
   */
  public WriteFragmentBuilder allocator(BufferAllocator allocator) {
    this.allocator = allocator;
    return this;
  }

  /**
   * Set the data to write using a VectorSchemaRoot.
   *
   * @param root the vector schema root containing the data
   * @return this builder
   */
  public WriteFragmentBuilder data(VectorSchemaRoot root) {
    Preconditions.checkState(
        this.arrowArrayStream == null, "Cannot set both VectorSchemaRoot and ArrowArrayStream");
    this.vectorSchemaRoot = root;
    return this;
  }

  /**
   * Set the data to write using an ArrowArrayStream.
   *
   * @param stream the arrow array stream containing the data
   * @return this builder
   */
  public WriteFragmentBuilder data(ArrowArrayStream stream) {
    Preconditions.checkState(
        this.vectorSchemaRoot == null, "Cannot set both VectorSchemaRoot and ArrowArrayStream");
    this.arrowArrayStream = stream;
    return this;
  }

  /**
   * Set the write parameters.
   *
   * @param params the write parameters
   * @return this builder
   */
  public WriteFragmentBuilder writeParams(WriteParams params) {
    this.writeParams = params;
    return this;
  }

  /**
   * Set storage options for object store access.
   *
   * @param storageOptions the storage options
   * @return this builder
   */
  public WriteFragmentBuilder storageOptions(Map<String, String> storageOptions) {
    ensureWriteParamsBuilder();
    this.writeParamsBuilder.withStorageOptions(storageOptions);
    return this;
  }

  /**
   * Set the storage options provider for dynamic credential refresh.
   *
   * @param provider the storage options provider
   * @return this builder
   */
  public WriteFragmentBuilder storageOptionsProvider(StorageOptionsProvider provider) {
    this.storageOptionsProvider = provider;
    return this;
  }

  /**
   * Set the maximum number of rows per file.
   *
   * @param maxRowsPerFile maximum rows per file
   * @return this builder
   */
  public WriteFragmentBuilder maxRowsPerFile(int maxRowsPerFile) {
    ensureWriteParamsBuilder();
    this.writeParamsBuilder.withMaxRowsPerFile(maxRowsPerFile);
    return this;
  }

  /**
   * Set the maximum number of rows per group.
   *
   * @param maxRowsPerGroup maximum rows per group
   * @return this builder
   */
  public WriteFragmentBuilder maxRowsPerGroup(int maxRowsPerGroup) {
    ensureWriteParamsBuilder();
    this.writeParamsBuilder.withMaxRowsPerGroup(maxRowsPerGroup);
    return this;
  }

  /**
   * Set the maximum number of bytes per file.
   *
   * @param maxBytesPerFile maximum bytes per file
   * @return this builder
   */
  public WriteFragmentBuilder maxBytesPerFile(long maxBytesPerFile) {
    ensureWriteParamsBuilder();
    this.writeParamsBuilder.withMaxBytesPerFile(maxBytesPerFile);
    return this;
  }

  /**
   * Set the write mode.
   *
   * @param mode the write mode
   * @return this builder
   */
  public WriteFragmentBuilder mode(WriteParams.WriteMode mode) {
    ensureWriteParamsBuilder();
    this.writeParamsBuilder.withMode(mode);
    return this;
  }

  /**
   * Enable or disable stable row IDs.
   *
   * @param enable whether to enable stable row IDs
   * @return this builder
   */
  public WriteFragmentBuilder enableStableRowIds(boolean enable) {
    ensureWriteParamsBuilder();
    this.writeParamsBuilder.withEnableStableRowIds(enable);
    return this;
  }

  /**
   * Set the data storage version.
   *
   * @param version the data storage version (e.g., "legacy", "stable", "2.0")
   * @return this builder
   */
  public WriteFragmentBuilder dataStorageVersion(String version) {
    ensureWriteParamsBuilder();
    this.writeParamsBuilder.withDataStorageVersion(version);
    return this;
  }

  /**
   * Execute the fragment write operation.
   *
   * @return the list of fragment metadata for the created fragments
   */
  public List<FragmentMetadata> execute() {
    validate();

    // Build the write params if builder was used
    WriteParams finalWriteParams = buildWriteParams();

    if (vectorSchemaRoot != null) {
      return Fragment.create(
          datasetUri, allocator, vectorSchemaRoot, finalWriteParams, storageOptionsProvider);
    } else {
      return Fragment.create(
          datasetUri, arrowArrayStream, finalWriteParams, storageOptionsProvider);
    }
  }

  private void ensureWriteParamsBuilder() {
    if (this.writeParamsBuilder == null) {
      this.writeParamsBuilder = new WriteParams.Builder();
    }
  }

  private WriteParams buildWriteParams() {
    if (writeParams != null) {
      return writeParams;
    } else if (writeParamsBuilder != null) {
      return writeParamsBuilder.build();
    } else {
      return new WriteParams.Builder().build();
    }
  }

  private void validate() {
    Preconditions.checkNotNull(datasetUri, "datasetUri is required");
    Preconditions.checkState(
        vectorSchemaRoot != null || arrowArrayStream != null,
        "Either VectorSchemaRoot or ArrowArrayStream must be provided");
    Preconditions.checkState(
        vectorSchemaRoot == null || arrowArrayStream == null,
        "Cannot set both VectorSchemaRoot and ArrowArrayStream");
    Preconditions.checkState(
        vectorSchemaRoot == null || allocator != null,
        "allocator is required when using VectorSchemaRoot");
    Preconditions.checkState(
        writeParams == null || writeParamsBuilder == null,
        "Cannot use both writeParams() and individual parameter methods");
  }
}
