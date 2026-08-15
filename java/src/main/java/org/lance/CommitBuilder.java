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

import org.lance.merge.UncommittedMergeInsertResult;
import org.lance.namespace.LanceNamespace;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.util.Preconditions;

import java.time.Duration;
import java.util.List;
import java.util.Map;

/**
 * Builder for committing a {@link Transaction} to a Lance dataset.
 *
 * <p>Supports two modes:
 *
 * <ul>
 *   <li><strong>Dataset-based commit</strong>: commits against an existing dataset.
 *   <li><strong>URI-based commit</strong>: creates or updates a dataset at a URI.
 * </ul>
 *
 * <p>Example usage (dataset-based):
 *
 * <pre>{@code
 * try (Transaction txn = new Transaction.Builder()
 *     .readVersion(dataset.version())
 *     .operation(Append.builder().fragments(fragments).build())
 *     .build();
 *     Dataset committed = new CommitBuilder(dataset).execute(txn)) {
 *     // use committed dataset
 * }
 * }</pre>
 *
 * <p>Example usage (URI-based):
 *
 * <pre>{@code
 * try (Transaction txn = new Transaction.Builder()
 *     .operation(Overwrite.builder().fragments(fragments).schema(schema).build())
 *     .build();
 *     Dataset committed = new CommitBuilder(uri, allocator).execute(txn)) {
 *     // use committed dataset
 * }
 * }</pre>
 */
public class CommitBuilder {
  static {
    JniLoader.ensureLoaded();
  }

  private final Dataset dataset;
  private final String uri;
  private final BufferAllocator allocator;

  private Map<String, String> writeParams;
  private LanceNamespace namespaceClient;
  private List<String> tableId;
  private boolean namespaceClientManagedVersioning = false;
  private boolean enableV2ManifestPaths = true;
  private boolean detached = false;
  private Boolean useStableRowIds;
  private String storageFormat;
  private int maxRetries = 0;
  private boolean skipAutoCleanup = false;
  // -1 disables the timeout; any positive value is the timeout in nanoseconds.
  private long commitTimeoutNanos = Duration.ofMinutes(30).toNanos();
  private byte[] affectedRows;

  /**
   * Create a commit builder for committing against an existing dataset.
   *
   * @param dataset the existing dataset to commit against
   */
  public CommitBuilder(Dataset dataset) {
    Preconditions.checkNotNull(dataset, "Dataset must not be null");
    this.dataset = dataset;
    this.uri = null;
    this.allocator = null;
  }

  /**
   * Create a commit builder for creating or updating a dataset at the given URI.
   *
   * @param uri the target URI for the dataset
   * @param allocator the Arrow buffer allocator for schema export
   */
  public CommitBuilder(String uri, BufferAllocator allocator) {
    Preconditions.checkNotNull(uri, "URI must not be null");
    Preconditions.checkNotNull(allocator, "Allocator must not be null");
    this.dataset = null;
    this.uri = uri;
    this.allocator = allocator;
  }

  /**
   * Set write parameters for object storage configuration.
   *
   * @param writeParams map of storage option key-value pairs
   * @return this builder instance
   */
  public CommitBuilder writeParams(Map<String, String> writeParams) {
    this.writeParams = writeParams;
    return this;
  }

  /**
   * Set the namespace client and table ID for managed versioning.
   *
   * @param namespaceClient the namespace client
   * @param tableId the table ID parts
   * @return this builder instance
   */
  public CommitBuilder namespace(LanceNamespace namespaceClient, List<String> tableId) {
    this.namespaceClient = namespaceClient;
    this.tableId = tableId;
    return this;
  }

  /**
   * Set the namespace client for managed versioning.
   *
   * @param namespaceClient the LanceNamespace client instance
   * @return this builder instance
   */
  public CommitBuilder namespaceClient(LanceNamespace namespaceClient) {
    this.namespaceClient = namespaceClient;
    return this;
  }

  /**
   * Set the table ID for namespace client-based commit handling.
   *
   * @param tableId the table identifier (e.g., ["workspace", "table_name"])
   * @return this builder instance
   */
  public CommitBuilder tableId(List<String> tableId) {
    this.tableId = tableId;
    return this;
  }

  /**
   * Enable or disable namespace-managed versioning.
   *
   * @param managed whether the namespace manages versioning
   * @return this builder instance
   */
  public CommitBuilder namespaceClientManagedVersioning(boolean managed) {
    this.namespaceClientManagedVersioning = managed;
    return this;
  }

  /**
   * Enable or disable V2 manifest paths.
   *
   * @param enable whether to enable V2 manifest paths
   * @return this builder instance
   */
  public CommitBuilder enableV2ManifestPaths(boolean enable) {
    this.enableV2ManifestPaths = enable;
    return this;
  }

  /**
   * Set detached mode for the commit.
   *
   * @param detached whether the commit is detached
   * @return this builder instance
   */
  public CommitBuilder detached(boolean detached) {
    this.detached = detached;
    return this;
  }

  /**
   * Enable or disable stable row IDs.
   *
   * @param useStableRowIds whether to use stable row IDs
   * @return this builder instance
   */
  public CommitBuilder useStableRowIds(boolean useStableRowIds) {
    this.useStableRowIds = useStableRowIds;
    return this;
  }

  /**
   * Enable or disable stable row IDs.
   *
   * @param useStableRowIds whether to use stable row IDs, or null for default
   * @return this builder instance
   */
  public CommitBuilder useStableRowIds(Boolean useStableRowIds) {
    this.useStableRowIds = useStableRowIds;
    return this;
  }

  /**
   * Set the storage format version.
   *
   * @param storageFormat format version string (e.g., "0.1", "0.2", "2.0", "legacy", "stable")
   * @return this builder instance
   */
  public CommitBuilder storageFormat(String storageFormat) {
    this.storageFormat = storageFormat;
    return this;
  }

  /**
   * Set the maximum number of retries for transaction conflict resolution.
   *
   * @param maxRetries the maximum retry count
   * @return this builder instance
   */
  public CommitBuilder maxRetries(int maxRetries) {
    this.maxRetries = maxRetries;
    return this;
  }

  /**
   * Set whether to skip automatic cleanup of unreferenced files after commit.
   *
   * @param skipAutoCleanup true to skip cleanup, false to run cleanup (default)
   * @return this builder instance
   */
  public CommitBuilder skipAutoCleanup(boolean skipAutoCleanup) {
    this.skipAutoCleanup = skipAutoCleanup;
    return this;
  }

  /**
   * Set a timeout for the commit operation.
   *
   * <p>If the commit (including retries on conflict) does not complete within {@code timeout},
   * {@link #execute(Transaction)} will fail. Pass {@code null} to disable the timeout entirely. The
   * default is 30 minutes.
   *
   * @param timeout the commit timeout, or {@code null} to disable
   * @return this builder instance
   * @throws IllegalArgumentException if {@code timeout} is zero or negative
   */
  public CommitBuilder commitTimeout(Duration timeout) {
    if (timeout == null) {
      this.commitTimeoutNanos = -1L;
    } else {
      Preconditions.checkArgument(
          !timeout.isZero() && !timeout.isNegative(),
          "commit timeout must be a positive duration; pass null to disable");
      this.commitTimeoutNanos = timeout.toNanos();
    }
    return this;
  }

  /**
   * Set the serialized affected row addresses for fast conflict resolution.
   *
   * @param affectedRows the serialized RowAddrTreeMap bytes
   * @return this builder instance
   */
  public CommitBuilder withAffectedRows(byte[] affectedRows) {
    this.affectedRows = affectedRows;
    return this;
  }

  public byte[] getAffectedRows() {
    return affectedRows;
  }

  /**
   * Execute the commit with the given uncommitted merge insert result.
   *
   * @param uncommitted the uncommitted result containing transaction and metadata
   * @return a new Dataset at the committed version
   */
  public Dataset execute(UncommittedMergeInsertResult uncommitted) {
    Preconditions.checkNotNull(uncommitted, "Uncommitted result must not be null");
    if (uncommitted.affectedRows() != null && this.affectedRows == null) {
      this.affectedRows = uncommitted.affectedRows();
    }
    return execute(uncommitted.transaction());
  }

  /**
   * Execute the commit with the given transaction.
   *
   * <p>The caller is responsible for closing the transaction (via try-with-resources or {@link
   * Transaction#close()}) to release any native resources held by the operation.
   *
   * @param transaction the transaction to commit
   * @return a new Dataset at the committed version
   */
  public Dataset execute(Transaction transaction) {
    Preconditions.checkNotNull(transaction, "Transaction must not be null");
    if (dataset != null) {
      Dataset result =
          nativeCommitToDataset(
              dataset,
              transaction,
              detached,
              enableV2ManifestPaths,
              writeParams,
              useStableRowIds,
              storageFormat,
              maxRetries,
              skipAutoCleanup,
              namespaceClient,
              tableId,
              namespaceClientManagedVersioning,
              commitTimeoutNanos,
              affectedRows);
      result.setAllocator(dataset.allocator());
      return result;
    }
    if (uri != null) {
      Dataset result =
          nativeCommitToUri(
              uri,
              transaction,
              detached,
              enableV2ManifestPaths,
              namespaceClient,
              tableId,
              allocator,
              writeParams,
              useStableRowIds,
              storageFormat,
              maxRetries,
              skipAutoCleanup,
              namespaceClientManagedVersioning,
              commitTimeoutNanos,
              affectedRows);
      result.setAllocator(allocator);
      return result;
    }
    throw new IllegalStateException("CommitBuilder requires either a dataset or a URI");
  }

  private static native Dataset nativeCommitToDataset(
      Dataset dataset,
      Transaction transaction,
      boolean detached,
      boolean enableV2ManifestPaths,
      Map<String, String> writeParams,
      Boolean useStableRowIds,
      String storageFormat,
      int maxRetries,
      boolean skipAutoCleanup,
      Object namespace,
      Object tableId,
      boolean namespaceClientManagedVersioning,
      long commitTimeoutNanos,
      byte[] affectedRows);

  private static native Dataset nativeCommitToUri(
      String uri,
      Transaction transaction,
      boolean detached,
      boolean enableV2ManifestPaths,
      Object namespace,
      Object tableId,
      Object allocator,
      Map<String, String> writeParams,
      Boolean useStableRowIds,
      String storageFormat,
      int maxRetries,
      boolean skipAutoCleanup,
      boolean namespaceClientManagedVersioning,
      long commitTimeoutNanos,
      byte[] affectedRows);
}
