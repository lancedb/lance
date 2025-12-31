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
package org.lance.namespace;

import org.lance.JniLoader;
import org.lance.namespace.model.*;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.arrow.memory.BufferAllocator;

import java.io.Closeable;
import java.util.Map;

/**
 * DirectoryNamespace implementation that provides Lance namespace functionality for directory-based
 * storage.
 *
 * <p>Supported storage backends:
 *
 * <ul>
 *   <li>Local filesystem
 *   <li>AWS S3 (s3://bucket/path)
 *   <li>Azure Blob Storage (az://container/path)
 *   <li>Google Cloud Storage (gs://bucket/path)
 * </ul>
 *
 * <p>This class wraps the native Rust implementation and provides a Java interface that implements
 * the LanceNamespace interface from lance-namespace-core.
 *
 * <p>Configuration properties:
 *
 * <ul>
 *   <li>root (required): Root directory path or URI (e.g., /path/to/dir, s3://bucket/path,
 *       az://container/path, gs://bucket/path)
 *   <li>manifest_enabled (optional): "true" or "false" (default: true)
 *   <li>dir_listing_enabled (optional): "true" or "false" (default: true)
 *   <li>inline_optimization_enabled (optional): "true" or "false" (default: true)
 *   <li>storage.* (optional): Storage options for cloud providers (e.g., storage.region=us-east-1
 *       for S3, storage.account_name=myaccount for Azure)
 * </ul>
 *
 * <p>Credential vending properties (requires credential-vendor-* features to be enabled):
 *
 * <p>When credential vendor properties are configured, describeTable() will return vended temporary
 * credentials. The vendor type is auto-selected based on the table location URI: s3:// for AWS,
 * gs:// for GCP, az:// for Azure.
 *
 * <ul>
 *   <li>Common properties:
 *       <ul>
 *         <li>credential_vendor.enabled (required): Set to "true" to enable credential vending
 *         <li>credential_vendor.permission (optional): read, write, or admin (default: read)
 *       </ul>
 *   <li>AWS-specific properties (for s3:// locations):
 *       <ul>
 *         <li>credential_vendor.aws_role_arn (required): IAM role ARN to assume
 *         <li>credential_vendor.aws_external_id (optional): External ID for assume role
 *         <li>credential_vendor.aws_region (optional): AWS region
 *         <li>credential_vendor.aws_role_session_name (optional): Role session name
 *         <li>credential_vendor.aws_duration_millis (optional): Duration in ms (default: 3600000,
 *             range: 15min-12hrs)
 *       </ul>
 *   <li>GCP-specific properties (for gs:// locations):
 *       <ul>
 *         <li>credential_vendor.gcp_service_account (optional): Service account to impersonate
 *         <li>Note: GCP uses Application Default Credentials (ADC). To use a service account key
 *             file, set the GOOGLE_APPLICATION_CREDENTIALS environment variable before starting.
 *         <li>Note: GCP token duration cannot be configured; it's determined by the STS endpoint
 *       </ul>
 *   <li>Azure-specific properties (for az:// locations):
 *       <ul>
 *         <li>credential_vendor.azure_account_name (required): Azure storage account name
 *         <li>credential_vendor.azure_tenant_id (optional): Azure tenant ID
 *         <li>credential_vendor.azure_duration_millis (optional): Duration in ms (default: 3600000,
 *             up to 7 days)
 *       </ul>
 * </ul>
 *
 * <p>Example usage (local filesystem):
 *
 * <pre>{@code
 * Map<String, String> properties = new HashMap<>();
 * properties.put("root", "/tmp/lance-data");
 * properties.put("manifest_enabled", "true");
 *
 * DirectoryNamespace namespace = new DirectoryNamespace();
 * namespace.initialize(properties, allocator);
 *
 * // Use namespace...
 * ListTablesResponse tables = namespace.listTables(request);
 *
 * // Clean up
 * namespace.close();
 * }</pre>
 *
 * <p>Example usage (AWS S3):
 *
 * <pre>{@code
 * Map<String, String> properties = new HashMap<>();
 * properties.put("root", "s3://my-bucket/lance-data");
 * properties.put("storage.region", "us-east-1");
 * // AWS credentials can be provided via environment variables or IAM roles
 *
 * DirectoryNamespace namespace = new DirectoryNamespace();
 * namespace.initialize(properties, allocator);
 * // Use namespace...
 * namespace.close();
 * }</pre>
 *
 * <p>Example usage (AWS S3 with credential vending):
 *
 * <pre>{@code
 * Map<String, String> properties = new HashMap<>();
 * properties.put("root", "s3://my-bucket/lance-data");
 * properties.put("credential_vendor.enabled", "true");
 * properties.put("credential_vendor.aws_role_arn", "arn:aws:iam::123456789012:role/MyRole");
 * properties.put("credential_vendor.aws_duration_millis", "3600000");  // 1 hour
 *
 * DirectoryNamespace namespace = new DirectoryNamespace();
 * namespace.initialize(properties, allocator);
 * // describeTable() will now return vended credentials (AWS vendor auto-selected from s3:// URI)
 * namespace.close();
 * }</pre>
 */
public class DirectoryNamespace implements LanceNamespace, Closeable {
  static {
    JniLoader.ensureLoaded();
  }

  private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

  private long nativeDirectoryNamespaceHandle;
  private BufferAllocator allocator;

  /** Creates a new DirectoryNamespace. Must call initialize() before use. */
  public DirectoryNamespace() {}

  @Override
  public void initialize(Map<String, String> configProperties, BufferAllocator allocator) {
    if (this.nativeDirectoryNamespaceHandle != 0) {
      throw new IllegalStateException("DirectoryNamespace already initialized");
    }
    this.allocator = allocator;
    this.nativeDirectoryNamespaceHandle = createNative(configProperties);
  }

  @Override
  public String namespaceId() {
    ensureInitialized();
    return namespaceIdNative(nativeDirectoryNamespaceHandle);
  }

  @Override
  public ListNamespacesResponse listNamespaces(ListNamespacesRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = listNamespacesNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, ListNamespacesResponse.class);
  }

  @Override
  public DescribeNamespaceResponse describeNamespace(DescribeNamespaceRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = describeNamespaceNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, DescribeNamespaceResponse.class);
  }

  @Override
  public CreateNamespaceResponse createNamespace(CreateNamespaceRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = createNamespaceNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, CreateNamespaceResponse.class);
  }

  @Override
  public DropNamespaceResponse dropNamespace(DropNamespaceRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = dropNamespaceNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, DropNamespaceResponse.class);
  }

  @Override
  public void namespaceExists(NamespaceExistsRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    namespaceExistsNative(nativeDirectoryNamespaceHandle, requestJson);
  }

  @Override
  public ListTablesResponse listTables(ListTablesRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = listTablesNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, ListTablesResponse.class);
  }

  @Override
  public DescribeTableResponse describeTable(DescribeTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = describeTableNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, DescribeTableResponse.class);
  }

  @Override
  public RegisterTableResponse registerTable(RegisterTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = registerTableNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, RegisterTableResponse.class);
  }

  @Override
  public void tableExists(TableExistsRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    tableExistsNative(nativeDirectoryNamespaceHandle, requestJson);
  }

  @Override
  public DropTableResponse dropTable(DropTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = dropTableNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, DropTableResponse.class);
  }

  @Override
  public DeregisterTableResponse deregisterTable(DeregisterTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = deregisterTableNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, DeregisterTableResponse.class);
  }

  @Override
  public Long countTableRows(CountTableRowsRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    return countTableRowsNative(nativeDirectoryNamespaceHandle, requestJson);
  }

  @Override
  public CreateTableResponse createTable(CreateTableRequest request, byte[] requestData) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson =
        createTableNative(nativeDirectoryNamespaceHandle, requestJson, requestData);
    return fromJson(responseJson, CreateTableResponse.class);
  }

  @Override
  public CreateEmptyTableResponse createEmptyTable(CreateEmptyTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = createEmptyTableNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, CreateEmptyTableResponse.class);
  }

  @Override
  public DeclareTableResponse declareTable(DeclareTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = declareTableNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, DeclareTableResponse.class);
  }

  @Override
  public InsertIntoTableResponse insertIntoTable(
      InsertIntoTableRequest request, byte[] requestData) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson =
        insertIntoTableNative(nativeDirectoryNamespaceHandle, requestJson, requestData);
    return fromJson(responseJson, InsertIntoTableResponse.class);
  }

  @Override
  public MergeInsertIntoTableResponse mergeInsertIntoTable(
      MergeInsertIntoTableRequest request, byte[] requestData) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson =
        mergeInsertIntoTableNative(nativeDirectoryNamespaceHandle, requestJson, requestData);
    return fromJson(responseJson, MergeInsertIntoTableResponse.class);
  }

  @Override
  public UpdateTableResponse updateTable(UpdateTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = updateTableNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, UpdateTableResponse.class);
  }

  @Override
  public DeleteFromTableResponse deleteFromTable(DeleteFromTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = deleteFromTableNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, DeleteFromTableResponse.class);
  }

  @Override
  public byte[] queryTable(QueryTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    return queryTableNative(nativeDirectoryNamespaceHandle, requestJson);
  }

  @Override
  public CreateTableIndexResponse createTableIndex(CreateTableIndexRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = createTableIndexNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, CreateTableIndexResponse.class);
  }

  @Override
  public ListTableIndicesResponse listTableIndices(ListTableIndicesRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = listTableIndicesNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, ListTableIndicesResponse.class);
  }

  @Override
  public DescribeTableIndexStatsResponse describeTableIndexStats(
      DescribeTableIndexStatsRequest request, String indexName) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson =
        describeTableIndexStatsNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, DescribeTableIndexStatsResponse.class);
  }

  @Override
  public DescribeTransactionResponse describeTransaction(DescribeTransactionRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = describeTransactionNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, DescribeTransactionResponse.class);
  }

  @Override
  public AlterTransactionResponse alterTransaction(AlterTransactionRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = alterTransactionNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, AlterTransactionResponse.class);
  }

  @Override
  public void close() {
    if (nativeDirectoryNamespaceHandle != 0) {
      releaseNative(nativeDirectoryNamespaceHandle);
      nativeDirectoryNamespaceHandle = 0;
    }
  }

  private void ensureInitialized() {
    if (nativeDirectoryNamespaceHandle == 0) {
      throw new IllegalStateException(
          "DirectoryNamespace not initialized. Call initialize() first.");
    }
  }

  private static String toJson(Object obj) {
    try {
      return OBJECT_MAPPER.writeValueAsString(obj);
    } catch (JsonProcessingException e) {
      throw new RuntimeException("Failed to serialize request to JSON", e);
    }
  }

  private static <T> T fromJson(String json, Class<T> clazz) {
    try {
      return OBJECT_MAPPER.readValue(json, clazz);
    } catch (JsonProcessingException e) {
      throw new RuntimeException("Failed to deserialize response from JSON", e);
    }
  }

  // Native methods
  private native long createNative(Map<String, String> properties);

  private native void releaseNative(long handle);

  private native String namespaceIdNative(long handle);

  private native String listNamespacesNative(long handle, String requestJson);

  private native String describeNamespaceNative(long handle, String requestJson);

  private native String createNamespaceNative(long handle, String requestJson);

  private native String dropNamespaceNative(long handle, String requestJson);

  private native void namespaceExistsNative(long handle, String requestJson);

  private native String listTablesNative(long handle, String requestJson);

  private native String describeTableNative(long handle, String requestJson);

  private native String registerTableNative(long handle, String requestJson);

  private native void tableExistsNative(long handle, String requestJson);

  private native String dropTableNative(long handle, String requestJson);

  private native String deregisterTableNative(long handle, String requestJson);

  private native long countTableRowsNative(long handle, String requestJson);

  private native String createTableNative(long handle, String requestJson, byte[] requestData);

  private native String createEmptyTableNative(long handle, String requestJson);

  private native String declareTableNative(long handle, String requestJson);

  private native String insertIntoTableNative(long handle, String requestJson, byte[] requestData);

  private native String mergeInsertIntoTableNative(
      long handle, String requestJson, byte[] requestData);

  private native String updateTableNative(long handle, String requestJson);

  private native String deleteFromTableNative(long handle, String requestJson);

  private native byte[] queryTableNative(long handle, String requestJson);

  private native String createTableIndexNative(long handle, String requestJson);

  private native String listTableIndicesNative(long handle, String requestJson);

  private native String describeTableIndexStatsNative(long handle, String requestJson);

  private native String describeTransactionNative(long handle, String requestJson);

  private native String alterTransactionNative(long handle, String requestJson);
}
