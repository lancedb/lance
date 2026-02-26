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

import org.lance.namespace.DirectoryNamespace;
import org.lance.namespace.LanceNamespace;
import org.lance.namespace.LanceNamespaceStorageOptionsProvider;
import org.lance.namespace.model.CreateEmptyTableRequest;
import org.lance.namespace.model.CreateEmptyTableResponse;
import org.lance.namespace.model.DeclareTableRequest;
import org.lance.namespace.model.DeclareTableResponse;
import org.lance.namespace.model.DescribeTableRequest;
import org.lance.namespace.model.DescribeTableResponse;
import org.lance.operation.Append;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.CreateBucketRequest;
import software.amazon.awssdk.services.s3.model.DeleteBucketRequest;
import software.amazon.awssdk.services.s3.model.DeleteObjectRequest;
import software.amazon.awssdk.services.s3.model.ListObjectsV2Request;
import software.amazon.awssdk.services.s3.model.S3Object;

import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Integration tests for Lance with S3 and credential refresh using StorageOptionsProvider.
 *
 * <p>This test simulates a tracking credential provider that returns incrementing credentials and
 * verifies that the credential refresh mechanism works correctly.
 *
 * <p>These tests require LocalStack to be running. Run with: docker compose up -d
 *
 * <p>Set LANCE_INTEGRATION_TEST=1 environment variable to enable these tests.
 */
@EnabledIfEnvironmentVariable(named = "LANCE_INTEGRATION_TEST", matches = "1")
public class NamespaceIntegrationTest {

  private static final String ENDPOINT_URL = "http://localhost:4566";
  private static final String REGION = "us-east-1";
  private static final String ACCESS_KEY = "ACCESS_KEY";
  private static final String SECRET_KEY = "SECRET_KEY";
  private static final String BUCKET_NAME = "lance-namespace-integtest-java";

  private static S3Client s3Client;

  @BeforeAll
  static void setup() {
    s3Client =
        S3Client.builder()
            .endpointOverride(URI.create(ENDPOINT_URL))
            .region(Region.of(REGION))
            .credentialsProvider(
                StaticCredentialsProvider.create(
                    AwsBasicCredentials.create(ACCESS_KEY, SECRET_KEY)))
            .forcePathStyle(true) // Required for LocalStack
            .build();

    // Delete bucket if it exists from previous run
    try {
      deleteBucket();
    } catch (Exception e) {
      // Ignore if bucket doesn't exist
    }

    // Create test bucket
    s3Client.createBucket(CreateBucketRequest.builder().bucket(BUCKET_NAME).build());
  }

  @AfterAll
  static void tearDown() {
    if (s3Client != null) {
      try {
        deleteBucket();
      } catch (Exception e) {
        // Ignore cleanup errors
      }
      s3Client.close();
    }
  }

  private static void deleteBucket() {
    // Delete all objects first
    List<S3Object> objects =
        s3Client
            .listObjectsV2(ListObjectsV2Request.builder().bucket(BUCKET_NAME).build())
            .contents();
    for (S3Object obj : objects) {
      s3Client.deleteObject(
          DeleteObjectRequest.builder().bucket(BUCKET_NAME).key(obj.key()).build());
    }
    s3Client.deleteBucket(DeleteBucketRequest.builder().bucket(BUCKET_NAME).build());
  }

  /**
   * Tracking LanceNamespace implementation for testing.
   *
   * <p>This implementation wraps DirectoryNamespace and tracks API calls. It returns incrementing
   * credentials with expiration timestamps to test the credential refresh mechanism.
   */
  static class TrackingNamespace implements LanceNamespace {
    private final String bucketName;
    private final Map<String, String> baseStorageOptions;
    private final int credentialExpiresInSeconds;
    private final AtomicInteger describeCallCount = new AtomicInteger(0);
    private final AtomicInteger createCallCount = new AtomicInteger(0);
    private final DirectoryNamespace inner;

    public TrackingNamespace(
        String bucketName, Map<String, String> storageOptions, int credentialExpiresInSeconds) {
      this.bucketName = bucketName;
      this.baseStorageOptions = new HashMap<>(storageOptions);
      this.credentialExpiresInSeconds = credentialExpiresInSeconds;

      // Create underlying DirectoryNamespace with storage options
      Map<String, String> dirProps = new HashMap<>();
      for (Map.Entry<String, String> entry : storageOptions.entrySet()) {
        dirProps.put("storage." + entry.getKey(), entry.getValue());
      }

      // Set root based on bucket type
      if (bucketName.startsWith("/") || bucketName.startsWith("file://")) {
        dirProps.put("root", bucketName + "/namespace_root");
      } else {
        dirProps.put("root", "s3://" + bucketName + "/namespace_root");
      }

      this.inner = new DirectoryNamespace();
      try (BufferAllocator allocator = new RootAllocator()) {
        this.inner.initialize(dirProps, allocator);
      }
    }

    public int getDescribeCallCount() {
      return describeCallCount.get();
    }

    public int getCreateCallCount() {
      return createCallCount.get();
    }

    @Override
    public void initialize(Map<String, String> configProperties, BufferAllocator allocator) {
      // Already initialized in constructor
    }

    @Override
    public String namespaceId() {
      return "TrackingNamespace { inner: " + inner.namespaceId() + " }";
    }

    /**
     * Modifies storage options to add incrementing credentials with expiration timestamp.
     *
     * @param storageOptions Original storage options
     * @param count Call count to use for credential generation
     * @return Modified storage options with new credentials
     */
    private Map<String, String> modifyStorageOptions(
        Map<String, String> storageOptions, int count) {
      Map<String, String> modified =
          storageOptions != null ? new HashMap<>(storageOptions) : new HashMap<>();

      modified.put("aws_access_key_id", "AKID_" + count);
      modified.put("aws_secret_access_key", "SECRET_" + count);
      modified.put("aws_session_token", "TOKEN_" + count);

      long expiresAtMillis = System.currentTimeMillis() + (credentialExpiresInSeconds * 1000L);
      modified.put("expires_at_millis", String.valueOf(expiresAtMillis));
      // Set refresh offset to 1 second (1000ms) for short-lived credential tests
      modified.put("refresh_offset_millis", "1000");

      return modified;
    }

    @Override
    public CreateEmptyTableResponse createEmptyTable(CreateEmptyTableRequest request) {
      int count = createCallCount.incrementAndGet();

      CreateEmptyTableResponse response = inner.createEmptyTable(request);
      response.setStorageOptions(modifyStorageOptions(response.getStorageOptions(), count));

      return response;
    }

    @Override
    public DeclareTableResponse declareTable(DeclareTableRequest request) {
      int count = createCallCount.incrementAndGet();

      DeclareTableResponse response = inner.declareTable(request);
      response.setStorageOptions(modifyStorageOptions(response.getStorageOptions(), count));

      return response;
    }

    @Override
    public DescribeTableResponse describeTable(DescribeTableRequest request) {
      int count = describeCallCount.incrementAndGet();

      DescribeTableResponse response = inner.describeTable(request);
      response.setStorageOptions(modifyStorageOptions(response.getStorageOptions(), count));

      return response;
    }
  }

  @Test
  void testOpenDatasetWithoutRefresh() throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      // Set up storage options
      Map<String, String> storageOptions = new HashMap<>();
      storageOptions.put("allow_http", "true");
      storageOptions.put("aws_access_key_id", ACCESS_KEY);
      storageOptions.put("aws_secret_access_key", SECRET_KEY);
      storageOptions.put("aws_endpoint", ENDPOINT_URL);
      storageOptions.put("aws_region", REGION);

      // Create tracking namespace with 60-second expiration (long enough to not expire during test)
      TrackingNamespace namespace = new TrackingNamespace(BUCKET_NAME, storageOptions, 60);
      String tableName = UUID.randomUUID().toString();

      // Create schema and data
      Schema schema =
          new Schema(
              Arrays.asList(
                  new Field("a", FieldType.nullable(new ArrowType.Int(32, true)), null),
                  new Field("b", FieldType.nullable(new ArrowType.Int(32, true)), null)));

      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        IntVector aVector = (IntVector) root.getVector("a");
        IntVector bVector = (IntVector) root.getVector("b");

        aVector.allocateNew(2);
        bVector.allocateNew(2);

        aVector.set(0, 1);
        bVector.set(0, 2);
        aVector.set(1, 10);
        bVector.set(1, 20);

        aVector.setValueCount(2);
        bVector.setValueCount(2);
        root.setRowCount(2);

        // Create a test reader that returns our VectorSchemaRoot
        ArrowReader testReader =
            new ArrowReader(allocator) {
              boolean firstRead = true;

              @Override
              public boolean loadNextBatch() {
                if (firstRead) {
                  firstRead = false;
                  return true;
                }
                return false;
              }

              @Override
              public long bytesRead() {
                return 0;
              }

              @Override
              protected void closeReadSource() {}

              @Override
              protected Schema readSchema() {
                return schema;
              }

              @Override
              public VectorSchemaRoot getVectorSchemaRoot() {
                return root;
              }
            };

        // Create dataset through namespace
        try (Dataset dataset =
            Dataset.write()
                .allocator(allocator)
                .reader(testReader)
                .namespace(namespace)
                .tableId(Arrays.asList(tableName))
                .mode(WriteParams.WriteMode.CREATE)
                .execute()) {
          assertEquals(2, dataset.countRows());
        }
      }

      // Verify createEmptyTable was called
      assertEquals(1, namespace.getCreateCallCount(), "createEmptyTable should be called once");

      // Open dataset through namespace WITH refresh enabled
      ReadOptions readOptions = new ReadOptions.Builder().build();

      int callCountBeforeOpen = namespace.getDescribeCallCount();
      try (Dataset dsFromNamespace =
          Dataset.open()
              .allocator(allocator)
              .namespace(namespace)
              .tableId(Arrays.asList(tableName))
              .readOptions(readOptions)
              .build()) {
        // With the fix, describeTable should only be called once during open
        // to get the table location and initial storage options
        int callCountAfterOpen = namespace.getDescribeCallCount();
        assertEquals(
            1,
            callCountAfterOpen - callCountBeforeOpen,
            "describeTable should be called exactly once during open, got: "
                + (callCountAfterOpen - callCountBeforeOpen));

        // Verify we can read the data multiple times
        assertEquals(2, dsFromNamespace.countRows());
        assertEquals(2, dsFromNamespace.countRows());
        assertEquals(2, dsFromNamespace.countRows());

        // Perform operations that access S3
        List<Fragment> fragments = dsFromNamespace.getFragments();
        assertEquals(1, fragments.size());
        List<Version> versions = dsFromNamespace.listVersions();
        assertEquals(1, versions.size());

        // With the fix, credentials are cached so no additional calls are made
        int finalCallCount = namespace.getDescribeCallCount();
        int totalCalls = finalCallCount - callCountBeforeOpen;
        assertEquals(
            1,
            totalCalls,
            "describeTable should only be called once total (credentials are cached), got: "
                + totalCalls);
      }
    }
  }

  @Test
  void testStorageOptionsProviderWithRefresh() throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      // Set up storage options
      Map<String, String> storageOptions = new HashMap<>();
      storageOptions.put("allow_http", "true");
      storageOptions.put("aws_access_key_id", ACCESS_KEY);
      storageOptions.put("aws_secret_access_key", SECRET_KEY);
      storageOptions.put("aws_endpoint", ENDPOINT_URL);
      storageOptions.put("aws_region", REGION);

      // Create tracking namespace with 5-second expiration for faster testing
      TrackingNamespace namespace = new TrackingNamespace(BUCKET_NAME, storageOptions, 5);
      String tableName = UUID.randomUUID().toString();

      // Create schema and data
      Schema schema =
          new Schema(
              Arrays.asList(
                  new Field("a", FieldType.nullable(new ArrowType.Int(32, true)), null),
                  new Field("b", FieldType.nullable(new ArrowType.Int(32, true)), null)));

      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        IntVector aVector = (IntVector) root.getVector("a");
        IntVector bVector = (IntVector) root.getVector("b");

        aVector.allocateNew(2);
        bVector.allocateNew(2);

        aVector.set(0, 1);
        bVector.set(0, 2);
        aVector.set(1, 10);
        bVector.set(1, 20);

        aVector.setValueCount(2);
        bVector.setValueCount(2);
        root.setRowCount(2);

        // Create a test reader that returns our VectorSchemaRoot
        ArrowReader testReader =
            new ArrowReader(allocator) {
              boolean firstRead = true;

              @Override
              public boolean loadNextBatch() {
                if (firstRead) {
                  firstRead = false;
                  return true;
                }
                return false;
              }

              @Override
              public long bytesRead() {
                return 0;
              }

              @Override
              protected void closeReadSource() {}

              @Override
              protected Schema readSchema() {
                return schema;
              }

              @Override
              public VectorSchemaRoot getVectorSchemaRoot() {
                return root;
              }
            };

        // Create dataset through namespace with refresh enabled
        try (Dataset dataset =
            Dataset.write()
                .allocator(allocator)
                .reader(testReader)
                .namespace(namespace)
                .tableId(Arrays.asList(tableName))
                .mode(WriteParams.WriteMode.CREATE)
                .execute()) {
          assertEquals(2, dataset.countRows());
        }
      }

      // Verify createEmptyTable was called
      assertEquals(1, namespace.getCreateCallCount(), "createEmptyTable should be called once");

      // Open dataset through namespace with refresh enabled
      ReadOptions readOptions = new ReadOptions.Builder().build();

      int callCountBeforeOpen = namespace.getDescribeCallCount();
      try (Dataset dsFromNamespace =
          Dataset.open()
              .allocator(allocator)
              .namespace(namespace)
              .tableId(Arrays.asList(tableName))
              .readOptions(readOptions)
              .build()) {
        // With the fix, describeTable should only be called once during open
        int callCountAfterOpen = namespace.getDescribeCallCount();
        assertEquals(
            1,
            callCountAfterOpen - callCountBeforeOpen,
            "describeTable should be called exactly once during open, got: "
                + (callCountAfterOpen - callCountBeforeOpen));

        // Verify we can read the data
        assertEquals(2, dsFromNamespace.countRows());

        // Record call count after initial reads
        int callCountAfterInitialReads = namespace.getDescribeCallCount();
        int callsAfterFirstRead = callCountAfterInitialReads - callCountBeforeOpen;
        assertEquals(
            1,
            callsAfterFirstRead,
            "describeTable should still be 1 (credentials are cached), got: "
                + callsAfterFirstRead);

        // Wait for credentials to be close to expiring (4 seconds - past the 3s refresh threshold)
        Thread.sleep(4000);

        // Perform read operations after expiration
        // Access fragments and versions which require S3 access and trigger credential refresh
        assertEquals(2, dsFromNamespace.countRows());
        List<Fragment> fragments = dsFromNamespace.getFragments();
        assertEquals(1, fragments.size());
        List<Version> versions = dsFromNamespace.listVersions();
        assertEquals(1, versions.size());

        int finalCallCount = namespace.getDescribeCallCount();
        int totalCallsAfterExpiration = finalCallCount - callCountBeforeOpen;
        assertEquals(
            2,
            totalCallsAfterExpiration,
            "Credentials should be refreshed once after expiration. "
                + "Expected 2 total calls (1 initial + 1 refresh), got: "
                + totalCallsAfterExpiration);
      }
    }
  }

  @Test
  void testWriteDatasetBuilderWithNamespaceCreate() throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      // Set up storage options
      Map<String, String> storageOptions = new HashMap<>();
      storageOptions.put("allow_http", "true");
      storageOptions.put("aws_access_key_id", ACCESS_KEY);
      storageOptions.put("aws_secret_access_key", SECRET_KEY);
      storageOptions.put("aws_endpoint", ENDPOINT_URL);
      storageOptions.put("aws_region", REGION);

      // Create tracking namespace
      TrackingNamespace namespace = new TrackingNamespace(BUCKET_NAME, storageOptions, 60);
      String tableName = UUID.randomUUID().toString();

      // Create schema and data
      Schema schema =
          new Schema(
              Arrays.asList(
                  new Field("a", FieldType.nullable(new ArrowType.Int(32, true)), null),
                  new Field("b", FieldType.nullable(new ArrowType.Int(32, true)), null)));

      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        IntVector aVector = (IntVector) root.getVector("a");
        IntVector bVector = (IntVector) root.getVector("b");

        aVector.allocateNew(2);
        bVector.allocateNew(2);

        aVector.set(0, 1);
        bVector.set(0, 2);
        aVector.set(1, 10);
        bVector.set(1, 20);

        aVector.setValueCount(2);
        bVector.setValueCount(2);
        root.setRowCount(2);

        // Create a test reader that returns our VectorSchemaRoot
        ArrowReader testReader =
            new ArrowReader(allocator) {
              boolean firstRead = true;

              @Override
              public boolean loadNextBatch() {
                if (firstRead) {
                  firstRead = false;
                  return true;
                }
                return false;
              }

              @Override
              public long bytesRead() {
                return 0;
              }

              @Override
              protected void closeReadSource() {}

              @Override
              protected Schema readSchema() {
                return schema;
              }

              @Override
              public VectorSchemaRoot getVectorSchemaRoot() {
                return root;
              }
            };

        int callCountBefore = namespace.getCreateCallCount();

        // Use the write builder to create a dataset through namespace
        try (Dataset dataset =
            Dataset.write()
                .allocator(allocator)
                .reader(testReader)
                .namespace(namespace)
                .tableId(Arrays.asList(tableName))
                .mode(WriteParams.WriteMode.CREATE)
                .execute()) {

          // Verify createEmptyTable was called
          int callCountAfter = namespace.getCreateCallCount();
          assertEquals(
              1, callCountAfter - callCountBefore, "createEmptyTable should be called once");

          // Verify dataset was created successfully
          assertEquals(2, dataset.countRows());
          assertEquals(schema, dataset.getSchema());
        }
      }
    }
  }

  @Test
  void testWriteDatasetBuilderWithNamespaceCreateCallCounts() throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      // Set up storage options
      Map<String, String> storageOptions = new HashMap<>();
      storageOptions.put("allow_http", "true");
      storageOptions.put("aws_access_key_id", ACCESS_KEY);
      storageOptions.put("aws_secret_access_key", SECRET_KEY);
      storageOptions.put("aws_endpoint", ENDPOINT_URL);
      storageOptions.put("aws_region", REGION);

      // Create tracking namespace with 60-second expiration (long enough that no refresh happens)
      // Credentials expire at T+60s. With a 1s refresh offset, refresh would happen at T+59s.
      // Since writes complete well under 59 seconds, NO credential refresh should occur.
      TrackingNamespace namespace = new TrackingNamespace(BUCKET_NAME, storageOptions, 60);
      String tableName = UUID.randomUUID().toString();

      // Verify initial call counts
      assertEquals(0, namespace.getCreateCallCount(), "createEmptyTable should not be called yet");
      assertEquals(0, namespace.getDescribeCallCount(), "describeTable should not be called yet");

      // Create schema and data
      Schema schema =
          new Schema(
              Arrays.asList(
                  new Field("a", FieldType.nullable(new ArrowType.Int(32, true)), null),
                  new Field("b", FieldType.nullable(new ArrowType.Int(32, true)), null)));

      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        IntVector aVector = (IntVector) root.getVector("a");
        IntVector bVector = (IntVector) root.getVector("b");

        aVector.allocateNew(2);
        bVector.allocateNew(2);

        aVector.set(0, 1);
        bVector.set(0, 2);
        aVector.set(1, 10);
        bVector.set(1, 20);

        aVector.setValueCount(2);
        bVector.setValueCount(2);
        root.setRowCount(2);

        // Create a test reader that returns our VectorSchemaRoot
        ArrowReader testReader =
            new ArrowReader(allocator) {
              boolean firstRead = true;

              @Override
              public boolean loadNextBatch() {
                if (firstRead) {
                  firstRead = false;
                  return true;
                }
                return false;
              }

              @Override
              public long bytesRead() {
                return 0;
              }

              @Override
              protected void closeReadSource() {}

              @Override
              protected Schema readSchema() {
                return schema;
              }

              @Override
              public VectorSchemaRoot getVectorSchemaRoot() {
                return root;
              }
            };

        // Use the write builder to create a dataset through namespace
        // Write completes instantly, so NO describeTable call should happen for refresh.
        try (Dataset dataset =
            Dataset.write()
                .allocator(allocator)
                .reader(testReader)
                .namespace(namespace)
                .tableId(Arrays.asList(tableName))
                .mode(WriteParams.WriteMode.CREATE)
                .execute()) {

          // Verify createEmptyTable was called exactly ONCE
          assertEquals(
              1, namespace.getCreateCallCount(), "createEmptyTable should be called exactly once");

          // Verify describeTable was NOT called during CREATE
          // Initial credentials come from createEmptyTable response, and since credentials
          // don't expire during the fast write, NO refresh (describeTable) is needed
          assertEquals(
              0,
              namespace.getDescribeCallCount(),
              "describeTable should NOT be called during CREATE - "
                  + "initial credentials come from createEmptyTable response and don't expire");

          // Verify dataset was created successfully
          assertEquals(2, dataset.countRows());
          assertEquals(schema, dataset.getSchema());
        }
      }

      // Verify counts after dataset is closed
      assertEquals(
          1, namespace.getCreateCallCount(), "createEmptyTable should still be 1 after close");
      assertEquals(
          0,
          namespace.getDescribeCallCount(),
          "describeTable should still be 0 after close (no refresh needed)");

      // Now open the dataset through namespace with long-lived credentials (60s expiration)
      ReadOptions readOptions = new ReadOptions.Builder().build();

      try (Dataset dsFromNamespace =
          Dataset.open()
              .allocator(allocator)
              .namespace(namespace)
              .tableId(Arrays.asList(tableName))
              .readOptions(readOptions)
              .build()) {

        // createEmptyTable should NOT be called during open (only during CREATE)
        assertEquals(
            1,
            namespace.getCreateCallCount(),
            "createEmptyTable should still be 1 (not called during open)");

        // describeTable is called exactly ONCE during open to get table location
        assertEquals(
            1,
            namespace.getDescribeCallCount(),
            "describeTable should be called exactly once during open");

        // Verify we can read the data multiple times
        assertEquals(2, dsFromNamespace.countRows());
        assertEquals(2, dsFromNamespace.countRows());
        assertEquals(2, dsFromNamespace.countRows());

        // After multiple reads, no additional describeTable calls should be made
        // (credentials are cached and don't expire during this fast test)
        assertEquals(
            1,
            namespace.getDescribeCallCount(),
            "describeTable should still be 1 after reads (credentials cached, no refresh needed)");
      }

      // Final verification
      assertEquals(1, namespace.getCreateCallCount(), "Final: createEmptyTable = 1");
      assertEquals(1, namespace.getDescribeCallCount(), "Final: describeTable = 1");
    }
  }

  @Test
  void testWriteDatasetBuilderWithNamespaceAppend() throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      // Set up storage options
      Map<String, String> storageOptions = new HashMap<>();
      storageOptions.put("allow_http", "true");
      storageOptions.put("aws_access_key_id", ACCESS_KEY);
      storageOptions.put("aws_secret_access_key", SECRET_KEY);
      storageOptions.put("aws_endpoint", ENDPOINT_URL);
      storageOptions.put("aws_region", REGION);

      // Create tracking namespace
      TrackingNamespace namespace = new TrackingNamespace(BUCKET_NAME, storageOptions, 60);
      String tableName = UUID.randomUUID().toString();

      Schema schema =
          new Schema(
              Arrays.asList(
                  new Field("a", FieldType.nullable(new ArrowType.Int(32, true)), null),
                  new Field("b", FieldType.nullable(new ArrowType.Int(32, true)), null)));

      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        IntVector aVector = (IntVector) root.getVector("a");
        IntVector bVector = (IntVector) root.getVector("b");

        aVector.allocateNew(2);
        bVector.allocateNew(2);

        aVector.set(0, 1);
        bVector.set(0, 2);
        aVector.set(1, 10);
        bVector.set(1, 20);

        aVector.setValueCount(2);
        bVector.setValueCount(2);
        root.setRowCount(2);

        // Create a test reader that returns our VectorSchemaRoot
        ArrowReader testReader =
            new ArrowReader(allocator) {
              boolean firstRead = true;

              @Override
              public boolean loadNextBatch() {
                if (firstRead) {
                  firstRead = false;
                  return true;
                }
                return false;
              }

              @Override
              public long bytesRead() {
                return 0;
              }

              @Override
              protected void closeReadSource() {}

              @Override
              protected Schema readSchema() {
                return schema;
              }

              @Override
              public VectorSchemaRoot getVectorSchemaRoot() {
                return root;
              }
            };

        // Create initial dataset through namespace
        try (Dataset dataset =
            Dataset.write()
                .allocator(allocator)
                .reader(testReader)
                .namespace(namespace)
                .tableId(Arrays.asList(tableName))
                .mode(WriteParams.WriteMode.CREATE)
                .execute()) {
          assertEquals(2, dataset.countRows());
        }

        assertEquals(1, namespace.getCreateCallCount(), "createEmptyTable should be called once");
        int initialDescribeCount = namespace.getDescribeCallCount();

        // Now append data using the write builder with namespace
        ArrowReader appendReader =
            new ArrowReader(allocator) {
              boolean firstRead = true;

              @Override
              public boolean loadNextBatch() {
                if (firstRead) {
                  firstRead = false;
                  return true;
                }
                return false;
              }

              @Override
              public long bytesRead() {
                return 0;
              }

              @Override
              protected void closeReadSource() {}

              @Override
              protected Schema readSchema() {
                return schema;
              }

              @Override
              public VectorSchemaRoot getVectorSchemaRoot() {
                return root;
              }
            };

        // Use the write builder to append to dataset through namespace
        try (Dataset dataset =
            Dataset.write()
                .allocator(allocator)
                .reader(appendReader)
                .namespace(namespace)
                .tableId(Arrays.asList(tableName))
                .mode(WriteParams.WriteMode.APPEND)
                .execute()) {

          // Verify describeTable was called
          int callCountAfter = namespace.getDescribeCallCount();
          assertEquals(
              1,
              callCountAfter - initialDescribeCount,
              "describeTable should be called once for append");

          // Verify data was appended successfully
          assertEquals(4, dataset.countRows()); // Original 2 + appended 2
        }
      }
    }
  }

  @Test
  void testWriteDatasetBuilderWithNamespaceOverwrite() throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      // Set up storage options
      Map<String, String> storageOptions = new HashMap<>();
      storageOptions.put("allow_http", "true");
      storageOptions.put("aws_access_key_id", ACCESS_KEY);
      storageOptions.put("aws_secret_access_key", SECRET_KEY);
      storageOptions.put("aws_endpoint", ENDPOINT_URL);
      storageOptions.put("aws_region", REGION);

      // Create tracking namespace
      TrackingNamespace namespace = new TrackingNamespace(BUCKET_NAME, storageOptions, 60);
      String tableName = UUID.randomUUID().toString();

      Schema schema =
          new Schema(
              Arrays.asList(
                  new Field("a", FieldType.nullable(new ArrowType.Int(32, true)), null),
                  new Field("b", FieldType.nullable(new ArrowType.Int(32, true)), null)));

      // Create initial dataset with 1 row
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        IntVector aVector = (IntVector) root.getVector("a");
        IntVector bVector = (IntVector) root.getVector("b");

        aVector.allocateNew(1);
        bVector.allocateNew(1);

        aVector.set(0, 1);
        bVector.set(0, 2);

        aVector.setValueCount(1);
        bVector.setValueCount(1);
        root.setRowCount(1);

        ArrowReader createReader =
            new ArrowReader(allocator) {
              boolean firstRead = true;

              @Override
              public boolean loadNextBatch() {
                if (firstRead) {
                  firstRead = false;
                  return true;
                }
                return false;
              }

              @Override
              public long bytesRead() {
                return 0;
              }

              @Override
              protected void closeReadSource() {}

              @Override
              protected Schema readSchema() {
                return schema;
              }

              @Override
              public VectorSchemaRoot getVectorSchemaRoot() {
                return root;
              }
            };

        try (Dataset dataset =
            Dataset.write()
                .allocator(allocator)
                .reader(createReader)
                .namespace(namespace)
                .tableId(Arrays.asList(tableName))
                .mode(WriteParams.WriteMode.CREATE)
                .execute()) {
          assertEquals(1, dataset.countRows());
        }

        assertEquals(1, namespace.getCreateCallCount(), "createEmptyTable should be called once");
        assertEquals(0, namespace.getDescribeCallCount(), "describeTable should not be called yet");

        // Now overwrite with 2 rows
        aVector.allocateNew(2);
        bVector.allocateNew(2);

        aVector.set(0, 10);
        bVector.set(0, 20);
        aVector.set(1, 100);
        bVector.set(1, 200);

        aVector.setValueCount(2);
        bVector.setValueCount(2);
        root.setRowCount(2);

        ArrowReader overwriteReader =
            new ArrowReader(allocator) {
              boolean firstRead = true;

              @Override
              public boolean loadNextBatch() {
                if (firstRead) {
                  firstRead = false;
                  return true;
                }
                return false;
              }

              @Override
              public long bytesRead() {
                return 0;
              }

              @Override
              protected void closeReadSource() {}

              @Override
              protected Schema readSchema() {
                return schema;
              }

              @Override
              public VectorSchemaRoot getVectorSchemaRoot() {
                return root;
              }
            };

        try (Dataset dataset =
            Dataset.write()
                .allocator(allocator)
                .reader(overwriteReader)
                .namespace(namespace)
                .tableId(Arrays.asList(tableName))
                .mode(WriteParams.WriteMode.OVERWRITE)
                .execute()) {

          // Verify describeTable was called for overwrite
          assertEquals(1, namespace.getCreateCallCount(), "createEmptyTable should still be 1");
          int describeCountAfterOverwrite = namespace.getDescribeCallCount();
          assertEquals(
              1, describeCountAfterOverwrite, "describeTable should be called once for overwrite");

          // Verify data was overwritten successfully
          assertEquals(2, dataset.countRows());
          assertEquals(
              2, dataset.listVersions().size()); // Version 1 (create) + Version 2 (overwrite)
        }

        // Verify we can open and read the dataset through namespace
        try (Dataset ds =
            Dataset.open()
                .allocator(allocator)
                .namespace(namespace)
                .tableId(Arrays.asList(tableName))
                .build()) {
          assertEquals(2, ds.countRows(), "Should have 2 rows after overwrite");
          assertEquals(2, ds.listVersions().size(), "Should have 2 versions");
        }
      }
    }
  }

  @Test
  void testDistributedWriteWithNamespace() throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      // Set up storage options
      Map<String, String> storageOptions = new HashMap<>();
      storageOptions.put("allow_http", "true");
      storageOptions.put("aws_access_key_id", ACCESS_KEY);
      storageOptions.put("aws_secret_access_key", SECRET_KEY);
      storageOptions.put("aws_endpoint", ENDPOINT_URL);
      storageOptions.put("aws_region", REGION);

      // Create tracking namespace
      TrackingNamespace namespace = new TrackingNamespace(BUCKET_NAME, storageOptions, 60);
      String tableName = UUID.randomUUID().toString();

      Schema schema =
          new Schema(
              Arrays.asList(
                  new Field("a", FieldType.nullable(new ArrowType.Int(32, true)), null),
                  new Field("b", FieldType.nullable(new ArrowType.Int(32, true)), null)));

      // Step 1: Create empty table via namespace
      CreateEmptyTableRequest request = new CreateEmptyTableRequest();
      request.setId(Arrays.asList(tableName));
      CreateEmptyTableResponse response = namespace.createEmptyTable(request);

      assertEquals(1, namespace.getCreateCallCount(), "createEmptyTable should be called once");
      assertEquals(0, namespace.getDescribeCallCount(), "describeTable should not be called yet");

      String tableUri = response.getLocation();
      Map<String, String> namespaceStorageOptions = response.getStorageOptions();

      // Merge storage options
      Map<String, String> mergedOptions = new HashMap<>(storageOptions);
      if (namespaceStorageOptions != null) {
        mergedOptions.putAll(namespaceStorageOptions);
      }

      // Create storage options provider
      LanceNamespaceStorageOptionsProvider storageOptionsProvider =
          new LanceNamespaceStorageOptionsProvider(namespace, Arrays.asList(tableName));

      WriteParams writeParams = new WriteParams.Builder().withStorageOptions(mergedOptions).build();

      // Step 2: Write multiple fragments in parallel (simulated)
      List<FragmentMetadata> allFragments = new ArrayList<>();

      // Fragment 1: 2 rows
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        IntVector aVector = (IntVector) root.getVector("a");
        IntVector bVector = (IntVector) root.getVector("b");

        aVector.allocateNew(2);
        bVector.allocateNew(2);
        aVector.set(0, 1);
        bVector.set(0, 2);
        aVector.set(1, 3);
        bVector.set(1, 4);
        aVector.setValueCount(2);
        bVector.setValueCount(2);
        root.setRowCount(2);

        List<FragmentMetadata> fragment1 =
            Fragment.create(tableUri, allocator, root, writeParams, storageOptionsProvider);
        allFragments.addAll(fragment1);
      }

      // Fragment 2: 2 rows
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        IntVector aVector = (IntVector) root.getVector("a");
        IntVector bVector = (IntVector) root.getVector("b");

        aVector.allocateNew(2);
        bVector.allocateNew(2);
        aVector.set(0, 10);
        bVector.set(0, 20);
        aVector.set(1, 30);
        bVector.set(1, 40);
        aVector.setValueCount(2);
        bVector.setValueCount(2);
        root.setRowCount(2);

        List<FragmentMetadata> fragment2 =
            Fragment.create(tableUri, allocator, root, writeParams, storageOptionsProvider);
        allFragments.addAll(fragment2);
      }

      // Fragment 3: 1 row
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        IntVector aVector = (IntVector) root.getVector("a");
        IntVector bVector = (IntVector) root.getVector("b");

        aVector.allocateNew(1);
        bVector.allocateNew(1);
        aVector.set(0, 100);
        bVector.set(0, 200);
        aVector.setValueCount(1);
        bVector.setValueCount(1);
        root.setRowCount(1);

        List<FragmentMetadata> fragment3 =
            Fragment.create(tableUri, allocator, root, writeParams, storageOptionsProvider);
        allFragments.addAll(fragment3);
      }

      // Step 3: Commit all fragments as one operation
      FragmentOperation.Overwrite overwriteOp =
          new FragmentOperation.Overwrite(allFragments, schema);

      try (Dataset dataset =
          Dataset.commit(allocator, tableUri, overwriteOp, Optional.empty(), mergedOptions)) {
        assertEquals(5, dataset.countRows(), "Should have 5 total rows from all fragments");
        assertEquals(1, dataset.listVersions().size(), "Should have 1 version after commit");
      }

      // Step 4: Open dataset through namespace and verify
      try (Dataset dsFromNamespace =
          Dataset.open()
              .allocator(allocator)
              .namespace(namespace)
              .tableId(Arrays.asList(tableName))
              .build()) {
        assertEquals(5, dsFromNamespace.countRows(), "Should read 5 rows through namespace");
      }
    }
  }

  @Test
  void testFragmentCreateAndCommitWithNamespace() throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      // Set up storage options
      Map<String, String> storageOptions = new HashMap<>();
      storageOptions.put("allow_http", "true");
      storageOptions.put("aws_access_key_id", ACCESS_KEY);
      storageOptions.put("aws_secret_access_key", SECRET_KEY);
      storageOptions.put("aws_endpoint", ENDPOINT_URL);
      storageOptions.put("aws_region", REGION);

      // Create tracking namespace with 60-second expiration
      TrackingNamespace namespace = new TrackingNamespace(BUCKET_NAME, storageOptions, 60);
      String tableName = UUID.randomUUID().toString();

      Schema schema =
          new Schema(
              Arrays.asList(
                  new Field("id", FieldType.nullable(new ArrowType.Int(32, true)), null),
                  new Field("value", FieldType.nullable(new ArrowType.Int(32, true)), null)));

      // Create empty table via namespace
      CreateEmptyTableRequest request = new CreateEmptyTableRequest();
      request.setId(Arrays.asList(tableName));
      CreateEmptyTableResponse response = namespace.createEmptyTable(request);

      assertEquals(1, namespace.getCreateCallCount(), "createEmptyTable should be called once");

      String tableUri = response.getLocation();
      Map<String, String> namespaceStorageOptions = response.getStorageOptions();

      // Merge storage options
      Map<String, String> mergedOptions = new HashMap<>(storageOptions);
      if (namespaceStorageOptions != null) {
        mergedOptions.putAll(namespaceStorageOptions);
      }

      // Create storage options provider
      LanceNamespaceStorageOptionsProvider provider =
          new LanceNamespaceStorageOptionsProvider(namespace, Arrays.asList(tableName));

      WriteParams writeParams = new WriteParams.Builder().withStorageOptions(mergedOptions).build();

      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        IntVector idVector = (IntVector) root.getVector("id");
        IntVector valueVector = (IntVector) root.getVector("value");

        // Write first fragment
        idVector.allocateNew(3);
        valueVector.allocateNew(3);

        idVector.set(0, 1);
        valueVector.set(0, 100);
        idVector.set(1, 2);
        valueVector.set(1, 200);
        idVector.set(2, 3);
        valueVector.set(2, 300);

        idVector.setValueCount(3);
        valueVector.setValueCount(3);
        root.setRowCount(3);

        // Create fragment with StorageOptionsProvider
        List<FragmentMetadata> fragments1 =
            Fragment.create(tableUri, allocator, root, writeParams, provider);

        assertEquals(1, fragments1.size());

        // Write second fragment with different data
        idVector.set(0, 4);
        valueVector.set(0, 400);
        idVector.set(1, 5);
        valueVector.set(1, 500);
        idVector.set(2, 6);
        valueVector.set(2, 600);
        root.setRowCount(3);

        // Create another fragment with the same provider
        List<FragmentMetadata> fragments2 =
            Fragment.create(tableUri, allocator, root, writeParams, provider);

        assertEquals(1, fragments2.size());

        // Commit first fragment to the dataset using Overwrite (for empty table)
        FragmentOperation.Overwrite overwriteOp =
            new FragmentOperation.Overwrite(fragments1, schema);
        try (Dataset updatedDataset =
            Dataset.commit(allocator, tableUri, overwriteOp, Optional.empty(), mergedOptions)) {
          assertEquals(1, updatedDataset.version());
          assertEquals(3, updatedDataset.countRows());

          // Append second fragment
          FragmentOperation.Append appendOp2 = new FragmentOperation.Append(fragments2);
          try (Dataset finalDataset =
              Dataset.commit(allocator, tableUri, appendOp2, Optional.of(1L), mergedOptions)) {
            assertEquals(2, finalDataset.version());
            assertEquals(6, finalDataset.countRows());
          }
        }
      }

      // Verify we can open and read the dataset through namespace
      try (Dataset ds =
          Dataset.open()
              .allocator(allocator)
              .namespace(namespace)
              .tableId(Arrays.asList(tableName))
              .build()) {
        assertEquals(6, ds.countRows(), "Should have 6 rows total");
        assertEquals(2, ds.listVersions().size(), "Should have 2 versions");
      }
    }
  }

  @Test
  void testTransactionCommitWithNamespace() throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      // Set up storage options
      Map<String, String> storageOptions = new HashMap<>();
      storageOptions.put("allow_http", "true");
      storageOptions.put("aws_access_key_id", ACCESS_KEY);
      storageOptions.put("aws_secret_access_key", SECRET_KEY);
      storageOptions.put("aws_endpoint", ENDPOINT_URL);
      storageOptions.put("aws_region", REGION);

      // Create tracking namespace
      TrackingNamespace namespace = new TrackingNamespace(BUCKET_NAME, storageOptions, 60);
      String tableName = UUID.randomUUID().toString();

      Schema schema =
          new Schema(
              Arrays.asList(
                  new Field("id", FieldType.nullable(new ArrowType.Int(32, true)), null),
                  new Field("name", FieldType.nullable(new ArrowType.Utf8()), null)));

      // Create empty table via namespace
      CreateEmptyTableRequest request = new CreateEmptyTableRequest();
      request.setId(Arrays.asList(tableName));
      CreateEmptyTableResponse response = namespace.createEmptyTable(request);

      String tableUri = response.getLocation();
      Map<String, String> namespaceStorageOptions = response.getStorageOptions();

      // Merge storage options
      Map<String, String> mergedOptions = new HashMap<>(storageOptions);
      if (namespaceStorageOptions != null) {
        mergedOptions.putAll(namespaceStorageOptions);
      }

      // Create storage options provider
      LanceNamespaceStorageOptionsProvider provider =
          new LanceNamespaceStorageOptionsProvider(namespace, Arrays.asList(tableName));

      // First, write some initial data using Fragment.create and commit
      WriteParams writeParams = new WriteParams.Builder().withStorageOptions(mergedOptions).build();

      List<FragmentMetadata> initialFragments;
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        IntVector idVector = (IntVector) root.getVector("id");
        org.apache.arrow.vector.VarCharVector nameVector =
            (org.apache.arrow.vector.VarCharVector) root.getVector("name");

        idVector.allocateNew(2);
        nameVector.allocateNew(2);

        idVector.set(0, 1);
        nameVector.setSafe(0, "Alice".getBytes());
        idVector.set(1, 2);
        nameVector.setSafe(1, "Bob".getBytes());

        idVector.setValueCount(2);
        nameVector.setValueCount(2);
        root.setRowCount(2);

        initialFragments = Fragment.create(tableUri, allocator, root, writeParams, provider);
      }

      // Commit initial fragments
      FragmentOperation.Overwrite overwriteOp =
          new FragmentOperation.Overwrite(initialFragments, schema);
      try (Dataset dataset =
          Dataset.commit(allocator, tableUri, overwriteOp, Optional.empty(), mergedOptions)) {
        assertEquals(1, dataset.version());
        assertEquals(2, dataset.countRows());
      }

      // Now test Transaction.commit with provider
      // Open dataset with provider using mergedOptions (which has expires_at_millis)
      ReadOptions readOptions =
          new ReadOptions.Builder()
              .setStorageOptions(mergedOptions)
              .setStorageOptionsProvider(provider)
              .build();

      try (Dataset datasetWithProvider = Dataset.open(allocator, tableUri, readOptions)) {
        // Create more fragments to append
        List<FragmentMetadata> newFragments;
        try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
          IntVector idVector = (IntVector) root.getVector("id");
          org.apache.arrow.vector.VarCharVector nameVector =
              (org.apache.arrow.vector.VarCharVector) root.getVector("name");

          idVector.allocateNew(2);
          nameVector.allocateNew(2);

          idVector.set(0, 3);
          nameVector.setSafe(0, "Charlie".getBytes());
          idVector.set(1, 4);
          nameVector.setSafe(1, "Diana".getBytes());

          idVector.setValueCount(2);
          nameVector.setValueCount(2);
          root.setRowCount(2);

          newFragments = Fragment.create(tableUri, allocator, root, writeParams, provider);
        }

        // Create and commit transaction
        Append appendOp = Append.builder().fragments(newFragments).build();
        try (Transaction transaction =
            new Transaction.Builder()
                .readVersion(datasetWithProvider.version())
                .operation(appendOp)
                .build()) {
          try (Dataset committedDataset =
              new CommitBuilder(datasetWithProvider).execute(transaction)) {
            assertEquals(2, committedDataset.version());
            assertEquals(4, committedDataset.countRows());
          }
        }
      }

      // Verify we can open and read the dataset through namespace
      try (Dataset ds =
          Dataset.open()
              .allocator(allocator)
              .namespace(namespace)
              .tableId(Arrays.asList(tableName))
              .build()) {
        assertEquals(4, ds.countRows(), "Should have 4 rows total");
        assertEquals(2, ds.listVersions().size(), "Should have 2 versions");
      }
    }
  }
}
