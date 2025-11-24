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

import com.lancedb.lance.namespace.LanceNamespace;
import com.lancedb.lance.namespace.model.DescribeTableRequest;
import com.lancedb.lance.namespace.model.DescribeTableResponse;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
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
 * <p>This test simulates a mock credential provider that returns incrementing credentials and
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
   * Mock LanceNamespace implementation for testing.
   *
   * <p>This implementation: - Returns table location and storage options via describeTable() -
   * Tracks the number of times describeTable has been called - Returns credentials with short
   * expiration times for testing refresh
   */
  static class MockLanceNamespace implements LanceNamespace {
    private final Map<String, String> tableLocations = new HashMap<>();
    private final Map<String, String> baseStorageOptions;
    private final int credentialExpiresInSeconds;
    private final AtomicInteger callCount = new AtomicInteger(0);

    public MockLanceNamespace(Map<String, String> storageOptions, int credentialExpiresInSeconds) {
      this.baseStorageOptions = new HashMap<>(storageOptions);
      this.credentialExpiresInSeconds = credentialExpiresInSeconds;
    }

    @Override
    public void initialize(Map<String, String> configProperties, BufferAllocator allocator) {
      // Not needed for test
    }

    public void registerTable(String tableName, String location) {
      tableLocations.put(tableName, location);
    }

    public int getCallCount() {
      return callCount.get();
    }

    @Override
    public String namespaceId() {
      return "MockLanceNamespace { }";
    }

    @Override
    public DescribeTableResponse describeTable(DescribeTableRequest request) {
      int count = callCount.incrementAndGet();

      String tableName = String.join("/", request.getId());
      String location = tableLocations.get(tableName);
      if (location == null) {
        throw new IllegalArgumentException("Table not found: " + tableName);
      }

      // Create storage options with expiration
      Map<String, String> storageOptions = new HashMap<>(baseStorageOptions);
      long expiresAtMillis = System.currentTimeMillis() + (credentialExpiresInSeconds * 1000L);
      storageOptions.put("expires_at_millis", String.valueOf(expiresAtMillis));

      DescribeTableResponse response = new DescribeTableResponse();
      response.setLocation(location);
      response.setStorageOptions(storageOptions);
      if (request.getVersion() != null) {
        response.setVersion(request.getVersion());
      }

      return response;
    }
  }

  @Test
  void testOpenDatasetWithoutRefresh() throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      // Create test dataset directly on S3
      String tableName = UUID.randomUUID().toString();
      String tableUri = "s3://" + BUCKET_NAME + "/" + tableName + ".lance";

      Map<String, String> storageOptions = new HashMap<>();
      storageOptions.put("allow_http", "true");
      storageOptions.put("aws_access_key_id", ACCESS_KEY);
      storageOptions.put("aws_secret_access_key", SECRET_KEY);
      storageOptions.put("aws_endpoint", ENDPOINT_URL);
      storageOptions.put("aws_region", REGION);

      // Create schema and write dataset
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

        WriteParams writeParams =
            new WriteParams.Builder().withStorageOptions(storageOptions).build();

        // Create dataset using Dataset.create
        try (Dataset dataset = Dataset.create(allocator, tableUri, schema, writeParams)) {
          // Add data via fragments
          List<FragmentMetadata> fragments =
              Fragment.create(tableUri, allocator, root, writeParams);
          FragmentOperation.Append appendOp = new FragmentOperation.Append(fragments);
          try (Dataset updatedDataset =
              Dataset.commit(allocator, tableUri, appendOp, Optional.of(1L), storageOptions)) {
            assertEquals(2, updatedDataset.version());
            assertEquals(2, updatedDataset.countRows());
          }
        }
      }

      // Create mock namespace with 60-second expiration (long enough to not expire during test)
      MockLanceNamespace namespace = new MockLanceNamespace(storageOptions, 60);
      namespace.registerTable(tableName, tableUri);

      // Open dataset through namespace WITH refresh enabled
      // Use 10-second refresh offset, so credentials effectively expire at T+50s
      ReadOptions readOptions =
          new ReadOptions.Builder()
              .setS3CredentialsRefreshOffsetSeconds(10) // Refresh 10s before expiration
              .build();

      int callCountBeforeOpen = namespace.getCallCount();
      try (Dataset dsFromNamespace =
          Dataset.open()
              .allocator(allocator)
              .namespace(namespace)
              .tableId(Arrays.asList(tableName))
              .readOptions(readOptions)
              .build()) {
        // With the fix, describeTable should only be called once during open
        // to get the table location and initial storage options
        int callCountAfterOpen = namespace.getCallCount();
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
        assertEquals(2, versions.size());

        // With the fix, credentials are cached so no additional calls are made
        int finalCallCount = namespace.getCallCount();
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
      // Create test dataset
      String tableName = UUID.randomUUID().toString();
      String tableUri = "s3://" + BUCKET_NAME + "/" + tableName + ".lance";

      Map<String, String> storageOptions = new HashMap<>();
      storageOptions.put("allow_http", "true");
      storageOptions.put("aws_access_key_id", ACCESS_KEY);
      storageOptions.put("aws_secret_access_key", SECRET_KEY);
      storageOptions.put("aws_endpoint", ENDPOINT_URL);
      storageOptions.put("aws_region", REGION);

      // Create schema and write dataset
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

        WriteParams writeParams =
            new WriteParams.Builder().withStorageOptions(storageOptions).build();

        // Create dataset using Dataset.create
        try (Dataset dataset = Dataset.create(allocator, tableUri, schema, writeParams)) {
          // Add data via fragments
          List<FragmentMetadata> fragments =
              Fragment.create(tableUri, allocator, root, writeParams);
          FragmentOperation.Append appendOp = new FragmentOperation.Append(fragments);
          try (Dataset updatedDataset =
              Dataset.commit(allocator, tableUri, appendOp, Optional.of(1L), storageOptions)) {
            assertEquals(2, updatedDataset.countRows());
          }
        }
      }

      // Create mock namespace with 5-second expiration for faster testing
      MockLanceNamespace namespace = new MockLanceNamespace(storageOptions, 5);
      namespace.registerTable(tableName, tableUri);

      // Open dataset through namespace with refresh enabled
      // Use 2-second refresh offset so credentials effectively expire at T+3s (5s - 2s)
      ReadOptions readOptions =
          new ReadOptions.Builder()
              .setS3CredentialsRefreshOffsetSeconds(2) // Refresh 2s before expiration
              .build();

      int callCountBeforeOpen = namespace.getCallCount();
      try (Dataset dsFromNamespace =
          Dataset.open()
              .allocator(allocator)
              .namespace(namespace)
              .tableId(Arrays.asList(tableName))
              .readOptions(readOptions)
              .build()) {
        // With the fix, describeTable should only be called once during open
        int callCountAfterOpen = namespace.getCallCount();
        assertEquals(
            1,
            callCountAfterOpen - callCountBeforeOpen,
            "describeTable should be called exactly once during open, got: "
                + (callCountAfterOpen - callCountBeforeOpen));

        // Verify we can read the data
        assertEquals(2, dsFromNamespace.countRows());

        // Record call count after initial reads
        int callCountAfterInitialReads = namespace.getCallCount();
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
        assertEquals(2, versions.size());

        int finalCallCount = namespace.getCallCount();
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
}
