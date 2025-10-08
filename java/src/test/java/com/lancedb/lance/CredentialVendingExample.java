// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

package com.lancedb.lance;

import java.util.HashMap;
import java.util.Map;

/**
 * Example: Using Credential Vending with Lance Datasets in Java
 *
 * This example demonstrates how to implement custom credential vendors
 * for automatic credential refresh with cloud storage backends.
 *
 * <p>Credential vending allows Lance to automatically refresh cloud storage
 * credentials before they expire, enabling long-running operations on
 * datasets stored in S3, Azure, or GCS.</p>
 */
public class CredentialVendingExample {

  // ==========================================================================
  // Example 1: Custom Credential Vendor Interface
  // ==========================================================================

  /**
   * Interface that must be implemented by credential vendors.
   *
   * <p>Lance will call getCredentials() to fetch fresh credentials when needed.</p>
   */
  public interface CredentialVendor {
    /**
     * Fetch fresh credentials from your credential service.
     *
     * @return Map containing:
     *         - "storage_options": Map&lt;String, String&gt; with credential keys
     *         - "expires_at_millis": Long with expiration timestamp
     */
    Map<String, Object> getCredentials();
  }

  // ==========================================================================
  // Example 2: Custom Implementation
  // ==========================================================================

  /**
   * Example custom credential vendor implementation.
   *
   * <p>This example shows how to implement a credential vendor that fetches
   * credentials from any source (AWS STS, HashiCorp Vault, custom service, etc.)</p>
   */
  public static class CustomCredentialVendor implements CredentialVendor {
    private final String credentialsServiceUrl;
    private int refreshCount = 0;

    public CustomCredentialVendor(String credentialsServiceUrl) {
      this.credentialsServiceUrl = credentialsServiceUrl;
    }

    @Override
    public Map<String, Object> getCredentials() {
      refreshCount++;

      // In a real implementation, you would:
      // 1. Call your credential service API
      // 2. Parse the response
      // 3. Return the credentials with expiration time

      // Example: Simulating AWS STS AssumeRole response
      Map<String, String> credentials = fetchFromService();

      // Build storage options
      Map<String, String> storageOptions = new HashMap<>();
      storageOptions.put("aws_access_key_id", credentials.get("AccessKeyId"));
      storageOptions.put("aws_secret_access_key", credentials.get("SecretAccessKey"));
      storageOptions.put("aws_session_token", credentials.get("SessionToken"));
      storageOptions.put("expires_at_millis", credentials.get("ExpiresAtMillis"));

      // Build result
      Map<String, Object> result = new HashMap<>();
      result.put("storage_options", storageOptions);
      result.put("expires_at_millis", Long.parseLong(credentials.get("ExpiresAtMillis")));

      return result;
    }

    private Map<String, String> fetchFromService() {
      // In real code, this would make an HTTP request
      // For example, using OkHttp, Apache HttpClient, or java.net.http
      /*
      HttpClient client = HttpClient.newHttpClient();
      HttpRequest request = HttpRequest.newBuilder()
          .uri(URI.create(credentialsServiceUrl))
          .POST(HttpRequest.BodyPublishers.ofString("{\"role\": \"data-reader\"}"))
          .build();
      HttpResponse<String> response = client.send(request,
          HttpResponse.BodyHandlers.ofString());
      return parseJsonResponse(response.body());
      */

      // Mock credentials (valid for 1 hour)
      long expiresAt = System.currentTimeMillis() + 3600_000;

      Map<String, String> credentials = new HashMap<>();
      credentials.put("AccessKeyId", "ASIA_MOCK_" + refreshCount);
      credentials.put("SecretAccessKey", "mock_secret_key");
      credentials.put("SessionToken", "mock_session_token");
      credentials.put("ExpiresAtMillis", String.valueOf(expiresAt));

      return credentials;
    }

    public int getRefreshCount() {
      return refreshCount;
    }
  }

  // ==========================================================================
  // Example 3: Usage with Lance Dataset
  // ==========================================================================

  /**
   * Example showing how to use a custom credential vendor with a Lance dataset.
   */
  public static void exampleCustomVendor() {
    System.out.println("======================================================================");
    System.out.println("Example 1: Custom Java Credential Vendor");
    System.out.println("======================================================================");

    // Create your custom vendor
    CustomCredentialVendor vendor = new CustomCredentialVendor(
        "https://my-creds-service.com/get-credentials"
    );

    // Use it with a dataset (via JNI)
    // Note: This would be used with real S3/cloud URIs
    /*
    // Pseudo-code showing how the JNI binding would work:
    Dataset dataset = Dataset.open(
        "s3://my-bucket/my-dataset.lance",
        new ReadOptions.Builder()
            .credentialVendor(vendor)
            .build()
    );
    */

    System.out.println("✓ Custom vendor created");
    System.out.println("  Service URL: " + vendor.credentialsServiceUrl);
    System.out.println("  Refresh count: " + vendor.getRefreshCount());
    System.out.println();

    // Test credential fetching
    Map<String, Object> creds = vendor.getCredentials();
    @SuppressWarnings("unchecked")
    Map<String, String> storageOptions = (Map<String, String>) creds.get("storage_options");
    Long expiresAt = (Long) creds.get("expires_at_millis");

    System.out.println("✓ Credentials fetched:");
    System.out.println("  Access Key: " + storageOptions.get("aws_access_key_id"));
    System.out.println("  Expires: " + new java.util.Date(expiresAt));
    System.out.println();
  }

  // ==========================================================================
  // Example 4: Multi-Account Credential Vendor
  // ==========================================================================

  /**
   * Advanced example: Vendor that handles multiple AWS accounts.
   *
   * <p>This shows how you might implement cross-account access or
   * environment-specific credential management.</p>
   */
  public static class MultiAccountCredentialVendor implements CredentialVendor {
    private final Map<String, String> accountMapping;
    private String currentAccount;

    /**
     * Create a multi-account credential vendor.
     *
     * @param accountMapping Maps dataset prefixes to AWS role ARNs
     */
    public MultiAccountCredentialVendor(Map<String, String> accountMapping) {
      this.accountMapping = accountMapping;
    }

    @Override
    public Map<String, Object> getCredentials() {
      // Determine which account based on dataset path
      // In real implementation, you'd use AWS SDK's STS.assumeRole()

      long expiresAt = System.currentTimeMillis() + 3600_000;

      Map<String, String> storageOptions = new HashMap<>();
      storageOptions.put("aws_access_key_id", "ASIA_CROSS_ACCOUNT");
      storageOptions.put("aws_secret_access_key", "secret");
      storageOptions.put("aws_session_token", "token");
      storageOptions.put("expires_at_millis", String.valueOf(expiresAt));

      Map<String, Object> result = new HashMap<>();
      result.put("storage_options", storageOptions);
      result.put("expires_at_millis", expiresAt);

      return result;
    }
  }

  /**
   * Example showing advanced credential vending patterns.
   */
  public static void exampleAdvancedUseCases() {
    System.out.println("======================================================================");
    System.out.println("Example 2: Advanced Use Cases");
    System.out.println("======================================================================");

    // Multi-account scenario
    Map<String, String> accountMapping = new HashMap<>();
    accountMapping.put("s3://prod-bucket", "arn:aws:iam::123456789012:role/ProdDataReader");
    accountMapping.put("s3://dev-bucket", "arn:aws:iam::987654321098:role/DevDataReader");

    MultiAccountCredentialVendor vendor = new MultiAccountCredentialVendor(accountMapping);

    System.out.println("✓ Multi-account vendor created");
    System.out.println("  Supports cross-account access patterns");
    System.out.println();

    // Example with AWS SDK STS
    System.out.println("Integration with AWS SDK:");
    System.out.println("""
        import software.amazon.awssdk.services.sts.StsClient;
        import software.amazon.awssdk.services.sts.model.AssumeRoleRequest;
        import software.amazon.awssdk.services.sts.model.AssumeRoleResponse;

        StsClient stsClient = StsClient.create();
        AssumeRoleRequest roleRequest = AssumeRoleRequest.builder()
            .roleArn("arn:aws:iam::123456789012:role/DataReader")
            .roleSessionName("lance-session")
            .durationSeconds(3600)
            .build();

        AssumeRoleResponse roleResponse = stsClient.assumeRole(roleRequest);
        Credentials credentials = roleResponse.credentials();

        // Use credentials.accessKeyId(), secretAccessKey(), sessionToken()
        """);
    System.out.println();
  }

  // ==========================================================================
  // Example 5: LanceNamespace Integration (Future)
  // ==========================================================================

  /**
   * Example showing how LanceNamespace integration would work.
   *
   * <p>Note: This is示范代码 showing the planned API. Full implementation
   * requires JNI bindings for lance-namespace.</p>
   */
  public static void exampleNamespaceIntegration() {
    System.out.println("======================================================================");
    System.out.println("Example 3: LanceNamespace Integration (Planned API)");
    System.out.println("======================================================================");

    System.out.println("Future API for namespace-based credential vending:");
    System.out.println("""

        // Connect to namespace service
        LanceNamespace namespace = LanceNamespace.connect("rest", Map.of(
            "url", "https://api.lancedb.com",
            "api_key", "your-api-key"
        ));

        // Describe table to get metadata including credentials
        TableInfo tableInfo = namespace.describeTable(
            List.of("workspace", "table"),
            null  // version
        );

        // tableInfo contains:
        // - location: "s3://bucket/path/table.lance"
        // - storageOptions: Map with credentials
        // - version: table version

        // Create credential vendor from namespace
        CredentialVendor vendor = new LanceNamespaceCredentialVendor(
            namespace,
            List.of("workspace", "table")
        );

        // Use with dataset - automatic credential refresh!
        Dataset dataset = Dataset.open(
            tableInfo.getLocation(),
            new ReadOptions.Builder()
                .credentialVendor(vendor)
                .build()
        );
        """);

    System.out.println("\nDirectory-based namespace (for local/testing):");
    System.out.println("""

        // Connect to directory-based namespace
        LanceNamespace namespace = LanceNamespace.connect("dir", Map.of(
            "path", "/tmp/lance-namespace"
        ));

        // Register a table
        namespace.registerTable(
            List.of("my_workspace", "my_table"),
            "file:///path/to/table.lance",
            schema
        );

        // Later, describe it to get credentials
        TableInfo info = namespace.describeTable(
            List.of("my_workspace", "my_table"),
            null
        );
        """);

    System.out.println();
  }

  // ==========================================================================
  // Main - Run Examples
  // ==========================================================================

  /**
   * Run all examples.
   */
  public static void main(String[] args) {
    System.out.println();
    System.out.println("**********************************************************************");
    System.out.println("Lance Credential Vending Examples (Java)");
    System.out.println("**********************************************************************");
    System.out.println();

    // Run examples
    exampleCustomVendor();
    exampleAdvancedUseCases();
    exampleNamespaceIntegration();

    System.out.println("======================================================================");
    System.out.println("Summary");
    System.out.println("======================================================================");
    System.out.println("""
        Key Takeaways:

        1. Custom Vendors: Implement CredentialVendor interface
        2. getCredentials(): Return Map with storage_options and expires_at_millis
        3. Automatic Refresh: Lance refreshes credentials before they expire
        4. JNI Integration: Pass vendor instance via ReadOptions

        Use Cases:
        - Long-running queries on S3/cloud data
        - Cross-account access in AWS
        - Integration with corporate credential services
        - LanceDB Cloud integration

        For more information:
        - Documentation: https://lancedb.github.io/lance/
        - GitHub: https://github.com/lancedb/lance
        """);
    System.out.println();
  }
}
