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

import org.lance.Dataset;
import org.lance.ReadOptions;
import org.lance.WriteParams;
import org.lance.namespace.model.*;
import org.lance.namespace.model.DescribeTableVersionRequest;
import org.lance.namespace.model.DescribeTableVersionResponse;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/** Tests for DirectoryNamespace implementation. */
public class DirectoryNamespaceTest {
  @TempDir Path tempDir;

  private BufferAllocator allocator;
  private DirectoryNamespace namespace;

  @BeforeEach
  void setUp() {
    allocator = new RootAllocator(Long.MAX_VALUE);
    namespace = new DirectoryNamespace();

    Map<String, String> config = new HashMap<>();
    config.put("root", tempDir.toString());
    namespace.initialize(config, allocator);
  }

  @AfterEach
  void tearDown() {
    if (namespace != null) {
      namespace.close();
    }
    if (allocator != null) {
      allocator.close();
    }
  }

  private byte[] createTestTableData() throws Exception {
    Schema schema =
        new Schema(
            Arrays.asList(
                new Field("id", FieldType.nullable(new ArrowType.Int(32, true)), null),
                new Field("name", FieldType.nullable(new ArrowType.Utf8()), null),
                new Field("age", FieldType.nullable(new ArrowType.Int(32, true)), null)));

    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      IntVector idVector = (IntVector) root.getVector("id");
      VarCharVector nameVector = (VarCharVector) root.getVector("name");
      IntVector ageVector = (IntVector) root.getVector("age");

      // Allocate space for 3 rows
      idVector.allocateNew(3);
      nameVector.allocateNew(3);
      ageVector.allocateNew(3);

      idVector.set(0, 1);
      nameVector.set(0, "Alice".getBytes());
      ageVector.set(0, 30);

      idVector.set(1, 2);
      nameVector.set(1, "Bob".getBytes());
      ageVector.set(1, 25);

      idVector.set(2, 3);
      nameVector.set(2, "Charlie".getBytes());
      ageVector.set(2, 35);

      // Set value counts
      idVector.setValueCount(3);
      nameVector.setValueCount(3);
      ageVector.setValueCount(3);
      root.setRowCount(3);

      ByteArrayOutputStream out = new ByteArrayOutputStream();
      try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, out)) {
        writer.writeBatch();
      }
      return out.toByteArray();
    }
  }

  @Test
  void testNamespaceId() {
    String namespaceId = namespace.namespaceId();
    assertNotNull(namespaceId);
    assertTrue(namespaceId.contains("DirectoryNamespace"));
  }

  @Test
  void testCreateAndListNamespaces() {
    // Create a namespace
    CreateNamespaceRequest createReq = new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    CreateNamespaceResponse createResp = namespace.createNamespace(createReq);
    assertNotNull(createResp);

    // List namespaces
    ListNamespacesRequest listReq = new ListNamespacesRequest();
    ListNamespacesResponse listResp = namespace.listNamespaces(listReq);
    assertNotNull(listResp);
    assertNotNull(listResp.getNamespaces());
    assertTrue(listResp.getNamespaces().contains("workspace"));
  }

  @Test
  void testDescribeNamespace() {
    // Create a namespace
    CreateNamespaceRequest createReq = new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespace.createNamespace(createReq);

    // Describe namespace
    DescribeNamespaceRequest descReq =
        new DescribeNamespaceRequest().id(Arrays.asList("workspace"));
    DescribeNamespaceResponse descResp = namespace.describeNamespace(descReq);
    assertNotNull(descResp);
    assertNotNull(descResp.getProperties());
  }

  @Test
  void testNamespaceExists() {
    // Create a namespace
    CreateNamespaceRequest createReq = new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespace.createNamespace(createReq);

    // Check existence
    NamespaceExistsRequest existsReq = new NamespaceExistsRequest().id(Arrays.asList("workspace"));
    assertDoesNotThrow(() -> namespace.namespaceExists(existsReq));

    // Check non-existent namespace
    NamespaceExistsRequest notExistsReq =
        new NamespaceExistsRequest().id(Arrays.asList("nonexistent"));
    assertThrows(RuntimeException.class, () -> namespace.namespaceExists(notExistsReq));
  }

  @Test
  void testDropNamespace() {
    // Create a namespace
    CreateNamespaceRequest createReq = new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespace.createNamespace(createReq);

    // Drop namespace
    DropNamespaceRequest dropReq = new DropNamespaceRequest().id(Arrays.asList("workspace"));
    DropNamespaceResponse dropResp = namespace.dropNamespace(dropReq);
    assertNotNull(dropResp);

    // Verify it's gone
    NamespaceExistsRequest existsReq = new NamespaceExistsRequest().id(Arrays.asList("workspace"));
    assertThrows(RuntimeException.class, () -> namespace.namespaceExists(existsReq));
  }

  @Test
  void testCreateTable() throws Exception {
    // Create parent namespace
    CreateNamespaceRequest createNsReq =
        new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespace.createNamespace(createNsReq);

    // Create table with data
    byte[] tableData = createTestTableData();
    CreateTableRequest createReq =
        new CreateTableRequest().id(Arrays.asList("workspace", "test_table"));
    CreateTableResponse createResp = namespace.createTable(createReq, tableData);

    assertNotNull(createResp);
    assertNotNull(createResp.getLocation());
    assertTrue(createResp.getLocation().contains("test_table"));
    assertEquals(Long.valueOf(1), createResp.getVersion());
  }

  @Test
  void testListTables() throws Exception {
    // Create parent namespace
    CreateNamespaceRequest createNsReq =
        new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespace.createNamespace(createNsReq);

    // Create a table
    byte[] tableData = createTestTableData();
    CreateTableRequest createReq =
        new CreateTableRequest().id(Arrays.asList("workspace", "test_table"));
    namespace.createTable(createReq, tableData);

    // List tables
    ListTablesRequest listReq = new ListTablesRequest().id(Arrays.asList("workspace"));
    ListTablesResponse listResp = namespace.listTables(listReq);

    assertNotNull(listResp);
    assertNotNull(listResp.getTables());
    assertTrue(listResp.getTables().contains("test_table"));
  }

  @Test
  void testDescribeTable() throws Exception {
    // Create parent namespace
    CreateNamespaceRequest createNsReq =
        new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespace.createNamespace(createNsReq);

    // Create a table
    byte[] tableData = createTestTableData();
    CreateTableRequest createReq =
        new CreateTableRequest().id(Arrays.asList("workspace", "test_table"));
    namespace.createTable(createReq, tableData);

    // Describe table
    DescribeTableRequest descReq =
        new DescribeTableRequest().id(Arrays.asList("workspace", "test_table"));
    DescribeTableResponse descResp = namespace.describeTable(descReq);

    assertNotNull(descResp);
    assertNotNull(descResp.getLocation());
    assertTrue(descResp.getLocation().contains("test_table"));
  }

  @Test
  void testTableExists() throws Exception {
    // Create parent namespace
    CreateNamespaceRequest createNsReq =
        new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespace.createNamespace(createNsReq);

    // Create a table
    byte[] tableData = createTestTableData();
    CreateTableRequest createReq =
        new CreateTableRequest().id(Arrays.asList("workspace", "test_table"));
    namespace.createTable(createReq, tableData);

    // Check existence
    TableExistsRequest existsReq =
        new TableExistsRequest().id(Arrays.asList("workspace", "test_table"));
    assertDoesNotThrow(() -> namespace.tableExists(existsReq));

    // Check non-existent table
    TableExistsRequest notExistsReq =
        new TableExistsRequest().id(Arrays.asList("workspace", "nonexistent"));
    assertThrows(RuntimeException.class, () -> namespace.tableExists(notExistsReq));
  }

  @Test
  void testDropTable() throws Exception {
    // Create parent namespace
    CreateNamespaceRequest createNsReq =
        new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespace.createNamespace(createNsReq);

    // Create a table
    byte[] tableData = createTestTableData();
    CreateTableRequest createReq =
        new CreateTableRequest().id(Arrays.asList("workspace", "test_table"));
    namespace.createTable(createReq, tableData);

    // Drop table
    DropTableRequest dropReq = new DropTableRequest().id(Arrays.asList("workspace", "test_table"));
    DropTableResponse dropResp = namespace.dropTable(dropReq);
    assertNotNull(dropResp);

    // Verify it's gone
    TableExistsRequest existsReq =
        new TableExistsRequest().id(Arrays.asList("workspace", "test_table"));
    assertThrows(RuntimeException.class, () -> namespace.tableExists(existsReq));
  }

  @Test
  void testCreateEmptyTable() {
    // Create parent namespace
    CreateNamespaceRequest createNsReq =
        new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespace.createNamespace(createNsReq);

    // Create empty table (metadata-only operation)
    CreateEmptyTableRequest createReq =
        new CreateEmptyTableRequest().id(Arrays.asList("workspace", "empty_table"));

    CreateEmptyTableResponse createResp = namespace.createEmptyTable(createReq);

    assertNotNull(createResp);
    assertNotNull(createResp.getLocation());
  }

  @Test
  void testDescribeTableReturnsManagedVersioningWhenTrackingEnabled() throws Exception {
    // Create namespace with table_version_tracking_enabled and manifest_enabled
    DirectoryNamespace trackingNs = new DirectoryNamespace();
    Map<String, String> config = new HashMap<>();
    config.put("root", tempDir.toString());
    config.put("table_version_tracking_enabled", "true");
    config.put("manifest_enabled", "true");
    trackingNs.initialize(config, allocator);

    try {
      // Create parent namespace
      CreateNamespaceRequest createNsReq =
          new CreateNamespaceRequest().id(Arrays.asList("workspace"));
      trackingNs.createNamespace(createNsReq);

      // Create a table
      byte[] tableData = createTestTableData();
      CreateTableRequest createReq =
          new CreateTableRequest().id(Arrays.asList("workspace", "test_table"));
      trackingNs.createTable(createReq, tableData);

      // Describe table should return managedVersioning=true
      DescribeTableRequest descReq =
          new DescribeTableRequest().id(Arrays.asList("workspace", "test_table"));
      DescribeTableResponse descResp = trackingNs.describeTable(descReq);

      assertNotNull(descResp);
      assertNotNull(descResp.getLocation());
      assertTrue(
          Boolean.TRUE.equals(descResp.getManagedVersioning()),
          "Expected managedVersioning=true, got " + descResp.getManagedVersioning());
    } finally {
      trackingNs.close();
    }
  }

  @Test
  void testDescribeTableVersion() throws Exception {
    // Use multi-level table ID with manifest_enabled
    DirectoryNamespace trackingNs = new DirectoryNamespace();
    Map<String, String> config = new HashMap<>();
    config.put("root", tempDir.toString());
    config.put("manifest_enabled", "true");
    trackingNs.initialize(config, allocator);

    try {
      // Create parent namespace
      CreateNamespaceRequest createNsReq =
          new CreateNamespaceRequest().id(Arrays.asList("workspace"));
      trackingNs.createNamespace(createNsReq);

      // Create a table with multi-level ID
      byte[] tableData = createTestTableData();
      CreateTableRequest createReq =
          new CreateTableRequest().id(Arrays.asList("workspace", "test_table"));
      trackingNs.createTable(createReq, tableData);

      // Describe table version
      DescribeTableVersionRequest descReq =
          new DescribeTableVersionRequest()
              .id(Arrays.asList("workspace", "test_table"))
              .version(1L);
      DescribeTableVersionResponse descResp = trackingNs.describeTableVersion(descReq);

      assertNotNull(descResp);
      assertNotNull(descResp.getVersion());
      assertEquals(Long.valueOf(1), descResp.getVersion().getVersion());
      assertNotNull(descResp.getVersion().getManifestPath());
    } finally {
      trackingNs.close();
    }
  }

  /**
   * Inner class that wraps DirectoryNamespace and tracks API calls for testing managed versioning.
   */
  static class TableVersionTrackingNamespace implements LanceNamespace, java.io.Closeable {
    private final DirectoryNamespace inner;
    private final AtomicInteger createTableVersionCount = new AtomicInteger(0);
    private final AtomicInteger describeTableVersionCount = new AtomicInteger(0);
    private final AtomicInteger listTableVersionsCount = new AtomicInteger(0);

    public TableVersionTrackingNamespace(Path root) {
      Map<String, String> dirProps = new HashMap<>();
      dirProps.put("root", root.toString());
      dirProps.put("table_version_tracking_enabled", "true");
      dirProps.put("manifest_enabled", "true");

      this.inner = new DirectoryNamespace();
      try (BufferAllocator allocator = new RootAllocator()) {
        this.inner.initialize(dirProps, allocator);
      }
    }

    public int getCreateTableVersionCount() {
      return createTableVersionCount.get();
    }

    public int getDescribeTableVersionCount() {
      return describeTableVersionCount.get();
    }

    public int getListTableVersionsCount() {
      return listTableVersionsCount.get();
    }

    public long getNativeHandle() {
      return inner.getNativeHandle();
    }

    @Override
    public void initialize(Map<String, String> configProperties, BufferAllocator allocator) {
      // Already initialized in constructor
    }

    @Override
    public String namespaceId() {
      return "TableVersionTrackingNamespace { inner: " + inner.namespaceId() + " }";
    }

    @Override
    public CreateEmptyTableResponse createEmptyTable(CreateEmptyTableRequest request) {
      return inner.createEmptyTable(request);
    }

    @Override
    public DeclareTableResponse declareTable(DeclareTableRequest request) {
      return inner.declareTable(request);
    }

    @Override
    public DescribeTableResponse describeTable(DescribeTableRequest request) {
      return inner.describeTable(request);
    }

    @Override
    public CreateTableVersionResponse createTableVersion(CreateTableVersionRequest request) {
      createTableVersionCount.incrementAndGet();
      return inner.createTableVersion(request);
    }

    @Override
    public DescribeTableVersionResponse describeTableVersion(DescribeTableVersionRequest request) {
      describeTableVersionCount.incrementAndGet();
      return inner.describeTableVersion(request);
    }

    @Override
    public ListTableVersionsResponse listTableVersions(ListTableVersionsRequest request) {
      listTableVersionsCount.incrementAndGet();
      return inner.listTableVersions(request);
    }

    @Override
    public BatchDeleteTableVersionsResponse batchDeleteTableVersions(
        BatchDeleteTableVersionsRequest request) {
      return inner.batchDeleteTableVersions(request);
    }

    @Override
    public void close() {
      inner.close();
    }
  }

  @Test
  void testExternalManifestStoreInvokesNamespaceApis(@TempDir Path managedVersioningTempDir)
      throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      // Create namespace with table_version_tracking_enabled
      TableVersionTrackingNamespace namespace =
          new TableVersionTrackingNamespace(managedVersioningTempDir);
      String tableName = "test_table";
      java.util.List<String> tableId = Arrays.asList(tableName);

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
                .tableId(tableId)
                .mode(WriteParams.WriteMode.CREATE)
                .execute()) {
          assertEquals(2, dataset.countRows());
          assertEquals(1, dataset.version());
        }
      }

      // Verify describe_table returns managed_versioning=true
      DescribeTableRequest descReq = new DescribeTableRequest();
      descReq.setId(tableId);
      DescribeTableResponse descResp = namespace.describeTable(descReq);

      assertEquals(
          Boolean.TRUE,
          descResp.getManagedVersioning(),
          "Expected managedVersioning=true when table_version_tracking_enabled");

      // Open dataset through namespace - this should call list_table_versions for latest
      int initialListCount = namespace.getListTableVersionsCount();
      try (Dataset dsFromNamespace =
          Dataset.open().allocator(allocator).namespace(namespace).tableId(tableId).build()) {

        assertEquals(2, dsFromNamespace.countRows());
        assertEquals(1, dsFromNamespace.version());
      }
      assertEquals(
          initialListCount + 1,
          namespace.getListTableVersionsCount(),
          "list_table_versions should have been called once when opening latest version");

      // Verify create_table_version was called once during CREATE
      assertEquals(
          1,
          namespace.getCreateTableVersionCount(),
          "create_table_version should have been called once during CREATE");

      try (VectorSchemaRoot appendRoot = VectorSchemaRoot.create(schema, allocator)) {
        IntVector aVector = (IntVector) appendRoot.getVector("a");
        IntVector bVector = (IntVector) appendRoot.getVector("b");

        aVector.allocateNew(2);
        bVector.allocateNew(2);

        aVector.set(0, 100);
        bVector.set(0, 200);
        aVector.set(1, 1000);
        bVector.set(1, 2000);

        aVector.setValueCount(2);
        bVector.setValueCount(2);
        appendRoot.setRowCount(2);

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
                return appendRoot;
              }
            };

        // Append through namespace
        try (Dataset dataset =
            Dataset.write()
                .allocator(allocator)
                .reader(appendReader)
                .namespace(namespace)
                .tableId(tableId)
                .mode(WriteParams.WriteMode.APPEND)
                .execute()) {
          assertEquals(4, dataset.countRows());
          assertEquals(2, dataset.version());
        }
      }

      assertEquals(
          2,
          namespace.getCreateTableVersionCount(),
          "create_table_version should have been called twice (once for CREATE, once for APPEND)");

      // Open latest version - should call list_table_versions
      int listCountBeforeLatest = namespace.getListTableVersionsCount();
      try (Dataset latestDs =
          Dataset.open().allocator(allocator).namespace(namespace).tableId(tableId).build()) {

        assertEquals(4, latestDs.countRows());
        assertEquals(2, latestDs.version());
      }
      assertEquals(
          listCountBeforeLatest + 1,
          namespace.getListTableVersionsCount(),
          "list_table_versions should have been called once when opening latest version");

      // Open specific version (version 1) - should call describe_table_version
      int describeCountBeforeV1 = namespace.getDescribeTableVersionCount();
      try (Dataset v1Ds =
          Dataset.open()
              .allocator(allocator)
              .namespace(namespace)
              .tableId(tableId)
              .readOptions(new ReadOptions.Builder().setVersion(1L).build())
              .build()) {

        assertEquals(2, v1Ds.countRows());
        assertEquals(1, v1Ds.version());
      }
      assertEquals(
          describeCountBeforeV1 + 1,
          namespace.getDescribeTableVersionCount(),
          "describe_table_version should have been called once when opening version 1");

      namespace.close();
    }
  }

  @Test
  void testConcurrentCreateAndDropWithSingleInstance() throws Exception {
    // Initialize namespace first - create parent namespace to ensure __manifest table
    // is created before concurrent operations
    CreateNamespaceRequest createNsReq = new CreateNamespaceRequest().id(Arrays.asList("test_ns"));
    namespace.createNamespace(createNsReq);

    int numTables = 10;
    ExecutorService executor = Executors.newFixedThreadPool(numTables);
    CountDownLatch startLatch = new CountDownLatch(1);
    CountDownLatch doneLatch = new CountDownLatch(numTables);
    AtomicInteger successCount = new AtomicInteger(0);
    AtomicInteger failCount = new AtomicInteger(0);

    for (int i = 0; i < numTables; i++) {
      final int tableIndex = i;
      executor.submit(
          () -> {
            try {
              startLatch.await();

              String tableName = "concurrent_table_" + tableIndex;
              byte[] tableData = createTestTableData();

              CreateTableRequest createReq =
                  new CreateTableRequest().id(Arrays.asList("test_ns", tableName));
              namespace.createTable(createReq, tableData);

              DropTableRequest dropReq =
                  new DropTableRequest().id(Arrays.asList("test_ns", tableName));
              namespace.dropTable(dropReq);

              successCount.incrementAndGet();
            } catch (Exception e) {
              failCount.incrementAndGet();
            } finally {
              doneLatch.countDown();
            }
          });
    }

    startLatch.countDown();
    assertTrue(doneLatch.await(60, TimeUnit.SECONDS), "Timed out waiting for tasks to complete");

    executor.shutdown();
    assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));

    assertEquals(numTables, successCount.get(), "All tasks should succeed");
    assertEquals(0, failCount.get(), "No tasks should fail");

    ListTablesRequest listReq = new ListTablesRequest().id(Arrays.asList("test_ns"));
    ListTablesResponse listResp = namespace.listTables(listReq);
    assertEquals(0, listResp.getTables().size(), "All tables should be dropped");
  }

  @Test
  void testConcurrentCreateAndDropWithMultipleInstances() throws Exception {
    // Initialize namespace first with a single instance to ensure __manifest
    // table is created and parent namespace exists before concurrent operations
    DirectoryNamespace initNs = new DirectoryNamespace();
    Map<String, String> initConfig = new HashMap<>();
    initConfig.put("root", tempDir.toString());
    initConfig.put("inline_optimization_enabled", "false");
    initNs.initialize(initConfig, allocator);

    CreateNamespaceRequest createNsReq = new CreateNamespaceRequest().id(Arrays.asList("test_ns"));
    initNs.createNamespace(createNsReq);
    initNs.close();

    int numTables = 10;
    ExecutorService executor = Executors.newFixedThreadPool(numTables);
    CountDownLatch startLatch = new CountDownLatch(1);
    CountDownLatch doneLatch = new CountDownLatch(numTables);
    AtomicInteger successCount = new AtomicInteger(0);
    AtomicInteger failCount = new AtomicInteger(0);
    List<DirectoryNamespace> namespaces = new ArrayList<>();

    for (int i = 0; i < numTables; i++) {
      final int tableIndex = i;
      executor.submit(
          () -> {
            DirectoryNamespace localNs = null;
            try {
              startLatch.await();

              localNs = new DirectoryNamespace();
              Map<String, String> config = new HashMap<>();
              config.put("root", tempDir.toString());
              config.put("inline_optimization_enabled", "false");
              localNs.initialize(config, allocator);

              synchronized (namespaces) {
                namespaces.add(localNs);
              }

              String tableName = "multi_ns_table_" + tableIndex;
              byte[] tableData = createTestTableData();

              CreateTableRequest createReq =
                  new CreateTableRequest().id(Arrays.asList("test_ns", tableName));
              localNs.createTable(createReq, tableData);

              DropTableRequest dropReq =
                  new DropTableRequest().id(Arrays.asList("test_ns", tableName));
              localNs.dropTable(dropReq);

              successCount.incrementAndGet();
            } catch (Exception e) {
              failCount.incrementAndGet();
            } finally {
              doneLatch.countDown();
            }
          });
    }

    startLatch.countDown();
    assertTrue(doneLatch.await(60, TimeUnit.SECONDS), "Timed out waiting for tasks to complete");

    executor.shutdown();
    assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));

    // Close all namespace instances
    for (DirectoryNamespace ns : namespaces) {
      try {
        ns.close();
      } catch (Exception e) {
        // Ignore
      }
    }

    assertEquals(numTables, successCount.get(), "All tasks should succeed");
    assertEquals(0, failCount.get(), "No tasks should fail");

    // Verify with a fresh namespace
    DirectoryNamespace verifyNs = new DirectoryNamespace();
    Map<String, String> config = new HashMap<>();
    config.put("root", tempDir.toString());
    verifyNs.initialize(config, allocator);

    ListTablesRequest listReq = new ListTablesRequest().id(Arrays.asList("test_ns"));
    ListTablesResponse listResp = verifyNs.listTables(listReq);
    assertEquals(0, listResp.getTables().size(), "All tables should be dropped");

    verifyNs.close();
  }

  @Test
  void testConcurrentCreateThenDropFromDifferentInstance() throws Exception {
    // Initialize namespace first with a single instance to ensure __manifest
    // table is created and parent namespace exists before concurrent operations
    DirectoryNamespace initNs = new DirectoryNamespace();
    Map<String, String> initConfig = new HashMap<>();
    initConfig.put("root", tempDir.toString());
    initConfig.put("inline_optimization_enabled", "false");
    initNs.initialize(initConfig, allocator);

    CreateNamespaceRequest createNsReq = new CreateNamespaceRequest().id(Arrays.asList("test_ns"));
    initNs.createNamespace(createNsReq);
    initNs.close();

    int numTables = 10;

    // First, create all tables using separate namespace instances
    ExecutorService createExecutor = Executors.newFixedThreadPool(numTables);
    CountDownLatch createStartLatch = new CountDownLatch(1);
    CountDownLatch createDoneLatch = new CountDownLatch(numTables);
    AtomicInteger createSuccessCount = new AtomicInteger(0);
    List<DirectoryNamespace> createNamespaces = new ArrayList<>();

    for (int i = 0; i < numTables; i++) {
      final int tableIndex = i;
      createExecutor.submit(
          () -> {
            DirectoryNamespace localNs = null;
            try {
              createStartLatch.await();

              localNs = new DirectoryNamespace();
              Map<String, String> config = new HashMap<>();
              config.put("root", tempDir.toString());
              config.put("inline_optimization_enabled", "false");
              localNs.initialize(config, allocator);

              synchronized (createNamespaces) {
                createNamespaces.add(localNs);
              }

              String tableName = "cross_instance_table_" + tableIndex;
              byte[] tableData = createTestTableData();

              CreateTableRequest createReq =
                  new CreateTableRequest().id(Arrays.asList("test_ns", tableName));
              localNs.createTable(createReq, tableData);

              createSuccessCount.incrementAndGet();
            } catch (Exception e) {
              // Ignore - test will fail on assertion
            } finally {
              createDoneLatch.countDown();
            }
          });
    }

    createStartLatch.countDown();
    assertTrue(createDoneLatch.await(60, TimeUnit.SECONDS), "Timed out waiting for creates");
    createExecutor.shutdown();

    assertEquals(numTables, createSuccessCount.get(), "All creates should succeed");

    // Close create namespaces
    for (DirectoryNamespace ns : createNamespaces) {
      try {
        ns.close();
      } catch (Exception e) {
        // Ignore
      }
    }

    // Now drop all tables using NEW namespace instances
    ExecutorService dropExecutor = Executors.newFixedThreadPool(numTables);
    CountDownLatch dropStartLatch = new CountDownLatch(1);
    CountDownLatch dropDoneLatch = new CountDownLatch(numTables);
    AtomicInteger dropSuccessCount = new AtomicInteger(0);
    AtomicInteger dropFailCount = new AtomicInteger(0);
    List<DirectoryNamespace> dropNamespaces = new ArrayList<>();

    for (int i = 0; i < numTables; i++) {
      final int tableIndex = i;
      dropExecutor.submit(
          () -> {
            DirectoryNamespace localNs = null;
            try {
              dropStartLatch.await();

              localNs = new DirectoryNamespace();
              Map<String, String> config = new HashMap<>();
              config.put("root", tempDir.toString());
              config.put("inline_optimization_enabled", "false");
              localNs.initialize(config, allocator);

              synchronized (dropNamespaces) {
                dropNamespaces.add(localNs);
              }

              String tableName = "cross_instance_table_" + tableIndex;

              DropTableRequest dropReq =
                  new DropTableRequest().id(Arrays.asList("test_ns", tableName));
              localNs.dropTable(dropReq);

              dropSuccessCount.incrementAndGet();
            } catch (Exception e) {
              dropFailCount.incrementAndGet();
            } finally {
              dropDoneLatch.countDown();
            }
          });
    }

    dropStartLatch.countDown();
    assertTrue(dropDoneLatch.await(60, TimeUnit.SECONDS), "Timed out waiting for drops");
    dropExecutor.shutdown();

    // Close drop namespaces
    for (DirectoryNamespace ns : dropNamespaces) {
      try {
        ns.close();
      } catch (Exception e) {
        // Ignore
      }
    }

    assertEquals(numTables, dropSuccessCount.get(), "All drops should succeed");
    assertEquals(0, dropFailCount.get(), "No drops should fail");
  }
}
