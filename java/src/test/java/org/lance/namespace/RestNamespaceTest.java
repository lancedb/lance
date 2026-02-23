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

import org.lance.namespace.model.*;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
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
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for RestNamespace implementation using RestAdapter with DirectoryNamespace backend.
 *
 * <p>This mirrors DirectoryNamespaceTest to ensure parity between DirectoryNamespace and
 * RestNamespace implementations.
 */
public class RestNamespaceTest {
  @TempDir Path tempDir;

  private BufferAllocator allocator;
  private RestAdapter adapter;
  private RestNamespace namespace;
  private int port;

  @BeforeEach
  void setUp() {
    allocator = new RootAllocator(Long.MAX_VALUE);

    // Create backend configuration for DirectoryNamespace
    Map<String, String> backendConfig = new HashMap<>();
    backendConfig.put("root", tempDir.toString());

    // Create and start REST adapter (port 0 lets OS assign available port)
    adapter = new RestAdapter("dir", backendConfig, "127.0.0.1", 0);
    adapter.start();
    port = adapter.getPort();

    // Create REST namespace client
    namespace = new RestNamespace();
    Map<String, String> clientConfig = new HashMap<>();
    clientConfig.put("uri", "http://127.0.0.1:" + port);
    namespace.initialize(clientConfig, allocator);
  }

  @AfterEach
  void tearDown() {
    if (namespace != null) {
      namespace.close();
    }
    if (adapter != null) {
      adapter.close();
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
    assertTrue(namespaceId.contains("RestNamespace"));
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
  void testRenameTable() throws Exception {
    // Create parent namespace
    CreateNamespaceRequest createNsReq =
        new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespace.createNamespace(createNsReq);

    // Create a table
    byte[] tableData = createTestTableData();
    CreateTableRequest createReq =
        new CreateTableRequest().id(Arrays.asList("workspace", "test_table"));
    namespace.createTable(createReq, tableData);

    // TODO: underlying dir namespace doesn't support rename yet...

    // // Rename the table
    // RenameTableRequest renameReq =
    //     new RenameTableRequest()
    //         .id(Arrays.asList("workspace", "test_table"))
    //         .newNamespaceId(Arrays.asList("workspace"))
    //         .newTableName("test_table_renamed");

    // RenameTableResponse renameRes = namespace.renameTable(renameReq);
    // assertNotNull(renameRes);

    // // Verify table with old name no longer exists
    // TableExistsRequest oldExistsReq =
    //     new TableExistsRequest().id(Arrays.asList("workspace", "test_table"));
    // assertThrows(RuntimeException.class, () -> namespace.tableExists(oldExistsReq));

    // // Verify table with new name exists
    // TableExistsRequest existsReq =
    //     new TableExistsRequest().id(Arrays.asList("workspace", "test_table_renamed"));
    // assertDoesNotThrow(() -> namespace.tableExists(existsReq));
  }
}
