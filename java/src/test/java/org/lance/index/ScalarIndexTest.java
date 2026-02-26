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
package org.lance.index;

import org.lance.CommitBuilder;
import org.lance.Dataset;
import org.lance.Fragment;
import org.lance.TestUtils;
import org.lance.Transaction;
import org.lance.WriteParams;
import org.lance.index.scalar.ScalarIndexParams;
import org.lance.ipc.LanceScanner;
import org.lance.ipc.ScanOptions;
import org.lance.operation.CreateIndex;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ScalarIndexTest {

  @Test
  public void testCreateBTreeIndex(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("btree_test").toString();
    Schema schema =
        new Schema(
            Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true)),
                Field.nullable("name", new ArrowType.Utf8())),
            null);

    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset dataset =
          Dataset.create(allocator, datasetPath, schema, new WriteParams.Builder().build())) {

        // Create BTree scalar index parameters
        ScalarIndexParams scalarParams = ScalarIndexParams.create("btree", "{\"zone_size\": 2048}");

        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();

        // Create BTree index on 'id' column
        Index index =
            dataset.createIndex(
                Collections.singletonList("id"),
                IndexType.BTREE,
                Optional.of("btree_id_index"),
                indexParams,
                true);

        // Verify the returned Index object
        assertEquals("btree_id_index", index.name());
        assertNotNull(index.uuid());
        assertFalse(index.fields().isEmpty());

        // Verify index was created and is in the list
        assertTrue(
            dataset.listIndexes().contains("btree_id_index"),
            "Expected 'btree_id_index' to be in the list of indexes: " + dataset.listIndexes());

        // TODO: Verify zone_size parameter was applied
        // Currently the Java API doesn't expose index configuration details,
        // but we could add a getIndexDetails() method in the future to verify
        // that the zone_size parameter was correctly set to 2048
      }
    }
  }

  @Test
  public void testCreateBTreeIndexDistributively(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("build_index_distributedly").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      // 1. write two fragments
      testDataset.write(1, 10).close();
      try (Dataset dataset = testDataset.write(2, 10)) {
        List<Fragment> fragments = dataset.getFragments();
        assertEquals(2, dataset.getFragments().size());

        ScalarIndexParams scalarParams = ScalarIndexParams.create("btree", "{\"zone_size\": 2048}");
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();
        UUID uuid = UUID.randomUUID();

        // 2. partially create index
        dataset.createIndex(
            IndexOptions.builder(Collections.singletonList("name"), IndexType.BTREE, indexParams)
                .withIndexName("test_index")
                .withIndexUUID(uuid.toString())
                .withFragmentIds(Collections.singletonList(fragments.get(0).getId()))
                .build());
        dataset.createIndex(
            IndexOptions.builder(Collections.singletonList("name"), IndexType.BTREE, indexParams)
                .withIndexName("test_index")
                .withIndexUUID(uuid.toString())
                .withFragmentIds(Collections.singletonList(fragments.get(1).getId()))
                .build());

        // then no index should have been created
        assertFalse(
            dataset.listIndexes().contains("test_index"),
            "Partially created index should not present");

        // 3. merge metadata, which will still not be committed
        dataset.mergeIndexMetadata(uuid.toString(), IndexType.BTREE, Optional.empty());

        // 4. commit the index
        int fieldId =
            dataset.getLanceSchema().fields().stream()
                .filter(f -> f.getName().equals("name"))
                .findAny()
                .orElseThrow(() -> new RuntimeException("Cannot find 'name' field for TestDataset"))
                .getId();

        long datasetVersion = dataset.version();

        Index index =
            Index.builder()
                .uuid(uuid)
                .name("test_index")
                .fields(Collections.singletonList(fieldId))
                .datasetVersion(datasetVersion)
                .indexVersion(0)
                .fragments(fragments.stream().map(Fragment::getId).collect(Collectors.toList()))
                .build();

        CreateIndex createIndexOp =
            CreateIndex.builder().withNewIndices(Collections.singletonList(index)).build();

        try (Transaction createIndexTx =
            new Transaction.Builder()
                .readVersion(datasetVersion)
                .operation(createIndexOp)
                .build()) {
          try (Dataset newDataset = new CommitBuilder(dataset).execute(createIndexTx)) {
            // new dataset should contain that index
            assertEquals(datasetVersion + 1, newDataset.version());
            assertTrue(newDataset.listIndexes().contains("test_index"));
          }
        }
      }
    }
  }

  @Test
  public void testRangedBTreeIndex(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("ranged_btree_map").toString();
    UUID indexUUID = UUID.randomUUID();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      // 1. write some data
      try (Dataset dataset = testDataset.write(1, 200)) {

        // 2. scan data out
        List<long[]> data = new ArrayList<>();
        try (LanceScanner scanner =
                dataset.newScan(
                    new ScanOptions.Builder()
                        .withRowId(true)
                        .columns(Collections.singletonList("id"))
                        .build());
            ArrowReader arrowReader = scanner.scanBatches(); ) {
          while (arrowReader.loadNextBatch()) {
            VectorSchemaRoot root = arrowReader.getVectorSchemaRoot();
            UInt8Vector rowIdVec = (UInt8Vector) root.getVector("_rowid");
            IntVector idVec = (IntVector) root.getVector("id");
            for (int i = 0; i < root.getRowCount(); i++) {
              data.add(new long[] {idVec.get(i), rowIdVec.get(i)});
            }
          }
        }

        // 3. sort data globally (This will be done by computing engines in production)
        data.sort((d1, d2) -> (int) (d1[0] - d2[0]));
        int mid = data.size() / 2;

        // 4. divide sorted data into ranges and build index for each range
        createBtreeIndexForRange(dataset, data.subList(0, mid), 1, allocator, indexUUID);
        createBtreeIndexForRange(dataset, data.subList(mid, data.size()), 2, allocator, indexUUID);

        // 5. merge index.
        dataset.mergeIndexMetadata(indexUUID.toString(), IndexType.BTREE, Optional.empty());

        // 6. commit index
        long datasetVersion = dataset.version();
        int fieldId =
            dataset.getLanceSchema().fields().stream()
                .filter(f -> f.getName().equals("id"))
                .findAny()
                .orElseThrow(() -> new RuntimeException("Cannot find 'id' field for TestDataset"))
                .getId();
        Index index =
            Index.builder()
                .uuid(indexUUID)
                .name("test_index")
                .fields(Collections.singletonList(fieldId))
                .datasetVersion(datasetVersion)
                .indexVersion(0)
                .fragments(
                    dataset.getFragments().stream()
                        .map(Fragment::getId)
                        .collect(Collectors.toList()))
                .build();

        CreateIndex createIndexOp =
            CreateIndex.builder().withNewIndices(Collections.singletonList(index)).build();

        try (Transaction createIndexTx =
            new Transaction.Builder()
                .readVersion(datasetVersion)
                .operation(createIndexOp)
                .build()) {
          try (Dataset newDataset = new CommitBuilder(dataset).execute(createIndexTx)) {
            // new dataset should contain that index
            assertEquals(datasetVersion + 1, newDataset.version());
            assertTrue(newDataset.listIndexes().contains("test_index"));

            // 7. compare results
            // force use index should get the right value
            ScanOptions scanOptions =
                new ScanOptions.Builder().withRowId(true).filter("id in (10, 20, 30)").build();
            try (LanceScanner scanner = newDataset.newScan(scanOptions);
                ArrowReader arrowReader = scanner.scanBatches(); ) {
              List<Integer> ids = new ArrayList<>();
              while (arrowReader.loadNextBatch()) {
                VectorSchemaRoot root = arrowReader.getVectorSchemaRoot();
                IntVector idVec = (IntVector) root.getVector("id");
                for (int i = 0; i < idVec.getValueCount(); i++) {
                  ids.add(idVec.get(i));
                }
              }
              Collections.sort(ids);
              Assertions.assertIterableEquals(Arrays.asList(10, 20, 30), ids);
            }
          }
        }
      }
    }
  }

  private void createBtreeIndexForRange(
      Dataset dataset,
      List<long[]> preprocessedData,
      int rangeId,
      BufferAllocator allocator,
      UUID indexUUID) {
    // Note that the indexing column is called 'value' in btree.
    Schema schema =
        new Schema(
            Arrays.asList(
                Field.nullable("value", new ArrowType.Int(32, true)),
                Field.nullable("_rowid", new ArrowType.Int(64, false))),
            null);
    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      root.allocateNew();
      IntVector idVec = (IntVector) root.getVector("value");
      UInt8Vector rowIdVec = (UInt8Vector) root.getVector("_rowid");
      for (int i = 0; i < preprocessedData.size(); i++) {
        long[] dataPair = preprocessedData.get(i);
        idVec.set(i, (int) dataPair[0]);
        rowIdVec.setSafe(i, dataPair[1]);
      }
      root.setRowCount(preprocessedData.size());

      ByteArrayOutputStream out = new ByteArrayOutputStream();
      try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, out)) {
        writer.start();
        writer.writeBatch();
        writer.end();
      } catch (IOException e) {
        throw new RuntimeException("Cannot write schema root", e);
      }

      byte[] arrowData = out.toByteArray();
      ByteArrayInputStream in = new ByteArrayInputStream(arrowData);

      try (ArrowStreamReader reader = new ArrowStreamReader(in, allocator);
          ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
        Data.exportArrayStream(allocator, reader, stream);

        ScalarIndexParams scalarParams =
            ScalarIndexParams.create("btree", String.format("{\"range_id\": %s}", rangeId));
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();
        dataset.createIndex(
            IndexOptions.builder(Collections.singletonList("id"), IndexType.BTREE, indexParams)
                .withIndexUUID(indexUUID.toString())
                .withPreprocessedData(stream)
                .build());
      } catch (Exception e) {
        throw new RuntimeException("Cannot read arrow stream.", e);
      }
    }
  }

  @Test
  public void testCreateZonemapIndex(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("zonemap_test").toString();
    Schema schema =
        new Schema(
            Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true)),
                Field.nullable("value", new ArrowType.Utf8())),
            null);

    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset dataset =
          Dataset.create(allocator, datasetPath, schema, new WriteParams.Builder().build())) {

        // Create Zonemap scalar index parameters with rows_per_zone setting
        ScalarIndexParams scalarParams =
            ScalarIndexParams.create("zonemap", "{\"rows_per_zone\": 1024}");

        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();

        // Create Zonemap index on 'value' column
        Index index =
            dataset.createIndex(
                Collections.singletonList("value"),
                IndexType.ZONEMAP,
                Optional.of("zonemap_value_index"),
                indexParams,
                true);

        // Verify the returned Index object
        assertEquals("zonemap_value_index", index.name());
        assertNotNull(index.uuid());

        // Verify index was created
        assertTrue(
            dataset.listIndexes().contains("zonemap_value_index"),
            "Expected 'zonemap_value_index' to be in the list of indexes: "
                + dataset.listIndexes());

        // TODO: Verify rows_per_zone parameter was applied
        // Currently the Java API doesn't expose index configuration details,
        // but we could add a getIndexDetails() method in the future to verify
        // that the rows_per_zone parameter was correctly set to 1024
      }
    }
  }
}
