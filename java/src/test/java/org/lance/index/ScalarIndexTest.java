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

import org.lance.Dataset;
import org.lance.Fragment;
import org.lance.TestUtils;
import org.lance.WriteParams;
import org.lance.index.scalar.FMIndexParams;
import org.lance.index.scalar.ScalarIndexParams;
import org.lance.ipc.LanceScanner;
import org.lance.ipc.ScanOptions;

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
        assertEquals(2, fragments.size());

        ScalarIndexParams scalarParams = ScalarIndexParams.create("btree", "{\"zone_size\": 2048}");
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();
        String indexName = "test_index";

        List<Index> segments = new ArrayList<>();
        for (Fragment fragment : fragments) {
          segments.add(
              dataset.createIndex(
                  IndexOptions.builder(
                          Collections.singletonList("name"), IndexType.BTREE, indexParams)
                      .withIndexName(indexName)
                      .withFragmentIds(Collections.singletonList(fragment.getId()))
                      .build()));
        }

        assertFalse(
            dataset.listIndexes().contains(indexName),
            "Partially created index should not present");

        List<Index> committed = dataset.commitExistingIndexSegments(indexName, "name", segments);
        assertEquals(2, committed.size());
        assertTrue(dataset.listIndexes().contains(indexName));

        assertEquals(2, dataset.countIndexedRows(indexName, "name = 'Person 5'", Optional.empty()));
        assertEquals(
            10,
            dataset.countIndexedRows(
                indexName, "name >= 'Person 3' AND name < 'Person 8'", Optional.empty()));
      }
    }
  }

  @Test
  public void testRangedBTreeIndex(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("ranged_btree_map").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      testDataset.write(1, 100).close();
      try (Dataset dataset = testDataset.write(2, 100)) {
        List<Fragment> fragments = dataset.getFragments();
        assertEquals(2, fragments.size());

        List<Index> segments = new ArrayList<>();
        for (Fragment fragment : fragments) {
          List<long[]> data = new ArrayList<>();
          try (LanceScanner scanner =
                  dataset.newScan(
                      new ScanOptions.Builder()
                          .fragmentIds(Collections.singletonList(fragment.getId()))
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

          data.sort((d1, d2) -> Long.compare(d1[0], d2[0]));
          segments.add(createBtreeIndexFromPreprocessedData(dataset, data, fragment, allocator));
        }

        String indexName = "test_index";
        List<Index> committed = dataset.commitExistingIndexSegments(indexName, "id", segments);
        assertEquals(2, committed.size());
        assertTrue(dataset.listIndexes().contains(indexName));

        assertEquals(
            6, dataset.countIndexedRows(indexName, "id in (10, 20, 30)", Optional.empty()));
        assertEquals(
            20, dataset.countIndexedRows(indexName, "id >= 50 AND id < 60", Optional.empty()));
      }
    }
  }

  private Index createBtreeIndexFromPreprocessedData(
      Dataset dataset,
      List<long[]> preprocessedData,
      Fragment fragment,
      BufferAllocator allocator) {
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
        idVec.setSafe(i, (int) dataPair[0]);
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

        ScalarIndexParams scalarParams = ScalarIndexParams.create("btree", "{\"zone_size\": 64}");
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();
        return dataset.createIndex(
            IndexOptions.builder(Collections.singletonList("id"), IndexType.BTREE, indexParams)
                .withIndexName("test_index")
                .withFragmentIds(Collections.singletonList(fragment.getId()))
                .withPreprocessedData(stream)
                .build());
      } catch (Exception e) {
        throw new RuntimeException("Cannot read arrow stream.", e);
      }
    }
  }

  @Test
  public void testBtreeMergeIndexMetadataSoftBreak(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("btree_merge_metadata_soft_break").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      testDataset.write(1, 10).close();
      try (Dataset dataset = testDataset.write(2, 10)) {
        Exception ex =
            Assertions.assertThrows(
                Exception.class,
                () ->
                    dataset.mergeIndexMetadata(
                        UUID.randomUUID().toString(), IndexType.BTREE, Optional.empty()));
        assertTrue(
            ex.getMessage() != null
                && ex.getMessage().contains("no longer supports merge_index_metadata"),
            "expected BTree merge_index_metadata soft-break error, got: " + ex.getMessage());
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

  @Test
  public void testCreateFMIndexDistributively(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("fm_index_distributed").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      // Write two fragments, each with names "Person 0" .. "Person 9".
      testDataset.write(1, 10).close();
      try (Dataset dataset = testDataset.write(2, 10)) {
        List<Fragment> fragments = dataset.getFragments();
        assertEquals(2, fragments.size());

        IndexParams indexParams =
            IndexParams.builder().setScalarIndexParams(FMIndexParams.builder().build()).build();
        String indexName = "fm_name_index";

        // Build one uncommitted FM segment per fragment.
        List<Index> segments = new ArrayList<>();
        for (Fragment fragment : fragments) {
          segments.add(
              dataset.createIndex(
                  IndexOptions.builder(Collections.singletonList("name"), IndexType.FM, indexParams)
                      .withIndexName(indexName)
                      .withFragmentIds(Collections.singletonList(fragment.getId()))
                      .build()));
        }

        assertFalse(
            dataset.listIndexes().contains(indexName),
            "Partially created index should not present");

        // FM segments support merge before commit.
        Index merged = dataset.mergeExistingIndexSegments(segments);

        List<Index> committed =
            dataset.commitExistingIndexSegments(
                indexName, "name", Collections.singletonList(merged));
        assertEquals(1, committed.size());
        assertTrue(dataset.listIndexes().contains(indexName));

        // FM-Index answers exact substring search via `contains`.
        assertEquals(
            2, dataset.countIndexedRows(indexName, "contains(name, 'Person 5')", Optional.empty()));
        assertEquals(
            20, dataset.countIndexedRows(indexName, "contains(name, 'Person')", Optional.empty()));
      }
    }
  }
}
