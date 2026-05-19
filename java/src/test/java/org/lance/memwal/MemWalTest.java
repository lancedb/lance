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
package org.lance.memwal;

import org.lance.Dataset;
import org.lance.index.DistanceType;
import org.lance.index.IndexParams;
import org.lance.index.IndexType;
import org.lance.index.vector.VectorIndexParams;
import org.lance.merge.MergeInsertParams;
import org.lance.merge.MergeInsertResult;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.FixedSizeListVector;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.channels.Channels;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Integration tests for the MemWAL Java bindings, mirroring python/python/tests/test_mem_wal.py.
 */
public class MemWalTest {
  private static final Map<String, String> PK_META =
      Collections.singletonMap("lance-schema:unenforced-primary-key", "true");

  private static final Schema LOOKUP_SCHEMA =
      new Schema(
          Arrays.asList(
              new Field(
                  "id", new FieldType(false, new ArrowType.Int(64, true), null, PK_META), null),
              Field.nullable("name", new ArrowType.Utf8())));
  private static final Schema APPEND_ONLY_SCHEMA =
      new Schema(
          Arrays.asList(
              new Field(
                  "id",
                  new FieldType(false, new ArrowType.Int(64, true), null, Collections.emptyMap()),
                  null),
              Field.nullable("name", new ArrowType.Utf8())));

  /** Build a single-batch root where {@code name = "{prefix}_{id}"}. */
  private static VectorSchemaRoot lookupRoot(BufferAllocator allocator, long[] ids, String prefix) {
    VectorSchemaRoot root = VectorSchemaRoot.create(LOOKUP_SCHEMA, allocator);
    BigIntVector idVector = (BigIntVector) root.getVector("id");
    VarCharVector nameVector = (VarCharVector) root.getVector("name");
    idVector.allocateNew(ids.length);
    nameVector.allocateNew();
    for (int i = 0; i < ids.length; i++) {
      idVector.set(i, ids[i]);
      nameVector.setSafe(i, (prefix + "_" + ids[i]).getBytes(StandardCharsets.UTF_8));
    }
    root.setRowCount(ids.length);
    return root;
  }

  /** Build a single-batch append-only root without primary-key metadata. */
  private static VectorSchemaRoot appendOnlyRoot(
      BufferAllocator allocator, long[] ids, String prefix) {
    VectorSchemaRoot root = VectorSchemaRoot.create(APPEND_ONLY_SCHEMA, allocator);
    BigIntVector idVector = (BigIntVector) root.getVector("id");
    VarCharVector nameVector = (VarCharVector) root.getVector("name");
    idVector.allocateNew(ids.length);
    nameVector.allocateNew();
    for (int i = 0; i < ids.length; i++) {
      idVector.set(i, ids[i]);
      nameVector.setSafe(i, (prefix + "_" + ids[i]).getBytes(StandardCharsets.UTF_8));
    }
    root.setRowCount(ids.length);
    return root;
  }

  /** Wrap an in-memory root into an {@link ArrowReader} via the Arrow IPC stream format. */
  private static ArrowReader toReader(BufferAllocator allocator, VectorSchemaRoot root)
      throws Exception {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, Channels.newChannel(out))) {
      writer.start();
      writer.writeBatch();
      writer.end();
    }
    return new ArrowStreamReader(new ByteArrayInputStream(out.toByteArray()), allocator);
  }

  /** Write a base dataset of `(id, name)` rows at {@code path}. */
  private static Dataset writeLookupDataset(
      BufferAllocator allocator, String path, long[] ids, String prefix) throws Exception {
    try (VectorSchemaRoot root = lookupRoot(allocator, ids, prefix);
        ArrowReader reader = toReader(allocator, root)) {
      return Dataset.write().allocator(allocator).reader(reader).uri(path).execute();
    }
  }

  /** Write an append-only base dataset of `(id, name)` rows at {@code path}. */
  private static Dataset writeAppendOnlyDataset(
      BufferAllocator allocator, String path, long[] ids, String prefix) throws Exception {
    try (VectorSchemaRoot root = appendOnlyRoot(allocator, ids, prefix);
        ArrowReader reader = toReader(allocator, root)) {
      return Dataset.write().allocator(allocator).reader(reader).uri(path).execute();
    }
  }

  /** Read an LSM scanner fully into an {@code id -> name} map. */
  private static Map<Long, String> readByName(ArrowReader reader) throws Exception {
    Map<Long, String> byId = new HashMap<>();
    while (reader.loadNextBatch()) {
      VectorSchemaRoot root = reader.getVectorSchemaRoot();
      BigIntVector idVector = (BigIntVector) root.getVector("id");
      VarCharVector nameVector = (VarCharVector) root.getVector("name");
      for (int i = 0; i < root.getRowCount(); i++) {
        byId.put(idVector.get(i), new String(nameVector.get(i), StandardCharsets.UTF_8));
      }
    }
    return byId;
  }

  @Test
  void testMemWalIndexDetailsNoneBeforeInit(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("base").toString();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = writeLookupDataset(allocator, path, new long[] {1, 2, 3}, "base")) {
      assertFalse(dataset.memWalIndexDetails().isPresent());
    }
  }

  @Test
  void testInitializeMemWalUnsharded(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("base").toString();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = writeLookupDataset(allocator, path, new long[] {1, 2, 3}, "base")) {
      dataset.initializeMemWal(new InitializeMemWalParams().withUnsharded());

      Optional<MemWalIndexDetails> details = dataset.memWalIndexDetails();
      assertTrue(details.isPresent());
      assertEquals(Collections.emptyList(), details.get().maintainedIndexes());
    }
  }

  @Test
  void testInitializeMemWalBucketShardingWithoutPrimaryKey(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("append_only").toString();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = writeAppendOnlyDataset(allocator, path, new long[] {1, 2, 3}, "base")) {
      dataset.initializeMemWal(new InitializeMemWalParams().withBucketSharding("id", 4));

      Optional<MemWalIndexDetails> details = dataset.memWalIndexDetails();
      assertTrue(details.isPresent());
      assertEquals(4L, details.get().numShards());
      ShardingField field = details.get().shardingSpecs().get(0).fields().get(0);
      assertEquals("bucket", field.transform().get());
      assertEquals("4", field.parameters().get("num_buckets"));
    }
  }

  @Test
  void testInitializeMemWalRejectsConflictingSharding(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("base").toString();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = writeLookupDataset(allocator, path, new long[] {1}, "base")) {
      InitializeMemWalParams params =
          new InitializeMemWalParams().withUnsharded().withBucketSharding("id", 4);
      assertThrows(IllegalArgumentException.class, () -> dataset.initializeMemWal(params));
    }
  }

  @Test
  void testShardWriterPutAndLsmScanner(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("base").toString();
    String shardId = UUID.randomUUID().toString();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = writeLookupDataset(allocator, path, new long[] {0}, "base")) {
      dataset.initializeMemWal(new InitializeMemWalParams());

      ShardWriterConfig config =
          new ShardWriterConfig()
              .withDurableWrite(true)
              .withMaxWalBufferSize(1)
              .withMaxWalFlushIntervalMs(10)
              .withMaxMemtableBatches(1);

      try (ShardWriter writer = dataset.memWalWriter(shardId, config)) {
        assertEquals(shardId, writer.shardId());

        try (VectorSchemaRoot root = lookupRoot(allocator, new long[] {1, 2}, "writer");
            ArrowReader reader = toReader(allocator, root)) {
          writer.put(reader);
        }

        Map<Long, String> byId = Collections.emptyMap();
        long deadline = System.currentTimeMillis() + 10_000;
        while (System.currentTimeMillis() < deadline) {
          try (LsmScanner scanner = writer.lsmScanner();
              ArrowReader reader = scanner.scanBatches()) {
            byId = readByName(reader);
          }
          if ("writer_1".equals(byId.get(1L)) && "writer_2".equals(byId.get(2L))) {
            break;
          }
          Thread.sleep(50);
        }
        assertEquals("writer_1", byId.get(1L), "writer.lsmScanner() must see written rows");
        assertEquals("writer_2", byId.get(2L), "writer.lsmScanner() must see written rows");

        WriteStats stats = writer.stats();
        assertEquals(1, stats.putCount());

        MemTableStats memtableStats = writer.memtableStats();
        assertTrue(memtableStats.generation() >= 0);
      }
    }
  }

  @Test
  void testLsmScannerFromSnapshots(@TempDir Path tempDir) throws Exception {
    String basePath = tempDir.resolve("base").toString();
    String shardId = UUID.randomUUID().toString();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = writeLookupDataset(allocator, basePath, new long[] {1, 2, 3}, "base")) {
      dataset.initializeMemWal(new InitializeMemWalParams());

      // Flushed generation overwrites id=2.
      String genPath = basePath + "/_mem_wal/" + shardId + "/gen_1";
      writeLookupDataset(allocator, genPath, new long[] {2}, "gen1").close();

      ShardSnapshot snapshot =
          new ShardSnapshot(shardId).withFlushedGeneration(1, "gen_1").withCurrentGeneration(2);

      try (LsmScanner scanner =
              LsmScanner.fromSnapshots(dataset, Collections.singletonList(snapshot));
          ArrowReader reader = scanner.scanBatches()) {
        Map<Long, String> byId = readByName(reader);
        assertEquals(3, byId.size(), "Expected 3 deduplicated rows");
        assertEquals("base_1", byId.get(1L));
        assertEquals("gen1_2", byId.get(2L), "Flushed generation must win over base");
        assertEquals("base_3", byId.get(3L));
      }
    }
  }

  @Test
  void testPointLookup(@TempDir Path tempDir) throws Exception {
    String basePath = tempDir.resolve("base").toString();
    String shardId = UUID.randomUUID().toString();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = writeLookupDataset(allocator, basePath, new long[] {1, 2, 3}, "base")) {
      dataset.initializeMemWal(new InitializeMemWalParams());

      String genPath = basePath + "/_mem_wal/" + shardId + "/gen_1";
      writeLookupDataset(allocator, genPath, new long[] {2}, "gen1").close();

      ShardSnapshot snapshot =
          new ShardSnapshot(shardId).withFlushedGeneration(1, "gen_1").withCurrentGeneration(2);

      try (LsmPointLookupPlanner planner =
          new LsmPointLookupPlanner(dataset, Collections.singletonList(snapshot))) {
        // id=2 must resolve to the flushed-generation value.
        assertEquals("gen1_2", lookup(planner, allocator, 2L));
        // id=1 only exists in the base table.
        assertEquals("base_1", lookup(planner, allocator, 1L));
        // id=99 does not exist.
        assertEquals(null, lookup(planner, allocator, 99L));
      }
    }
  }

  /** Run a single-key point lookup and return the resolved name, or {@code null} if absent. */
  private static String lookup(LsmPointLookupPlanner planner, BufferAllocator allocator, long id)
      throws Exception {
    try (BigIntVector pkVector = new BigIntVector("id", allocator)) {
      pkVector.allocateNew(1);
      pkVector.set(0, id);
      pkVector.setValueCount(1);
      try (ExecutionPlan plan = planner.planLookup(pkVector);
          ArrowReader reader = plan.toReader()) {
        Map<Long, String> byId = readByName(reader);
        return byId.get(id);
      }
    }
  }

  @Test
  void testMergeInsertMarkGenerationsAsMerged(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("base").toString();
    String shardId = UUID.randomUUID().toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      Dataset dataset = writeLookupDataset(allocator, path, new long[] {1, 2, 3}, "base");
      try {
        dataset.initializeMemWal(new InitializeMemWalParams());

        MergeInsertParams params =
            new MergeInsertParams(Collections.singletonList("id"))
                .withMatchedUpdateAll()
                .withNotMatched(MergeInsertParams.WhenNotMatched.InsertAll)
                .markGenerationsAsMerged(
                    Collections.singletonList(new MergedGeneration(shardId, 1)));

        try (VectorSchemaRoot root = lookupRoot(allocator, new long[] {2, 4}, "merged");
            ArrowReader reader = toReader(allocator, root);
            ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
          Data.exportArrayStream(allocator, reader, stream);
          MergeInsertResult result = dataset.mergeInsert(params, stream);
          Dataset merged = result.dataset();
          try {
            assertEquals(4, merged.countRows(), "merge insert should yield 4 rows");
          } finally {
            merged.close();
          }
        }
      } finally {
        dataset.close();
      }
    }
  }

  private static final int VDIM = 32;

  private static Schema vectorSchema() {
    Field idField =
        new Field("id", new FieldType(false, new ArrowType.Int(32, true), null, PK_META), null);
    Field vecField =
        new Field(
            "vec",
            FieldType.nullable(new ArrowType.FixedSizeList(VDIM)),
            Collections.singletonList(
                new Field(
                    "item",
                    FieldType.nullable(new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)),
                    null)));
    return new Schema(Arrays.asList(idField, vecField));
  }

  @Test
  void testVectorSearch(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("vec").toString();
    int rows = 400;
    try (BufferAllocator allocator = new RootAllocator()) {
      Schema schema = vectorSchema();
      Dataset dataset;
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        IntVector idVector = (IntVector) root.getVector("id");
        FixedSizeListVector vecVector = (FixedSizeListVector) root.getVector("vec");
        Float4Vector items = (Float4Vector) vecVector.getDataVector();
        idVector.allocateNew(rows);
        vecVector.allocateNew();
        for (int i = 0; i < rows; i++) {
          idVector.set(i, i);
          vecVector.setNotNull(i);
          for (int d = 0; d < VDIM; d++) {
            items.setSafe(i * VDIM + d, (float) (i * VDIM + d));
          }
        }
        root.setRowCount(rows);
        try (ArrowReader reader = toReader(allocator, root)) {
          dataset = Dataset.write().allocator(allocator).reader(reader).uri(path).execute();
        }
      }

      try {
        IndexParams params =
            IndexParams.builder()
                .setVectorIndexParams(VectorIndexParams.ivfPq(2, 8, 2, DistanceType.L2, 2))
                .build();
        dataset.createIndex(
            Collections.singletonList("vec"),
            IndexType.VECTOR,
            Optional.of("vec_idx"),
            params,
            true);
        dataset.initializeMemWal(
            new InitializeMemWalParams()
                .withMaintainedIndexes(Collections.singletonList("vec_idx")));

        try (LsmVectorSearchPlanner planner =
            new LsmVectorSearchPlanner(dataset, Collections.emptyList(), "vec")) {
          int[] queryIds = {10, 50, 100, 200, 300, 350};
          int found = 0;
          for (int queryId : queryIds) {
            try (Float4Vector query = new Float4Vector("q", allocator)) {
              query.allocateNew(VDIM);
              for (int d = 0; d < VDIM; d++) {
                query.set(d, (float) (queryId * VDIM + d));
              }
              query.setValueCount(VDIM);
              try (ExecutionPlan plan = planner.planSearch(query, 10);
                  ArrowReader reader = plan.toReader()) {
                while (reader.loadNextBatch()) {
                  VectorSchemaRoot result = reader.getVectorSchemaRoot();
                  IntVector ids = (IntVector) result.getVector("id");
                  for (int i = 0; i < result.getRowCount(); i++) {
                    if (ids.get(i) == queryId) {
                      found++;
                    }
                  }
                }
              }
            }
          }
          double recall = (double) found / queryIds.length;
          assertTrue(recall >= 0.5, "vector search recall too low: " + recall);
        }
      } finally {
        dataset.close();
      }
    }
  }
}
