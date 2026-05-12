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
import org.lance.FragmentMetadata;
import org.lance.FragmentOperation;
import org.lance.WriteParams;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

/** Tests for {@link Dataset#computeZonemapBatch(String, long[], String, BufferAllocator)}. */
public class ComputeZonemapBatchTest {

  private static Schema intSchema() {
    return new Schema(
        Arrays.asList(
            Field.nullable("id", new ArrowType.Int(32, true)),
            Field.nullable("value", new ArrowType.Int(32, true))),
        null);
  }

  private Dataset writeIntFragment(
      BufferAllocator allocator, String path, long version, int startValue, int rowCount) {
    Schema schema = intSchema();
    List<FragmentMetadata> metas;
    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      root.allocateNew();
      IntVector idVec = (IntVector) root.getVector("id");
      IntVector valVec = (IntVector) root.getVector("value");
      for (int i = 0; i < rowCount; i++) {
        idVec.setSafe(i, startValue + i);
        valVec.setSafe(i, (startValue + i) * 10);
      }
      root.setRowCount(rowCount);
      metas = Fragment.create(path, allocator, root, new WriteParams.Builder().build());
    }
    FragmentOperation.Append appendOp = new FragmentOperation.Append(metas);
    return Dataset.commit(allocator, path, appendOp, Optional.of(version));
  }

  /**
   * Null fragmentIds + explicit rows_per_zone — should compute over every fragment. We pass an
   * explicit zone size (rather than relying on the default) because Lance honours the {@code
   * LANCE_ZONEMAP_DEFAULT_ROWS_PER_ZONE} env var as the default zone size, so a CI environment that
   * sets it would otherwise flip the expected zone count and break this test.
   */
  @Test
  public void allFragmentsExplicitParams(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("all_frags").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(allocator, path, intSchema(), new WriteParams.Builder().build())) {
        // empty
      }
      writeIntFragment(allocator, path, 1, 0, 20).close();
      writeIntFragment(allocator, path, 2, 20, 20).close();
      writeIntFragment(allocator, path, 3, 40, 20).close();

      try (Dataset dataset = Dataset.open(path, allocator);
          VectorSchemaRoot batch =
              dataset.computeZonemapBatch("value", null, "{\"rows_per_zone\": 8192}", allocator)) {
        Schema s = batch.getSchema();
        // Canonical schema: min, max, null_count, nan_count, fragment_id, zone_start, zone_length
        // Pin column types so a regression from UInt32→Int32 on null_count etc. is caught here
        // rather than silently round-tripping through FFI.
        assertEquals(7, s.getFields().size(), "canonical zonemap stats schema has 7 columns");
        // Field 0: min (Int32, nullable — an all-NULL batch is legitimate)
        assertEquals("min", s.getFields().get(0).getName());
        assertEquals(new ArrowType.Int(32, true), s.getFields().get(0).getType(), "min: Int32");
        assertTrue(s.getFields().get(0).isNullable(), "min must be nullable");
        // Field 1: max (Int32, nullable — same rationale as min)
        assertEquals("max", s.getFields().get(1).getName());
        assertEquals(new ArrowType.Int(32, true), s.getFields().get(1).getType(), "max: Int32");
        assertTrue(s.getFields().get(1).isNullable(), "max must be nullable");
        // Field 2: null_count (UInt32, NON-nullable — metadata column)
        assertEquals("null_count", s.getFields().get(2).getName());
        assertEquals(
            new ArrowType.Int(32, false), s.getFields().get(2).getType(), "null_count: UInt32");
        assertFalse(s.getFields().get(2).isNullable(), "null_count must be non-nullable");
        // Field 3: nan_count (UInt32, NON-nullable)
        assertEquals("nan_count", s.getFields().get(3).getName());
        assertEquals(
            new ArrowType.Int(32, false), s.getFields().get(3).getType(), "nan_count: UInt32");
        assertFalse(s.getFields().get(3).isNullable(), "nan_count must be non-nullable");
        // Field 4: fragment_id (UInt64, NON-nullable)
        assertEquals("fragment_id", s.getFields().get(4).getName());
        assertEquals(
            new ArrowType.Int(64, false), s.getFields().get(4).getType(), "fragment_id: UInt64");
        assertFalse(s.getFields().get(4).isNullable(), "fragment_id must be non-nullable");
        // Field 5: zone_start (UInt64, NON-nullable)
        assertEquals("zone_start", s.getFields().get(5).getName());
        assertEquals(
            new ArrowType.Int(64, false), s.getFields().get(5).getType(), "zone_start: UInt64");
        assertFalse(s.getFields().get(5).isNullable(), "zone_start must be non-nullable");
        // Field 6: zone_length (UInt64, NON-nullable)
        assertEquals("zone_length", s.getFields().get(6).getName());
        assertEquals(
            new ArrowType.Int(64, false), s.getFields().get(6).getType(), "zone_length: UInt64");
        assertFalse(s.getFields().get(6).isNullable(), "zone_length must be non-nullable");

        // Default rows_per_zone (8192) is >> 60 rows total, so we get one zone per fragment.
        assertEquals(3, batch.getRowCount(), "one zone per fragment under default zone size");
      }
    }
  }

  /** Explicit fragment subset + custom rows_per_zone — verifies both knobs reach the Rust side. */
  @Test
  public void fragmentSubsetWithSmallZoneSize(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("subset").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(allocator, path, intSchema(), new WriteParams.Builder().build())) {
        // empty
      }
      writeIntFragment(allocator, path, 1, 0, 20).close();
      writeIntFragment(allocator, path, 2, 20, 20).close();
      writeIntFragment(allocator, path, 3, 40, 20).close();

      // Pick the first and third committed fragment ids by discovery (same pattern as the
      // WriteZonemapIndexFromBatchesTest tests). Hardcoded {0L, 2L} would silently break if
      // Lance ever changed fragment-id allocation.
      java.util.List<Integer> fragIds = new java.util.ArrayList<>();
      try (Dataset dataset = Dataset.open(path, allocator)) {
        for (org.lance.Fragment f : dataset.getFragments()) {
          fragIds.add(f.getId());
        }
      }
      assertEquals(3, fragIds.size(), "expected 3 fragments committed");
      long firstFragId = fragIds.get(0).longValue();
      long thirdFragId = fragIds.get(2).longValue();

      try (Dataset dataset = Dataset.open(path, allocator);
          VectorSchemaRoot batch =
              dataset.computeZonemapBatch(
                  "value",
                  new long[] {firstFragId, thirdFragId},
                  "{\"rows_per_zone\": 5}",
                  allocator)) {
        // 2 fragments × ceil(20/5) = 4 zones each = 8 zones total.
        assertEquals(8, batch.getRowCount(), "two fragments × 4 zones each");

        // Verify the returned batch's fragment_id column actually reflects the REQUESTED
        // subset — not just any pair that happens to sum to 8 zones. A regression where the
        // JNI swapped fragment-id-arg order or ignored the array would pass the row-count
        // assertion above but fail here.
        org.apache.arrow.vector.UInt8Vector fragVec =
            (org.apache.arrow.vector.UInt8Vector) batch.getVector("fragment_id");
        java.util.Set<Long> frags = new java.util.HashSet<>();
        for (int row = 0; row < batch.getRowCount(); row++) {
          frags.add(fragVec.getValueAsLong(row));
        }
        assertEquals(
            new java.util.HashSet<>(java.util.Arrays.asList(firstFragId, thirdFragId)),
            frags,
            "fragment_id column must contain exactly the requested subset");
      }
    }
  }

  /**
   * Explicit empty fragmentIds array must be rejected: it would otherwise reach the JNI as {@code
   * Some(vec![])}, meaning "zero fragments to scan", which is silently distinct from {@code null}
   * ("all fragments"). Callers passing an empty list are almost certainly confused about the
   * all-vs-none distinction.
   */
  @Test
  public void emptyFragmentIdsArrayRejected(@TempDir Path tempDir) {
    String path = tempDir.resolve("empty_frags").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset dataset =
          Dataset.create(allocator, path, intSchema(), new WriteParams.Builder().build())) {
        IllegalArgumentException thrown =
            assertThrows(
                IllegalArgumentException.class,
                () -> dataset.computeZonemapBatch("value", new long[0], "", allocator));
        assertTrue(
            thrown.getMessage() != null && thrown.getMessage().toLowerCase().contains("empty"),
            "error must mention empty; got: " + thrown.getMessage());
      }
    }
  }

  @Test
  public void nullColumnRejected(@TempDir Path tempDir) {
    String path = tempDir.resolve("null_col").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset dataset =
          Dataset.create(allocator, path, intSchema(), new WriteParams.Builder().build())) {
        assertThrows(
            IllegalArgumentException.class,
            () -> dataset.computeZonemapBatch(null, null, "", allocator));
      }
    }
  }

  @Test
  public void emptyParamsJsonUsesDefaults(@TempDir Path tempDir) throws Exception {
    // Both "" and null paramsJson must take the default-params path without throwing.
    String path = tempDir.resolve("empty_params").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(allocator, path, intSchema(), new WriteParams.Builder().build())) {
        // empty
      }
      writeIntFragment(allocator, path, 1, 0, 30).close();

      try (Dataset dataset = Dataset.open(path, allocator)) {
        try (VectorSchemaRoot a = dataset.computeZonemapBatch("value", null, "", allocator);
            VectorSchemaRoot b = dataset.computeZonemapBatch("value", null, null, allocator)) {
          assertEquals(a.getRowCount(), b.getRowCount());
        }
      }
    }
  }
}
