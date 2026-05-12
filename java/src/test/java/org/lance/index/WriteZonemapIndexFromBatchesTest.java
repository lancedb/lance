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
import org.lance.index.scalar.ZoneStats;

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
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the build-time zonemap consolidation flow: {@link Dataset#computeZonemapBatch} (worker)
 * → {@link Dataset#writeZonemapIndexFromBatches} (driver) → {@link
 * Dataset#commitExistingIndexSegments} (driver).
 */
public class WriteZonemapIndexFromBatchesTest {

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
   * End-to-end: 4 fragments, two workers each compute batches for half, driver consolidates +
   * writes + commits one IndexMetadata, then read via getZonemapStats and verify it covers all 4
   * fragments via the single returned segment.
   */
  @Test
  public void endToEndConsolidation(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("e2e").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(allocator, path, intSchema(), new WriteParams.Builder().build())) {
        // empty
      }
      writeIntFragment(allocator, path, 1, 0, 20).close();
      writeIntFragment(allocator, path, 2, 20, 20).close();
      writeIntFragment(allocator, path, 3, 40, 20).close();
      writeIntFragment(allocator, path, 4, 60, 20).close();

      // Build a fragId → startValue map by reading getFragments() in commit-iteration order.
      // This is a SMALLER assumption than "fragment ids are monotonically allocated from 0":
      // we only need getFragments() to return fragments in the order they were committed,
      // not that the ids themselves are sequential. A future Lance change that allocates
      // non-monotonic ids (e.g. random UUIDs) would still satisfy this test as long as the
      // listing order matches the commit order. If even that assumption breaks, the right
      // fix is to read identifying data out of each fragment instead of relying on listing
      // order at all.
      java.util.Map<Integer, Integer> startValueByFragId = new java.util.HashMap<>();
      try (Dataset dataset = Dataset.open(path, allocator)) {
        List<org.lance.Fragment> frags = dataset.getFragments();
        assertEquals(4, frags.size(), "expected 4 fragments to be committed");
        for (int i = 0; i < frags.size(); i++) {
          // Fragments are returned in commit order; we wrote startValue=0/20/40/60 in that order.
          startValueByFragId.put(frags.get(i).getId(), i * 20);
        }
      }

      try (Dataset dataset = Dataset.open(path, allocator)) {
        // Worker 1: first two fragments by id. Worker 2: last two fragments by id.
        java.util.List<Integer> sortedFragIds =
            new java.util.ArrayList<>(startValueByFragId.keySet());
        java.util.Collections.sort(sortedFragIds);
        long[] w1Frags =
            new long[] {sortedFragIds.get(0).longValue(), sortedFragIds.get(1).longValue()};
        long[] w2Frags =
            new long[] {sortedFragIds.get(2).longValue(), sortedFragIds.get(3).longValue()};
        String paramsJson = "{\"rows_per_zone\": 5}";
        try (VectorSchemaRoot w1 =
                dataset.computeZonemapBatch("value", w1Frags, paramsJson, allocator);
            VectorSchemaRoot w2 =
                dataset.computeZonemapBatch("value", w2Frags, paramsJson, allocator)) {

          assertEquals(8, w1.getRowCount(), "worker 1: 2 frags × 4 zones");
          assertEquals(8, w2.getRowCount(), "worker 2: 2 frags × 4 zones");

          Index segment =
              dataset.writeZonemapIndexFromBatches(
                  "value_zm", "value", Arrays.asList(w1, w2), paramsJson, allocator);

          // Returned segment should have all 4 fragments in its bitmap and a real UUID.
          assertNotNull(segment.uuid(), "segment uuid must be non-null");
          assertTrue(segment.fragments().isPresent(), "segment fragments must be populated");
          Set<Integer> fragIds = new HashSet<>(segment.fragments().get());
          assertEquals(
              new HashSet<>(startValueByFragId.keySet()),
              fragIds,
              "bitmap must cover every input fragment");
          assertEquals("value_zm", segment.name());

          // Commit it. One IndexMetadata covering all 4 fragments — not 4 separate segments.
          dataset.commitExistingIndexSegments(
              "value_zm", "value", java.util.Collections.singletonList(segment));
        }
      }

      // Re-open and confirm (a) the committed manifest has exactly one IndexMetadata for
      // value_zm, and (b) getZonemapStats reflects the union of fragments through that single
      // segment. Combined with the earlier `segment.fragments() == {0,1,2,3}` assertion this
      // is what makes the consolidation property end-to-end: ONE segment, not ONE per
      // fragment, covers the full fragment set.
      try (Dataset dataset = Dataset.open(path, allocator)) {
        long segmentCount =
            dataset.getIndexes().stream().filter(i -> "value_zm".equals(i.name())).count();
        assertEquals(
            1,
            segmentCount,
            "committed manifest must hold exactly one IndexMetadata for value_zm; got "
                + segmentCount);

        List<ZoneStats> stats = dataset.getZonemapStats("value");
        assertEquals(16, stats.size(), "4 fragments × 4 zones each at rows_per_zone=5");

        Set<Integer> fragIds = new HashSet<>();
        for (ZoneStats z : stats) {
          fragIds.add(z.getFragmentId());
          // The dataset stores value = (startValue + i) * 10. startValue per fragment comes
          // from the startValueByFragId map captured at commit time — robust to Lance changing
          // its fragment-id allocation strategy (currently monotonic with commit order, but
          // we do not want to bake that into the test).
          Integer startValue = startValueByFragId.get(z.getFragmentId());
          assertNotNull(
              startValue, "fragment id " + z.getFragmentId() + " was not in the committed set");
          // PER-ZONE bound: zone covers rows [zone_start, zone_start + zone_length) within the
          // fragment, and values are (startValue + i)*10. A regression that emitted constant
          // fragment-wide min/max for every zone (ignoring zone_start) would pass a looser
          // fragment-range check but fail this per-zone bound.
          long zoneStart = z.getZoneStart();
          long zoneLength = z.getZoneLength();
          int expectedZoneMin = (startValue + (int) zoneStart) * 10;
          int expectedZoneMax = (startValue + (int) zoneStart + (int) zoneLength - 1) * 10;
          long zoneMin = ((Number) z.getMin()).longValue();
          long zoneMax = ((Number) z.getMax()).longValue();
          assertEquals(
              expectedZoneMin,
              zoneMin,
              "zone min for fragment " + z.getFragmentId() + " starting at row " + zoneStart);
          assertEquals(
              expectedZoneMax,
              zoneMax,
              "zone max for fragment " + z.getFragmentId() + " starting at row " + zoneStart);
        }
        assertEquals(
            new HashSet<>(startValueByFragId.keySet()),
            fragIds,
            "every fragment must appear in the committed consolidated zonemap");

        // Across each fragment's zones, min should be monotone-nondecreasing with zone_start
        // (data is in row-address order and values are monotone within a fragment).
        for (Integer fragmentId : startValueByFragId.keySet()) {
          final int frag = fragmentId;
          List<ZoneStats> perFrag = new java.util.ArrayList<>();
          for (ZoneStats z : stats) {
            if (z.getFragmentId() == frag) perFrag.add(z);
          }
          perFrag.sort(java.util.Comparator.comparingLong(ZoneStats::getZoneStart));
          long prevMin = Long.MIN_VALUE;
          for (ZoneStats z : perFrag) {
            long m = ((Number) z.getMin()).longValue();
            assertTrue(
                m >= prevMin,
                "fragment " + frag + ": zone min must be monotone-nondecreasing along zone_start");
            prevMin = m;
          }
        }
      }
    }
  }

  /**
   * Single-worker case: the natural Spark setup where {@code num_partitions=1} skips the
   * consolidation rationale but the API must still work. Single-batch concat is a no-op but we
   * still want a properly committed segment.
   */
  @Test
  public void singleWorkerBatch(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("single").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(allocator, path, intSchema(), new WriteParams.Builder().build())) {
        // empty
      }
      writeIntFragment(allocator, path, 1, 0, 20).close();
      writeIntFragment(allocator, path, 2, 20, 20).close();

      // Capture actual fragment ids in commit-iteration order — consistent with
      // endToEndConsolidation. Tests should not bake "fragment ids are 0,1,2,..." into the
      // bitmap-expectation assertion below.
      Set<Integer> committedFragIds = new HashSet<>();
      try (Dataset dataset = Dataset.open(path, allocator)) {
        for (Fragment f : dataset.getFragments()) {
          committedFragIds.add(f.getId());
        }
      }

      try (Dataset dataset = Dataset.open(path, allocator)) {
        String paramsJson = "{\"rows_per_zone\": 5}";
        try (VectorSchemaRoot w1 =
            dataset.computeZonemapBatch("value", null, paramsJson, allocator)) {
          Index segment =
              dataset.writeZonemapIndexFromBatches(
                  "value_zm",
                  "value",
                  java.util.Collections.singletonList(w1),
                  paramsJson,
                  allocator);
          assertEquals(committedFragIds, new HashSet<>(segment.fragments().get()));
          dataset.commitExistingIndexSegments(
              "value_zm", "value", java.util.Collections.singletonList(segment));
        }
      }

      try (Dataset dataset = Dataset.open(path, allocator)) {
        assertEquals(
            1, dataset.getIndexes().stream().filter(i -> "value_zm".equals(i.name())).count());
        assertEquals(8, dataset.getZonemapStats("value").size(), "2 frags × 4 zones");
      }
    }
  }

  /**
   * Fragment-hole scenario: workers cover {0, 2}; fragment 1 has data but was deliberately not
   * indexed. The committed segment's bitmap should hold a literal hole rather than expanding to {0,
   * 1, 2}.
   */
  @Test
  public void fragmentIdHolePreserved(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("hole").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(allocator, path, intSchema(), new WriteParams.Builder().build())) {
        // empty
      }
      writeIntFragment(allocator, path, 1, 0, 20).close();
      writeIntFragment(allocator, path, 2, 20, 20).close();
      writeIntFragment(allocator, path, 3, 40, 20).close();

      // Discover the three actual fragment ids in commit order — same pattern as
      // endToEndConsolidation / singleWorkerBatch. We index the FIRST and THIRD; the MIDDLE
      // one should remain a hole in the committed bitmap.
      java.util.List<Integer> fragIds = new java.util.ArrayList<>();
      try (Dataset dataset = Dataset.open(path, allocator)) {
        for (Fragment f : dataset.getFragments()) {
          fragIds.add(f.getId());
        }
      }
      assertEquals(3, fragIds.size(), "expected 3 fragments committed");
      int indexedA = fragIds.get(0);
      int holeFragId = fragIds.get(1);
      int indexedB = fragIds.get(2);
      Set<Integer> expectedBitmap = new HashSet<>(Arrays.asList(indexedA, indexedB));

      try (Dataset dataset = Dataset.open(path, allocator)) {
        // Note: middle fragment deliberately excluded — that's the hole this test pins.
        try (VectorSchemaRoot batch =
            dataset.computeZonemapBatch(
                "value",
                new long[] {(long) indexedA, (long) indexedB},
                "{\"rows_per_zone\": 5}",
                allocator)) {
          Index segment =
              dataset.writeZonemapIndexFromBatches(
                  "value_zm",
                  "value",
                  java.util.Collections.singletonList(batch),
                  "{\"rows_per_zone\": 5}",
                  allocator);
          assertEquals(
              expectedBitmap,
              new HashSet<>(segment.fragments().get()),
              "in-memory segment bitmap must not invent fragments");
          dataset.commitExistingIndexSegments(
              "value_zm", "value", java.util.Collections.singletonList(segment));
        }
      }

      // Reopen and verify the hole survives on disk through both the IndexMetadata bitmap and
      // the read-back zone-stats listing (which is what consumers actually see).
      try (Dataset dataset = Dataset.open(path, allocator)) {
        Index committed =
            dataset.getIndexes().stream()
                .filter(i -> "value_zm".equals(i.name()))
                .findFirst()
                .orElseThrow();
        assertEquals(
            expectedBitmap,
            new HashSet<>(committed.fragments().get()),
            "committed manifest must preserve the literal hole");
        assertFalse(
            committed.fragments().get().contains(holeFragId),
            "the deliberately unindexed fragment must NOT appear in the committed bitmap");

        Set<Integer> statFragIds = new HashSet<>();
        for (ZoneStats z : dataset.getZonemapStats("value")) {
          statFragIds.add(z.getFragmentId());
        }
        assertEquals(
            expectedBitmap,
            statFragIds,
            "getZonemapStats must not return zones for the unindexed fragment");
      }
    }
  }

  /**
   * All-NULL column: the doc claims min/max are nullable because an entire batch may be all-NULL.
   * Verify the round-trip through compute → write → commit → re-read works for that shape without
   * crashing.
   */
  @Test
  public void allNullColumn(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("all_null").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      Schema schema = intSchema();
      try (Dataset ds =
          Dataset.create(allocator, path, schema, new WriteParams.Builder().build())) {
        // empty
      }
      // Write a fragment whose 'value' column is entirely NULL.
      List<FragmentMetadata> metas;
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        root.allocateNew();
        IntVector idVec = (IntVector) root.getVector("id");
        IntVector valVec = (IntVector) root.getVector("value");
        for (int i = 0; i < 20; i++) {
          idVec.setSafe(i, i);
          valVec.setNull(i);
        }
        root.setRowCount(20);
        metas = Fragment.create(path, allocator, root, new WriteParams.Builder().build());
      }
      Dataset.commit(allocator, path, new FragmentOperation.Append(metas), Optional.of(1L)).close();

      try (Dataset dataset = Dataset.open(path, allocator);
          VectorSchemaRoot batch =
              dataset.computeZonemapBatch("value", null, "{\"rows_per_zone\": 5}", allocator)) {
        // Even with all-NULL data we still get one zone per zone-sized chunk.
        assertTrue(batch.getRowCount() > 0, "all-NULL fragment must still produce zones");
        Index segment =
            dataset.writeZonemapIndexFromBatches(
                "value_zm",
                "value",
                java.util.Collections.singletonList(batch),
                "{\"rows_per_zone\": 5}",
                allocator);
        dataset.commitExistingIndexSegments(
            "value_zm", "value", java.util.Collections.singletonList(segment));
      }

      try (Dataset dataset = Dataset.open(path, allocator)) {
        List<ZoneStats> stats = dataset.getZonemapStats("value");
        assertFalse(stats.isEmpty(), "all-NULL fragment must produce readable stats");
        for (ZoneStats z : stats) {
          assertEquals(
              z.getZoneLength(),
              z.getNullCount(),
              "every zone in an all-NULL column must have nullCount == zoneLength");
          assertNull(z.getMin(), "min must be null when every value is null");
          assertNull(z.getMax(), "max must be null when every value is null");
        }
      }
    }
  }

  @Test
  public void rejectsEmptyBatchList(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("empty").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset dataset =
          Dataset.create(allocator, path, intSchema(), new WriteParams.Builder().build())) {
        assertThrows(
            IllegalArgumentException.class,
            () ->
                dataset.writeZonemapIndexFromBatches(
                    "zm", "value", java.util.Collections.emptyList(), "", allocator));
      }
    }
  }

  @Test
  public void rejectsNullBatchInList(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("null_batch").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(allocator, path, intSchema(), new WriteParams.Builder().build())) {
        // empty
      }
      writeIntFragment(allocator, path, 1, 0, 10).close();
      try (Dataset dataset = Dataset.open(path, allocator);
          VectorSchemaRoot batch = dataset.computeZonemapBatch("value", null, "", allocator)) {
        List<VectorSchemaRoot> withNull = Arrays.asList(batch, null);
        assertThrows(
            NullPointerException.class,
            () -> dataset.writeZonemapIndexFromBatches("zm", "value", withNull, "", allocator));
      }
    }
  }

  /**
   * Exercises the defence-in-depth schema check on the Rust side: feed two batches whose schemas
   * differ (one from `value`, one from a different column) and expect a clear error rather than a
   * confusing concat_batches failure deeper in the stack.
   */
  @Test
  public void rejectsDivergentBatchSchemas(@TempDir Path tempDir) throws Exception {
    // Multi-column dataset so workers can compute zonemap batches with different min/max
    // value types (Int32 for `value`, Int32 — but on a different Lance column — for `id`).
    // The two output batches have identical Arrow schema shapes, so to force a real divergence
    // we use a column with a wider integer type for one worker.
    String path = tempDir.resolve("divergent").toString();
    Schema schema =
        new Schema(
            Arrays.asList(
                Field.nullable("id_i32", new ArrowType.Int(32, true)),
                Field.nullable("val_i64", new ArrowType.Int(64, true))),
            null);
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(allocator, path, schema, new WriteParams.Builder().build())) {
        // empty
      }
      // Write a tiny fragment so computeZonemapBatch has data over which to compute.
      List<FragmentMetadata> metas;
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        root.allocateNew();
        ((IntVector) root.getVector("id_i32")).setSafe(0, 1);
        ((org.apache.arrow.vector.BigIntVector) root.getVector("val_i64")).setSafe(0, 100L);
        root.setRowCount(1);
        metas = Fragment.create(path, allocator, root, new WriteParams.Builder().build());
      }
      Dataset.commit(allocator, path, new FragmentOperation.Append(metas), Optional.of(1L)).close();

      try (Dataset dataset = Dataset.open(path, allocator);
          VectorSchemaRoot wIdInt32 = dataset.computeZonemapBatch("id_i32", null, "", allocator);
          VectorSchemaRoot wValInt64 =
              dataset.computeZonemapBatch("val_i64", null, "", allocator)) {
        // The two batches have different min/max types (Int32 vs Int64). Native-side error
        // bubbles up as IllegalArgumentException (the Rust JNI maps Error::input_error there).
        // Tightening the expected class beats `Exception.class` — the latter would also accept
        // an unrelated IOException whose message happens to contain "schema".
        IllegalArgumentException thrown =
            assertThrows(
                IllegalArgumentException.class,
                () ->
                    dataset.writeZonemapIndexFromBatches(
                        "zm", "id_i32", Arrays.asList(wIdInt32, wValInt64), "", allocator));
        assertTrue(
            thrown.getMessage() != null && thrown.getMessage().toLowerCase().contains("schema"),
            "error must mention schema; got: " + thrown.getMessage());
      }
    }
  }

  /**
   * Coordinator-bug scenario: two workers were accidentally assigned overlapping fragment ids. The
   * current contract is that the consolidated batch faithfully reflects whatever the caller hands
   * it (no implicit dedup), so the resulting segment contains duplicated zones for the shared
   * fragment ids and the bitmap dedups (RoaringBitmap is a set). This test pins that observable
   * behaviour so a future change to either policy (silent-dedup vs explicit-reject) shows up here.
   */
  @Test
  public void overlappingFragmentIdsAcrossWorkers(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("overlap").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(allocator, path, intSchema(), new WriteParams.Builder().build())) {
        // empty
      }
      writeIntFragment(allocator, path, 1, 0, 20).close();
      writeIntFragment(allocator, path, 2, 20, 20).close();

      // Discover the two fragment ids in commit order. Worker 1 indexes both; worker 2
      // overlaps on the SECOND one — the coordinator-bug shape this test pins.
      java.util.List<Integer> fragIds = new java.util.ArrayList<>();
      try (Dataset dataset = Dataset.open(path, allocator)) {
        for (Fragment f : dataset.getFragments()) {
          fragIds.add(f.getId());
        }
      }
      assertEquals(2, fragIds.size(), "expected 2 fragments committed");
      int fragA = fragIds.get(0);
      int fragB = fragIds.get(1);

      try (Dataset dataset = Dataset.open(path, allocator)) {
        String paramsJson = "{\"rows_per_zone\": 5}";
        try (VectorSchemaRoot w1 =
                dataset.computeZonemapBatch(
                    "value", new long[] {(long) fragA, (long) fragB}, paramsJson, allocator);
            // Worker 2 overlaps with worker 1 on fragB — coordinator bug.
            VectorSchemaRoot w2 =
                dataset.computeZonemapBatch(
                    "value", new long[] {(long) fragB}, paramsJson, allocator)) {

          Index segment =
              dataset.writeZonemapIndexFromBatches(
                  "value_zm", "value", Arrays.asList(w1, w2), paramsJson, allocator);

          // Bitmap dedups (it's a set). The DATA in the segment contains duplicated zone rows
          // for fragB — that's a coordinator bug surfacing as confused read-time stats
          // rather than a write-time error.
          assertEquals(
              new HashSet<>(Arrays.asList(fragA, fragB)),
              new HashSet<>(segment.fragments().get()),
              "bitmap dedups to the union of fragment ids");
          dataset.commitExistingIndexSegments(
              "value_zm", "value", java.util.Collections.singletonList(segment));
        }
      }

      try (Dataset dataset = Dataset.open(path, allocator)) {
        List<ZoneStats> stats = dataset.getZonemapStats("value");
        // 12 zone rows total: 8 from worker 1 (fragA + fragB, 4 zones each) + 4 from worker
        // 2 (fragB, 4 zones). The duplication for fragB is preserved on disk.
        assertEquals(
            12,
            stats.size(),
            "overlapping fragment partitions produce duplicated zone rows — caller bug surfaces "
                + "at read time, not write time");
        // Verify the duplication is on the OVERLAPPED fragment, not somewhere unexpected. A bug
        // that produced 12 zones all on fragA would pass a size-only assertion.
        long zonesForA = stats.stream().filter(z -> z.getFragmentId() == fragA).count();
        long zonesForB = stats.stream().filter(z -> z.getFragmentId() == fragB).count();
        assertEquals(4, zonesForA, "fragA indexed once → 4 zones");
        assertEquals(8, zonesForB, "fragB indexed twice → 8 zones (duplicate evidence)");
      }
    }
  }

  /**
   * NaN is not conflated with NULL through the FFI / commit / re-read pipeline. A Float64 column
   * carrying NaN values but no NULLs must report {@code nullCount=0} on every zone — proving NaN
   * counting lives in its own column and is not being merged into the null lane somewhere along the
   * export path. The Java {@link ZoneStats} type currently exposes only {@code nullCount} (the
   * {@code nan_count} column is not surfaced), so the assertion is indirect; the test name reflects
   * what is actually verified.
   */
  @Test
  public void nanNotConflatedWithNull(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("nan").toString();
    Schema schema =
        new Schema(
            Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true)),
                Field.nullable(
                    "val_f64",
                    new ArrowType.FloatingPoint(
                        org.apache.arrow.vector.types.FloatingPointPrecision.DOUBLE))),
            null);
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(allocator, path, schema, new WriteParams.Builder().build())) {
        // empty
      }
      // Write a 20-row fragment where rows 5..10 are NaN.
      List<FragmentMetadata> metas;
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        root.allocateNew();
        IntVector idVec = (IntVector) root.getVector("id");
        org.apache.arrow.vector.Float8Vector valVec =
            (org.apache.arrow.vector.Float8Vector) root.getVector("val_f64");
        for (int i = 0; i < 20; i++) {
          idVec.setSafe(i, i);
          if (i >= 5 && i < 10) {
            valVec.setSafe(i, Double.NaN);
          } else {
            valVec.setSafe(i, (double) i);
          }
        }
        root.setRowCount(20);
        metas = Fragment.create(path, allocator, root, new WriteParams.Builder().build());
      }
      Dataset.commit(allocator, path, new FragmentOperation.Append(metas), Optional.of(1L)).close();

      try (Dataset dataset = Dataset.open(path, allocator);
          VectorSchemaRoot batch =
              dataset.computeZonemapBatch("val_f64", null, "{\"rows_per_zone\": 5}", allocator)) {
        Index segment =
            dataset.writeZonemapIndexFromBatches(
                "val_zm",
                "val_f64",
                java.util.Collections.singletonList(batch),
                "{\"rows_per_zone\": 5}",
                allocator);
        dataset.commitExistingIndexSegments(
            "val_zm", "val_f64", java.util.Collections.singletonList(segment));
      }

      try (Dataset dataset = Dataset.open(path, allocator)) {
        // 20 rows / 5 = 4 zones. Zone covering rows 5..10 has 5 NaN values; zone covering
        // 10..15 has 0 (since NaN range was 5..10 exclusive).
        // ZoneStats currently exposes nullCount but not nanCount via the public Java type;
        // verify indirectly by confirming that zones containing the NaN range still produce
        // valid min/max for the non-NaN portion and have zero null_count.
        List<ZoneStats> stats = dataset.getZonemapStats("val_f64");
        assertEquals(4, stats.size(), "20 rows / rows_per_zone=5 → 4 zones");
        for (ZoneStats z : stats) {
          // No row in this fragment is NULL; only NaN. So null_count must remain 0 for every
          // zone — confirming NaN is NOT being conflated with NULL through the FFI pipeline.
          assertEquals(
              0,
              z.getNullCount(),
              "zone "
                  + z.getZoneStart()
                  + ": NaN must not be counted as NULL — got nullCount="
                  + z.getNullCount());
        }
      }
    }
  }

  /**
   * Malformed paramsJson causes the JNI to error BEFORE the FFI from_raw loop runs — i.e. the Java
   * side has already allocated ArrowArray/ArrowSchema handles and exported batch data into them,
   * but Rust never consumes via from_raw. The finally block must release all of those handles
   * cleanly so RootAllocator.close() doesn't fail with a leak; if Arrow Java's {@code close()}
   * alone weren't enough, this test would fail at the RootAllocator's try-with-resources exit.
   */
  @Test
  public void invalidParamsJsonDoesNotLeakFfiHandles(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("bad_params").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(allocator, path, intSchema(), new WriteParams.Builder().build())) {
        // empty
      }
      writeIntFragment(allocator, path, 1, 0, 10).close();
      writeIntFragment(allocator, path, 2, 10, 10).close();

      // Discover fragment ids via getFragments() so this test doesn't bake in the assumption
      // that ids are allocated sequentially from 0 — same pattern as the other tests in this
      // suite. If Lance ever changes allocation strategy the test still exercises the
      // leak-window contract (malformed paramsJson AFTER successful FFI export).
      java.util.List<Long> discoveredFragIds = new java.util.ArrayList<>();
      try (Dataset dataset = Dataset.open(path, allocator)) {
        for (Fragment f : dataset.getFragments()) {
          discoveredFragIds.add((long) f.getId());
        }
      }
      assertEquals(2, discoveredFragIds.size(), "expected 2 fragments committed");

      try (Dataset dataset = Dataset.open(path, allocator);
          VectorSchemaRoot w1 =
              dataset.computeZonemapBatch(
                  "value", new long[] {discoveredFragIds.get(0)}, "", allocator);
          VectorSchemaRoot w2 =
              dataset.computeZonemapBatch(
                  "value", new long[] {discoveredFragIds.get(1)}, "", allocator)) {
        // Force the JNI's serde_json parse step to fail AFTER Java has finished exporting
        // both batches into FFI handles.
        IllegalArgumentException thrown =
            assertThrows(
                IllegalArgumentException.class,
                () ->
                    dataset.writeZonemapIndexFromBatches(
                        "zm", "value", Arrays.asList(w1, w2), "{not valid json", allocator));
        assertTrue(
            thrown.getMessage() != null && thrown.getMessage().toLowerCase().contains("params"),
            "error must mention params JSON; got: " + thrown.getMessage());
      }
    }
  }

  /**
   * Upper-bound fragment id rejection: {@code long} values greater than {@code u32::MAX}
   * (4294967295) must be rejected before the JNI silently truncates. Counterpart to the negative-id
   * test below; together they pin the full out-of-range contract.
   */
  @Test
  public void aboveU32MaxFragmentIdRejected(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("oversize_frag").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(allocator, path, intSchema(), new WriteParams.Builder().build())) {
        // empty
      }
      writeIntFragment(allocator, path, 1, 0, 10).close();
      try (Dataset dataset = Dataset.open(path, allocator)) {
        long aboveU32Max = (1L << 32); // u32::MAX is 2^32 - 1; one past = 2^32.
        IllegalArgumentException thrown =
            assertThrows(
                IllegalArgumentException.class,
                () ->
                    dataset.computeZonemapBatch("value", new long[] {aboveU32Max}, "", allocator));
        assertTrue(
            thrown.getMessage() != null && thrown.getMessage().toLowerCase().contains("fragment"),
            "error must mention fragment id; got: " + thrown.getMessage());
      }
    }
  }

  /**
   * Column-not-found is rejected at the Rust layer (Java only checks null/empty string). Verifies
   * the typo-catching error propagates back as IllegalArgumentException with a clear message rather
   * than e.g. an NPE deeper in the import path.
   */
  @Test
  public void unknownColumnRejected(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("unknown_col").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(allocator, path, intSchema(), new WriteParams.Builder().build())) {
        // empty
      }
      writeIntFragment(allocator, path, 1, 0, 10).close();
      try (Dataset dataset = Dataset.open(path, allocator)) {
        IllegalArgumentException thrown =
            assertThrows(
                IllegalArgumentException.class,
                () -> dataset.computeZonemapBatch("not_a_column", null, "", allocator));
        assertTrue(
            thrown.getMessage() != null
                && thrown.getMessage().toLowerCase().contains("not_a_column"),
            "error must name the missing column; got: " + thrown.getMessage());
      }
    }
  }

  /**
   * Negative fragment id rejection — Java accepts a {@code long[]} so a negative element is
   * reachable; the JNI must reject rather than silently u32-wrap.
   */
  @Test
  public void negativeFragmentIdRejected(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("neg_frag").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(allocator, path, intSchema(), new WriteParams.Builder().build())) {
        // empty
      }
      writeIntFragment(allocator, path, 1, 0, 10).close();
      try (Dataset dataset = Dataset.open(path, allocator)) {
        IllegalArgumentException thrown =
            assertThrows(
                IllegalArgumentException.class,
                () -> dataset.computeZonemapBatch("value", new long[] {-1L}, "", allocator));
        assertTrue(
            thrown.getMessage() != null && thrown.getMessage().toLowerCase().contains("fragment"),
            "error must mention fragment id; got: " + thrown.getMessage());
      }
    }
  }
}
