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
package org.lance.index.scalar;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.DateMilliVector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.TimeStampMicroVector;
import org.apache.arrow.vector.UInt4Vector;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Codec-level unit tests for {@link ZonemapStatsCodec}. Synthesizes Arrow IPC byte streams directly
 * — no Lance dataset, no JNI — so we can exercise edge cases the Rust producer is contractually
 * forbidden from emitting (null in a non-nullable id column, intentionally chosen item types).
 */
public class ZonemapStatsCodecTest {

  @FunctionalInterface
  private interface BatchPopulator {
    void populate(VectorSchemaRoot root);
  }

  /**
   * Encode 1+ batches as one IPC stream. Each populator must call {@code root.setRowCount(...)}.
   */
  private static byte[] encodeIpc(
      Schema schema, BufferAllocator allocator, BatchPopulator... batches) throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
        ArrowStreamWriter writer = new ArrowStreamWriter(root, null, out)) {
      writer.start();
      for (BatchPopulator pop : batches) {
        root.allocateNew();
        pop.populate(root);
        writer.writeBatch();
        for (int j = 0; j < root.getFieldVectors().size(); j++) {
          root.getFieldVectors().get(j).clear();
        }
      }
      writer.end();
    }
    return out.toByteArray();
  }

  /** Canonical zonemap schema (mirrors Rust {@code zonemap_stats_as_batch}). */
  private static Schema zonemapSchema(ArrowType itemsType) {
    return new Schema(
        Arrays.asList(
            Field.nullable("min", itemsType),
            Field.nullable("max", itemsType),
            new Field("null_count", FieldType.notNullable(new ArrowType.Int(32, false)), null),
            new Field("nan_count", FieldType.notNullable(new ArrowType.Int(32, false)), null),
            new Field("fragment_id", FieldType.notNullable(new ArrowType.Int(64, false)), null),
            new Field("zone_start", FieldType.notNullable(new ArrowType.Int(64, false)), null),
            new Field("zone_length", FieldType.notNullable(new ArrowType.Int(64, false)), null)),
        null);
  }

  @Test
  public void decode_emptyBytes_returnsEmptyList() throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      assertTrue(ZonemapStatsCodec.decode(new byte[0], allocator).isEmpty());
      assertTrue(ZonemapStatsCodec.decode(null, allocator).isEmpty());
    }
  }

  @ParameterizedTest
  @ValueSource(strings = {"fragment_id", "zone_start", "zone_length", "null_count"})
  public void decode_nullInNonNullableColumn_throwsIllegalStateException(String nullField)
      throws Exception {
    // Each of these four columns is non-nullable in the canonical schema; declare them
    // nullable here only to simulate "upstream produced a null" corruption — the decoder
    // must reject it explicitly instead of silently substituting 0.
    Schema schema =
        new Schema(
            Arrays.asList(
                Field.nullable("min", new ArrowType.Int(32, true)),
                Field.nullable("max", new ArrowType.Int(32, true)),
                Field.nullable("null_count", new ArrowType.Int(32, false)),
                Field.nullable("nan_count", new ArrowType.Int(32, false)),
                Field.nullable("fragment_id", new ArrowType.Int(64, false)),
                Field.nullable("zone_start", new ArrowType.Int(64, false)),
                Field.nullable("zone_length", new ArrowType.Int(64, false))),
            null);
    try (BufferAllocator allocator = new RootAllocator()) {
      byte[] ipc =
          encodeIpc(
              schema,
              allocator,
              root -> {
                ((IntVector) root.getVector("min")).setSafe(0, 1);
                ((IntVector) root.getVector("max")).setSafe(0, 100);
                UInt4Vector nullCount = (UInt4Vector) root.getVector("null_count");
                UInt4Vector nanCount = (UInt4Vector) root.getVector("nan_count");
                UInt8Vector fragId = (UInt8Vector) root.getVector("fragment_id");
                UInt8Vector start = (UInt8Vector) root.getVector("zone_start");
                UInt8Vector length = (UInt8Vector) root.getVector("zone_length");

                // Populate every non-target column with a valid value, then null out the target.
                if (!nullField.equals("null_count")) {
                  nullCount.setSafe(0, 0);
                }
                nanCount.setSafe(0, 0);
                if (!nullField.equals("fragment_id")) {
                  fragId.setSafe(0, 0L);
                }
                if (!nullField.equals("zone_start")) {
                  start.setSafe(0, 0L);
                }
                if (!nullField.equals("zone_length")) {
                  length.setSafe(0, 1024L);
                }

                switch (nullField) {
                  case "null_count":
                    nullCount.setNull(0);
                    break;
                  case "fragment_id":
                    fragId.setNull(0);
                    break;
                  case "zone_start":
                    start.setNull(0);
                    break;
                  case "zone_length":
                    length.setNull(0);
                    break;
                  default:
                    throw new IllegalArgumentException("unexpected field: " + nullField);
                }
                root.setRowCount(1);
              });

      IllegalStateException ex =
          assertThrows(IllegalStateException.class, () -> ZonemapStatsCodec.decode(ipc, allocator));
      assertTrue(
          ex.getMessage().contains(nullField),
          "error must name the offending field, got: " + ex.getMessage());
      assertTrue(
          ex.getMessage().contains("at row 0"),
          "error must include row index in 'at row N' form, got: " + ex.getMessage());
    }
  }

  @Test
  public void decode_dateDay_returnsLongDays() throws Exception {
    Schema schema = zonemapSchema(new ArrowType.Date(DateUnit.DAY));
    try (BufferAllocator allocator = new RootAllocator()) {
      byte[] ipc =
          encodeIpc(
              schema,
              allocator,
              root -> {
                ((DateDayVector) root.getVector("min")).setSafe(0, 19737);
                ((DateDayVector) root.getVector("max")).setSafe(0, 19800);
                ((UInt4Vector) root.getVector("null_count")).setSafe(0, 0);
                ((UInt4Vector) root.getVector("nan_count")).setSafe(0, 0);
                ((UInt8Vector) root.getVector("fragment_id")).setSafe(0, 0L);
                ((UInt8Vector) root.getVector("zone_start")).setSafe(0, 0L);
                ((UInt8Vector) root.getVector("zone_length")).setSafe(0, 1024L);
                root.setRowCount(1);
              });
      List<ZoneStats> got = ZonemapStatsCodec.decode(ipc, allocator);
      assertEquals(1, got.size());
      assertEquals(19737L, got.get(0).getMin());
      assertEquals(19800L, got.get(0).getMax());
    }
  }

  @Test
  public void decode_dateMilli_returnsLongMillis() throws Exception {
    Schema schema = zonemapSchema(new ArrowType.Date(DateUnit.MILLISECOND));
    try (BufferAllocator allocator = new RootAllocator()) {
      byte[] ipc =
          encodeIpc(
              schema,
              allocator,
              root -> {
                ((DateMilliVector) root.getVector("min")).setSafe(0, 1_700_000_000_000L);
                ((DateMilliVector) root.getVector("max")).setSafe(0, 1_700_000_999_000L);
                ((UInt4Vector) root.getVector("null_count")).setSafe(0, 0);
                ((UInt4Vector) root.getVector("nan_count")).setSafe(0, 0);
                ((UInt8Vector) root.getVector("fragment_id")).setSafe(0, 0L);
                ((UInt8Vector) root.getVector("zone_start")).setSafe(0, 0L);
                ((UInt8Vector) root.getVector("zone_length")).setSafe(0, 1024L);
                root.setRowCount(1);
              });
      List<ZoneStats> got = ZonemapStatsCodec.decode(ipc, allocator);
      assertEquals(1, got.size());
      assertEquals(1_700_000_000_000L, got.get(0).getMin());
      assertEquals(1_700_000_999_000L, got.get(0).getMax());
    }
  }

  @Test
  public void decode_timestampMicro_returnsLongMicros() throws Exception {
    // The bug arm: Timestamp(MICROSECOND, null) — non-TZ. Arrow Java's getObject returns
    // LocalDateTime here; the existing codec's `(Number) raw` cast crashes on it.
    Schema schema = zonemapSchema(new ArrowType.Timestamp(TimeUnit.MICROSECOND, null));
    try (BufferAllocator allocator = new RootAllocator()) {
      byte[] ipc =
          encodeIpc(
              schema,
              allocator,
              root -> {
                ((TimeStampMicroVector) root.getVector("min")).setSafe(0, 1_700_000_000_000_000L);
                ((TimeStampMicroVector) root.getVector("max")).setSafe(0, 1_700_000_999_999_999L);
                ((UInt4Vector) root.getVector("null_count")).setSafe(0, 0);
                ((UInt4Vector) root.getVector("nan_count")).setSafe(0, 0);
                ((UInt8Vector) root.getVector("fragment_id")).setSafe(0, 0L);
                ((UInt8Vector) root.getVector("zone_start")).setSafe(0, 0L);
                ((UInt8Vector) root.getVector("zone_length")).setSafe(0, 1024L);
                root.setRowCount(1);
              });
      List<ZoneStats> got = ZonemapStatsCodec.decode(ipc, allocator);
      assertEquals(1, got.size());
      assertEquals(1_700_000_000_000_000L, got.get(0).getMin());
      assertEquals(1_700_000_999_999_999L, got.get(0).getMax());
    }
  }

  @Test
  public void decode_multipleBatches_preservesOrder() throws Exception {
    Schema schema = zonemapSchema(new ArrowType.Int(32, true));
    try (BufferAllocator allocator = new RootAllocator()) {
      byte[] ipc =
          encodeIpc(
              schema,
              allocator,
              // Batch A: fragment 0, two zones.
              root -> {
                IntVector min = (IntVector) root.getVector("min");
                IntVector max = (IntVector) root.getVector("max");
                UInt4Vector nullCount = (UInt4Vector) root.getVector("null_count");
                UInt4Vector nanCount = (UInt4Vector) root.getVector("nan_count");
                UInt8Vector fragId = (UInt8Vector) root.getVector("fragment_id");
                UInt8Vector start = (UInt8Vector) root.getVector("zone_start");
                UInt8Vector length = (UInt8Vector) root.getVector("zone_length");
                min.setSafe(0, 0);
                max.setSafe(0, 99);
                min.setSafe(1, 100);
                max.setSafe(1, 199);
                nullCount.setSafe(0, 0);
                nullCount.setSafe(1, 0);
                nanCount.setSafe(0, 0);
                nanCount.setSafe(1, 0);
                fragId.setSafe(0, 0L);
                fragId.setSafe(1, 0L);
                start.setSafe(0, 0L);
                start.setSafe(1, 100L);
                length.setSafe(0, 100L);
                length.setSafe(1, 100L);
                root.setRowCount(2);
              },
              // Batch B: fragment 1, one zone.
              root -> {
                ((IntVector) root.getVector("min")).setSafe(0, 200);
                ((IntVector) root.getVector("max")).setSafe(0, 299);
                ((UInt4Vector) root.getVector("null_count")).setSafe(0, 0);
                ((UInt4Vector) root.getVector("nan_count")).setSafe(0, 0);
                ((UInt8Vector) root.getVector("fragment_id")).setSafe(0, 1L);
                ((UInt8Vector) root.getVector("zone_start")).setSafe(0, 0L);
                ((UInt8Vector) root.getVector("zone_length")).setSafe(0, 100L);
                root.setRowCount(1);
              });

      List<ZoneStats> got = ZonemapStatsCodec.decode(ipc, allocator);
      assertEquals(3, got.size(), "two zones from batch A + one from batch B");
      assertEquals(0, got.get(0).getFragmentId());
      assertEquals(0L, got.get(0).getZoneStart());
      assertEquals(0, got.get(1).getFragmentId());
      assertEquals(100L, got.get(1).getZoneStart());
      assertEquals(1, got.get(2).getFragmentId());
      assertEquals(0L, got.get(2).getZoneStart());
      assertEquals(0L, got.get(0).getMin());
      assertEquals(99L, got.get(0).getMax());
      assertEquals(100L, got.get(1).getMin());
      assertEquals(299L, got.get(2).getMax());
    }
  }
}
