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
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.DateMilliVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.TimeStampMicroTZVector;
import org.apache.arrow.vector.TimeStampMicroVector;
import org.apache.arrow.vector.TimeStampMilliTZVector;
import org.apache.arrow.vector.TimeStampMilliVector;
import org.apache.arrow.vector.TimeStampNanoTZVector;
import org.apache.arrow.vector.TimeStampNanoVector;
import org.apache.arrow.vector.TimeStampSecTZVector;
import org.apache.arrow.vector.TimeStampSecVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.types.Types;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Decodes the Arrow IPC byte stream returned by {@code nativeGetZonemapStatsIpc} into a {@code
 * List<ZoneStats>}.
 *
 * <p>This codec exists to keep {@code List<ZoneStats>} as the public surface of {@link
 * org.lance.Dataset#getZonemapStats(String)} while letting the JNI hot path return a single Arrow
 * IPC byte array instead of constructing N {@code ZoneStats} JNI objects per call. The type →
 * {@code Comparable} mapping mirrors the Rust {@code scalar_value_to_java} helper.
 */
public final class ZonemapStatsCodec {

  private ZonemapStatsCodec() {}

  /**
   * Decode the IPC payload produced by the JNI call.
   *
   * @param ipcBytes Arrow IPC stream, possibly empty (length 0) when the column has no zonemap
   * @param allocator the {@link BufferAllocator} owning intermediate buffers; the caller retains
   *     ownership and is responsible for closing it
   * @return zone stats in the order the Rust side produced them
   */
  public static List<ZoneStats> decode(byte[] ipcBytes, BufferAllocator allocator)
      throws IOException {
    if (ipcBytes == null || ipcBytes.length == 0) {
      return Collections.emptyList();
    }

    ArrayList<ZoneStats> out = new ArrayList<>();
    try (ByteArrayInputStream in = new ByteArrayInputStream(ipcBytes);
        ArrowStreamReader reader = new ArrowStreamReader(in, allocator)) {
      while (reader.loadNextBatch()) {
        VectorSchemaRoot root = reader.getVectorSchemaRoot();
        FieldVector fragmentIdVec = root.getVector("fragment_id");
        FieldVector zoneStartVec = root.getVector("zone_start");
        FieldVector zoneLengthVec = root.getVector("zone_length");
        FieldVector nullCountVec = root.getVector("null_count");
        FieldVector minVec = root.getVector("min");
        FieldVector maxVec = root.getVector("max");

        int n = root.getRowCount();
        out.ensureCapacity(out.size() + n);
        for (int i = 0; i < n; i++) {
          int fragmentId = (int) toLong(fragmentIdVec, i, "fragment_id");
          long zoneStart = toLong(zoneStartVec, i, "zone_start");
          long zoneLength = toLong(zoneLengthVec, i, "zone_length");
          long nullCount = toLong(nullCountVec, i, "null_count");
          Comparable<?> min = toComparable(minVec, i);
          Comparable<?> max = toComparable(maxVec, i);
          out.add(new ZoneStats(fragmentId, zoneStart, zoneLength, min, max, nullCount));
        }
      }
    }
    return out;
  }

  /**
   * Coerce a primitive integer-family vector value at row {@code i} to {@code long}.
   *
   * <p>The four columns this is called on (fragment_id, zone_start, zone_length, null_count) are
   * declared non-nullable in the canonical zonemap schema produced on the Rust side. A null
   * observed here means corruption upstream — surface it loudly rather than silently substituting
   * {@code 0}.
   */
  private static long toLong(FieldVector v, int i, String fieldName) {
    if (v.isNull(i)) {
      throw new IllegalStateException(
          "unexpected null in non-nullable column '" + fieldName + "' at row " + i);
    }
    Object obj = v.getObject(i);
    if (obj instanceof Number) {
      return ((Number) obj).longValue();
    }
    throw new IllegalStateException(
        "expected numeric vector for '"
            + fieldName
            + "', got "
            + v.getField().getType()
            + " at row "
            + i);
  }

  /**
   * Convert one row of an arbitrary Arrow vector to the {@code Comparable} the public {@code
   * ZoneStats} API exposes. Mirrors {@code scalar_value_to_java} on the Rust side: int family →
   * {@link Long}, float family → {@link Double}, utf8 → {@link String}, bool → {@link Boolean},
   * date / timestamp → {@link Long} (epoch units preserved as in Rust).
   *
   * <p>Date and timestamp arms must use the typed-vector {@code .get(i)} API rather than {@code
   * getObject(i)}: in Arrow Java, {@code DateDayVector#getObject} returns {@link
   * java.time.LocalDate}, {@code DateMilliVector#getObject} and the non-TZ {@code TimeStamp*Vector}
   * variants return {@link java.time.LocalDateTime}, neither of which is a {@link Number} — casting
   * through {@code Number} crashes with {@code ClassCastException}.
   */
  private static Comparable<?> toComparable(FieldVector v, int i) {
    if (v.isNull(i)) {
      return null;
    }
    Types.MinorType minor = v.getMinorType();
    switch (minor) {
      case TINYINT:
      case SMALLINT:
      case INT:
      case BIGINT:
      case UINT1:
      case UINT2:
      case UINT4:
      case UINT8:
        return ((Number) v.getObject(i)).longValue();

      case DATEDAY:
        return (long) ((DateDayVector) v).get(i);
      case DATEMILLI:
        return ((DateMilliVector) v).get(i);
      case TIMESTAMPSEC:
        return ((TimeStampSecVector) v).get(i);
      case TIMESTAMPMILLI:
        return ((TimeStampMilliVector) v).get(i);
      case TIMESTAMPMICRO:
        return ((TimeStampMicroVector) v).get(i);
      case TIMESTAMPNANO:
        return ((TimeStampNanoVector) v).get(i);
      case TIMESTAMPSECTZ:
        return ((TimeStampSecTZVector) v).get(i);
      case TIMESTAMPMILLITZ:
        return ((TimeStampMilliTZVector) v).get(i);
      case TIMESTAMPMICROTZ:
        return ((TimeStampMicroTZVector) v).get(i);
      case TIMESTAMPNANOTZ:
        return ((TimeStampNanoTZVector) v).get(i);

      case FLOAT2:
      case FLOAT4:
      case FLOAT8:
        return ((Number) v.getObject(i)).doubleValue();

      case BIT:
        {
          Object raw = v.getObject(i);
          if (raw instanceof Boolean) {
            return (Boolean) raw;
          }
          return ((Number) raw).intValue() != 0;
        }

      case VARCHAR:
      case LARGEVARCHAR:
        {
          Object raw = v.getObject(i);
          if (raw instanceof byte[]) {
            return new String((byte[]) raw, StandardCharsets.UTF_8);
          }
          return raw.toString();
        }

      default:
        // Conservative parity with Rust's `_ => Ok(JObject::null())` arm.
        return null;
    }
  }
}
