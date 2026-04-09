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
package org.lance.fragment;

import com.google.common.base.MoreObjects;

import java.io.ByteArrayOutputStream;
import java.io.Serializable;
import java.util.Objects;

public class RowIdMeta implements Serializable {
  private static final long serialVersionUID = -6532828695072614148L;

  private final String metadata;

  public RowIdMeta(String metadata) {
    this.metadata = metadata;
  }

  /**
   * Creates a RowIdMeta from an array of stable row IDs. Encodes them as an inline RowIdSequence
   * protobuf wrapped in the JSON format expected by lance-core.
   */
  public static RowIdMeta fromRowIds(long[] rowIds) {
    byte[] values = new byte[rowIds.length * 8];
    for (int r = 0; r < rowIds.length; r++) {
      long id = rowIds[r];
      int base = r * 8;
      for (int i = 0; i < 8; i++) {
        values[base + i] = (byte) ((id >>> (8 * i)) & 0xFF);
      }
    }
    // RowIdSequence protobuf nesting:
    // segment(1) > encoded(5) > u64array(3) > bytes(2)
    byte[] proto = lenDelimited(1, lenDelimited(5, lenDelimited(3, lenDelimited(2, values))));
    StringBuilder sb = new StringBuilder(12 + proto.length * 4);
    sb.append("{\"Inline\":[");
    for (int i = 0; i < proto.length; i++) {
      if (i > 0) sb.append(',');
      sb.append(proto[i] & 0xFF);
    }
    sb.append("]}");
    return new RowIdMeta(sb.toString());
  }

  private static byte[] lenDelimited(int fieldNumber, byte[] data) {
    int tag = (fieldNumber << 3) | 2;
    ByteArrayOutputStream out = new ByteArrayOutputStream(1 + 5 + data.length);
    writeVarint(out, tag);
    writeVarint(out, data.length);
    out.write(data, 0, data.length);
    return out.toByteArray();
  }

  private static void writeVarint(ByteArrayOutputStream out, int value) {
    while ((value & ~0x7F) != 0) {
      out.write((value & 0x7F) | 0x80);
      value >>>= 7;
    }
    out.write(value);
  }

  public String getMetadata() {
    return metadata;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null || getClass() != obj.getClass()) {
      return false;
    }
    RowIdMeta that = (RowIdMeta) obj;
    return Objects.equals(metadata, that.metadata);
  }

  @Override
  public int hashCode() {
    return Objects.hash(metadata);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("metadata", metadata).toString();
  }
}
