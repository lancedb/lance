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

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class RowIdMetaTest {

  @Test
  void testFromRowIdsSingleRow() {
    RowIdMeta meta = RowIdMeta.fromRowIds(new long[] {42});
    String json = meta.getMetadata();
    assertTrue(json.startsWith("{\"Inline\":["));
    assertTrue(json.endsWith("]}"));
  }

  @Test
  void testFromRowIdsMultipleRows() {
    RowIdMeta meta = RowIdMeta.fromRowIds(new long[] {0, 1, 2, 100, Long.MAX_VALUE});
    assertNotNull(meta);
    String json = meta.getMetadata();
    assertFalse(json.isEmpty());
    assertTrue(json.startsWith("{\"Inline\":["));
    assertTrue(json.endsWith("]}"));
  }

  @Test
  void testFromRowIdsEmpty() {
    RowIdMeta meta = RowIdMeta.fromRowIds(new long[] {});
    String json = meta.getMetadata();
    assertTrue(json.startsWith("{\"Inline\":["));
    assertTrue(json.endsWith("]}"));
  }

  @Test
  void testFromRowIdsRoundTrip() {
    long[] ids = {10, 20, 30};
    RowIdMeta first = RowIdMeta.fromRowIds(ids);
    RowIdMeta second = RowIdMeta.fromRowIds(ids);
    assertEquals(first, second);
  }

  @Test
  void testFromRowIdsDeterministic() {
    long[] ids = {10, 20, 30};
    String a = RowIdMeta.fromRowIds(ids).getMetadata();
    String b = RowIdMeta.fromRowIds(ids).getMetadata();
    assertEquals(a, b);
  }

  @Test
  void testFromRowIdsProtoStructure() {
    long[] rowIds = {1};
    String json = RowIdMeta.fromRowIds(rowIds).getMetadata();

    int start = json.indexOf('[') + 1;
    int end = json.lastIndexOf(']');
    String[] parts = json.substring(start, end).split(",");
    byte[] proto = new byte[parts.length];
    for (int i = 0; i < parts.length; i++) {
      proto[i] = (byte) Integer.parseInt(parts[i].trim());
    }

    // Outermost: field 1, wire type 2 (length-delimited) → tag byte = 0x0a
    assertEquals((byte) 0x0a, proto[0]);

    // Walk 4 nested length-delimited fields to reach the payload
    int pos = 0;
    for (int level = 0; level < 4; level++) {
      int tag = proto[pos++] & 0xFF;
      assertEquals(2, tag & 0x07, "wire type must be 2 (length-delimited) at level " + level);
      // decode varint length
      int len = 0;
      int shift = 0;
      while (true) {
        int b = proto[pos++] & 0xFF;
        len |= (b & 0x7F) << shift;
        if ((b & 0x80) == 0) break;
        shift += 7;
      }
      if (level < 3) {
        assertEquals(
            proto.length - pos, len, "length at level " + level + " should span remaining bytes");
      } else {
        // innermost: payload is exactly rowIds.length * 8 bytes
        assertEquals(rowIds.length * 8, len);
      }
    }

    // Verify the last 8 bytes are little-endian encoding of 1
    byte[] expected = {1, 0, 0, 0, 0, 0, 0, 0};
    byte[] actual = new byte[8];
    System.arraycopy(proto, proto.length - 8, actual, 0, 8);
    assertArrayEquals(expected, actual);
  }

  @Test
  void testEquals() {
    RowIdMeta a = new RowIdMeta("test");
    RowIdMeta b = new RowIdMeta("test");
    RowIdMeta c = new RowIdMeta("other");

    assertEquals(a, b);
    assertNotEquals(a, c);
    assertNotEquals(a, null);
    assertNotEquals(a, "test");
    assertEquals(a, a);
  }

  @Test
  void testHashCodeConsistency() {
    RowIdMeta a = new RowIdMeta("test");
    RowIdMeta b = new RowIdMeta("test");
    assertEquals(a.hashCode(), b.hashCode());
  }

  @Test
  void testToString() {
    RowIdMeta meta = new RowIdMeta("someMetadata");
    String str = meta.toString();
    assertTrue(str.contains("RowIdMeta"));
    assertTrue(str.contains("someMetadata"));
  }
}
