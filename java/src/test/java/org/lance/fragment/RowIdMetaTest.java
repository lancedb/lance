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
