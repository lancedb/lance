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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class VersionMetaTest {

  @Test
  void testConstructorAndGetter() {
    VersionMeta meta = new VersionMeta("test");
    assertEquals("test", meta.getMetadata());
  }

  @Test
  void testNullMetadata() {
    VersionMeta meta = new VersionMeta(null);
    assertNull(meta.getMetadata());
  }

  @Test
  void testEquals() {
    VersionMeta a = new VersionMeta("x");
    VersionMeta b = new VersionMeta("x");
    assertEquals(a, b);
    assertEquals(b, a);

    VersionMeta c = new VersionMeta("y");
    assertNotEquals(a, c);

    assertFalse(a.equals(null));

    assertFalse(a.equals("x"));
  }

  @Test
  void testHashCode() {
    VersionMeta a = new VersionMeta("same");
    VersionMeta b = new VersionMeta("same");
    assertEquals(a.hashCode(), b.hashCode());
  }

  @Test
  void testHashCodeWithNull() {
    VersionMeta meta = new VersionMeta(null);
    meta.hashCode();
  }

  @Test
  void testToString() {
    VersionMeta meta = new VersionMeta("hello");
    String s = meta.toString();
    assertTrue(s.contains("VersionMeta"));
    assertTrue(s.contains("hello"));
  }

  @Test
  void testJsonMetadataPreservation() {
    String json = "{\"Inline\":[10,20,30]}";
    VersionMeta meta = new VersionMeta(json);
    assertEquals(json, meta.getMetadata());
  }
}
