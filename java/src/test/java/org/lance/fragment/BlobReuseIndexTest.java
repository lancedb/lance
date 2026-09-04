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

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class BlobReuseIndexTest {
  @Test
  void dataFileCarriesBlobReuseIndex() {
    BlobReuseSource source =
        new BlobReuseSource(7, "source.blob", new long[] {1, 9}, new long[] {2, 10});
    BlobReuseIndex index = new BlobReuseIndex(List.of(source));
    DataFile file = new DataFile("file.lance", new int[] {0}, new int[] {0}, 2, 2, 100L, 7, index);

    assertEquals(index, file.getBlobReuseIndex().orElseThrow());
    assertEquals(7, source.getBaseId().orElseThrow());
    assertArrayEquals(new long[] {1, 9}, source.getLocalIds());
    assertArrayEquals(new long[] {2, 10}, source.getPhysicalIds());
  }

  @Test
  void sourceArraysAreDefensivelyCopied() {
    long[] localIds = {1};
    long[] physicalIds = {2};
    BlobReuseSource source = new BlobReuseSource(null, "source.blob", localIds, physicalIds);

    localIds[0] = 9;
    physicalIds[0] = 10;
    source.getLocalIds()[0] = 11;
    source.getPhysicalIds()[0] = 12;

    assertArrayEquals(new long[] {1}, source.getLocalIds());
    assertArrayEquals(new long[] {2}, source.getPhysicalIds());
  }
}
