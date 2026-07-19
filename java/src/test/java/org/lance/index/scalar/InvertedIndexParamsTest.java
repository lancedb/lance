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

import org.lance.util.JsonUtils;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class InvertedIndexParamsTest {

  @Test
  void testIcuSplitTokenizerVariant() {
    ScalarIndexParams params = InvertedIndexParams.builder().baseTokenizer("icu/split").build();

    assertEquals("inverted", params.getIndexType());
    String jsonParams = params.getJsonParams().orElseThrow(AssertionError::new);
    assertTrue(jsonParams.contains("\"base_tokenizer\":\"icu/split\""));
  }

  @Test
  void defaultBlockSizeIsSerialized() {
    ScalarIndexParams params = InvertedIndexParams.builder().build();

    Map<String, Object> json = JsonUtils.fromJson(params.getJsonParams().orElseThrow());
    assertEquals(128, ((Number) json.get("block_size")).intValue());
  }

  @Test
  void blockSizeIsSerialized() {
    ScalarIndexParams params = InvertedIndexParams.builder().blockSize(128).build();

    assertEquals("inverted", params.getIndexType());
    Map<String, Object> json = JsonUtils.fromJson(params.getJsonParams().orElseThrow());
    assertEquals(128, ((Number) json.get("block_size")).intValue());
  }

  @Test
  void invalidBlockSizeIsRejected() {
    assertThrows(
        IllegalArgumentException.class, () -> InvertedIndexParams.builder().blockSize(129));
    assertThrows(
        IllegalArgumentException.class, () -> InvertedIndexParams.builder().blockSize(512));
  }

  @Test
  void formatVersionThreeRequiresBlockSize256() {
    ScalarIndexParams params =
        InvertedIndexParams.builder().blockSize(256).formatVersion(3).build();

    Map<String, Object> json = JsonUtils.fromJson(params.getJsonParams().orElseThrow());
    assertEquals(256, ((Number) json.get("block_size")).intValue());
    assertEquals(3, ((Number) json.get("format_version")).intValue());

    assertThrows(
        IllegalArgumentException.class,
        () -> InvertedIndexParams.builder().formatVersion(3).build());
    assertThrows(
        IllegalArgumentException.class,
        () -> InvertedIndexParams.builder().blockSize(256).formatVersion(2).build());
  }
}
