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
package org.lance.util;

import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Utility helpers for constructing JSON fields using Arrow extension metadata.
 *
 * <p>This class aligns with the Arrow JSON extension type (extension name {@code "arrow.json"}) for
 * Utf8 and LargeUtf8 fields that logically carry JSON text.
 *
 * <p>When writing data, fields annotated with {@code arrow.json} are converted by Lance into its
 * internal JSONB representation (physically stored as {@code LargeBinary} with extension name
 * {@code "lance.json"}). When reading, Lance converts {@code lance.json} back into {@code
 * arrow.json} (Utf8), so callers always work with JSON text rather than binary JSON.
 *
 * <p>The {@code lance.json} storage type is intentionally not exposed via helpers in this class to
 * keep the internal JSONB format an implementation detail.
 *
 * <p>See also the Arrow extension type documentation:
 * https://arrow.apache.org/docs/format/Extensions.html
 */
public final class JsonFields {

  /**
   * Field metadata key used by Arrow to store the extension type name ({@code
   * ARROW:extension:name}).
   */
  private static final String EXTENSION_NAME_KEY = "ARROW:extension:name";

  /**
   * Arrow JSON extension type name ({@code arrow.json}) used to mark Utf8/LargeUtf8 fields as
   * carrying JSON text, whose semantics are interpreted and converted by Lance.
   */
  private static final String ARROW_JSON_EXTENSION_NAME = "arrow.json";

  private JsonFields() {}

  /**
   * Create a Utf8 field annotated as an Arrow JSON extension field.
   *
   * <p>The resulting field uses the {@code arrow.json} extension and relies on Lance to convert
   * between JSON text and its internal JSONB representation on write and read.
   *
   * @param name the field name
   * @param nullable whether the field is nullable
   * @return a Field with Utf8 storage type and arrow.json extension metadata
   */
  public static Field jsonUtf8(String name, boolean nullable) {
    return new Field(name, jsonFieldType(new ArrowType.Utf8(), nullable), Collections.emptyList());
  }

  /**
   * Create a LargeUtf8 field annotated as an Arrow JSON extension field.
   *
   * <p>The resulting field uses the {@code arrow.json} extension and relies on Lance to convert
   * between JSON text and its internal JSONB representation on write and read.
   *
   * @param name the field name
   * @param nullable whether the field is nullable
   * @return a Field with LargeUtf8 storage type and arrow.json extension metadata
   */
  public static Field jsonLargeUtf8(String name, boolean nullable) {
    return new Field(
        name, jsonFieldType(new ArrowType.LargeUtf8(), nullable), Collections.emptyList());
  }

  private static FieldType jsonFieldType(ArrowType storageType, boolean nullable) {
    return new FieldType(nullable, storageType, null, jsonExtensionMetadata());
  }

  private static Map<String, String> jsonExtensionMetadata() {
    Map<String, String> metadata = new HashMap<>();
    metadata.put(EXTENSION_NAME_KEY, ARROW_JSON_EXTENSION_NAME);
    return Collections.unmodifiableMap(metadata);
  }
}
