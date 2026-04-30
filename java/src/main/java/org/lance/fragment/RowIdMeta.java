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

import org.lance.JniLoader;

import com.google.common.base.MoreObjects;

import java.io.Serializable;
import java.util.Objects;

public class RowIdMeta implements Serializable {
  private static final long serialVersionUID = -6532828695072614148L;

  static {
    JniLoader.ensureLoaded();
  }

  private final String metadata;

  public RowIdMeta(String metadata) {
    this.metadata = metadata;
  }

  /**
   * Creates a RowIdMeta from an array of stable row IDs by delegating to the Rust {@code
   * write_row_ids} encoder via JNI. The returned metadata is a JSON string wrapping the
   * protobuf-encoded RowIdSequence, matching the format expected by lance-core.
   *
   * @param rowIds stable row IDs to encode
   * @return RowIdMeta containing the serialized inline representation
   */
  public static RowIdMeta fromRowIds(long[] rowIds) {
    return new RowIdMeta(nativeEncodeRowIds(rowIds));
  }

  private static native String nativeEncodeRowIds(long[] rowIds);

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
