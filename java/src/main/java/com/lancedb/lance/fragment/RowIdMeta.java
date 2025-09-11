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
package com.lancedb.lance.fragment;

import com.google.common.base.MoreObjects;

import java.io.Serializable;
import java.util.Objects;

public class RowIdMeta implements Serializable {
  private static final long serialVersionUID = -6532828695072614148L;

  private final String metadata;

  public RowIdMeta(String metadata) {
    this.metadata = metadata;
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
  public String toString() {
    return MoreObjects.toStringHelper(this).add("metadata", metadata).toString();
  }
}
