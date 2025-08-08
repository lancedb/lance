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
package com.lancedb.lance.operation;

import org.apache.commons.lang3.builder.ToStringBuilder;

import java.util.Objects;
import java.util.UUID;

/**
 * A rewritten index mapping from old index UUID to new index UUID. This class corresponds to the
 * Rust RewrittenIndex struct in lance/rust/lance/src/dataset/transaction.rs.
 */
public class RewrittenIndex {
  private final UUID oldId;
  private final UUID newId;

  private RewrittenIndex(UUID oldId, UUID newId) {
    this.oldId = oldId;
    this.newId = newId;
  }

  public UUID getOldId() {
    return oldId;
  }

  public UUID getNewId() {
    return newId;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    RewrittenIndex that = (RewrittenIndex) o;
    return Objects.equals(oldId, that.oldId) && Objects.equals(newId, that.newId);
  }

  @Override
  public int hashCode() {
    return Objects.hash(oldId, newId);
  }

  @Override
  public String toString() {
    return new ToStringBuilder(this).append("oldId", oldId).append("newId", newId).toString();
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {
    private UUID oldId;
    private UUID newId;

    private Builder() {}

    public Builder oldId(UUID oldId) {
      this.oldId = oldId;
      return this;
    }

    public Builder newId(UUID newId) {
      this.newId = newId;
      return this;
    }

    public RewrittenIndex build() {
      return new RewrittenIndex(oldId, newId);
    }
  }
}
