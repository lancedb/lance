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

import com.lancedb.lance.index.Index;

import com.google.common.base.MoreObjects;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Create Index Operation. This class corresponds to the Rust CreateIndex struct in
 * lance/rust/lance/src/dataset/transaction.rs.
 */
public class CreateIndex implements Operation {
  private final List<Index> newIndices;
  private final List<Index> removedIndices;

  private CreateIndex(List<Index> newIndices, List<Index> removedIndices) {
    this.newIndices = newIndices;
    this.removedIndices = removedIndices;
  }

  public List<Index> getNewIndices() {
    return newIndices;
  }

  public List<Index> getRemovedIndices() {
    return removedIndices;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    CreateIndex that = (CreateIndex) o;
    return Objects.equals(newIndices, that.newIndices)
        && Objects.equals(removedIndices, that.removedIndices);
  }

  @Override
  public int hashCode() {
    return Objects.hash(newIndices, removedIndices);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("newIndices", newIndices)
        .add("removedIndices", removedIndices)
        .toString();
  }

  @Override
  public String name() {
    return "CreateIndex";
  }

  public static Builder builder() {
    return new Builder();
  }

  /** Builder class. */
  public static class Builder {
    private List<Index> newIndices = Collections.emptyList();
    private List<Index> removedIndices = Collections.emptyList();

    private Builder() {}

    public Builder withNewIndices(List<Index> newIndices) {
      this.newIndices = newIndices;
      return this;
    }

    public Builder withRemovedIndices(List<Index> removedIndices) {
      this.removedIndices = removedIndices;
      return this;
    }

    public CreateIndex build() {
      return new CreateIndex(newIndices, removedIndices);
    }
  }
}
