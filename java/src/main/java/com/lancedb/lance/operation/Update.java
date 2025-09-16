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

import com.lancedb.lance.FragmentMetadata;

import com.google.common.base.MoreObjects;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

public class Update implements Operation {
  private final List<Long> removedFragmentIds;
  private final List<FragmentMetadata> updatedFragments;
  private final List<FragmentMetadata> newFragments;

  private Update(
      List<Long> removedFragmentIds,
      List<FragmentMetadata> updatedFragments,
      List<FragmentMetadata> newFragments) {
    this.removedFragmentIds = removedFragmentIds;
    this.updatedFragments = updatedFragments;
    this.newFragments = newFragments;
  }

  public static Builder builder() {
    return new Builder();
  }

  public List<Long> removedFragmentIds() {
    return removedFragmentIds;
  }

  public List<FragmentMetadata> updatedFragments() {
    return updatedFragments;
  }

  public List<FragmentMetadata> newFragments() {
    return newFragments;
  }

  @Override
  public String name() {
    return "Update";
  }

  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("removedFragmentIds", removedFragmentIds)
        .add("updatedFragments", updatedFragments)
        .add("newFragments", newFragments)
        .toString();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    Update that = (Update) o;
    return Objects.equals(removedFragmentIds, that.removedFragmentIds)
        && Objects.equals(updatedFragments, that.updatedFragments)
        && Objects.equals(newFragments, that.newFragments);
  }

  public static class Builder {
    private List<Long> removedFragmentIds = Collections.emptyList();
    private List<FragmentMetadata> updatedFragments = Collections.emptyList();
    private List<FragmentMetadata> newFragments = Collections.emptyList();

    private Builder() {}

    public Builder removedFragmentIds(List<Long> removedFragmentIds) {
      this.removedFragmentIds = removedFragmentIds;
      return this;
    }

    public Builder updatedFragments(List<FragmentMetadata> updatedFragments) {
      this.updatedFragments = updatedFragments;
      return this;
    }

    public Builder newFragments(List<FragmentMetadata> newFragments) {
      this.newFragments = newFragments;
      return this;
    }

    public Update build() {
      return new Update(removedFragmentIds, updatedFragments, newFragments);
    }
  }
}
