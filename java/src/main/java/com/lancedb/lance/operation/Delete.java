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

public class Delete implements Operation {
  private final List<FragmentMetadata> updatedFragments;
  private final List<Long> deletedFragmentIds;
  private final String predicate;

  private Delete(
      List<FragmentMetadata> updatedFragments, List<Long> deletedFragmentIds, String predicate) {
    this.updatedFragments = updatedFragments;
    this.deletedFragmentIds = deletedFragmentIds;
    this.predicate = predicate;
  }

  public static Builder builder() {
    return new Builder();
  }

  public List<FragmentMetadata> updatedFragments() {
    return updatedFragments;
  }

  public List<Long> deletedFragmentIds() {
    return deletedFragmentIds;
  }

  public String predicate() {
    return predicate;
  }

  @Override
  public String name() {
    return "Delete";
  }

  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("updatedFragments", updatedFragments)
        .add("deletedFragmentIds", deletedFragmentIds)
        .add("predicate", predicate)
        .toString();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    Delete that = (Delete) o;
    return Objects.equals(updatedFragments, that.updatedFragments)
        && Objects.equals(deletedFragmentIds, that.deletedFragmentIds)
        && Objects.equals(predicate, that.predicate);
  }

  public static class Builder {
    private List<FragmentMetadata> updatedFragments = Collections.emptyList();
    private List<Long> deletedFragmentIds = Collections.emptyList();
    private String predicate = "";

    private Builder() {}

    public Builder updatedFragments(List<FragmentMetadata> updatedFragments) {
      this.updatedFragments = updatedFragments;
      return this;
    }

    public Builder deletedFragmentIds(List<Long> deletedFragmentIds) {
      this.deletedFragmentIds = deletedFragmentIds;
      return this;
    }

    public Builder predicate(String predicate) {
      this.predicate = predicate;
      return this;
    }

    public Delete build() {
      return new Delete(updatedFragments, deletedFragmentIds, predicate);
    }
  }
}
