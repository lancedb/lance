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

import java.util.List;
import java.util.Objects;

/**
 * A group of fragments to be rewritten. This class corresponds to the Rust RewriteGroup struct in
 * lance/rust/lance/src/dataset/transaction.rs.
 */
public class RewriteGroup {

  private final List<FragmentMetadata> oldFragments;
  private final List<FragmentMetadata> newFragments;

  private RewriteGroup(List<FragmentMetadata> oldFragments, List<FragmentMetadata> newFragments) {
    this.oldFragments = oldFragments;
    this.newFragments = newFragments;
  }

  public List<FragmentMetadata> oldFragments() {
    return oldFragments;
  }

  public List<FragmentMetadata> newFragments() {
    return newFragments;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    RewriteGroup that = (RewriteGroup) o;
    return Objects.equals(oldFragments, that.oldFragments)
        && Objects.equals(newFragments, that.newFragments);
  }

  @Override
  public int hashCode() {
    return Objects.hash(oldFragments, newFragments);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("oldFragments", oldFragments)
        .add("newFragments", newFragments)
        .toString();
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {
    private List<FragmentMetadata> oldFragments;
    private List<FragmentMetadata> newFragments;

    private Builder() {}

    public Builder oldFragments(List<FragmentMetadata> oldFragments) {
      this.oldFragments = oldFragments;
      return this;
    }

    public Builder newFragments(List<FragmentMetadata> newFragments) {
      this.newFragments = newFragments;
      return this;
    }

    public RewriteGroup build() {
      return new RewriteGroup(oldFragments, newFragments);
    }
  }
}
