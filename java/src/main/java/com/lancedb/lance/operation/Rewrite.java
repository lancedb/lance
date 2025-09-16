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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Rewrite operation for reorganizing data without changing content. This operation rewrites the
 * data but does not modify the content, e.g. for compaction or reordering. This modifies the
 * addresses of existing rows, so indices that cover rewritten fragments need to be remapped.
 */
public class Rewrite implements Operation {
  private final List<RewriteGroup> groups;
  private final List<RewrittenIndex> rewrittenIndices;
  private final Optional<Index> fragReuseIndex;

  private Rewrite(
      List<RewriteGroup> groups, List<RewrittenIndex> rewrittenIndices, Index fragReuseIndex) {
    this.groups = groups != null ? groups : new ArrayList<>();
    this.rewrittenIndices = rewrittenIndices != null ? rewrittenIndices : new ArrayList<>();
    this.fragReuseIndex = Optional.ofNullable(fragReuseIndex);
  }

  @Override
  public String name() {
    return "Rewrite";
  }

  /**
   * Get the rewrite groups.
   *
   * @return the rewrite groups
   */
  public List<RewriteGroup> groups() {
    return groups;
  }

  /**
   * Get the rewritten indices.
   *
   * @return the rewritten indices
   */
  public List<RewrittenIndex> rewrittenIndices() {
    return rewrittenIndices;
  }

  /**
   * Get the fragment reuse index.
   *
   * @return the fragment reuse index
   */
  public Optional<Index> fragReuseIndex() {
    return fragReuseIndex;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    Rewrite rewrite = (Rewrite) o;
    return Objects.equals(groups, rewrite.groups)
        && Objects.equals(rewrittenIndices, rewrite.rewrittenIndices)
        && Objects.equals(fragReuseIndex, rewrite.fragReuseIndex);
  }

  @Override
  public int hashCode() {
    return Objects.hash(groups, rewrittenIndices, fragReuseIndex);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("groups", groups)
        .add("rewrittenIndices", rewrittenIndices)
        .add("fragReuseIndex", fragReuseIndex)
        .toString();
  }

  /**
   * Create a new builder for Rewrite.
   *
   * @return a new builder
   */
  public static Builder builder() {
    return new Builder();
  }

  /** Builder for Rewrite. */
  public static class Builder {
    private List<RewriteGroup> groups;
    private List<RewrittenIndex> rewrittenIndices;
    private Index fragReuseIndex;

    private Builder() {}

    /**
     * Set the rewrite groups.
     *
     * @param groups the rewrite groups
     * @return this builder
     */
    public Builder groups(List<RewriteGroup> groups) {
      this.groups = groups;
      return this;
    }

    /**
     * Set the rewritten indices.
     *
     * @param rewrittenIndices the rewritten indices
     * @return this builder
     */
    public Builder rewrittenIndices(List<RewrittenIndex> rewrittenIndices) {
      this.rewrittenIndices = rewrittenIndices;
      return this;
    }

    /**
     * Set the fragment reuse index.
     *
     * @param fragReuseIndex the fragment reuse index
     * @return this builder
     */
    public Builder fragReuseIndex(Index fragReuseIndex) {
      this.fragReuseIndex = fragReuseIndex;
      return this;
    }

    /**
     * Build a new Rewrite operation.
     *
     * @return a new Rewrite operation
     */
    public Rewrite build() {
      return new Rewrite(groups, rewrittenIndices, fragReuseIndex);
    }
  }
}
