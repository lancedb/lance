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
package org.lance.index;

import java.util.Optional;

/**
 * Criteria for describing or selecting indices on a dataset.
 *
 * <p>This mirrors the semantics of the Rust {@code IndexCriteria} struct used by {@code
 * Dataset::describe_indices} and related APIs.
 */
public final class IndexCriteria {

  private final Optional<String> forColumn;
  private final Optional<String> hasName;
  private final boolean mustSupportFts;
  private final boolean mustSupportExactEquality;

  private IndexCriteria(Builder builder) {
    this.forColumn = Optional.ofNullable(builder.forColumn);
    this.hasName = Optional.ofNullable(builder.hasName);
    this.mustSupportFts = builder.mustSupportFts;
    this.mustSupportExactEquality = builder.mustSupportExactEquality;
  }

  /**
   * Optional column name to restrict indices to.
   *
   * <p>If present, only indices built on this column (and only this column) will be considered.
   */
  public Optional<String> getForColumn() {
    return forColumn;
  }

  /** Optional index name to restrict indices to. */
  public Optional<String> getHasName() {
    return hasName;
  }

  /** If true, only indices that support full-text search will be considered. */
  public boolean mustSupportFts() {
    return mustSupportFts;
  }

  /** If true, only indices that support exact equality predicates will be considered. */
  public boolean mustSupportExactEquality() {
    return mustSupportExactEquality;
  }

  /** Builder for {@link IndexCriteria}. */
  public static final class Builder {

    private String forColumn;
    private String hasName;
    private boolean mustSupportFts;
    private boolean mustSupportExactEquality;

    /** Restrict indices to those built on the given column. */
    public Builder forColumn(String forColumn) {
      this.forColumn = forColumn;
      return this;
    }

    /** Restrict indices to those with the given name. */
    public Builder hasName(String name) {
      this.hasName = name;
      return this;
    }

    /** Require indices to support full-text search. */
    public Builder mustSupportFts(boolean mustSupportFts) {
      this.mustSupportFts = mustSupportFts;
      return this;
    }

    /** Require indices to support exact equality predicates. */
    public Builder mustSupportExactEquality(boolean mustSupportExactEquality) {
      this.mustSupportExactEquality = mustSupportExactEquality;
      return this;
    }

    public IndexCriteria build() {
      return new IndexCriteria(this);
    }
  }
}
