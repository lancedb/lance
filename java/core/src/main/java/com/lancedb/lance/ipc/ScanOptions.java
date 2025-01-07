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
package com.lancedb.lance.ipc;

import org.apache.arrow.util.Preconditions;
import org.apache.commons.lang3.builder.ToStringBuilder;

import java.nio.ByteBuffer;
import java.util.List;
import java.util.Optional;

/** Lance scan options. */
public class ScanOptions {
  private final Optional<List<Integer>> fragmentIds;
  private final Optional<Long> batchSize;
  private final Optional<List<String>> columns;
  private final Optional<String> filter;
  private final Optional<ByteBuffer> substraitFilter;
  private final Optional<Long> limit;
  private final Optional<Long> offset;
  private final Optional<Query> nearest;
  private final boolean withRowId;
  private final boolean withRowAddress;
  private final int batchReadahead;
  private final Optional<List<ColumnOrdering>> columnOrderings;

  /**
   * Constructor for LanceScanOptions.
   *
   * @param fragmentIds the id of the fragments to scan
   * @param batchSize Maximum row number of each returned ArrowRecordBatch. Optional, use
   *     Optional.empty() if unspecified.
   * @param columns (Optional) Projected columns. Optional.empty() for scanning all columns.
   *     Otherwise, only columns present in the List will be scanned.
   * @param filter (Optional) Filter expression. Optional.empty() for no filter.
   * @param substraitFilter (Optional) Substrait filter expression.
   * @param limit (Optional) Maximum number of rows to return.
   * @param offset (Optional) Number of rows to skip before returning results.
   * @param withRowId Whether to include the row ID in the results.
   * @param withRowAddress Whether to include the row address in the results.
   * @param nearest (Optional) Nearest neighbor query.
   * @param batchReadahead Number of batches to read ahead.
   */
  public ScanOptions(
      Optional<List<Integer>> fragmentIds,
      Optional<Long> batchSize,
      Optional<List<String>> columns,
      Optional<String> filter,
      Optional<ByteBuffer> substraitFilter,
      Optional<Long> limit,
      Optional<Long> offset,
      Optional<Query> nearest,
      boolean withRowId,
      boolean withRowAddress,
      int batchReadahead,
      Optional<List<ColumnOrdering>> columnOrderings) {
    Preconditions.checkArgument(
        !(filter.isPresent() && substraitFilter.isPresent()),
        "cannot set both substrait filter and string filter");
    this.fragmentIds = fragmentIds;
    this.batchSize = batchSize;
    this.columns = columns;
    this.filter = filter;
    this.substraitFilter = substraitFilter;
    this.limit = limit;
    this.offset = offset;
    this.nearest = nearest;
    this.withRowId = withRowId;
    this.withRowAddress = withRowAddress;
    this.batchReadahead = batchReadahead;
    this.columnOrderings = columnOrderings;
  }

  /**
   * Get the fragment ids.
   *
   * @return Optional containing the fragment ids if specified, otherwise empty.
   */
  public Optional<List<Integer>> getFragmentIds() {
    return fragmentIds;
  }

  /**
   * Get the batch size.
   *
   * @return Optional containing the batch size if specified, otherwise empty.
   */
  public Optional<Long> getBatchSize() {
    return batchSize;
  }

  /**
   * Get the columns.
   *
   * @return Optional containing the columns if specified, otherwise empty.
   */
  public Optional<List<String>> getColumns() {
    return columns;
  }

  /**
   * Get the filter.
   *
   * @return Optional containing the filter if specified, otherwise empty.
   */
  public Optional<String> getFilter() {
    return filter;
  }

  /**
   * Get the substrait filter.
   *
   * @return Optional containing the substrait filter if specified, otherwise empty.
   */
  public Optional<ByteBuffer> getSubstraitFilter() {
    return substraitFilter;
  }

  /**
   * Get the limit.
   *
   * @return Optional containing the limit if specified, otherwise empty.
   */
  public Optional<Long> getLimit() {
    return limit;
  }

  /**
   * Get the offset.
   *
   * @return Optional containing the offset if specified, otherwise empty.
   */
  public Optional<Long> getOffset() {
    return offset;
  }

  /**
   * Get the nearest neighbor query.
   *
   * @return Optional containing the nearest neighbor query if specified, otherwise empty.
   */
  public Optional<Query> getNearest() {
    return nearest;
  }

  /**
   * Get whether to include the row ID.
   *
   * @return true if row ID should be included, false otherwise.
   */
  public boolean isWithRowId() {
    return withRowId;
  }

  /**
   * Get whether to include the row address.
   *
   * @return true if row address should be included, false otherwise.
   */
  public boolean isWithRowAddress() {
    return withRowAddress;
  }

  /**
   * Get the batch readahead.
   *
   * @return the number of batches to read ahead.
   */
  public int getBatchReadahead() {
    return batchReadahead;
  }

  public Optional<List<ColumnOrdering>> getColumnOrderings() {
    return columnOrderings;
  }

  @Override
  public String toString() {
    return new ToStringBuilder(this)
        .append("fragmentIds", fragmentIds.orElse(null))
        .append("batchSize", batchSize.orElse(null))
        .append("columns", columns.orElse(null))
        .append("filter", filter.orElse(null))
        .append(
            "substraitFilter",
            substraitFilter.map(buf -> "ByteBuffer[" + buf.remaining() + " bytes]").orElse(null))
        .append("limit", limit.orElse(null))
        .append("offset", offset.orElse(null))
        .append("nearest", nearest.orElse(null))
        .append("withRowId", withRowId)
        .append("WithRowAddress", withRowAddress)
        .append("batchReadahead", batchReadahead)
        .append("columnOrdering", columnOrderings)
        .toString();
  }

  /** Builder for constructing LanceScanOptions. */
  public static class Builder {
    private Optional<List<Integer>> fragmentIds = Optional.empty();
    private Optional<Long> batchSize = Optional.empty();
    private Optional<List<String>> columns = Optional.empty();
    private Optional<String> filter = Optional.empty();
    private Optional<ByteBuffer> substraitFilter = Optional.empty();
    private Optional<Long> limit = Optional.empty();
    private Optional<Long> offset = Optional.empty();
    private Optional<Query> nearest = Optional.empty();
    private boolean withRowId = false;
    private boolean withRowAddress = false;
    private int batchReadahead = 16;
    private Optional<List<ColumnOrdering>> columnOrderings = Optional.empty();

    public Builder() {}

    /**
     * Create a builder from another scan options.
     *
     * @param options another scan options
     */
    public Builder(ScanOptions options) {
      this.fragmentIds = options.getFragmentIds();
      this.batchSize = options.getBatchSize();
      this.columns = options.getColumns();
      this.filter = options.getFilter();
      this.substraitFilter = options.getSubstraitFilter();
      this.limit = options.getLimit();
      this.offset = options.getOffset();
      this.nearest = options.getNearest();
      this.withRowId = options.isWithRowId();
      this.withRowAddress = options.isWithRowAddress();
      this.batchReadahead = options.getBatchReadahead();
      this.columnOrderings = options.getColumnOrderings();
    }

    /**
     * Set the fragment ids.
     *
     * @param fragmentIds the id of the fragments to scan
     * @return Builder instance for method chaining.
     */
    public Builder fragmentIds(List<Integer> fragmentIds) {
      this.fragmentIds = Optional.of(fragmentIds);
      return this;
    }

    /**
     * Set the batch size.
     *
     * @param batchSize Maximum row number of each returned ArrowRecordBatch.
     * @return Builder instance for method chaining.
     */
    public Builder batchSize(long batchSize) {
      this.batchSize = Optional.of(batchSize);
      return this;
    }

    /**
     * Set the columns.
     *
     * @param columns Projected columns.
     * @return Builder instance for method chaining.
     */
    public Builder columns(List<String> columns) {
      this.columns = Optional.of(columns);
      return this;
    }

    /**
     * Set the filter.
     *
     * @param filter Filter expression.
     * @return Builder instance for method chaining.
     */
    public Builder filter(String filter) {
      this.filter = Optional.of(filter);
      return this;
    }

    /**
     * Set the substrait filter.
     *
     * @param substraitFilter Substrait filter expression.
     * @return Builder instance for method chaining.
     */
    public Builder substraitFilter(ByteBuffer substraitFilter) {
      this.substraitFilter = Optional.of(substraitFilter);
      return this;
    }

    /**
     * Set the limit.
     *
     * @param limit Maximum number of rows to return.
     * @return Builder instance for method chaining.
     */
    public Builder limit(long limit) {
      this.limit = Optional.of(limit);
      return this;
    }

    /**
     * Set the offset.
     *
     * @param offset Number of rows to skip before returning results.
     * @return Builder instance for method chaining.
     */
    public Builder offset(long offset) {
      this.offset = Optional.of(offset);
      return this;
    }

    /**
     * Set the nearest neighbor query.
     *
     * @param nearest The nearest neighbor query.
     * @return Builder instance for method chaining.
     */
    public Builder nearest(Query nearest) {
      this.nearest = Optional.of(nearest);
      return this;
    }

    /**
     * Set whether to include the row ID.
     *
     * @param withRowId true to include row ID, false otherwise.
     * @return Builder instance for method chaining.
     */
    public Builder withRowId(boolean withRowId) {
      this.withRowId = withRowId;
      return this;
    }

    /**
     * Set whether to include the row addr.
     *
     * @param withRowAddress true to include row ID, false otherwise.
     * @return Builder instance for method chaining.
     */
    public Builder withRowAddress(boolean withRowAddress) {
      this.withRowAddress = withRowAddress;
      return this;
    }

    /**
     * Set the batch readahead.
     *
     * @param batchReadahead Number of batches to read ahead.
     * @return Builder instance for method chaining.
     */
    public Builder batchReadahead(int batchReadahead) {
      this.batchReadahead = batchReadahead;
      return this;
    }

    public Builder setColumnOrderings(List<ColumnOrdering> columnOrderings) {
      this.columnOrderings = Optional.of(columnOrderings);
      return this;
    }

    /**
     * Build the LanceScanOptions instance.
     *
     * @return LanceScanOptions instance with the specified parameters.
     */
    public ScanOptions build() {
      return new ScanOptions(
          fragmentIds,
          batchSize,
          columns,
          filter,
          substraitFilter,
          limit,
          offset,
          nearest,
          withRowId,
          withRowAddress,
          batchReadahead,
          columnOrderings);
    }
  }
}
