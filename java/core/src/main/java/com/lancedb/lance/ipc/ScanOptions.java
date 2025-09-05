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

import com.lancedb.lance.util.ToStringHelper;

import org.apache.arrow.util.Preconditions;

import java.nio.ByteBuffer;
import java.util.List;
import java.util.Optional;

/** Lance scan options. */
public class ScanOptions {
  private final List<Integer> fragmentIds;
  private final Long batchSize;
  private final List<String> columns;
  private final String filter;
  private final ByteBuffer substraitFilter;
  private final Long limit;
  private final Long offset;
  private final Query nearest;
  private final boolean withRowId;
  private final boolean withRowAddress;
  private final int batchReadahead;
  private final List<ColumnOrdering> columnOrderings;

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
  private ScanOptions(
      List<Integer> fragmentIds,
      Long batchSize,
      List<String> columns,
      String filter,
      ByteBuffer substraitFilter,
      Long limit,
      Long offset,
      Query nearest,
      boolean withRowId,
      boolean withRowAddress,
      int batchReadahead,
      List<ColumnOrdering> columnOrderings) {
    Preconditions.checkArgument(
        !(filter != null && substraitFilter != null),
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
    return Optional.ofNullable(fragmentIds);
  }

  /**
   * Get the batch size.
   *
   * @return Optional containing the batch size if specified, otherwise empty.
   */
  public Optional<Long> getBatchSize() {
    return Optional.ofNullable(batchSize);
  }

  /**
   * Get the columns.
   *
   * @return Optional containing the columns if specified, otherwise empty.
   */
  public Optional<List<String>> getColumns() {
    return Optional.ofNullable(columns);
  }

  /**
   * Get the filter.
   *
   * @return Optional containing the filter if specified, otherwise empty.
   */
  public Optional<String> getFilter() {
    return Optional.ofNullable(filter);
  }

  /**
   * Get the substrait filter.
   *
   * @return Optional containing the substrait filter if specified, otherwise empty.
   */
  public Optional<ByteBuffer> getSubstraitFilter() {
    return Optional.ofNullable(substraitFilter);
  }

  /**
   * Get the limit.
   *
   * @return Optional containing the limit if specified, otherwise empty.
   */
  public Optional<Long> getLimit() {
    return Optional.ofNullable(limit);
  }

  /**
   * Get the offset.
   *
   * @return Optional containing the offset if specified, otherwise empty.
   */
  public Optional<Long> getOffset() {
    return Optional.ofNullable(offset);
  }

  /**
   * Get the nearest neighbor query.
   *
   * @return Optional containing the nearest neighbor query if specified, otherwise empty.
   */
  public Optional<Query> getNearest() {
    return Optional.ofNullable(nearest);
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
    return Optional.ofNullable(columnOrderings);
  }

  @Override
  public String toString() {
    return ToStringHelper.of(this)
        .add("fragmentIds", fragmentIds)
        .add("batchSize", batchSize)
        .add("columns", columns)
        .add("filter", filter)
        .add(
            "substraitFilter",
            getSubstraitFilter().map(buf -> "ByteBuffer[" + buf.remaining() + " bytes]"))
        .add("limit", limit)
        .add("offset", offset)
        .add("nearest", nearest)
        .add("withRowId", withRowId)
        .add("WithRowAddress", withRowAddress)
        .add("batchReadahead", batchReadahead)
        .add("columnOrdering", columnOrderings)
        .toString();
  }

  /** Builder for constructing LanceScanOptions. */
  public static class Builder {
    private List<Integer> fragmentIds;
    private Long batchSize;
    private List<String> columns;
    private String filter;
    private ByteBuffer substraitFilter;
    private Long limit;
    private Long offset;
    private Query nearest;
    private boolean withRowId = false;
    private boolean withRowAddress = false;
    private int batchReadahead = 16;
    private List<ColumnOrdering> columnOrderings;

    public Builder() {}

    /**
     * Create a builder from another scan options.
     *
     * @param options another scan options
     */
    public Builder(ScanOptions options) {
      this.fragmentIds = options.fragmentIds;
      this.batchSize = options.batchSize;
      this.columns = options.columns;
      this.filter = options.filter;
      this.substraitFilter = options.substraitFilter;
      this.limit = options.limit;
      this.offset = options.offset;
      this.nearest = options.nearest;
      this.withRowId = options.withRowId;
      this.withRowAddress = options.withRowAddress;
      this.batchReadahead = options.batchReadahead;
      this.columnOrderings = options.columnOrderings;
    }

    /**
     * Set the fragment ids.
     *
     * @param fragmentIds the id of the fragments to scan
     * @return Builder instance for method chaining.
     */
    public Builder fragmentIds(List<Integer> fragmentIds) {
      this.fragmentIds = fragmentIds;
      return this;
    }

    /**
     * Set the batch size.
     *
     * @param batchSize Maximum row number of each returned ArrowRecordBatch.
     * @return Builder instance for method chaining.
     */
    public Builder batchSize(long batchSize) {
      this.batchSize = batchSize;
      return this;
    }

    /**
     * Set the columns.
     *
     * @param columns Projected columns.
     * @return Builder instance for method chaining.
     */
    public Builder columns(List<String> columns) {
      this.columns = columns;
      return this;
    }

    /**
     * Set the filter.
     *
     * @param filter Filter expression.
     * @return Builder instance for method chaining.
     */
    public Builder filter(String filter) {
      this.filter = filter;
      return this;
    }

    /**
     * Set the substrait filter.
     *
     * @param substraitFilter Substrait filter expression.
     * @return Builder instance for method chaining.
     */
    public Builder substraitFilter(ByteBuffer substraitFilter) {
      this.substraitFilter = substraitFilter;
      return this;
    }

    /**
     * Set the limit.
     *
     * @param limit Maximum number of rows to return.
     * @return Builder instance for method chaining.
     */
    public Builder limit(long limit) {
      this.limit = limit;
      return this;
    }

    /**
     * Set the offset.
     *
     * @param offset Number of rows to skip before returning results.
     * @return Builder instance for method chaining.
     */
    public Builder offset(long offset) {
      this.offset = offset;
      return this;
    }

    /**
     * Set the nearest neighbor query.
     *
     * @param nearest The nearest neighbor query.
     * @return Builder instance for method chaining.
     */
    public Builder nearest(Query nearest) {
      this.nearest = nearest;
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
      this.columnOrderings = columnOrderings;
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
