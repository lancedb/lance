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

import java.nio.ByteBuffer;
import java.util.List;
import java.util.Optional;
import org.apache.arrow.util.Preconditions;

/**
 * Lance scan options.
 */
public class ScanOptions {
  private final Optional<Long> batchSize;
  private final Optional<List<String>> columns;
  private final Optional<String> filter;
  private final Optional<ByteBuffer> substraitFilter;

  /**
   * Constructor for LanceScanOptions.
   *
   * @param batchSize Maximum row number of each returned ArrowRecordBatch.
   *                  Optional, use Optional.empty() if unspecified.
   * @param columns   (Optional) Projected columns. Optional.empty() for scanning all columns.
   *                  Otherwise, only columns present in the List will be scanned.
   * @param filter    (Optional) Filter expression. Optional.empty() for no filter.
   * @param substraitFilter (Optional) Substrait filter expression.
   */
  public ScanOptions(Optional<Long> batchSize, Optional<List<String>> columns,
      Optional<String> filter, Optional<ByteBuffer> substraitFilter) {
    Preconditions.checkArgument(!(filter.isPresent() && substraitFilter.isPresent()),
        "cannot set both substrait filter and string filter");
    this.batchSize = batchSize;
    this.columns = columns;
    this.filter = filter;
    this.substraitFilter = substraitFilter;
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
   * Get the projected columns.
   *
   * @return Optional containing the list of projected columns if specified, otherwise empty.
   */
  public Optional<List<String>> getColumns() {
    return columns;
  }

  /**
   * Get the filter expression.
   *
   * @return Optional containing the filter expression if specified, otherwise empty.
   */
  public Optional<String> getFilter() {
    return filter;
  }

  /**
   * Get the substrait filter expression.
   *
   * @return Optional containing the substrait filter expression if specified, otherwise empty.
   */
  public Optional<ByteBuffer> getSubstraitFilter() {
    return substraitFilter;
  }

  /**
   * Builder for constructing LanceScanOptions.
   */
  public static class Builder {
    private Optional<Long> batchSize = Optional.empty();
    private Optional<List<String>> columns = Optional.empty();
    private Optional<String> filter = Optional.empty();
    private Optional<ByteBuffer> substraitFilter = Optional.empty();

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
     * Set the projected columns.
     *
     * @param columns List of projected columns.
     * @return Builder instance for method chaining.
     */
    public Builder columns(List<String> columns) {
      Preconditions.checkNotNull(columns);
      this.columns = Optional.of(columns);
      return this;
    }

    /**
     * Set the filter expression.
     *
     * @param filter Filter expression.
     * @return Builder instance for method chaining.
     */
    public Builder filter(String filter) {
      Preconditions.checkNotNull(filter);
      this.filter = Optional.of(filter);
      return this;
    }

    /**
     * Set the substrait filter expression.
     *
     * @param substraitFilter Filter expression.
     * @return Builder instance for method chaining.
     */
    public Builder substraitFilter(ByteBuffer substraitFilter) {
      Preconditions.checkNotNull(substraitFilter);
      this.substraitFilter = Optional.of(substraitFilter);
      return this;
    }

    /**
     * Build the LanceScanOptions instance.
     *
     * @return LanceScanOptions instance with the specified parameters.
     */
    public ScanOptions build() {
      return new ScanOptions(batchSize, columns, filter, substraitFilter);
    }
  }
}
