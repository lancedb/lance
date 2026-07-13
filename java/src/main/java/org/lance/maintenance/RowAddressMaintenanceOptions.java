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
package org.lance.maintenance;

import java.util.Optional;

/** Shared rewrite options for storage-version-2.3 row-address maintenance. */
public final class RowAddressMaintenanceOptions {
  private final Optional<Long> targetRowsPerFragment;
  private final Optional<Long> maxRowsPerGroup;
  private final Optional<Long> maxBytesPerFile;
  private final Optional<Long> batchSize;
  private final Optional<Long> ioBufferSize;

  private RowAddressMaintenanceOptions(Builder builder) {
    this.targetRowsPerFragment = builder.targetRowsPerFragment;
    this.maxRowsPerGroup = builder.maxRowsPerGroup;
    this.maxBytesPerFile = builder.maxBytesPerFile;
    this.batchSize = builder.batchSize;
    this.ioBufferSize = builder.ioBufferSize;
  }

  /** Maximum rows per output fragment, or empty to use the Rust default of 1,000,000. */
  public Optional<Long> getTargetRowsPerFragment() {
    return targetRowsPerFragment;
  }

  /** Maximum rows per writer batch group, or empty to use the Rust default of 1,024. */
  public Optional<Long> getMaxRowsPerGroup() {
    return maxRowsPerGroup;
  }

  /**
   * Maximum physical bytes per output file.
   *
   * <p>Placement normalization rejects this option because exact admission requires deterministic
   * row-count boundaries.
   */
  public Optional<Long> getMaxBytesPerFile() {
    return maxBytesPerFile;
  }

  /** Input scan batch size. */
  public Optional<Long> getBatchSize() {
    return batchSize;
  }

  /** Input scan I/O buffer size in bytes. */
  public Optional<Long> getIoBufferSize() {
    return ioBufferSize;
  }

  /** Create an empty options builder that delegates defaults to Rust. */
  public static Builder builder() {
    return new Builder();
  }

  /** Builder for explicit row-address maintenance options. */
  public static final class Builder {
    private Optional<Long> targetRowsPerFragment = Optional.empty();
    private Optional<Long> maxRowsPerGroup = Optional.empty();
    private Optional<Long> maxBytesPerFile = Optional.empty();
    private Optional<Long> batchSize = Optional.empty();
    private Optional<Long> ioBufferSize = Optional.empty();

    private Builder() {}

    /** Set the maximum rows per output fragment. */
    public Builder withTargetRowsPerFragment(long value) {
      this.targetRowsPerFragment = Optional.of(value);
      return this;
    }

    /** Set the maximum rows per writer batch group. */
    public Builder withMaxRowsPerGroup(long value) {
      this.maxRowsPerGroup = Optional.of(value);
      return this;
    }

    /** Set the maximum physical bytes per output file. */
    public Builder withMaxBytesPerFile(long value) {
      this.maxBytesPerFile = Optional.of(value);
      return this;
    }

    /** Set the input scan batch size. */
    public Builder withBatchSize(long value) {
      this.batchSize = Optional.of(value);
      return this;
    }

    /** Set the input scan I/O buffer size in bytes. */
    public Builder withIoBufferSize(long value) {
      this.ioBufferSize = Optional.of(value);
      return this;
    }

    /** Build immutable maintenance options. */
    public RowAddressMaintenanceOptions build() {
      return new RowAddressMaintenanceOptions(this);
    }
  }
}
