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
package org.lance.memwal;

import com.google.common.base.MoreObjects;

import java.util.Optional;

/** Statistics of the active MemTable held by a {@link ShardWriter}. */
public class MemTableStats {
  private final long rowCount;
  private final long batchCount;
  private final long estimatedSizeBytes;
  private final long generation;
  private final Optional<Long> maxBufferedBatchPosition;
  private final Optional<Long> maxFlushedBatchPosition;
  private final Optional<Long> pendingWalStartBatchPosition;
  private final Optional<Long> pendingWalEndBatchPosition;
  private final long pendingWalBatchCount;
  private final long pendingWalRowCount;
  private final long pendingWalEstimatedBytes;

  public MemTableStats(
      long rowCount,
      long batchCount,
      long estimatedSizeBytes,
      long generation,
      Long maxBufferedBatchPosition,
      Long maxFlushedBatchPosition,
      Long pendingWalStartBatchPosition,
      Long pendingWalEndBatchPosition,
      long pendingWalBatchCount,
      long pendingWalRowCount,
      long pendingWalEstimatedBytes) {
    this.rowCount = rowCount;
    this.batchCount = batchCount;
    this.estimatedSizeBytes = estimatedSizeBytes;
    this.generation = generation;
    this.maxBufferedBatchPosition = Optional.ofNullable(maxBufferedBatchPosition);
    this.maxFlushedBatchPosition = Optional.ofNullable(maxFlushedBatchPosition);
    this.pendingWalStartBatchPosition = Optional.ofNullable(pendingWalStartBatchPosition);
    this.pendingWalEndBatchPosition = Optional.ofNullable(pendingWalEndBatchPosition);
    this.pendingWalBatchCount = pendingWalBatchCount;
    this.pendingWalRowCount = pendingWalRowCount;
    this.pendingWalEstimatedBytes = pendingWalEstimatedBytes;
  }

  /** Number of rows currently buffered in the active MemTable. */
  public long rowCount() {
    return rowCount;
  }

  /** Number of record batches currently buffered in the active MemTable. */
  public long batchCount() {
    return batchCount;
  }

  /** Estimated in-memory size of the active MemTable, in bytes. */
  public long estimatedSizeBytes() {
    return estimatedSizeBytes;
  }

  /** Generation number of the active MemTable. */
  public long generation() {
    return generation;
  }

  /** Highest WAL batch position buffered into the MemTable, if any. */
  public Optional<Long> maxBufferedBatchPosition() {
    return maxBufferedBatchPosition;
  }

  /** Highest WAL batch position flushed from the MemTable, if any. */
  public Optional<Long> maxFlushedBatchPosition() {
    return maxFlushedBatchPosition;
  }

  /** First WAL batch position pending flush, if any. */
  public Optional<Long> pendingWalStartBatchPosition() {
    return pendingWalStartBatchPosition;
  }

  /** Last WAL batch position pending flush, if any. */
  public Optional<Long> pendingWalEndBatchPosition() {
    return pendingWalEndBatchPosition;
  }

  /** Number of WAL batches pending flush. */
  public long pendingWalBatchCount() {
    return pendingWalBatchCount;
  }

  /** Number of rows in WAL batches pending flush. */
  public long pendingWalRowCount() {
    return pendingWalRowCount;
  }

  /** Estimated bytes of WAL batches pending flush. */
  public long pendingWalEstimatedBytes() {
    return pendingWalEstimatedBytes;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("rowCount", rowCount)
        .add("batchCount", batchCount)
        .add("estimatedSizeBytes", estimatedSizeBytes)
        .add("generation", generation)
        .add("maxBufferedBatchPosition", maxBufferedBatchPosition.orElse(null))
        .add("maxFlushedBatchPosition", maxFlushedBatchPosition.orElse(null))
        .add("pendingWalStartBatchPosition", pendingWalStartBatchPosition.orElse(null))
        .add("pendingWalEndBatchPosition", pendingWalEndBatchPosition.orElse(null))
        .add("pendingWalBatchCount", pendingWalBatchCount)
        .add("pendingWalRowCount", pendingWalRowCount)
        .add("pendingWalEstimatedBytes", pendingWalEstimatedBytes)
        .toString();
  }
}
