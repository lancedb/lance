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

import java.util.Optional;

/**
 * Optional configuration for a {@link ShardWriter}.
 *
 * <p>Every value is optional; unset values fall back to the Lance default writer configuration. Use
 * the {@code with*} methods to build a configuration, then pass it to {@link
 * org.lance.Dataset#memWalWriter} or {@link InitializeMemWalParams#withWriterConfig}.
 */
public class ShardWriterConfig {
  private Optional<Boolean> durableWrite = Optional.empty();
  private Optional<Boolean> syncIndexedWrite = Optional.empty();
  private Optional<Long> maxWalBufferSize = Optional.empty();
  private Optional<Long> maxWalFlushIntervalMs = Optional.empty();
  private Optional<Long> maxMemtableSize = Optional.empty();
  private Optional<Long> maxMemtableRows = Optional.empty();
  private Optional<Long> maxMemtableBatches = Optional.empty();
  private Optional<Long> maxUnflushedMemtableBytes = Optional.empty();
  private Optional<Long> manifestScanBatchSize = Optional.empty();
  private Optional<Long> asyncIndexBufferRows = Optional.empty();
  private Optional<Long> asyncIndexIntervalMs = Optional.empty();
  private Optional<Long> backpressureLogIntervalMs = Optional.empty();
  private Optional<Long> statsLogIntervalMs = Optional.empty();

  /** Whether each {@code put} is durably persisted to the WAL before returning. */
  public ShardWriterConfig withDurableWrite(boolean durableWrite) {
    this.durableWrite = Optional.of(durableWrite);
    return this;
  }

  /** Whether indexed writes are applied synchronously. */
  public ShardWriterConfig withSyncIndexedWrite(boolean syncIndexedWrite) {
    this.syncIndexedWrite = Optional.of(syncIndexedWrite);
    return this;
  }

  /** Maximum size of the in-memory WAL buffer, in bytes. */
  public ShardWriterConfig withMaxWalBufferSize(long maxWalBufferSize) {
    this.maxWalBufferSize = Optional.of(maxWalBufferSize);
    return this;
  }

  /** Maximum interval between WAL flushes, in milliseconds. */
  public ShardWriterConfig withMaxWalFlushIntervalMs(long maxWalFlushIntervalMs) {
    this.maxWalFlushIntervalMs = Optional.of(maxWalFlushIntervalMs);
    return this;
  }

  /** Maximum size of a MemTable before it is flushed, in bytes. */
  public ShardWriterConfig withMaxMemtableSize(long maxMemtableSize) {
    this.maxMemtableSize = Optional.of(maxMemtableSize);
    return this;
  }

  /** Maximum number of rows in a MemTable before it is flushed. */
  public ShardWriterConfig withMaxMemtableRows(long maxMemtableRows) {
    this.maxMemtableRows = Optional.of(maxMemtableRows);
    return this;
  }

  /** Maximum number of record batches in a MemTable before it is flushed. */
  public ShardWriterConfig withMaxMemtableBatches(long maxMemtableBatches) {
    this.maxMemtableBatches = Optional.of(maxMemtableBatches);
    return this;
  }

  /** Maximum unflushed MemTable bytes allowed before applying backpressure. */
  public ShardWriterConfig withMaxUnflushedMemtableBytes(long maxUnflushedMemtableBytes) {
    this.maxUnflushedMemtableBytes = Optional.of(maxUnflushedMemtableBytes);
    return this;
  }

  /** Batch size used when scanning the WAL manifest. */
  public ShardWriterConfig withManifestScanBatchSize(long manifestScanBatchSize) {
    this.manifestScanBatchSize = Optional.of(manifestScanBatchSize);
    return this;
  }

  /** Number of rows buffered before an asynchronous index update is triggered. */
  public ShardWriterConfig withAsyncIndexBufferRows(long asyncIndexBufferRows) {
    this.asyncIndexBufferRows = Optional.of(asyncIndexBufferRows);
    return this;
  }

  /** Interval between asynchronous index updates, in milliseconds. */
  public ShardWriterConfig withAsyncIndexIntervalMs(long asyncIndexIntervalMs) {
    this.asyncIndexIntervalMs = Optional.of(asyncIndexIntervalMs);
    return this;
  }

  /** Interval between backpressure log messages, in milliseconds. */
  public ShardWriterConfig withBackpressureLogIntervalMs(long backpressureLogIntervalMs) {
    this.backpressureLogIntervalMs = Optional.of(backpressureLogIntervalMs);
    return this;
  }

  /** Interval between statistics log messages, in milliseconds; {@code 0} disables logging. */
  public ShardWriterConfig withStatsLogIntervalMs(long statsLogIntervalMs) {
    this.statsLogIntervalMs = Optional.of(statsLogIntervalMs);
    return this;
  }

  public Optional<Boolean> durableWrite() {
    return durableWrite;
  }

  public Optional<Boolean> syncIndexedWrite() {
    return syncIndexedWrite;
  }

  public Optional<Long> maxWalBufferSize() {
    return maxWalBufferSize;
  }

  public Optional<Long> maxWalFlushIntervalMs() {
    return maxWalFlushIntervalMs;
  }

  public Optional<Long> maxMemtableSize() {
    return maxMemtableSize;
  }

  public Optional<Long> maxMemtableRows() {
    return maxMemtableRows;
  }

  public Optional<Long> maxMemtableBatches() {
    return maxMemtableBatches;
  }

  public Optional<Long> maxUnflushedMemtableBytes() {
    return maxUnflushedMemtableBytes;
  }

  public Optional<Long> manifestScanBatchSize() {
    return manifestScanBatchSize;
  }

  public Optional<Long> asyncIndexBufferRows() {
    return asyncIndexBufferRows;
  }

  public Optional<Long> asyncIndexIntervalMs() {
    return asyncIndexIntervalMs;
  }

  public Optional<Long> backpressureLogIntervalMs() {
    return backpressureLogIntervalMs;
  }

  public Optional<Long> statsLogIntervalMs() {
    return statsLogIntervalMs;
  }
}
