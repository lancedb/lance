package com.lancedb.lance.blob;

public class BlobColumnStatistics {
  private final long totalBlobs;
  private final long totalSize;
  private final long averageSize;
  private final long minSize;
  private final long maxSize;
  private final long nullCount;

  public BlobColumnStatistics(long totalBlobs, long totalSize, long averageSize,
      long minSize, long maxSize, long nullCount) {
    this.totalBlobs = totalBlobs;
    this.totalSize = totalSize;
    this.averageSize = averageSize;
    this.minSize = minSize;
    this.maxSize = maxSize;
    this.nullCount = nullCount;
  }

  /**
   * Get the total number of blobs in this column.
   *
   * @return the total number of blobs
   */
  public long getTotalBlobs() {
    return totalBlobs;
  }

  /**
   * Get the total size of all blobs in this column.
   *
   * @return the total size in bytes
   */
  public long getTotalSize() {
    return totalSize;
  }

  /**
   * Get the average size of blobs in this column.
   *
   * @return the average size in bytes
   */
  public long getAverageSize() {
    return averageSize;
  }

  /**
   * Get the minimum blob size in this column.
   *
   * @return the minimum size in bytes
   */
  public long getMinSize() {
    return minSize;
  }

  /**
   * Get the maximum blob size in this column.
   *
   * @return the maximum size in bytes
   */
  public long getMaxSize() {
    return maxSize;
  }

  /**
   * Get the number of null values in this column.
   *
   * @return the null count
   */
  public long getNullCount() {
    return nullCount;
  }

  @Override
  public String toString() {
    return String.format(
        "BlobColumnStatistics{totalBlobs=%d, totalSize=%d, averageSize=%d, " +
            "minSize=%d, maxSize=%d, nullCount=%d}",
        totalBlobs, totalSize, averageSize, minSize, maxSize, nullCount
    );
  }
}