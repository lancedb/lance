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
package org.lance.compaction;

import org.lance.FragmentMetadata;

import javax.annotation.Nullable;

import java.io.Serializable;
import java.util.List;

/**
 * Rewrite Result of a single compaction task. It will be passed across different workers and be
 * committed later.
 */
public class RewriteResult implements Serializable {
  private final CompactionMetrics metrics;
  private final List<FragmentMetadata> newFragments;
  private final List<FragmentMetadata> originalFragments;
  private final long readVersion;

  // Serialized RoaringTreemap of row addresses read from the original fragments.
  // null for stable row IDs.
  @Nullable private final byte[] rowAddrs;
  @Nullable private final byte[] logicalRowIds;
  @Nullable private final byte[] retiredLogicalRowIds;

  public RewriteResult(
      CompactionMetrics metrics,
      List<FragmentMetadata> newFragments,
      List<FragmentMetadata> originalFragments,
      long readVersion,
      byte[] rowAddrs) {
    this(metrics, newFragments, originalFragments, readVersion, rowAddrs, null, null);
  }

  public RewriteResult(
      CompactionMetrics metrics,
      List<FragmentMetadata> newFragments,
      List<FragmentMetadata> originalFragments,
      long readVersion,
      @Nullable byte[] rowAddrs,
      @Nullable byte[] logicalRowIds,
      @Nullable byte[] retiredLogicalRowIds) {
    this.metrics = metrics;
    this.newFragments = newFragments;
    this.originalFragments = originalFragments;
    this.readVersion = readVersion;
    this.rowAddrs = rowAddrs;
    this.logicalRowIds = logicalRowIds;
    this.retiredLogicalRowIds = retiredLogicalRowIds;
  }

  public long getReadVersion() {
    return readVersion;
  }

  public CompactionMetrics getMetrics() {
    return metrics;
  }

  @Nullable
  public byte[] getRowAddrs() {
    return rowAddrs;
  }

  /** Actual logical row-address sequence emitted by a storage-version-2.3 rewrite. */
  @Nullable
  public byte[] getLogicalRowIds() {
    return logicalRowIds;
  }

  /** Logical identities whose deleted physical rows were reclaimed. */
  @Nullable
  public byte[] getRetiredLogicalRowIds() {
    return retiredLogicalRowIds;
  }

  public List<FragmentMetadata> getNewFragments() {
    return newFragments;
  }

  public List<FragmentMetadata> getOriginalFragments() {
    return originalFragments;
  }
}
