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
package com.lancedb.lance.compaction;

import com.lancedb.lance.FragmentMetadata;

import javax.annotation.Nullable;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

/**
 * Rewrite Result of a single compaction task. It will be passed across different workers and be
 * committed later.
 */
public class RewriteResult implements Serializable {
  private final CompactionMetrics metrics;
  private final List<FragmentMetadata> newFragments;
  private final List<FragmentMetadata> originalFragments;
  private final long readVersion;

  // null if index remap is deferred after compaction
  @Nullable private final Map<Long, Long> rowIdMap;

  // null if index remap is part of compaction
  @Nullable private final byte[] changedRowAddrs;

  public RewriteResult(
      CompactionMetrics metrics,
      List<FragmentMetadata> newFragments,
      List<FragmentMetadata> originalFragments,
      long readVersion,
      Map<Long, Long> rowIdMap,
      byte[] changedRowAddrs) {
    this.metrics = metrics;
    this.newFragments = newFragments;
    this.originalFragments = originalFragments;
    this.readVersion = readVersion;
    this.rowIdMap = rowIdMap;
    this.changedRowAddrs = changedRowAddrs;
  }

  public long getReadVersion() {
    return readVersion;
  }

  public CompactionMetrics getMetrics() {
    return metrics;
  }

  @Nullable
  public byte[] getChangedRowAddrs() {
    return changedRowAddrs;
  }

  public List<FragmentMetadata> getNewFragments() {
    return newFragments;
  }

  public List<FragmentMetadata> getOriginalFragments() {
    return originalFragments;
  }

  @Nullable
  public Map<Long, Long> getRowIdMap() {
    return rowIdMap;
  }
}
