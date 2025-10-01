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

import com.lancedb.lance.Dataset;
import com.lancedb.lance.JniLoader;

import com.google.common.base.Preconditions;

import java.util.List;
import java.util.Optional;

/** The entrypoint of distributed compaction-related methods. */
public class Compaction {
  static {
    JniLoader.ensureLoaded();
  }

  public static CompactionPlan planCompaction(
      Dataset dataset, CompactionOptions compactionOptions) {
    Preconditions.checkNotNull(dataset);
    Preconditions.checkNotNull(compactionOptions);

    return nativePlanCompaction(
        dataset,
        compactionOptions.getTargetRowsPerFragment(),
        compactionOptions.getMaxRowsPerGroup(),
        compactionOptions.getMaxBytesPerFile(),
        compactionOptions.getMaterializeDeletions(),
        compactionOptions.getMaterializeDeletionsThreshold(),
        compactionOptions.getNumThreads(),
        compactionOptions.getBatchSize(),
        compactionOptions.getDeferIndexRemap());
  }

  public static CompactionMetrics commitCompaction(
      Dataset dataset, List<RewriteResult> rewriteResults, CompactionOptions compactionOptions) {
    Preconditions.checkNotNull(dataset);
    Preconditions.checkNotNull(rewriteResults);
    Preconditions.checkNotNull(compactionOptions);
    return nativeCommitCompaction(
        dataset,
        rewriteResults,
        compactionOptions.getTargetRowsPerFragment(),
        compactionOptions.getMaxRowsPerGroup(),
        compactionOptions.getMaxBytesPerFile(),
        compactionOptions.getMaterializeDeletions(),
        compactionOptions.getMaterializeDeletionsThreshold(),
        compactionOptions.getNumThreads(),
        compactionOptions.getBatchSize(),
        compactionOptions.getDeferIndexRemap());
  }

  public static native CompactionMetrics nativeCommitCompaction(
      Dataset dataset,
      List<RewriteResult> rewriteResults,
      Optional<Long> targetRowsPerFragment,
      Optional<Long> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile,
      Optional<Boolean> materializeDeletions,
      Optional<Float> materializeDeletionsThreshold,
      Optional<Long> numThreads,
      Optional<Long> batchSize,
      Optional<Boolean> deferIndexRemap);

  private static native CompactionPlan nativePlanCompaction(
      Dataset dataset,
      Optional<Long> targetRowsPerFragment,
      Optional<Long> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile,
      Optional<Boolean> materializeDeletions,
      Optional<Float> materializeDeletionsThreshold,
      Optional<Long> numThreads,
      Optional<Long> batchSize,
      Optional<Boolean> deferIndexRemap);
}
