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
package org.lance;

import org.lance.index.IndexBuildProgress;

import java.util.Optional;

/**
 * Marks the current thread as executing an index progress callback for a specific Dataset.
 *
 * <p>This context lets Dataset reject conflicting callback re-entry without rejecting unrelated
 * concurrent callers.
 */
final class ContextIndexBuildProgress implements IndexBuildProgress {
  private static final ThreadLocal<Dataset> CURRENT_DATASET = new ThreadLocal<>();

  private final Dataset dataset;
  private final IndexBuildProgress delegate;

  ContextIndexBuildProgress(Dataset dataset, IndexBuildProgress delegate) {
    this.dataset = dataset;
    this.delegate = delegate;
  }

  static boolean isCurrent(Dataset dataset) {
    return CURRENT_DATASET.get() == dataset;
  }

  @Override
  public void stageStart(String stage, Optional<Long> total, String unit) {
    CURRENT_DATASET.set(dataset);
    try {
      delegate.stageStart(stage, total, unit);
    } finally {
      CURRENT_DATASET.remove();
    }
  }

  @Override
  public void stageProgress(String stage, long completed) {
    CURRENT_DATASET.set(dataset);
    try {
      delegate.stageProgress(stage, completed);
    } finally {
      CURRENT_DATASET.remove();
    }
  }

  @Override
  public void stageComplete(String stage) {
    CURRENT_DATASET.set(dataset);
    try {
      delegate.stageComplete(stage);
    } finally {
      CURRENT_DATASET.remove();
    }
  }
}
