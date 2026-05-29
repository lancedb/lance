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
package org.lance.update;

import org.lance.Dataset;

import com.google.common.base.MoreObjects;

/**
 * Result of {@link org.lance.Dataset#update(UpdateParams)}.
 *
 * <p>Contains the new {@link Dataset} produced by the commit and the number of rows affected.
 */
public final class UpdateResult {
  private final Dataset dataset;
  private final long numRowsUpdated;
  // Snapshot the dataset identity at construction time so toString() stays safe to call after the
  // caller has closed the dataset (closed datasets reject native uri()/version() calls).
  private final String datasetUri;
  private final long datasetVersion;

  public UpdateResult(Dataset dataset, long numRowsUpdated) {
    this.dataset = dataset;
    this.numRowsUpdated = numRowsUpdated;
    this.datasetUri = dataset.uri();
    this.datasetVersion = dataset.version();
  }

  /** Returns the new dataset reflecting the committed update. */
  public Dataset getDataset() {
    return dataset;
  }

  /** Returns the number of rows that matched the predicate and were updated. */
  public long getNumRowsUpdated() {
    return numRowsUpdated;
  }

  @Override
  public String toString() {
    // Use the snapshot taken at construction time; the caller may have already closed `dataset`.
    return MoreObjects.toStringHelper(this)
        .add("datasetUri", datasetUri)
        .add("datasetVersion", datasetVersion)
        .add("numRowsUpdated", numRowsUpdated)
        .toString();
  }
}
