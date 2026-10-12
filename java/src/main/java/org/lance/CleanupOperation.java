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

import org.lance.cleanup.CleanupExplanation;
import org.lance.cleanup.CleanupPolicy;
import org.lance.cleanup.RemovalStats;

import org.apache.arrow.util.Preconditions;

import java.util.Optional;

/**
 * A cleanup operation with separate read-only explain and destructive execute actions.
 *
 * <p>This is not a deletion plan. Calling {@link #execute()} re-evaluates the current dataset and
 * reference state before deleting files.
 */
public class CleanupOperation {
  private final Dataset dataset;
  private final CleanupPolicy policy;
  private Optional<Long> deleteConcurrency = Optional.empty();
  private Optional<Long> deleteRateLimit = Optional.empty();
  private Optional<Long> maxCandidateFiles = Optional.empty();

  CleanupOperation(Dataset dataset, CleanupPolicy policy) {
    this.dataset = Preconditions.checkNotNull(dataset, "dataset cannot be null");
    this.policy = Preconditions.checkNotNull(policy, "policy cannot be null");
  }

  /**
   * Set the maximum in-flight file deletes shared with all cascaded branches.
   *
   * <p>Defaults to the initiating store's I/O parallelism. Must be between 1 and Tokio's semaphore
   * limit (2^61 - 1 on 64-bit platforms), validated by Rust on execute/explain. Applies only to
   * this call, independently of QPS; does not update table configuration. Cascades ignore their own
   * concurrency and QPS settings, including invalid values. Separate calls do not share limits.
   *
   * <p>Example: {@code dataset.cleanup(policy).withDeleteConcurrency(32).execute()}.
   *
   * @param deleteConcurrency maximum concurrent file deletes
   * @return this operation for chaining
   */
  public CleanupOperation withDeleteConcurrency(long deleteConcurrency) {
    this.deleteConcurrency = Optional.of(deleteConcurrency);
    return this;
  }

  /**
   * Set delete requests per second, shared with all cascaded branches.
   *
   * <p>Overrides the policy's rate; without an override, the policy's rate applies. Must be
   * positive, validated by Rust on execute/explain. Does not update table configuration.
   *
   * <p>Example: {@code dataset.cleanup(policy).withDeleteRateLimit(100).execute()}.
   *
   * @param deleteRateLimit maximum delete requests per second
   * @return this operation for chaining
   */
  public CleanupOperation withDeleteRateLimit(long deleteRateLimit) {
    this.deleteRateLimit = Optional.of(deleteRateLimit);
    return this;
  }

  /**
   * Set the maximum number of candidate files included in the {@link #explain()} result.
   *
   * <p>Defaults to 1000 if not set. The aggregate {@link RemovalStats} returned by {@link
   * #explain()} still account for all files that would be removed regardless of this limit; only
   * the per-file {@code candidateFiles} list is truncated.
   *
   * @param maxCandidateFiles maximum number of candidate files to include; must be positive
   * @return this operation for chaining
   */
  public CleanupOperation withMaxCandidateFiles(long maxCandidateFiles) {
    Preconditions.checkArgument(
        maxCandidateFiles > 0, "maxCandidateFiles must be positive, got %s", maxCandidateFiles);
    this.maxCandidateFiles = Optional.of(maxCandidateFiles);
    return this;
  }

  /**
   * Explain what cleanup would remove without deleting files.
   *
   * @return cleanup explanation
   */
  public CleanupExplanation explain() {
    return dataset.explainCleanup(policy, maxCandidateFiles, deleteConcurrency, deleteRateLimit);
  }

  /**
   * Execute cleanup, re-evaluating the current dataset and reference state before deleting files.
   *
   * @return removal stats
   */
  public RemovalStats execute() {
    return dataset.executeCleanup(policy, deleteConcurrency, deleteRateLimit);
  }
}
