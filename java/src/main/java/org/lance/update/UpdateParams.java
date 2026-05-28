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

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Parameters describing an update operation.
 *
 * <p>An update operation is similar to SQL's {@code UPDATE} statement: each entry in {@link
 * #updates()} maps a target column name to a SQL expression evaluated for every row that matches
 * the optional {@link #whereClause()}.
 *
 * <p>The {@code where} clause accepts the dataset columns as well as the system columns {@code
 * _rowid}, {@code _rowaddr}, and {@code _rowoffset}, which is useful for targeting specific rows by
 * stable row id.
 */
public class UpdateParams {
  // Defaults are kept in sync with the Rust core defaults declared on
  // `lance::dataset::UpdateBuilder` (`conflict_retries = 10`,
  // `retry_timeout = 30s`). When the Rust defaults change, update both sides.
  private static final int DEFAULT_CONFLICT_RETRIES = 10;
  private static final long DEFAULT_RETRY_TIMEOUT_MS = 30_000L;

  private final Map<String, String> updates;
  private Optional<String> whereClause = Optional.empty();
  private int conflictRetries = DEFAULT_CONFLICT_RETRIES;
  private long retryTimeoutMs = DEFAULT_RETRY_TIMEOUT_MS;

  public UpdateParams(Map<String, String> updates) {
    Preconditions.checkNotNull(updates, "updates must not be null");
    Preconditions.checkArgument(!updates.isEmpty(), "updates must not be empty");
    this.updates = new HashMap<>(updates);
  }

  /**
   * Restrict the update to rows matching the given SQL predicate.
   *
   * <p>The predicate is evaluated against the dataset schema augmented with the system columns
   * {@code _rowid}, {@code _rowaddr}, and {@code _rowoffset}.
   *
   * @param whereClause SQL predicate, e.g. {@code "id = 1"} or {@code "_rowid IN (1, 2, 3)"}.
   * @return This UpdateParams instance.
   */
  public UpdateParams withWhere(String whereClause) {
    Preconditions.checkNotNull(whereClause, "whereClause must not be null");
    this.whereClause = Optional.of(whereClause);
    return this;
  }

  /**
   * Set number of times to retry the operation if there is contention.
   *
   * <p>Default mirrors the Rust core default ({@value #DEFAULT_CONFLICT_RETRIES}).
   *
   * @param retries Number of times to retry the operation if there is contention.
   * @return This UpdateParams instance.
   */
  public UpdateParams withConflictRetries(int retries) {
    Preconditions.checkArgument(retries >= 0, "retries must be non-negative");
    this.conflictRetries = retries;
    return this;
  }

  /**
   * Set the timeout in milliseconds used to limit retries.
   *
   * <p>This is the maximum time to spend on the operation before giving up. At least one attempt
   * will be made, regardless of how long it takes to complete. Default mirrors the Rust core
   * default ({@value #DEFAULT_RETRY_TIMEOUT_MS} ms).
   *
   * @param timeoutMs Timeout in milliseconds used to limit retries.
   * @return This UpdateParams instance.
   */
  public UpdateParams withRetryTimeoutMs(long timeoutMs) {
    Preconditions.checkArgument(timeoutMs >= 0, "timeoutMs must be non-negative");
    this.retryTimeoutMs = timeoutMs;
    return this;
  }

  /** Returns an unmodifiable view of the update expressions. */
  public Map<String, String> updates() {
    return Collections.unmodifiableMap(updates);
  }

  public Optional<String> whereClause() {
    return whereClause;
  }

  public int conflictRetries() {
    return conflictRetries;
  }

  public long retryTimeoutMs() {
    return retryTimeoutMs;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("updates", updates)
        .add("whereClause", whereClause.orElse(null))
        .add("conflictRetries", conflictRetries)
        .add("retryTimeoutMs", retryTimeoutMs)
        .toString();
  }
}
