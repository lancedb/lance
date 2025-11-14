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
package com.lancedb.lance.cleanup;

import java.util.Optional;

/**
 * Cleanup policy for dataset cleanup.
 *
 * <p>All fields are optional. We intentionally do not set default values here to avoid conflicting
 * with Rust-side defaults. Refer to Rust CleanupPolicy for defaults.
 */
public class CleanupPolicy {
  private final Optional<Long> beforeTimestampMillis;
  private final Optional<Long> beforeVersion;
  private final Optional<Boolean> deleteUnverified;
  private final Optional<Boolean> errorIfTaggedOldVersions;

  private CleanupPolicy(
      Optional<Long> beforeTimestampMillis,
      Optional<Long> beforeVersion,
      Optional<Boolean> deleteUnverified,
      Optional<Boolean> errorIfTaggedOldVersions) {
    this.beforeTimestampMillis = beforeTimestampMillis;
    this.beforeVersion = beforeVersion;
    this.deleteUnverified = deleteUnverified;
    this.errorIfTaggedOldVersions = errorIfTaggedOldVersions;
  }

  public static Builder builder() {
    return new Builder();
  }

  public Optional<Long> getBeforeTimestampMillis() {
    return beforeTimestampMillis;
  }

  public Optional<Long> getBeforeVersion() {
    return beforeVersion;
  }

  public Optional<Boolean> getDeleteUnverified() {
    return deleteUnverified;
  }

  public Optional<Boolean> getErrorIfTaggedOldVersions() {
    return errorIfTaggedOldVersions;
  }

  /** Builder for CleanupPolicy. */
  public static class Builder {
    private Optional<Long> beforeTimestampMillis = Optional.empty();
    private Optional<Long> beforeVersion = Optional.empty();
    private Optional<Boolean> deleteUnverified = Optional.empty();
    private Optional<Boolean> errorIfTaggedOldVersions = Optional.empty();

    private Builder() {}

    /** Set a timestamp threshold in milliseconds since UNIX epoch (UTC). */
    public Builder withBeforeTimestampMillis(long beforeTimestampMillis) {
      this.beforeTimestampMillis = Optional.of(beforeTimestampMillis);
      return this;
    }

    /** Set a version threshold; versions older than this will be cleaned. */
    public Builder withBeforeVersion(long beforeVersion) {
      this.beforeVersion = Optional.of(beforeVersion);
      return this;
    }

    /** If true, delete unverified data files even if they are recent. */
    public Builder withDeleteUnverified(boolean deleteUnverified) {
      this.deleteUnverified = Optional.of(deleteUnverified);
      return this;
    }

    /** If true, raise error when tagged versions are old and matched by policy. */
    public Builder withErrorIfTaggedOldVersions(boolean errorIfTaggedOldVersions) {
      this.errorIfTaggedOldVersions = Optional.of(errorIfTaggedOldVersions);
      return this;
    }

    public CleanupPolicy build() {
      return new CleanupPolicy(
          beforeTimestampMillis, beforeVersion, deleteUnverified, errorIfTaggedOldVersions);
    }
  }
}
