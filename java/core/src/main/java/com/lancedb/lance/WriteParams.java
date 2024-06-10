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

package com.lancedb.lance;

import java.util.Optional;

/**
 * Write Params for Write Operations of Lance.
 */
public class WriteParams {

  /**
   * Write Mode.
   */
  public enum WriteMode {
    CREATE,
    APPEND,
    OVERWRITE
  }

  private final Optional<Integer> maxRowsPerFile;
  private final Optional<Integer> maxRowsPerGroup;
  private final Optional<Long> maxBytesPerFile;
  private final Optional<WriteMode> mode;

  private WriteParams(Optional<Integer> maxRowsPerFile, Optional<Integer> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile, Optional<WriteMode> mode) {
    this.maxRowsPerFile = maxRowsPerFile;
    this.maxRowsPerGroup = maxRowsPerGroup;
    this.maxBytesPerFile = maxBytesPerFile;
    this.mode = mode;
  }

  public Optional<Integer> getMaxRowsPerFile() {
    return maxRowsPerFile;
  }

  public Optional<Integer> getMaxRowsPerGroup() {
    return maxRowsPerGroup;
  }

  public Optional<Long> getMaxBytesPerFile() {
    return maxBytesPerFile;
  }

  /** Get Mode with name. */
  public Optional<String> getMode() {
    return mode.map(Enum::name);
  }

  /**
   * A builder of WriteParams.
   */
  public static class Builder {
    private Optional<Integer> maxRowsPerFile = Optional.empty();
    private Optional<Integer> maxRowsPerGroup = Optional.empty();
    private Optional<Long> maxBytesPerFile = Optional.empty();
    private Optional<WriteMode> mode = Optional.empty();

    public Builder withMaxRowsPerFile(int maxRowsPerFile) {
      this.maxRowsPerFile = Optional.of(maxRowsPerFile);
      return this;
    }

    public Builder withMaxRowsPerGroup(int maxRowsPerGroup) {
      this.maxRowsPerGroup = Optional.of(maxRowsPerGroup);
      return this;
    }

    public Builder withMaxBytesPerFile(long maxBytesPerFile) {
      this.maxBytesPerFile = Optional.of(maxBytesPerFile);
      return this;
    }

    public Builder withMode(WriteMode mode) {
      this.mode = Optional.of(mode);
      return this;
    }

    public WriteParams build() {
      return new WriteParams(maxRowsPerFile, maxRowsPerGroup, maxBytesPerFile, mode);
    }
  }
}