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

import com.lancedb.lance.util.ToStringHelper;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/** Write Params for Write Operations of Lance. */
public class WriteParams {

  /** Write Mode. */
  public enum WriteMode {
    CREATE,
    APPEND,
    OVERWRITE
  }

  private final Integer maxRowsPerFile;
  private final Integer maxRowsPerGroup;
  private final Long maxBytesPerFile;
  private final WriteMode mode;
  private final Map<String, String> storageOptions;

  private WriteParams(
      Integer maxRowsPerFile,
      Integer maxRowsPerGroup,
      Long maxBytesPerFile,
      WriteMode mode,
      Map<String, String> storageOptions) {
    this.maxRowsPerFile = maxRowsPerFile;
    this.maxRowsPerGroup = maxRowsPerGroup;
    this.maxBytesPerFile = maxBytesPerFile;
    this.mode = mode;
    this.storageOptions = storageOptions;
  }

  public Optional<Integer> getMaxRowsPerFile() {
    return Optional.ofNullable(maxRowsPerFile);
  }

  public Optional<Integer> getMaxRowsPerGroup() {
    return Optional.ofNullable(maxRowsPerGroup);
  }

  public Optional<Long> getMaxBytesPerFile() {
    return Optional.ofNullable(maxBytesPerFile);
  }

  /**
   * Get Mode with name.
   *
   * @return mode
   */
  public Optional<String> getMode() {
    return Optional.ofNullable(mode).map(Enum::name);
  }

  public Map<String, String> getStorageOptions() {
    return storageOptions;
  }

  @Override
  public String toString() {
    return ToStringHelper.of(this)
        .add("maxRowsPerFile", maxRowsPerFile)
        .add("maxRowsPerGroup", maxRowsPerGroup)
        .add("maxBytesPerFile", maxBytesPerFile)
        .add("mode", mode)
        .toString();
  }

  /** A builder of WriteParams. */
  public static class Builder {
    private Integer maxRowsPerFile;
    private Integer maxRowsPerGroup;
    private Long maxBytesPerFile;
    private WriteMode mode;
    private Map<String, String> storageOptions = new HashMap<>();

    public Builder withMaxRowsPerFile(int maxRowsPerFile) {
      this.maxRowsPerFile = maxRowsPerFile;
      return this;
    }

    public Builder withMaxRowsPerGroup(int maxRowsPerGroup) {
      this.maxRowsPerGroup = maxRowsPerGroup;
      return this;
    }

    public Builder withMaxBytesPerFile(long maxBytesPerFile) {
      this.maxBytesPerFile = maxBytesPerFile;
      return this;
    }

    public Builder withMode(WriteMode mode) {
      this.mode = mode;
      return this;
    }

    public Builder withStorageOptions(Map<String, String> storageOptions) {
      this.storageOptions = storageOptions;
      return this;
    }

    public WriteParams build() {
      return new WriteParams(
          maxRowsPerFile, maxRowsPerGroup, maxBytesPerFile, mode, storageOptions);
    }
  }
}
