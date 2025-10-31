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

import com.google.common.base.MoreObjects;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Write Params for Write Operations of Lance. Please check the <a
 * href="https://docs.rs/lance/latest/src/lance/dataset/write.rs.html#152">rust code</a> for all
 * default values.
 */
public class WriteParams {

  /** Write Mode. */
  public enum WriteMode {
    CREATE,
    APPEND,
    OVERWRITE
  }

  public enum LanceFileVersion {
    LEGACY("legacy"),
    V0_1("0.1"),
    V2_0("2.0"),
    STABLE("stable"),
    V2_1("2.1"),
    NEXT("next"),
    V2_2("2.2");

    private final String versionString;

    LanceFileVersion(String versionString) {
      this.versionString = versionString;
    }

    public String getVersionString() {
      return versionString;
    }
  }

  private final Optional<Integer> maxRowsPerFile;
  private final Optional<Integer> maxRowsPerGroup;
  private final Optional<Long> maxBytesPerFile;
  private final Optional<WriteMode> mode;
  private final Optional<Boolean> enableStableRowIds;
  private final Optional<LanceFileVersion> dataStorageVersion;
  private Map<String, String> storageOptions = new HashMap<>();

  private WriteParams(
      Optional<Integer> maxRowsPerFile,
      Optional<Integer> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile,
      Optional<WriteMode> mode,
      Optional<Boolean> enableStableRowIds,
      Optional<LanceFileVersion> dataStorageVersion,
      Map<String, String> storageOptions) {
    this.maxRowsPerFile = maxRowsPerFile;
    this.maxRowsPerGroup = maxRowsPerGroup;
    this.maxBytesPerFile = maxBytesPerFile;
    this.mode = mode;
    this.enableStableRowIds = enableStableRowIds;
    this.dataStorageVersion = dataStorageVersion;
    this.storageOptions = storageOptions;
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

  /**
   * Get Mode with name.
   *
   * @return mode
   */
  public Optional<String> getMode() {
    return mode.map(Enum::name);
  }

  public Optional<Boolean> getEnableStableRowIds() {
    return enableStableRowIds;
  }

  public Optional<String> getDataStorageVersion() {
    return dataStorageVersion.map(LanceFileVersion::getVersionString);
  }

  public Map<String, String> getStorageOptions() {
    return storageOptions;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("maxRowsPerFile", maxRowsPerFile.orElse(null))
        .add("maxRowsPerGroup", maxRowsPerGroup.orElse(null))
        .add("maxBytesPerFile", maxBytesPerFile.orElse(null))
        .add("mode", mode.orElse(null))
        .add("dataStorageVersion", dataStorageVersion.orElse(null))
        .toString();
  }

  /** A builder of WriteParams. */
  public static class Builder {
    private Optional<Integer> maxRowsPerFile = Optional.empty();
    private Optional<Integer> maxRowsPerGroup = Optional.empty();
    private Optional<Long> maxBytesPerFile = Optional.empty();
    private Optional<WriteMode> mode = Optional.empty();
    private Optional<Boolean> enableStableRowIds = Optional.empty();
    private Optional<LanceFileVersion> dataStorageVersion = Optional.empty();
    private Map<String, String> storageOptions = new HashMap<>();

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

    public Builder withDataStorageVersion(LanceFileVersion dataStorageVersion) {
      this.dataStorageVersion = Optional.of(dataStorageVersion);
      return this;
    }

    public Builder withStorageOptions(Map<String, String> storageOptions) {
      this.storageOptions = storageOptions;
      return this;
    }

    public Builder withEnableStableRowIds(boolean enableStableRowIds) {
      this.enableStableRowIds = Optional.of(enableStableRowIds);
      return this;
    }

    public WriteParams build() {
      return new WriteParams(
          maxRowsPerFile,
          maxRowsPerGroup,
          maxBytesPerFile,
          mode,
          enableStableRowIds,
          dataStorageVersion,
          storageOptions);
    }
  }
}
