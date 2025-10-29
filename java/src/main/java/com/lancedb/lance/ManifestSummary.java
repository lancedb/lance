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

import java.util.Map;

/** Statistical summary of a dataset manifest for a specific version. */
public class ManifestSummary {
  private static final String TOTAL_FRAGMENTS_KEY = "total_fragments";
  private static final String TOTAL_DATA_FILES_KEY = "total_data_files";
  private static final String TOTAL_FILES_SIZE_KEY = "total_files_size";
  private static final String TOTAL_DELETION_FILES_KEY = "total_deletion_files";
  private static final String TOTAL_DATA_FILE_ROWS_KEY = "total_data_file_rows";
  private static final String TOTAL_DELETION_FILE_ROWS_KEY = "total_deletion_file_rows";
  private static final String TOTAL_ROWS_KEY = "total_rows";

  private final long totalFragments;
  private final long totalDataFiles;
  private final long totalFilesSize;
  private final long totalDeletionFiles;
  private final long totalDataFileRows;
  private final long totalDeletionFileRows;
  private final long totalRows;

  public ManifestSummary(
      long totalFragments,
      long totalDataFiles,
      long totalFilesSize,
      long totalDeletionFiles,
      long totalDataFileRows,
      long totalDeletionFileRows,
      long totalRows) {
    this.totalFragments = totalFragments;
    this.totalDataFiles = totalDataFiles;
    this.totalFilesSize = totalFilesSize;
    this.totalDeletionFiles = totalDeletionFiles;
    this.totalDataFileRows = totalDataFileRows;
    this.totalDeletionFileRows = totalDeletionFileRows;
    this.totalRows = totalRows;
  }

  public long getTotalDataFileRows() {
    return totalDataFileRows;
  }

  public long getTotalDataFiles() {
    return totalDataFiles;
  }

  public long getTotalDeletionFileRows() {
    return totalDeletionFileRows;
  }

  public long getTotalDeletionFiles() {
    return totalDeletionFiles;
  }

  public long getTotalFilesSize() {
    return totalFilesSize;
  }

  public long getTotalFragments() {
    return totalFragments;
  }

  public long getTotalRows() {
    return totalRows;
  }

  public static ManifestSummary fromMetadata(Map<String, String> map) {
    long totalFragments = Long.parseLong(map.getOrDefault(TOTAL_FRAGMENTS_KEY, "0"));
    long totalDataFiles = Long.parseLong(map.getOrDefault(TOTAL_DATA_FILES_KEY, "0"));
    long totalFilesSize = Long.parseLong(map.getOrDefault(TOTAL_FILES_SIZE_KEY, "0"));
    long totalDeletionFiles = Long.parseLong(map.getOrDefault(TOTAL_DELETION_FILES_KEY, "0"));
    long totalDataFileRows = Long.parseLong(map.getOrDefault(TOTAL_DATA_FILE_ROWS_KEY, "0"));
    long totalDeletionFileRows =
        Long.parseLong(map.getOrDefault(TOTAL_DELETION_FILE_ROWS_KEY, "0"));
    long totalRows = Long.parseLong(map.getOrDefault(TOTAL_ROWS_KEY, "0"));

    return new ManifestSummary(
        totalFragments,
        totalDataFiles,
        totalFilesSize,
        totalDeletionFiles,
        totalDataFileRows,
        totalDeletionFileRows,
        totalRows);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("totalFragments", totalFragments)
        .add("totalDataFiles", totalDataFiles)
        .add("totalFilesSize", totalFilesSize)
        .add("totalDeletionFiles", totalDeletionFiles)
        .add("totalDataFileRows", totalDataFileRows)
        .add("totalDeletionFileRows", totalDeletionFileRows)
        .add("totalRows", totalRows)
        .toString();
  }
}
