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
package com.lancedb.lance.fragment;

import org.apache.commons.lang3.builder.ToStringBuilder;

import java.io.Serializable;

public class DataFile implements Serializable {
  private static final long serialVersionUID = -2827710928026343591L;
  private final String path;
  private final int[] fields;
  private final int[] columnIndices;
  private final int fileMajorVersion;
  private final int fileMinorVersion;

  public DataFile(
      String path, int[] fields, int[] columnIndices, int fileMajorVersion, int fileMinorVersion) {
    this.path = path;
    this.fields = fields;
    this.columnIndices = columnIndices;
    this.fileMajorVersion = fileMajorVersion;
    this.fileMinorVersion = fileMinorVersion;
  }

  public String getPath() {
    return path;
  }

  public int[] getFields() {
    return fields;
  }

  public int[] getColumnIndices() {
    return columnIndices;
  }

  public int getFileMajorVersion() {
    return fileMajorVersion;
  }

  public int getFileMinorVersion() {
    return fileMinorVersion;
  }

  @Override
  public String toString() {
    return new ToStringBuilder(this)
        .append("path", path)
        .append("fields", fields)
        .append("columnIndices", columnIndices)
        .append("fileMajorVersion", fileMajorVersion)
        .append("fileMinorVersion", fileMinorVersion)
        .toString();
  }
}
