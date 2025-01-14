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

public class DeletionFile implements Serializable {
  private static final long serialVersionUID = 3786348766842875859L;

  private final long id;
  private final long readVersion;
  private final Long numDeletedRows;
  private final DeletionFileType fileType;

  public DeletionFile(long id, long readVersion, Long numDeletedRows, DeletionFileType fileType) {
    this.id = id;
    this.readVersion = readVersion;
    this.numDeletedRows = numDeletedRows;
    this.fileType = fileType;
  }

  public long getId() {
    return id;
  }

  public long getReadVersion() {
    return readVersion;
  }

  public Long getNumDeletedRows() {
    return numDeletedRows;
  }

  public DeletionFileType getFileType() {
    return fileType;
  }

  @Override
  public String toString() {
    return new ToStringBuilder(this)
        .append("id", id)
        .append("readVersion", readVersion)
        .append("numDeletedRows", numDeletedRows)
        .append("fileType", fileType)
        .toString();
  }
}
