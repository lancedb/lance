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

import com.lancedb.lance.fragment.DataFile;
import com.lancedb.lance.fragment.DeletionFile;
import com.lancedb.lance.fragment.RowIdMeta;

import com.google.common.base.MoreObjects;

import java.io.Serializable;
import java.util.List;
import java.util.Objects;

/** Metadata of a Fragment in the dataset. Matching to lance Fragment. */
public class FragmentMetadata implements Serializable {
  private static final long serialVersionUID = -5886811251944130460L;
  private final int id;
  private final List<DataFile> files;
  private final long physicalRows;
  private final DeletionFile deletionFile;
  private final RowIdMeta rowIdMeta;

  public FragmentMetadata(
      int id,
      List<DataFile> files,
      Long physicalRows,
      DeletionFile deletionFile,
      RowIdMeta rowIdMeta) {
    this.id = id;
    this.files = files;
    this.physicalRows = physicalRows;
    this.deletionFile = deletionFile;
    this.rowIdMeta = rowIdMeta;
  }

  public int getId() {
    return id;
  }

  public List<DataFile> getFiles() {
    return files;
  }

  public long getPhysicalRows() {
    return physicalRows;
  }

  public DeletionFile getDeletionFile() {
    return deletionFile;
  }

  public long getNumDeletions() {
    if (deletionFile == null) {
      return 0;
    }
    Long deleted = deletionFile.getNumDeletedRows();
    if (deleted == null) {
      return 0;
    }
    return deleted;
  }

  public long getNumRows() {
    return getPhysicalRows() - getNumDeletions();
  }

  public RowIdMeta getRowIdMeta() {
    return rowIdMeta;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    FragmentMetadata that = (FragmentMetadata) o;
    return id == that.id
        && physicalRows == that.physicalRows
        && Objects.equals(this.files, that.files)
        && Objects.equals(deletionFile, that.deletionFile)
        && Objects.equals(rowIdMeta, that.rowIdMeta);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("id", id)
        .add("physicalRows", physicalRows)
        .add("files", files)
        .add("deletionFile", deletionFile)
        .add("rowIdMeta", rowIdMeta)
        .toString();
  }
}
