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
package org.lance.maintenance;

import com.google.common.base.MoreObjects;

/** Observable cost of explicit storage-version-2.3 row-address maintenance. */
public final class RowAddressMaintenanceMetrics {
  private final long fragmentsRemoved;
  private final long fragmentsAdded;
  private final long dataFilesWritten;
  private final long locatorObjectsWritten;
  private final long locatorBytesWritten;
  private final long rowsRewritten;

  public RowAddressMaintenanceMetrics(
      long fragmentsRemoved,
      long fragmentsAdded,
      long dataFilesWritten,
      long locatorObjectsWritten,
      long locatorBytesWritten,
      long rowsRewritten) {
    this.fragmentsRemoved = fragmentsRemoved;
    this.fragmentsAdded = fragmentsAdded;
    this.dataFilesWritten = dataFilesWritten;
    this.locatorObjectsWritten = locatorObjectsWritten;
    this.locatorBytesWritten = locatorBytesWritten;
    this.rowsRewritten = rowsRewritten;
  }

  /** Return the number of physical fragments replaced. */
  public long getFragmentsRemoved() {
    return fragmentsRemoved;
  }

  /** Return the number of physical fragments created. */
  public long getFragmentsAdded() {
    return fragmentsAdded;
  }

  /** Return the number of user-data files written. */
  public long getDataFilesWritten() {
    return dataFilesWritten;
  }

  /** Return the number of locator and hidden {@code _rowid} objects written. */
  public long getLocatorObjectsWritten() {
    return locatorObjectsWritten;
  }

  /** Return bytes written to locator and hidden {@code _rowid} objects. */
  public long getLocatorBytesWritten() {
    return locatorBytesWritten;
  }

  /** Return the number of live rows physically rewritten. */
  public long getRowsRewritten() {
    return rowsRewritten;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("fragmentsRemoved", fragmentsRemoved)
        .add("fragmentsAdded", fragmentsAdded)
        .add("dataFilesWritten", dataFilesWritten)
        .add("locatorObjectsWritten", locatorObjectsWritten)
        .add("locatorBytesWritten", locatorBytesWritten)
        .add("rowsRewritten", rowsRewritten)
        .toString();
  }
}
