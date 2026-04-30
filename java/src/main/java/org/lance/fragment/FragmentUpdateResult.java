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
package org.lance.fragment;

import org.lance.FragmentMetadata;

import com.google.common.base.MoreObjects;
import org.apache.arrow.c.ArrowArrayStream;

/**
 * Result of {@link org.lance.Fragment#updateColumns(ArrowArrayStream, String, String)
 * Fragment.updateColumns()}.
 */
public class FragmentUpdateResult {
  private final FragmentMetadata updatedFragment;
  private final long[] fieldsModified;

  /** Local physical row offsets within the fragment that received updates (see RowAddress). */
  private final long[] updatedRowOffsets;

  /** Two-argument form for callers that do not track per-row offsets; offsets default to empty. */
  public FragmentUpdateResult(FragmentMetadata updatedFragment, long[] updatedFieldIds) {
    this(updatedFragment, updatedFieldIds, new long[0]);
  }

  public FragmentUpdateResult(
      FragmentMetadata updatedFragment, long[] updatedFieldIds, long[] updatedRowOffsets) {
    this.updatedFragment = updatedFragment;
    this.fieldsModified = updatedFieldIds;
    this.updatedRowOffsets = updatedRowOffsets;
  }

  public FragmentMetadata getUpdatedFragment() {
    return updatedFragment;
  }

  public long[] getFieldsModified() {
    return fieldsModified;
  }

  /** Physical row offsets (0-based within the fragment) whose columns were rewritten. */
  public long[] getUpdatedRowOffsets() {
    return updatedRowOffsets;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("fragmentMetadata", updatedFragment)
        .add("updatedFieldIds", fieldsModified)
        .add("updatedRowOffsets", updatedRowOffsets)
        .toString();
  }
}
