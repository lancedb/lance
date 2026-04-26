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
  private final long[] matchedOffsets;

  public FragmentUpdateResult(FragmentMetadata updatedFragment, long[] updatedFieldIds, long[] matchedOffsets) {
    this.updatedFragment = updatedFragment;
    this.fieldsModified = updatedFieldIds;
    this.matchedOffsets = matchedOffsets;
  }

  public FragmentUpdateResult(FragmentMetadata updatedFragment, long[] updatedFieldIds) {
    this(updatedFragment, updatedFieldIds, new long[0]);
  }

  public FragmentMetadata getUpdatedFragment() {
    return updatedFragment;
  }

  public long[] getFieldsModified() {
    return fieldsModified;
  }

  /**
   * Physical row offsets within the fragment that were matched (updated) by the join.
   * These are 0-based indices and can be passed as {@code updatedRowOffsets} when
   * committing with {@link org.lance.operation.Update}.
   */
  public long[] getMatchedOffsets() {
    return matchedOffsets;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("fragmentMetadata", updatedFragment)
        .add("updatedFieldIds", fieldsModified)
        .add("matchedOffsets", matchedOffsets)
        .toString();
  }
}
