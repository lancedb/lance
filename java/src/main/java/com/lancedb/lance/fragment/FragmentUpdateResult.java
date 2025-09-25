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

import com.lancedb.lance.FragmentMetadata;

import com.google.common.base.MoreObjects;
import org.apache.arrow.c.ArrowArrayStream;

/**
 * Result of {@link com.lancedb.lance.Fragment#updateColumns(ArrowArrayStream, String, String)
 * Fragment.updateColumns()}.
 */
public class FragmentUpdateResult {
  private final FragmentMetadata updatedFragment;
  private final long[] fieldsModified;

  public FragmentUpdateResult(FragmentMetadata updatedFragment, long[] updatedFieldIds) {
    this.updatedFragment = updatedFragment;
    this.fieldsModified = updatedFieldIds;
  }

  public FragmentMetadata getUpdatedFragment() {
    return updatedFragment;
  }

  public long[] getFieldsModified() {
    return fieldsModified;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("fragmentMetadata", updatedFragment)
        .add("updatedFieldIds", fieldsModified)
        .toString();
  }
}
