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
package org.lance.index;

/** Address domain stored in an index segment's postings. */
public enum RowReferenceDomain {
  PHYSICAL_ROW_ADDRESS(1),
  LEGACY_STABLE_ROW_ID(2),
  STABLE_LOGICAL_ROW_ADDRESS(3);

  private final int value;

  RowReferenceDomain(int value) {
    this.value = value;
  }

  /** Return the protobuf wire value expected by the native layer. */
  public int getValue() {
    return value;
  }
}
