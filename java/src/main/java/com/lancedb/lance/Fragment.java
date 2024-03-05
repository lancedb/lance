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

/** Data Fragment. */
public class Fragment {
  // Only keep fragmentId for reference, so we don't need to make this
  // object to be {@link Closable} to track Rust native object.
  private final int fragmentId;

  /** Pointer to the {@link Dataset} instance in Java. */
  private final Dataset dataset;

  /** Private constructor, calling from JNI. */
  Fragment(Dataset dataset, int fragmentId) {
    this.dataset = dataset;
    this.fragmentId = fragmentId;
  }

  private native int countRowsNative(Dataset dataset, long fragmentId);

  public int getFragmentId() {
    return this.fragmentId;
  }

  public String toString() {
    return String.format("Fragment(id=%d)", this.fragmentId);
  }

  /** Count rows in this Fragment. */
  public int countRows() {
    return countRowsNative(this.dataset, this.fragmentId);
  }
}
