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

/** Data Fragment.
 *
 */
public class Fragment {
  // Only keep fragmentId for reference, so we dont need to make this
  // object to be Closable to track Rust native object.
  private long fragmentId;

  /** Private constructor, calling from JNI. */
  private Fragment(long fragmentId) {
    this.fragmentId = fragmentId;
  }

  public long getFragmentId() {
    return this.fragmentId;
  }

  public String toString() {
    return String.format("Fragment(id=%d)", this.fragmentId);
  }
}
