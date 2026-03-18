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
package org.lance.compaction;

/** Controls how data is rewritten during compaction. */
public enum CompactionMode {
  /** Decode and re-encode data (default). */
  REENCODE("reencode"),
  /** Try binary copy if fragments are compatible, fall back to reencode otherwise. */
  TRY_BINARY_COPY("try_binary_copy"),
  /** Use binary copy or fail if fragments are not compatible. */
  FORCE_BINARY_COPY("force_binary_copy");

  private final String value;

  CompactionMode(String value) {
    this.value = value;
  }

  public String getValue() {
    return value;
  }
}
