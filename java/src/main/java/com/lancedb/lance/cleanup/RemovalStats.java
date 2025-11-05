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
package com.lancedb.lance.cleanup;

/** Statistics returned by dataset cleanup. */
public class RemovalStats {
  private final long bytesRemoved;
  private final long oldVersions;

  public RemovalStats(long bytesRemoved, long oldVersions) {
    this.bytesRemoved = bytesRemoved;
    this.oldVersions = oldVersions;
  }

  public long getBytesRemoved() {
    return bytesRemoved;
  }

  public long getOldVersions() {
    return oldVersions;
  }
}
