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
package org.lance.memwal;

import com.google.common.base.MoreObjects;

/** A flushed MemWAL generation and the storage path of its Lance files. */
public class FlushedGeneration {
  private final long generation;
  private final String path;

  public FlushedGeneration(long generation, String path) {
    this.generation = generation;
    this.path = path;
  }

  /** The generation number of this flushed MemTable. */
  public long generation() {
    return generation;
  }

  /** The storage path of the flushed Lance files. */
  public String path() {
    return path;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("generation", generation)
        .add("path", path)
        .toString();
  }
}
