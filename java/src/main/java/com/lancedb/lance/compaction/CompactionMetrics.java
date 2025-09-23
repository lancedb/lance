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
package com.lancedb.lance.compaction;

import com.google.common.base.MoreObjects;

import java.io.Serializable;

/** The compaction metrics. */
public class CompactionMetrics implements Serializable {
  private final long fragmentsRemoved;
  private final long fragmentsAdded;
  private final long filesRemoved;
  private final long filesAdded;

  public CompactionMetrics(
      long fragmentsRemoved, long fragmentsAdded, long filesRemoved, long filesAdded) {
    this.filesRemoved = filesRemoved;
    this.fragmentsAdded = fragmentsAdded;
    this.fragmentsRemoved = fragmentsRemoved;
    this.filesAdded = filesAdded;
  }

  public long getFilesAdded() {
    return filesAdded;
  }

  public long getFilesRemoved() {
    return filesRemoved;
  }

  public long getFragmentsAdded() {
    return fragmentsAdded;
  }

  public long getFragmentsRemoved() {
    return fragmentsRemoved;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("fragmentsRemoved", fragmentsRemoved)
        .add("fragmentsAdded", fragmentsAdded)
        .add("filesRemoved", filesRemoved)
        .add("filesAdded", filesAdded)
        .toString();
  }
}
