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

import org.lance.FragmentMetadata;

import javax.annotation.Nullable;

import java.io.Serializable;
import java.util.List;

/** Data of compaction task. */
public class TaskData implements Serializable {
  private final List<FragmentMetadata> fragments;
  @Nullable private final byte[] v23Plan;

  public TaskData(List<FragmentMetadata> fragments) {
    this(fragments, null);
  }

  public TaskData(List<FragmentMetadata> fragments, @Nullable byte[] v23Plan) {
    this.fragments = fragments;
    this.v23Plan = v23Plan;
  }

  public List<FragmentMetadata> getFragments() {
    return fragments;
  }

  /** Opaque exact storage-version-2.3 compaction preflight plan. */
  @Nullable
  public byte[] getV23Plan() {
    return v23Plan;
  }
}
