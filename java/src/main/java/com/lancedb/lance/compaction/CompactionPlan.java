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

import java.util.List;
import java.util.stream.Collectors;

/** The compaction plan. */
public class CompactionPlan {
  private final List<TaskData> tasks;
  private final long readVersion;
  private final CompactionOptions compactionOptions;

  public CompactionPlan(
      List<TaskData> tasks, long readVersion, CompactionOptions compactionOptions) {
    this.tasks = tasks;
    this.readVersion = readVersion;
    this.compactionOptions = compactionOptions;
  }

  public long getReadVersion() {
    return readVersion;
  }

  public CompactionOptions getCompactionOptions() {
    return compactionOptions;
  }

  public List<CompactionTask> getCompactionTasks() {
    return tasks.stream()
        .map(task -> new CompactionTask(task, readVersion, compactionOptions))
        .collect(Collectors.toList());
  }
}
