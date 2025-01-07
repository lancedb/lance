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
package com.lancedb.lance.ipc;

import java.io.Serializable;

public class FieldStatistics implements Serializable {
  private final int id;
  // The size of the data in bytes
  private final long dataSize;

  public FieldStatistics(int id, long dataSize) {
    this.id = id;
    this.dataSize = dataSize;
  }

  public int getId() {
    return id;
  }

  public long getDataSize() {
    return dataSize;
  }

  @Override
  public String toString() {
    return "FieldStatistics{" + "id=" + id + ", dataSize=" + dataSize + '}';
  }
}
