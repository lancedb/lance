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

import java.time.ZonedDateTime;
import java.util.TreeMap;

public class Version {
  private final long id;
  private final ZonedDateTime dataTime;
  private final TreeMap<String, String> metadata;

  public Version(long id, ZonedDateTime dataTime, TreeMap<String, String> metadata) {
    this.id = id;
    this.dataTime = dataTime;
    this.metadata = metadata;
  }

  public ZonedDateTime getDataTime() {
    return dataTime;
  }

  public TreeMap<String, String> getMetadata() {
    return metadata;
  }

  public long getId() {
    return id;
  }

  public String toString() {
    return "Version: " + id + ", dataTime: " + dataTime + ", metadata: " + metadata;
  }

  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    Version version = (Version) o;
    return id == version.id
        && dataTime.isEqual(version.dataTime)
        && metadata.equals(version.metadata);
  }

  public int hashCode() {
    return (int) (id ^ dataTime.hashCode());
  }
}
