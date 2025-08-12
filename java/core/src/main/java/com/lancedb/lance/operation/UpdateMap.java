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
package com.lancedb.lance.operation;

import java.util.Map;

/** Represents updates to a metadata map where null values indicate deletion. */
public class UpdateMap {

  private final Map<String, String> updates;
  private final boolean replace;

  private UpdateMap(Map<String, String> updates, boolean replace) {
    this.updates = updates;
    this.replace = replace;
  }

  public Map<String, String> updates() {
    return updates;
  }

  public boolean replace() {
    return replace;
  }

  @Override
  public String toString() {
    return "UpdateMap{" + "updates=" + updates + ", replace=" + replace + '}';
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {
    private Map<String, String> updates;
    private boolean replace = false;

    public Builder() {}

    public Builder updates(Map<String, String> updates) {
      this.updates = updates;
      return this;
    }

    public Builder replace(boolean replace) {
      this.replace = replace;
      return this;
    }

    public UpdateMap build() {
      return new UpdateMap(updates, replace);
    }
  }
}
