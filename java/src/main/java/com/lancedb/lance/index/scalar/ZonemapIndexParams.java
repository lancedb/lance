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
package com.lancedb.lance.index.scalar;

import com.google.common.base.MoreObjects;

/** Parameters for creating Zonemap scalar indices. */
public class ZonemapIndexParams {

  private ZonemapIndexParams(Builder builder) {
    // Currently no specific parameters for zonemap index
  }

  public static class Builder {
    public Builder() {}

    public ZonemapIndexParams build() {
      return new ZonemapIndexParams(this);
    }
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).toString();
  }
}
