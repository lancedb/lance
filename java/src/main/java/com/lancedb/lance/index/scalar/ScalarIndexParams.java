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

import java.util.Optional;

/** Parameters for creating scalar indices. */
public class ScalarIndexParams {
  private final Optional<BTreeIndexParams> btreeParams;
  private final Optional<ZonemapIndexParams> zonemapParams;

  private ScalarIndexParams(Builder builder) {
    this.btreeParams = builder.btreeParams;
    this.zonemapParams = builder.zonemapParams;
  }

  public static class Builder {
    private Optional<BTreeIndexParams> btreeParams = Optional.empty();
    private Optional<ZonemapIndexParams> zonemapParams = Optional.empty();

    public Builder() {}

    /**
     * Set B-Tree index parameters.
     *
     * @param btreeParams B-Tree index parameters
     * @return this builder
     */
    public Builder setBTreeParams(BTreeIndexParams btreeParams) {
      this.btreeParams = Optional.of(btreeParams);
      return this;
    }

    /**
     * Set Zonemap index parameters.
     *
     * @param zonemapParams Zonemap index parameters
     * @return this builder
     */
    public Builder setZonemapParams(ZonemapIndexParams zonemapParams) {
      this.zonemapParams = Optional.of(zonemapParams);
      return this;
    }

    public ScalarIndexParams build() {
      return new ScalarIndexParams(this);
    }
  }

  public Optional<BTreeIndexParams> getBTreeParams() {
    return btreeParams;
  }

  public Optional<ZonemapIndexParams> getZonemapParams() {
    return zonemapParams;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("btreeParams", btreeParams.orElse(null))
        .add("zonemapParams", zonemapParams.orElse(null))
        .toString();
  }
}
