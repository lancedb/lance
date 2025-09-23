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

import com.google.common.base.MoreObjects;

/** ReserveFragments operation to reserve fragment IDs for future use. */
public class ReserveFragments implements Operation {

  private final int numFragments;

  private ReserveFragments(int numFragments) {
    this.numFragments = numFragments;
  }

  @Override
  public String name() {
    return "ReserveFragments";
  }

  /**
   * Get the number of fragments to reserve.
   *
   * @return the number of fragments
   */
  public int numFragments() {
    return numFragments;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("numFragments", numFragments).toString();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    ReserveFragments that = (ReserveFragments) o;
    return numFragments == that.numFragments;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {
    private int numFragments;

    public Builder() {}

    public Builder numFragments(int numFragments) {
      this.numFragments = numFragments;
      return this;
    }

    public ReserveFragments build() {
      return new ReserveFragments(numFragments);
    }
  }
}
