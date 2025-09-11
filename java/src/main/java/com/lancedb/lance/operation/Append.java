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

import com.lancedb.lance.FragmentMetadata;

import com.google.common.base.MoreObjects;
import org.apache.arrow.util.Preconditions;

import java.util.List;
import java.util.Objects;

public class Append implements Operation {
  private final List<FragmentMetadata> fragments;

  private Append(List<FragmentMetadata> fragments) {
    Preconditions.checkArgument(
        fragments != null && !fragments.isEmpty(), "fragments cannot be null or empty");
    this.fragments = fragments;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    Append that = (Append) o;
    return Objects.equals(fragments, that.fragments);
  }

  public List<FragmentMetadata> fragments() {
    return fragments;
  }

  @Override
  public String name() {
    return "Append";
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("fragments", fragments).toString();
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {
    private List<FragmentMetadata> fragments;

    public Builder() {}

    public Builder fragments(List<FragmentMetadata> fragments) {
      this.fragments = fragments;
      return this;
    }

    public Append build() {
      return new Append(fragments);
    }
  }
}
