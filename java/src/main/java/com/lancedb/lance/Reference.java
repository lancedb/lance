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

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;

import java.util.Optional;

public class Reference {

  private final Optional<Long> versionNumber;
  private final Optional<String> branchName;
  private final Optional<String> tagName;

  public Reference(
      Optional<Long> versionNumber, Optional<String> branchName, Optional<String> tagName) {
    this.versionNumber = versionNumber;
    this.branchName = branchName;
    this.tagName = tagName;
  }

  public Optional<Long> getVersionNumber() {
    return versionNumber;
  }

  public Optional<String> getBranchName() {
    return branchName;
  }

  public Optional<String> getTagName() {
    return tagName;
  }

  public static Builder builder() {
    return new Builder();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("versionNumber", versionNumber.orElse(null))
        .add("branchName", branchName.orElse(null))
        .add("tagName", tagName.orElse(null))
        .toString();
  }

  public static class Builder {
    private Long versionNumber;
    private String branchName;
    private String tagName;

    private Builder() {}

    public Builder versionNumber(long versionNumber) {
      this.versionNumber = versionNumber;
      return this;
    }

    public Builder branchName(String branchName) {
      this.branchName = branchName;
      return this;
    }

    public Builder tagName(String tagName) {
      this.tagName = tagName;
      return this;
    }

    public Reference build() {
      Preconditions.checkArgument(
          (tagName == null && (versionNumber != null || branchName != null))
              || (tagName != null && (versionNumber == null && branchName == null)),
          "Invalid parameters: either specify tagName alone, or versionNumber/branchName together");
      return new Reference(
          Optional.ofNullable(versionNumber),
          Optional.ofNullable(branchName),
          Optional.ofNullable(tagName));
    }
  }
}
