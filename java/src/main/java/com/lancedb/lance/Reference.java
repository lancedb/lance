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

  public Optional<Long> versionNumber() {
    return versionNumber;
  }

  public Optional<String> branchName() {
    return branchName;
  }

  public Optional<String> tagName() {
    return tagName;
  }

  public static Reference ofMain(long versionNumber) {
    return new Reference(Optional.of(versionNumber), Optional.empty(), Optional.empty());
  }

  public static Reference ofBranch(String branchName) {
    return new Reference(Optional.empty(), Optional.of(branchName), Optional.empty());
  }

  public static Reference ofBranch(String branchName, long versionNumber) {
    return new Reference(Optional.of(versionNumber), Optional.of(branchName), Optional.empty());
  }

  public static Reference ofTag(String tagName) {
    return new Reference(Optional.empty(), Optional.empty(), Optional.of(tagName));
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("versionNumber", versionNumber.orElse(null))
        .add("branchName", branchName.orElse(null))
        .add("tagName", tagName.orElse(null))
        .toString();
  }
}
