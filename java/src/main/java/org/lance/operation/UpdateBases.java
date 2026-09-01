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
package org.lance.operation;

import org.lance.BasePath;

import com.google.common.base.MoreObjects;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** Add base paths to the dataset manifest. */
public final class UpdateBases implements Operation {
  private final List<BasePath> newBases;

  private UpdateBases(List<BasePath> newBases) {
    this.newBases = Objects.requireNonNull(newBases);
  }

  public List<BasePath> getNewBases() {
    return newBases;
  }

  @Override
  public String name() {
    return "UpdateBases";
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("newBases", newBases).toString();
  }

  public static Builder builder() {
    return new Builder();
  }

  public static final class Builder {
    private List<BasePath> newBases = Collections.emptyList();

    public Builder newBases(List<BasePath> newBases) {
      this.newBases = newBases;
      return this;
    }

    public UpdateBases build() {
      return new UpdateBases(newBases);
    }
  }
}
