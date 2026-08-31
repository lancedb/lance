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
package org.lance.fragment;

import java.io.Serializable;
import java.util.List;
import java.util.Objects;

/** Sparse mappings from local blob identifiers to reused physical sidecars. */
public final class BlobReuseIndex implements Serializable {
  private static final long serialVersionUID = 1L;

  private final List<BlobReuseSource> sources;

  /** Creates an immutable reuse index from its source mappings. */
  public BlobReuseIndex(List<BlobReuseSource> sources) {
    this.sources = List.copyOf(Objects.requireNonNull(sources, "sources"));
  }

  /** Returns the immutable source mappings. */
  public List<BlobReuseSource> getSources() {
    return sources;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    BlobReuseIndex that = (BlobReuseIndex) o;
    return Objects.equals(sources, that.sources);
  }

  @Override
  public int hashCode() {
    return Objects.hash(sources);
  }
}
