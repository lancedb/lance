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

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.types.pojo.Schema;

/** Fragment related operations. */
public abstract class FragmentOperation {
  protected static void validateFragments(List<FragmentMetadata> fragments) {
    if (fragments == null || fragments.isEmpty()) {
      throw new IllegalArgumentException("fragments cannot be null or empty");
    }
  }

  public abstract Dataset commit(BufferAllocator allocator, String path,
      Optional<Integer> readVersion);

  /** Fragment append operation. */
  public static class Append extends FragmentOperation {
    private final List<FragmentMetadata> fragments;

    public Append(List<FragmentMetadata> fragments) {
      validateFragments(fragments);
      this.fragments = fragments;
    }

    @Override
    public Dataset commit(BufferAllocator allocator, String path, Optional<Integer> readVersion) {
      return Dataset.commitAppend(path, readVersion,
          fragments.stream().map(FragmentMetadata::getJsonMetadata).collect(Collectors.toList()));
    }
  }
}
