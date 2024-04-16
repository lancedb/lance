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

  /** Fragment delete operation. */
  public static class Delete extends FragmentOperation {
    private final List<FragmentMetadata> updatedFragments;
    private final List<FragmentMetadata> deletedFragments;
    private final String predicate;

    /**
     * Delete fragments.
     *
     * @param updatedFragments updated fragments
     * @param deletedFragments deleted fragments
     * @param predicate A SQL predicate that specifies the rows to delete.
     */
    public Delete(List<FragmentMetadata> updatedFragments,
        List<FragmentMetadata> deletedFragments, String predicate) {
      validateFragments(updatedFragments);
      validateFragments(deletedFragments);
      this.updatedFragments = updatedFragments;
      this.deletedFragments = deletedFragments;
      this.predicate = predicate;
    }

    @Override
    public Dataset commit(BufferAllocator allocator, String path, Optional<Integer> readVersion) {
      throw new UnsupportedOperationException();
    }
  }

  /** Fragment merge operation. */
  public static class Merge extends FragmentOperation {
    private final List<FragmentMetadata> fragments;
    private final Schema schema;

    /**
     * Merge fragments.
     *
     * @param fragments fragments to merge
     * @param schema schema
     */
    public Merge(List<FragmentMetadata> fragments, Schema schema) {
      validateFragments(fragments);
      this.fragments = fragments;
      this.schema = schema;
    }

    @Override
    public Dataset commit(BufferAllocator allocator, String path, Optional<Integer> readVersion) {
      throw new UnsupportedOperationException();
    }
  }

  /** Fragment overwrite operation. */
  public static class Overwrite extends FragmentOperation {
    private final List<FragmentMetadata> fragments;
    private final Schema newSchema;

    /**
     * Overwrite fragments.
     *
     * @param fragments fragments to overwrite
     * @param newSchema new schema
     */
    public Overwrite(List<FragmentMetadata> fragments, Schema newSchema) {
      validateFragments(fragments);
      this.fragments = fragments;
      this.newSchema = newSchema;
    }

    @Override
    public Dataset commit(BufferAllocator allocator, String path, Optional<Integer> readVersion) {
      throw new UnsupportedOperationException();
    }
  }

  /** Fragment restore operation. */
  public static class Restore extends FragmentOperation {
    private final int version;

    /**
     * Fragments to restore.
     *
     * @param version the version to restore to
     */
    public Restore(int version) {
      this.version = version;
    }

    @Override
    public Dataset commit(BufferAllocator allocator, String path, Optional<Integer> readVersion) {
      throw new UnsupportedOperationException();
    }
  }
}
