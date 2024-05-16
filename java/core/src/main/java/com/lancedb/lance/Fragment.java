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

import java.util.Optional;
import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;

/** Fragment operations. */
public class Fragment {
  static {
    JniLoader.ensureLoaded();
  }

  /** Create a fragment from the given data in vector schema root. */
  public static FragmentMetadata create(String datasetUri, BufferAllocator allocator,
      VectorSchemaRoot root, Optional<Integer> fragmentId, WriteParams params) {
    try (ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator);
         ArrowArray arrowArray = ArrowArray.allocateNew(allocator)) {
      Data.exportVectorSchemaRoot(allocator, root, null, arrowArray, arrowSchema);
      return FragmentMetadata.fromJson(createWithFfiArray(datasetUri, arrowArray.memoryAddress(),
          arrowSchema.memoryAddress(), fragmentId, params.getMaxRowsPerFile(),
          params.getMaxRowsPerGroup(), params.getMaxBytesPerFile(), params.getMode()));
    }
  }

  /** Create a fragment from the given data. */
  public static FragmentMetadata create(String datasetUri, ArrowArrayStream stream,
      Optional<Integer> fragmentId, WriteParams params) {
    return FragmentMetadata.fromJson(createWithFfiStream(datasetUri,
        stream.memoryAddress(), fragmentId,
        params.getMaxRowsPerFile(), params.getMaxRowsPerGroup(),
        params.getMaxBytesPerFile(), params.getMode()));
  }

  /**
   * Create a fragment from the given arrow array and schema.
   *
   * @return the json serialized fragment metadata
   */
  private static native String createWithFfiArray(String datasetUri,
      long arrowArrayMemoryAddress, long arrowSchemaMemoryAddress, Optional<Integer> fragmentId,
      Optional<Integer> maxRowsPerFile, Optional<Integer> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile, Optional<String> mode);

  /**
   * Create a fragment from the given arrow stream.
   *
   * @return the json serialized fragment metadata
   */
  private static native String createWithFfiStream(String datasetUri, long arrowStreamMemoryAddress,
      Optional<Integer> fragmentId, Optional<Integer> maxRowsPerFile,
      Optional<Integer> maxRowsPerGroup, Optional<Long> maxBytesPerFile,
      Optional<String> mode);
}
