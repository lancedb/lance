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
package org.lance.memwal;

import org.lance.JniLoader;
import org.lance.schema.LanceField;
import org.lance.schema.LanceSchema;

import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/** Evaluates MemWAL sharding specs against Arrow record batches. */
public final class ShardingEvaluator {
  static {
    JniLoader.ensureLoaded();
  }

  private ShardingEvaluator() {}

  /**
   * Evaluate {@code spec} against {@code root}.
   *
   * @param allocator allocator used for Arrow C data interface structs
   * @param root input record batch
   * @param spec MemWAL sharding spec to evaluate
   * @param schema Lance table schema used to resolve spec source field IDs to input column names
   * @return an Arrow reader containing one result batch with the derived sharding fields
   */
  public static ArrowReader evaluate(
      BufferAllocator allocator, VectorSchemaRoot root, ShardingSpec spec, LanceSchema schema) {
    Preconditions.checkNotNull(allocator, "allocator must not be null");
    Preconditions.checkNotNull(root, "root must not be null");
    Preconditions.checkNotNull(spec, "spec must not be null");
    Preconditions.checkNotNull(schema, "schema must not be null");

    try (ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator);
        ArrowArray arrowArray = ArrowArray.allocateNew(allocator);
        ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
      Data.exportVectorSchemaRoot(allocator, root, null, arrowArray, arrowSchema);
      nativeEvaluate(
          arrowArray.memoryAddress(),
          arrowSchema.memoryAddress(),
          spec,
          sourceIdToColumnMap(schema),
          stream.memoryAddress());
      return Data.importArrayStream(allocator, stream);
    }
  }

  /** Evaluate {@code spec} against {@code root} when the spec embeds enough column information. */
  public static ArrowReader evaluate(
      BufferAllocator allocator, VectorSchemaRoot root, ShardingSpec spec) {
    return evaluate(allocator, root, spec, Collections.emptyMap());
  }

  private static ArrowReader evaluate(
      BufferAllocator allocator,
      VectorSchemaRoot root,
      ShardingSpec spec,
      Map<Integer, String> sourceIdToColumn) {
    Preconditions.checkNotNull(allocator, "allocator must not be null");
    Preconditions.checkNotNull(root, "root must not be null");
    Preconditions.checkNotNull(spec, "spec must not be null");
    Preconditions.checkNotNull(sourceIdToColumn, "sourceIdToColumn must not be null");

    try (ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator);
        ArrowArray arrowArray = ArrowArray.allocateNew(allocator);
        ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
      Data.exportVectorSchemaRoot(allocator, root, null, arrowArray, arrowSchema);
      nativeEvaluate(
          arrowArray.memoryAddress(),
          arrowSchema.memoryAddress(),
          spec,
          sourceIdToColumn,
          stream.memoryAddress());
      return Data.importArrayStream(allocator, stream);
    }
  }

  private static Map<Integer, String> sourceIdToColumnMap(LanceSchema schema) {
    Map<Integer, String> result = new HashMap<>();
    for (LanceField field : schema.fields()) {
      collectFieldIds(field, "", result);
    }
    return result;
  }

  private static void collectFieldIds(
      LanceField field, String prefix, Map<Integer, String> result) {
    String fullName = prefix.isEmpty() ? field.getName() : prefix + "." + field.getName();
    result.put(field.getId(), fullName);
    for (LanceField child : field.getChildren()) {
      collectFieldIds(child, fullName, result);
    }
  }

  private static native void nativeEvaluate(
      long arrowArrayAddress,
      long arrowSchemaAddress,
      ShardingSpec spec,
      Map<Integer, String> sourceIdToColumn,
      long streamAddress);
}
